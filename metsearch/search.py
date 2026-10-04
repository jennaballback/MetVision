"""Vector + hybrid retrieval over pgvector.

Ranking is done entirely in SQL/numpy; the LLM only ever produces `Filters`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import psycopg

from .embed import Embedder

RRF_K = 60  # standard reciprocal-rank-fusion constant
CANDIDATES = 100


@dataclass
class Filters:
    departments: list[str] = field(default_factory=list)   # department names, exact
    cultures: list[str] = field(default_factory=list)       # substring match, case-insensitive
    classifications: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    keywords: list[str] = field(default_factory=list)       # must appear in title/medium/tags text

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def sql(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.departments:
            clauses.append("d.name = ANY(%s)")
            params.append(self.departments)
        if self.cultures:
            clauses.append("o.culture ILIKE ANY(%s)")
            params.append([f"%{c}%" for c in self.cultures])
        if self.classifications:
            clauses.append("o.classification ILIKE ANY(%s)")
            params.append([f"%{c}%" for c in self.classifications])
        # Date ranges overlap: an object made 1790-1810 matches "1800s".
        if self.year_from is not None:
            clauses.append("o.end_year >= %s")
            params.append(self.year_from)
        if self.year_to is not None:
            clauses.append("o.begin_year <= %s")
            params.append(self.year_to)
        for kw in self.keywords:
            clauses.append("(o.search_tsv @@ plainto_tsquery('english', %s) OR EXISTS ("
                           "SELECT 1 FROM object_tags ot JOIN tags t USING (tag_id) "
                           "WHERE ot.object_id = o.object_id AND t.term ILIKE %s))")
            params += [kw, f"%{kw}%"]
        return (" AND ".join(clauses) or "TRUE"), params


RESULT_COLS = """o.object_id, o.title, o.artist, d.name AS department, o.object_date, o.begin_year,
    o.end_year, o.culture, o.period, o.medium, o.classification, o.image_small_url, o.object_url"""


def _prepare(conn: psycopg.Connection, filtered: bool) -> None:
    # Higher ef_search keeps recall up when a WHERE clause discards most HNSW candidates.
    conn.execute("SET hnsw.ef_search = %d" % (400 if filtered else 100))
    try:  # pgvector >= 0.8: keep scanning the graph until enough rows pass the filter
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except psycopg.Error:
        conn.rollback()


def _ranked(conn, column: str, vec: np.ndarray, filters: Filters, limit: int,
            exclude: int | None) -> list[int]:
    where, params = filters.sql()
    if exclude is not None:
        where += " AND o.object_id <> %s"
        params.append(exclude)
    rows = conn.execute(
        f"""SELECT o.object_id FROM object_embeddings e
            JOIN objects o USING (object_id) LEFT JOIN departments d USING (department_id)
            WHERE {where} AND e.{column} IS NOT NULL
            ORDER BY e.{column} <=> %s LIMIT %s""",
        (*params, vec, limit)).fetchall()
    return [r[0] for r in rows]


def _keyword_ranked(conn, text: str, filters: Filters, limit: int) -> list[int]:
    where, params = filters.sql()
    rows = conn.execute(
        f"""SELECT o.object_id FROM objects o LEFT JOIN departments d USING (department_id)
            JOIN object_embeddings e USING (object_id)
            WHERE {where} AND o.search_tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(o.search_tsv, plainto_tsquery('english', %s)) DESC LIMIT %s""",
        (*params, text, text, limit)).fetchall()
    return [r[0] for r in rows]


def rrf(*rankings: list[int], weights: list[float] | None = None) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, oid in enumerate(ranking):
            scores[oid] = scores.get(oid, 0.0) + w / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def fetch_rows(conn, ids_scores: list[tuple[int, float]]) -> list[dict]:
    if not ids_scores:
        return []
    ids = [i for i, _ in ids_scores]
    cur = conn.execute(
        f"SELECT {RESULT_COLS} FROM objects o LEFT JOIN departments d USING (department_id) "
        "WHERE o.object_id = ANY(%s)", (ids,))
    cols = [c.name for c in cur.description]
    by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    return [{**by_id[i], "score": s} for i, s in ids_scores if i in by_id]


def similar_to_object(conn, object_id: int, k: int = 10, filters: Filters | None = None,
                      mode: str = "image") -> list[dict]:
    """'Find artworks like this'. mode: 'image' (pure visual) or 'hybrid'
    (visual neighbours fused with metadata-text neighbours)."""
    filters = filters or Filters()
    row = conn.execute("SELECT image_embedding, text_embedding FROM object_embeddings WHERE object_id = %s",
                       (object_id,)).fetchone()
    if row is None:
        raise KeyError(f"object {object_id} has no embedding")
    _prepare(conn, not filters.is_empty())
    if mode == "image":
        ids = _ranked(conn, "image_embedding", row[0], filters, k, object_id)
        return fetch_rows(conn, [(i, 1.0 / (r + 1)) for r, i in enumerate(ids)])
    by_image = _ranked(conn, "image_embedding", row[0], filters, CANDIDATES, object_id)
    by_text = _ranked(conn, "text_embedding", row[1], filters, CANDIDATES, object_id)
    return fetch_rows(conn, rrf(by_image, by_text, weights=[1.0, 0.6])[:k])


def search_by_image(conn, vec: np.ndarray, k: int = 10, filters: Filters | None = None) -> list[dict]:
    """Uploaded photo -> nearest artworks by image embedding."""
    filters = filters or Filters()
    _prepare(conn, not filters.is_empty())
    ids = _ranked(conn, "image_embedding", vec, filters, k, None)
    return fetch_rows(conn, [(i, 1.0 / (r + 1)) for r, i in enumerate(ids)])


def search_by_text(conn, text: str, embedder: Embedder, k: int = 10,
                   filters: Filters | None = None) -> list[dict]:
    """Hybrid text search: CLIP text->image, CLIP text->metadata-text, and Postgres
    full-text keyword rank, fused with reciprocal rank fusion."""
    filters = filters or Filters()
    _prepare(conn, not filters.is_empty())
    if not text.strip():  # filters only: order by visual "typicality" is meaningless, use id
        where, params = filters.sql()
        ids = [r[0] for r in conn.execute(
            f"""SELECT o.object_id FROM objects o LEFT JOIN departments d USING (department_id)
                JOIN object_embeddings e USING (object_id) WHERE {where}
                ORDER BY o.object_id LIMIT %s""", (*params, k))]
        return fetch_rows(conn, [(i, 0.0) for i in ids])
    q = embedder.embed_texts([text])[0]
    by_image = _ranked(conn, "image_embedding", q, filters, CANDIDATES, None)
    by_text = _ranked(conn, "text_embedding", q, filters, CANDIDATES, None)
    by_kw = _keyword_ranked(conn, text, filters, CANDIDATES)
    return fetch_rows(conn, rrf(by_image, by_text, by_kw, weights=[1.0, 0.7, 0.5])[:k])
