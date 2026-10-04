"""The number: of each query object's 10 nearest neighbours, what fraction share its
department / culture / classification / period, versus 10 random objects?

Curator metadata is the ground truth. Queries whose own value is missing are
skipped for that attribute; a neighbour with a missing value counts as a miss.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from .search import _prepare

ATTRS = ["department", "culture", "classification", "period"]
SEED = 7


def _norm(v):
    return v.strip().lower() if isinstance(v, str) and v.strip() else None


def _load(conn) -> dict[int, dict]:
    cur = conn.execute(
        """SELECT o.object_id, d.name, o.culture, o.classification, o.period, o.title
           FROM object_embeddings e JOIN objects o USING (object_id)
           LEFT JOIN departments d USING (department_id)
           WHERE e.image_embedding IS NOT NULL""")
    return {r[0]: {"department": _norm(r[1]), "culture": _norm(r[2]), "classification": _norm(r[3]),
                   "period": _norm(r[4]), "title": r[5], "department_name": r[1]} for r in cur}


def _neighbours(conn, object_id: int, k: int, column: str) -> list[int]:
    return [r[0] for r in conn.execute(
        f"""SELECT e.object_id FROM object_embeddings e
            WHERE e.object_id <> %s AND e.{column} IS NOT NULL
            ORDER BY e.{column} <=> (SELECT {column} FROM object_embeddings WHERE object_id = %s)
            LIMIT %s""", (object_id, object_id, k))]


def _score(meta, query_id, neighbour_ids):
    q = meta[query_id]
    out = {}
    for a in ATTRS:
        if q[a] is None:
            continue
        out[a] = sum(meta[n][a] == q[a] for n in neighbour_ids) / len(neighbour_ids)
    out["any"] = sum(any(q[a] is not None and meta[n][a] == q[a] for a in ATTRS[:3])
                     for n in neighbour_ids) / len(neighbour_ids)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def run_eval(conn: psycopg.Connection, n_queries: int = 500, k: int = 10,
             column: str = "image_embedding", n_failures: int = 8) -> dict:
    meta = _load(conn)
    ids = sorted(meta)
    if len(ids) <= k:
        raise ValueError(f"need more than {k} embedded objects, have {len(ids)}")
    rng = random.Random(SEED)
    queries = rng.sample(ids, min(n_queries, len(ids)))
    _prepare(conn, filtered=False)

    model_scores, random_scores = defaultdict(list), defaultdict(list)
    per_dept = defaultdict(lambda: defaultdict(list))
    rows = []
    for qid in queries:
        nn = _neighbours(conn, qid, k, column)
        rnd = [i for i in rng.sample(ids, k + 1) if i != qid][:k]
        s_model, s_rand = _score(meta, qid, nn), _score(meta, qid, rnd)
        for a, v in s_model.items():
            model_scores[a].append(v)
            per_dept[meta[qid]["department_name"]][a].append(v)
        for a, v in s_rand.items():
            random_scores[a].append(v)
        rows.append({"query": qid, "neighbours": nn, **{f"model_{a}": v for a, v in s_model.items()}})

    summary = {}
    for a in ATTRS + ["any"]:
        m, r = _mean(model_scores[a]), _mean(random_scores[a])
        summary[a] = {"model": m, "random": r, "lift": (m / r) if m is not None and r else None,
                      "n_queries": len(model_scores[a])}

    # Failure cases: queries whose neighbours least often share any curator attribute.
    worst = sorted(rows, key=lambda r: (r["model_any"], r.get("model_classification", 1)))[:n_failures]
    failures = [{
        "query": {"object_id": r["query"], "title": meta[r["query"]]["title"],
                  **{a: meta[r["query"]][a] for a in ATTRS}},
        "match_any": r["model_any"],
        "neighbours": [{"object_id": n, "title": meta[n]["title"], **{a: meta[n][a] for a in ATTRS}}
                       for n in r["neighbours"]],
    } for r in worst]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": column, "k": k, "n_queries": len(queries), "pool_size": len(ids),
        "model_name": conn.execute("SELECT max(model_name) FROM object_embeddings").fetchone()[0],
        "summary": summary,
        "per_department": {d: {a: _mean(v) for a, v in s.items()} for d, s in per_dept.items()},
        "failures": failures,
    }


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(result: dict) -> str:
    lines = [
        f"# Neighbour-match evaluation ({result['embedding']})",
        "",
        f"{result['n_queries']} query objects from a pool of {result['pool_size']}; "
        f"k = {result['k']}; model `{result['model_name']}`; generated {result['generated_at']}.",
        "",
        "| Attribute | Model | Random | Lift | Queries with value |",
        "|---|---|---|---|---|",
    ]
    for a, s in result["summary"].items():
        lift = "n/a" if s["lift"] is None else f"{s['lift']:.1f}x"
        lines.append(f"| {a} | {_pct(s['model'])} | {_pct(s['random'])} | {lift} | {s['n_queries']} |")
    lines += ["", "`any` = neighbour shares at least one of department, culture, classification.", "",
              "## By department (model)", "", "| Department | " + " | ".join(ATTRS) + " |",
              "|---|" + "---|" * len(ATTRS)]
    for d, s in sorted(result["per_department"].items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {d} | " + " | ".join(_pct(s.get(a)) for a in ATTRS) + " |")
    lines += ["", "## Worst queries", ""]
    for f in result["failures"]:
        q = f["query"]
        lines.append(f"- **{q['title']}** ({q['object_id']}; {q['culture'] or '?'}, {q['classification'] or '?'}): "
                     f"{_pct(f['match_any'])} of neighbours share anything. Neighbours: "
                     + "; ".join(f"{n['title']} ({n['classification'] or '?'})" for n in f["neighbours"][:5]))
    return "\n".join(lines) + "\n"


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "results_" + result["embedding"].replace("_embedding", "")
    jp, mp = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jp.write_text(json.dumps(result, indent=2, default=str))
    mp.write_text(to_markdown(result))
    return jp, mp
