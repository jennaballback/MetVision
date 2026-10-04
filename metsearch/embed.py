"""Step 5: CLIP embeddings for images and for a short text description of each object.

Image and text vectors live in the same CLIP space, so a typed query ("a woman
reading by a window") can be compared against image vectors directly.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Protocol

import numpy as np
import psycopg
from PIL import Image

log = logging.getLogger(__name__)

DIM = 512


class Embedder(Protocol):
    name: str

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class ClipEmbedder:
    """open_clip ViT-B/32 on CPU (or GPU if present): ~5-15 images/s on a laptop CPU."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"open_clip/{model_name}/{pretrained}"

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
            feats = self.model.encode_image(batch).float().cpu().numpy()
        return _normalise(feats)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)  # truncates to CLIP's 77 tokens
            feats = self.model.encode_text(tokens).float().cpu().numpy()
        return _normalise(feats)


class FakeEmbedder:
    """Deterministic, dependency-free stand-in used by tests and offline dev.

    Images: a downsampled colour thumbnail projected to 512-d, so visually similar
    images really are close. Texts: hashed bag of words. The two spaces are NOT
    aligned, so cross-modal search is meaningless with this embedder.
    """

    name = "fake/color-thumb+hashed-bow"

    def __init__(self):
        rng = np.random.default_rng(0)
        self.proj = rng.standard_normal((8 * 8 * 3, DIM)).astype(np.float32)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        thumbs = np.stack([
            np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).ravel() / 255.0 - 0.5
            for im in images])
        return _normalise(thumbs @ self.proj)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % DIM] += 1 if (h >> 9) & 1 else -1
        out[np.all(out == 0, axis=1), 0] = 1
        return _normalise(out)


@lru_cache(maxsize=1)
def get_embedder(kind: str = "clip", model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> Embedder:
    if kind == "fake":
        return FakeEmbedder()
    return ClipEmbedder(model_name, pretrained)


def object_text(title, object_name, medium, culture, period, classification, artist, tags) -> str:
    """The string that gets text-embedded. Most informative fields first, since
    CLIP truncates at 77 tokens."""
    parts = [title, object_name, culture, period, classification, medium, artist]
    text = ". ".join(p for p in parts if p)
    if tags:
        text += ". Tags: " + ", ".join(tags)
    return text or "artwork"


def embed_pending(conn: psycopg.Connection, embedder: Embedder, batch_size: int = 32,
                  limit: int | None = None, run_id: int | None = None) -> int:
    """Embed every object that has a downloaded image but no embedding yet."""
    rows = conn.execute(
        """SELECT o.object_id, i.local_path, o.title, o.object_name, o.medium, o.culture, o.period,
                  o.classification, o.artist,
                  coalesce(array_agg(t.term ORDER BY t.term) FILTER (WHERE t.term IS NOT NULL), '{}')
           FROM objects o
           JOIN object_images i ON i.object_id = o.object_id AND i.status = 'ok'
           LEFT JOIN object_embeddings e ON e.object_id = o.object_id
           LEFT JOIN object_tags ot ON ot.object_id = o.object_id
           LEFT JOIN tags t ON t.tag_id = ot.tag_id
           WHERE e.object_id IS NULL
           GROUP BY o.object_id, i.local_path ORDER BY o.object_id"""
        + (" LIMIT %d" % int(limit) if limit else "")).fetchall()
    log.info("embedding %d objects with %s", len(rows), embedder.name)
    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images, ok_rows = [], []
        for r in batch:
            try:
                with Image.open(r[1]) as im:
                    images.append(im.convert("RGB"))
                ok_rows.append(r)
            except Exception as exc:  # noqa: BLE001
                log.warning("unreadable image for %s: %s", r[0], exc)
                conn.execute("UPDATE object_images SET status='failed', error=%s WHERE object_id=%s",
                             (f"unreadable: {exc}"[:500], r[0]))
        if not ok_rows:
            continue
        texts = [object_text(*r[2:9], r[9]) for r in ok_rows]
        img_vecs = embedder.embed_images(images)
        txt_vecs = embedder.embed_texts(texts)
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO object_embeddings (object_id, image_embedding, text_embedding, text_source, model_name)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (object_id) DO UPDATE SET image_embedding = EXCLUDED.image_embedding,
                     text_embedding = EXCLUDED.text_embedding, text_source = EXCLUDED.text_source,
                     model_name = EXCLUDED.model_name, created_at = now()""",
                [(r[0], iv, tv, t, embedder.name) for r, iv, tv, t in zip(ok_rows, img_vecs, txt_vecs, texts)])
        if run_id is not None:
            conn.execute("UPDATE ingest_runs SET objects_embedded = objects_embedded + %s WHERE run_id = %s",
                         (len(ok_rows), run_id))
        conn.commit()
        done += len(ok_rows)
        log.info("embedded %d/%d", done, len(rows))
    return done
