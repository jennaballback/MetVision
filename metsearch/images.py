"""Step 4: download the small image for every object, politely, into a disk cache."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import psycopg
import requests
from PIL import Image

from .met_api import RateLimiter

log = logging.getLogger(__name__)

MAX_SIDE = 512  # CLIP only needs 224px; keep the cache small


def image_path(image_dir: Path, object_id: int) -> Path:
    return image_dir / f"{object_id // 1000:04d}" / f"{object_id}.jpg"


def download_images(conn: psycopg.Connection, image_dir: Path, rate_per_second: float = 5,
                    user_agent: str = "met-visual-search", workers: int = 4,
                    limit: int | None = None, run_id: int | None = None) -> dict:
    """Download images for objects with no successful object_images row.

    Skips anything already on disk (e.g. after a DB reset), so it is cheap to re-run.
    """
    rows = conn.execute(
        """SELECT o.object_id, o.image_small_url FROM objects o
           LEFT JOIN object_images i ON i.object_id = o.object_id AND i.status = 'ok'
           WHERE i.object_id IS NULL ORDER BY o.object_id"""
        + (" LIMIT %d" % int(limit) if limit else "")).fetchall()
    limiter = RateLimiter(rate_per_second)
    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    def fetch(row):
        object_id, url = row
        path = image_path(image_dir, object_id)
        if path.exists():
            return object_id, url, str(path), "ok", None
        try:
            limiter.acquire()
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            img.save(tmp, "JPEG", quality=88)
            tmp.rename(path)
            return object_id, url, str(path), "ok", None
        except Exception as exc:  # noqa: BLE001 - record and move on
            return object_id, url, None, "failed", f"{type(exc).__name__}: {exc}"[:500]

    stats = {"ok": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (object_id, url, local, status, err) in enumerate(pool.map(fetch, rows), 1):
            conn.execute(
                """INSERT INTO object_images (object_id, local_path, source_url, status, error)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (object_id) DO UPDATE SET local_path = EXCLUDED.local_path,
                     source_url = EXCLUDED.source_url, status = EXCLUDED.status,
                     error = EXCLUDED.error, downloaded_at = now()""",
                (object_id, local, url, status, err))
            stats[status] += 1
            if i % 100 == 0:
                conn.commit()
                log.info("images %d/%d (%d failed)", i, len(rows), stats["failed"])
    if run_id is not None:
        conn.execute("UPDATE ingest_runs SET images_downloaded = images_downloaded + %s WHERE run_id = %s",
                     (stats["ok"], run_id))
    conn.commit()
    return stats
