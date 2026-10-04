"""Steps 1-3: discover object IDs, fetch metadata into staging, transform in SQL."""
from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import islice

import psycopg
from psycopg.types.json import Jsonb

from .db import SQL_DIR, update_run
from .met_api import MetClient

log = logging.getLogger(__name__)

SAMPLE_SEED = 42

DEPT_STATE_DDL = """
CREATE TABLE IF NOT EXISTS department_state (
    department_id      INT PRIMARY KEY,
    max_seen_object_id INT NOT NULL,     -- high-water mark for "new accession" detection
    last_full_discovery TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_incremental    DATE
)"""


def load_departments(conn: psycopg.Connection, client: MetClient) -> None:
    with conn.cursor() as cur:
        for d in client.departments():
            cur.execute(
                """INSERT INTO stg_departments (department_id, payload) VALUES (%s, %s)
                   ON CONFLICT (department_id) DO UPDATE SET payload = EXCLUDED.payload, fetched_at = now()""",
                (d["departmentId"], Jsonb(d)),
            )
    conn.commit()


def _enqueue(conn: psycopg.Connection, ids: list[int], department_id: int, run_id: int) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO stg_object_ids (object_id, department_id, run_id, needs_fetch)
               VALUES (%s, %s, %s, TRUE)
               ON CONFLICT (object_id) DO UPDATE SET needs_fetch = TRUE, run_id = EXCLUDED.run_id""",
            [(i, department_id, run_id) for i in ids],
        )
    conn.commit()
    return len(ids)


def discover_full(conn: psycopg.Connection, client: MetClient, department_ids: list[int],
                  cap_per_department: int, run_id: int) -> int:
    """Pick up to `cap_per_department` objects that have images, per department.

    The /search endpoint pre-filters to objects with images, which avoids fetching
    thousands of image-less records. We take a seeded random sample rather than the
    first N IDs because IDs cluster by accession batch (one donor, one site, ...).
    Already-known objects are kept, so re-running is idempotent and never shrinks the set.
    """
    conn.execute(DEPT_STATE_DDL)
    total = 0
    for dept in department_ids:
        with_images = sorted(client.object_ids_with_images(dept))
        all_ids = client.object_ids(dept)
        known = {r[0] for r in conn.execute(
            "SELECT object_id FROM stg_object_ids WHERE department_id = %s", (dept,))}
        room = max(0, cap_per_department - len(known))
        candidates = [i for i in with_images if i not in known]
        sample = random.Random(SAMPLE_SEED + dept).sample(candidates, min(room, len(candidates)))
        log.info("dept %s: %d ids, %d with images, %d known, enqueueing %d",
                 dept, len(all_ids), len(with_images), len(known), len(sample))
        total += _enqueue(conn, sample, dept, run_id)
        conn.execute(
            """INSERT INTO department_state (department_id, max_seen_object_id) VALUES (%s, %s)
               ON CONFLICT (department_id) DO UPDATE
               SET max_seen_object_id = GREATEST(department_state.max_seen_object_id, EXCLUDED.max_seen_object_id),
                   last_full_discovery = now()""",
            (dept, max(all_ids, default=0)),
        )
        conn.commit()
    update_run(conn, run_id, ids_discovered=total)
    return total


def discover_incremental(conn: psycopg.Connection, client: MetClient, department_ids: list[int],
                         since: date, run_id: int, max_new_per_department: int = 2000) -> int:
    """Queue (a) objects we already hold that the Met changed since `since`,
    (b) brand-new accessions: IDs above the department's high-water mark, and
    (c) objects we hold that vanished from the department's ID list (withdrawn;
    the re-fetch 404s and the transform deletes them)."""
    conn.execute(DEPT_STATE_DDL)
    total = 0
    for dept in department_ids:
        changed = client.object_ids(dept, updated_since=since)
        row = conn.execute("SELECT max_seen_object_id FROM department_state WHERE department_id = %s",
                           (dept,)).fetchone()
        high_water = row[0] if row else 0
        known = {r[0] for r in conn.execute(
            "SELECT object_id FROM stg_object_ids WHERE department_id = %s", (dept,))}
        current = set(client.object_ids(dept))
        refresh = [i for i in changed if i in known]
        new = sorted(i for i in changed if i > high_water and i not in known)[:max_new_per_department]
        withdrawn = sorted(known - current) if current else []  # empty list = API hiccup, skip
        log.info("dept %s since %s: %d changed, %d refresh, %d new, %d withdrawn", dept, since,
                 len(changed), len(refresh), len(new), len(withdrawn))
        total += _enqueue(conn, refresh + new + withdrawn, dept, run_id)
        conn.execute(
            """INSERT INTO department_state (department_id, max_seen_object_id, last_incremental)
               VALUES (%s, %s, %s)
               ON CONFLICT (department_id) DO UPDATE
               SET max_seen_object_id = GREATEST(department_state.max_seen_object_id, EXCLUDED.max_seen_object_id),
                   last_incremental = EXCLUDED.last_incremental""",
            (dept, max(changed + [high_water]), date.today()),
        )
        conn.commit()
    update_run(conn, run_id, ids_discovered=total)
    return total


def last_successful_run_date(conn: psycopg.Connection) -> date | None:
    row = conn.execute(
        "SELECT max(started_at)::date FROM ingest_runs WHERE status = 'succeeded'").fetchone()
    return row[0] if row else None


def _batched(it, n):
    it = iter(it)
    while chunk := list(islice(it, n)):
        yield chunk


def fetch_metadata(conn: psycopg.Connection, client: MetClient, run_id: int,
                   workers: int = 8, batch_size: int = 200, limit: int | None = None) -> int:
    """Fetch every queued object into stg_objects_raw.

    Resumable: the cursor is `stg_object_ids.needs_fetch`, flipped to FALSE in the
    same transaction that stores the response. A crash loses at most one batch.
    """
    pending = [r[0] for r in conn.execute(
        "SELECT object_id FROM stg_object_ids WHERE needs_fetch ORDER BY object_id"
        + (" LIMIT %s" % int(limit) if limit else ""))]
    log.info("fetching %d objects (%d workers, %.0f req/s)", len(pending), workers, client.limiter.rate)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in _batched(pending, batch_size):
            results = list(pool.map(lambda oid: (oid, *client.get_object(oid)), batch))
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO stg_objects_raw (object_id, http_status, payload, run_id, fetched_at)
                       VALUES (%s, %s, %s, %s, now())
                       ON CONFLICT (object_id) DO UPDATE SET http_status = EXCLUDED.http_status,
                           payload = EXCLUDED.payload, run_id = EXCLUDED.run_id, fetched_at = now()""",
                    [(oid, status, Jsonb(p) if p is not None else None, run_id) for oid, status, p in results],
                )
                cur.execute("UPDATE stg_object_ids SET needs_fetch = FALSE WHERE object_id = ANY(%s)", (batch,))
                cur.execute("UPDATE ingest_runs SET objects_fetched = objects_fetched + %s WHERE run_id = %s",
                            (len(batch), run_id))
            done += len(batch)
            log.info("fetched %d/%d", done, len(pending))
    return done


def transform(conn: psycopg.Connection) -> dict:
    """Staging -> core tables in a single transaction."""
    conn.commit()  # so the block below is a real transaction, not a savepoint
    with conn.transaction():
        conn.execute((SQL_DIR / "transform.sql").read_text())
    counts = conn.execute(
        """SELECT (SELECT count(*) FROM stg_objects_raw),
                  (SELECT count(*) FROM objects),
                  (SELECT count(*) FROM tags),
                  (SELECT count(*) FROM object_tags)""").fetchone()
    return dict(zip(["staged", "objects", "tags", "object_tags"], counts))
