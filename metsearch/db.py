"""Postgres helpers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from .config import get_settings

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def connect(url: str | None = None, *, vector: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(url or get_settings().database_url)
    if vector:
        register_vector(conn)
    return conn


@contextmanager
def connection(url: str | None = None) -> Iterator[psycopg.Connection]:
    conn = connect(url)
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: psycopg.Connection) -> None:
    """Create the extension and all tables. Safe to run repeatedly."""
    conn.execute((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.commit()
    register_vector(conn)


def start_run(conn: psycopg.Connection, kind: str, since_date=None) -> int:
    run_id = conn.execute(
        "INSERT INTO ingest_runs (kind, since_date) VALUES (%s, %s) RETURNING run_id",
        (kind, since_date),
    ).fetchone()[0]
    conn.commit()
    return run_id


def update_run(conn: psycopg.Connection, run_id: int, **counts: int) -> None:
    if not counts:
        return
    sets = ", ".join(f"{k} = {k} + %s" for k in counts)
    conn.execute(f"UPDATE ingest_runs SET {sets} WHERE run_id = %s", (*counts.values(), run_id))
    conn.commit()


def finish_run(conn: psycopg.Connection, run_id: int, status: str, error: str | None = None) -> None:
    conn.rollback()  # discard any half-done transaction before recording the outcome
    conn.execute(
        "UPDATE ingest_runs SET finished_at = now(), status = %s, error = %s WHERE run_id = %s",
        (status, error, run_id),
    )
    conn.commit()
