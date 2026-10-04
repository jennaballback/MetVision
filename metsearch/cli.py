"""Command line entry point: `python -m metsearch <command>`."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from . import db, evaluate, images, ingest
from .config import get_settings
from .embed import embed_pending, get_embedder
from .met_api import MetClient

log = logging.getLogger("metsearch")


def _client(s):
    return MetClient(s.met_api_base, s.requests_per_second, s.user_agent)


def _embedder(s):
    return get_embedder(s.embedder, s.clip_model, s.clip_pretrained)


def cmd_init_db(args, s, conn):
    print("schema ready")


def _pipeline(args, s, conn, kind: str):
    """Discover -> fetch -> transform -> images -> embed, recorded as one ingest run."""
    since = None
    if kind == "incremental":
        since = args.since or ((ingest.last_successful_run_date(conn) or date.today()) - timedelta(days=1))
    run_id = db.start_run(conn, kind, since)
    try:
        client = _client(s)
        ingest.load_departments(conn, client)
        if kind == "full":
            n = ingest.discover_full(conn, client, s.departments, args.cap or s.max_objects_per_department, run_id)
        else:
            n = ingest.discover_incremental(conn, client, s.departments, since, run_id)
        log.info("queued %d objects", n)
        ingest.fetch_metadata(conn, client, run_id, limit=args.limit)
        log.info("transform: %s", ingest.transform(conn))
        if not args.skip_images:
            log.info("images: %s", images.download_images(
                conn, s.image_dir, s.image_requests_per_second, s.user_agent, limit=args.limit, run_id=run_id))
            if not args.skip_embed:
                log.info("embedded %d", embed_pending(conn, _embedder(s), limit=args.limit, run_id=run_id))
        db.finish_run(conn, run_id, "succeeded")
    except BaseException as exc:
        db.finish_run(conn, run_id, "failed", repr(exc)[:2000])
        raise
    print(conn.execute("SELECT * FROM ingest_runs WHERE run_id = %s", (run_id,)).fetchone())


def cmd_run(args, s, conn):
    _pipeline(args, s, conn, "full")


def cmd_incremental(args, s, conn):
    _pipeline(args, s, conn, "incremental")


def cmd_discover(args, s, conn):
    run_id = db.start_run(conn, "full")
    client = _client(s)
    ingest.load_departments(conn, client)
    n = ingest.discover_full(conn, client, s.departments, args.cap or s.max_objects_per_department, run_id)
    db.finish_run(conn, run_id, "succeeded")
    print(f"queued {n} ids")


def cmd_fetch(args, s, conn):
    run_id = db.start_run(conn, "full")
    n = ingest.fetch_metadata(conn, _client(s), run_id, limit=args.limit)
    db.finish_run(conn, run_id, "succeeded")
    print(f"fetched {n}")


def cmd_transform(args, s, conn):
    print(ingest.transform(conn))


def cmd_images(args, s, conn):
    print(images.download_images(conn, s.image_dir, s.image_requests_per_second, s.user_agent, limit=args.limit))


def cmd_embed(args, s, conn):
    print(f"embedded {embed_pending(conn, _embedder(s), limit=args.limit)}")


def cmd_eval(args, s, conn):
    column = f"{args.embedding}_embedding"
    result = evaluate.run_eval(conn, n_queries=args.n, k=args.k, column=column)
    jp, mp = evaluate.write_report(result, Path(args.out))
    print(evaluate.to_markdown(result))
    print(f"wrote {jp} and {mp}")


def cmd_stats(args, s, conn):
    q = """SELECT d.name, count(o.*), count(i.*) FILTER (WHERE i.status='ok'), count(e.*)
           FROM objects o LEFT JOIN departments d USING (department_id)
           LEFT JOIN object_images i USING (object_id) LEFT JOIN object_embeddings e USING (object_id)
           GROUP BY d.name ORDER BY 1"""
    print(f"{'department':<30}{'objects':>10}{'images':>10}{'embedded':>10}")
    for name, n, im, em in conn.execute(q):
        print(f"{str(name):<30}{n:>10}{im:>10}{em:>10}")
    for r in conn.execute("SELECT run_id, kind, status, started_at::timestamp(0), ids_discovered, "
                          "objects_fetched, images_downloaded, objects_embedded FROM ingest_runs "
                          "ORDER BY run_id DESC LIMIT 5"):
        print(r)


def main(argv=None):
    p = argparse.ArgumentParser(prog="metsearch")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    for name, fn, help_ in [("run", cmd_run, "full pipeline for DEPARTMENTS, capped"),
                            ("incremental", cmd_incremental, "pick up objects changed/added since a date")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--cap", type=int, help="max objects per department (default MAX_OBJECTS_PER_DEPT)")
        sp.add_argument("--limit", type=int, help="stop each step after N items (smoke tests)")
        sp.add_argument("--since", type=date.fromisoformat, help="incremental: YYYY-MM-DD")
        sp.add_argument("--skip-images", action="store_true")
        sp.add_argument("--skip-embed", action="store_true")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("discover")
    sp.add_argument("--cap", type=int)
    sp.set_defaults(fn=cmd_discover)
    for name, fn in [("fetch", cmd_fetch), ("images", cmd_images), ("embed", cmd_embed)]:
        sp = sub.add_parser(name)
        sp.add_argument("--limit", type=int)
        sp.set_defaults(fn=fn)
    sub.add_parser("transform").set_defaults(fn=cmd_transform)
    sp = sub.add_parser("eval")
    sp.add_argument("--n", type=int, default=500)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--embedding", choices=["image", "text"], default="image")
    sp.add_argument("--out", default="eval")
    sp.set_defaults(fn=cmd_eval)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    conn = db.connect(s.database_url, vector=False)
    try:
        db.init_db(conn)  # idempotent; also registers the vector type
        args.fn(args, s, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
