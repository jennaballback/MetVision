import os
import uuid

import psycopg
from psycopg.conninfo import make_conninfo
import pytest

from metsearch import db
from tests.fake_met import FakeMet


@pytest.fixture(scope="session")
def pg_url(tmp_path_factory):
    """DATABASE_URL if set (CI service container), else a throwaway pgserver instance."""
    if os.getenv("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    return srv.get_uri()


@pytest.fixture
def conn(pg_url):
    """A fresh database per test."""
    name = "t_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    url = make_conninfo(pg_url, dbname=name)
    c = db.connect(url, vector=False)
    db.init_db(c)
    yield c
    c.close()
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE {name} WITH (FORCE)")


@pytest.fixture
def fake_met():
    f = FakeMet(per_department=60).start()
    yield f
    f.stop()
