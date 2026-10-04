from datetime import date
from types import SimpleNamespace

import pytest

from metsearch import db, evaluate, images, ingest
from metsearch.embed import FakeEmbedder, embed_pending
from metsearch.met_api import MetClient
from metsearch.nl_filter import parse_query, parse_tool_input
from metsearch.search import Filters, search_by_text, similar_to_object

DEPTS = [6, 11]


def client_for(fake):
    c = MetClient(fake.base + "/v1", rate_per_second=500, max_retries=3)
    return c


def run_full(conn, fake, tmp_path, cap=40):
    client = client_for(fake)
    run_id = db.start_run(conn, "full")
    ingest.load_departments(conn, client)
    ingest.discover_full(conn, client, DEPTS, cap, run_id)
    ingest.fetch_metadata(conn, client, run_id, workers=4, batch_size=15)
    ingest.transform(conn)
    images.download_images(conn, tmp_path / "img", rate_per_second=500, run_id=run_id)
    embed_pending(conn, FakeEmbedder(), batch_size=16, run_id=run_id)
    db.finish_run(conn, run_id, "succeeded")
    return run_id


def count(conn, sql, *params):
    return conn.execute(sql, params).fetchone()[0]


def test_full_pipeline_respects_cap_and_filters(conn, fake_met, tmp_path):
    run_full(conn, fake_met, tmp_path, cap=40)
    assert count(conn, "SELECT count(*) FROM stg_object_ids") == 80
    assert count(conn, "SELECT count(*) FROM stg_object_ids WHERE needs_fetch") == 0
    # every core object is public domain with an image, and was embedded
    assert count(conn, "SELECT count(*) FROM objects WHERE NOT is_public_domain OR image_small_url IS NULL") == 0
    n_obj = count(conn, "SELECT count(*) FROM objects")
    assert 0 < n_obj <= 80
    assert count(conn, "SELECT count(*) FROM object_embeddings") == n_obj
    assert count(conn, "SELECT count(*) FROM object_tags") > 0
    assert count(conn, "SELECT count(*) FROM objects WHERE begin_year IS NULL") == 0
    run = conn.execute("SELECT status, objects_fetched, objects_embedded FROM ingest_runs").fetchone()
    assert run == ("succeeded", 80, n_obj)


def test_transform_is_idempotent(conn, fake_met, tmp_path):
    run_full(conn, fake_met, tmp_path)
    before = conn.execute("SELECT count(*), sum(object_id) FROM objects").fetchone()
    tags_before = count(conn, "SELECT count(*) FROM object_tags")
    ingest.transform(conn)
    ingest.transform(conn)
    assert conn.execute("SELECT count(*), sum(object_id) FROM objects").fetchone() == before
    assert count(conn, "SELECT count(*) FROM object_tags") == tags_before
    assert count(conn, "SELECT count(*) FROM object_embeddings") == before[0]  # nothing invalidated


def test_fetch_resumes_and_retries(conn, fake_met):
    client = client_for(fake_met)
    run_id = db.start_run(conn, "full")
    ingest.load_departments(conn, client)
    ingest.discover_full(conn, client, [6], 30, run_id)
    ingest.fetch_metadata(conn, client, run_id, limit=10)       # "crash" after 10
    assert count(conn, "SELECT count(*) FROM stg_object_ids WHERE needs_fetch") == 20
    fake_met.fail_next = 2                                      # transient 503s get retried
    before = fake_met.requests
    ingest.fetch_metadata(conn, client, run_id)
    assert count(conn, "SELECT count(*) FROM stg_object_ids WHERE needs_fetch") == 0
    assert fake_met.requests - before == 22                     # only the remaining 20 (+2 retries)
    assert count(conn, "SELECT count(*) FROM stg_objects_raw") == 30


def test_discovery_rerun_does_not_grow_past_cap(conn, fake_met):
    client = client_for(fake_met)
    run_id = db.start_run(conn, "full")
    ingest.discover_full(conn, client, [6], 25, run_id)
    ingest.discover_full(conn, client, [6], 25, run_id)
    assert count(conn, "SELECT count(*) FROM stg_object_ids") == 25


def test_incremental_picks_up_new_and_withdrawn(conn, fake_met, tmp_path):
    run_full(conn, fake_met, tmp_path)
    n_before = count(conn, "SELECT count(*) FROM objects")
    new_id = fake_met.add_object(6)
    gone = conn.execute("SELECT min(object_id) FROM objects o JOIN departments d USING (department_id) "
                        "WHERE d.name = 'Asian Art'").fetchone()[0]
    fake_met.withdrawn.add(gone)

    client = client_for(fake_met)
    run_id = db.start_run(conn, "incremental", date(2026, 9, 30))
    queued = ingest.discover_incremental(conn, client, DEPTS, date(2026, 9, 30), run_id)
    assert queued == 2  # one new, one withdrawn
    ingest.fetch_metadata(conn, client, run_id)
    ingest.transform(conn)
    images.download_images(conn, tmp_path / "img", rate_per_second=500)
    embed_pending(conn, FakeEmbedder())
    assert count(conn, "SELECT count(*) FROM objects WHERE object_id = %s", new_id) == 1
    assert count(conn, "SELECT count(*) FROM objects WHERE object_id = %s", gone) == 0
    assert count(conn, "SELECT count(*) FROM object_embeddings WHERE object_id = %s", new_id) == 1
    assert count(conn, "SELECT count(*) FROM objects") == n_before


def test_search_and_filters(conn, fake_met, tmp_path):
    run_full(conn, fake_met, tmp_path)
    oid = conn.execute("SELECT object_id FROM object_embeddings ORDER BY object_id LIMIT 1").fetchone()[0]
    res = similar_to_object(conn, oid, k=10)
    assert len(res) == 10 and oid not in [r["object_id"] for r in res]
    hyb = similar_to_object(conn, oid, k=5, mode="hybrid")
    assert len(hyb) == 5

    f = Filters(cultures=["Japan"], classifications=["Prints"], year_from=1800, year_to=1899)
    res = search_by_text(conn, "bird", FakeEmbedder(), k=10, filters=f)
    assert res, "expected some Japanese prints from the 1800s"
    for r in res:
        assert r["culture"] == "Japan" and r["classification"] == "Prints"
        assert r["end_year"] >= 1800 and r["begin_year"] <= 1899
    # keyword + metadata-text rankings pull title matches to the top
    birds = count(conn, "SELECT count(*) FROM objects WHERE culture = 'Japan' AND classification = 'Prints' "
                        "AND title LIKE 'Bird%%' AND end_year >= 1800 AND begin_year <= 1899")
    top = [r["title"] for r in res[:max(1, min(birds, 3))]]
    assert birds == 0 or all(t.startswith("Bird") for t in top), top

    res = search_by_text(conn, "", FakeEmbedder(), k=5, filters=Filters(departments=["European Paintings"]))
    assert res and all(r["department"] == "European Paintings" for r in res)


def test_eval_beats_random(conn, fake_met, tmp_path):
    run_full(conn, fake_met, tmp_path, cap=60)
    result = evaluate.run_eval(conn, n_queries=50, k=10)
    s = result["summary"]
    assert s["classification"]["model"] > s["classification"]["random"]
    assert s["department"]["lift"] > 1.2
    assert len(result["failures"]) > 0
    md = evaluate.to_markdown(result)
    assert "| department |" in md


def test_parse_tool_input_cleans_values():
    p = parse_tool_input({"free_text": " birds ", "departments": ["Asian Art", "Made Up"],
                          "cultures": ["Japan"], "classifications": ["Prints"],
                          "year_from": 1899, "year_to": 1800}, ["Asian Art"])
    assert p.free_text == "birds"
    assert p.filters.departments == ["Asian Art"]
    assert (p.filters.year_from, p.filters.year_to) == (1800, 1899)


class _FakeAnthropic:
    def __init__(self, content, stop_reason="tool_use"):
        resp = SimpleNamespace(content=content, stop_reason=stop_reason)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self.calls.append(kw) or resp))


def test_parse_query_with_tool_call():
    block = SimpleNamespace(type="tool_use", name="apply_search_filters", input={
        "free_text": "birds", "departments": [], "cultures": ["Japan"], "classifications": ["Prints"],
        "year_from": 1800, "year_to": 1899})
    client = _FakeAnthropic([block])
    p = parse_query("Japanese woodblock prints of birds from the 1800s", ["Asian Art"], client=client)
    assert not p.fallback and p.free_text == "birds" and p.filters.cultures == ["Japan"]
    sent = client.calls[0]
    assert sent["tools"][0]["strict"] is True and sent["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("content,stop", [([SimpleNamespace(type="text", text="hi")], "end_turn"),
                                          ([], "refusal")])
def test_parse_query_falls_back(content, stop):
    p = parse_query("cats", ["Asian Art"], client=_FakeAnthropic(content, stop))
    assert p.fallback and p.free_text == "cats" and p.filters.is_empty()
