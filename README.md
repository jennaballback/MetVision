# MetVision

MetVision is visual search over the Met's open-access collection. Click an artwork, upload a photo, or type "a woman reading by a window", and get the ten most
visually similar pieces from the Metropolitan Museum's open-access collection, with optional
filters for department, culture, classification and date.

Underneath: a rate-limited, resumable ingestion pipeline from the Met Collection API into
Postgres, CLIP image and text embeddings in pgvector, hybrid retrieval, a Claude-powered
natural-language filter parser, a weekly incremental job, and an evaluation against curator
metadata.

> **Results** (first real run, 5 Nov 2026: 7,070 public-domain works from Asian Art, Drawings and
> Prints, and European Paintings; 500 query objects, 10 nearest neighbours by CLIP image embedding):
>
> | Neighbour shares the query's… | MetVision | Random 10 | Lift |
> |---|---|---|---|
> | classification | 72.8% | 9.2% | 7.9x |
> | period | 32.8% | 7.4% | 4.4x |
> | culture | 64.7% | 23.3% | 2.8x |
> | department | 97.1% | 56.3% | 1.7x |
>
> CLIP never sees the curators' labels, yet a work's visual neighbours share its classification
> 7.9x more often than chance. Department lift is low only because Asian Art is most of the pool.
> Full report: run `python -m metsearch eval --n 500` (writes `eval/results_image.md`).
> (`eval/sample_results_fixture.md` is from synthetic test data and only shows the report format.)

## Architecture

```
Met API ──► stg_object_ids ──► stg_objects_raw (JSONB) ──SQL──► objects / departments / tags / object_tags
  (rate limited, retries,         (resumable cursor:                         │
   seeded sample, capped)          needs_fetch flag)                         ▼
                                                              data/images (small JPEGs, cached)
                                                                             │
                                                              CLIP ViT-B/32 (image + metadata text)
                                                                             ▼
                                                     object_embeddings (vector(512) ×2, HNSW)
                                                                             │
                       Claude (tool use) ──► Filters ──► SQL WHERE + vector ORDER BY ──► Streamlit
```

| Step | Code | Notes |
|---|---|---|
| Discover | `metsearch/ingest.py: discover_and_fetch` | Department ID list in a seeded random order, queued in rounds sized from the share of usable records seen so far, until each department has about `MAX_OBJECTS_PER_DEPT` public-domain objects with images. Re-running is idempotent. |
| Fetch | `ingest.fetch_metadata` | Token-bucket rate limiter (default 6 req/s). When the Met answers 403/429 (its "too fast" signal), every worker pauses (30s, doubling, up to 5 min) and the rate halves; 5xx errors get short exponential backoff, raw JSON into staging. The cursor is `stg_object_ids.needs_fetch`, flipped in the same transaction that stores the response, so a crash loses at most one batch. |
| Transform | `sql/transform.sql` | Staging → core with idempotent upserts in one transaction. Only public-domain objects with an image are promoted. Withdrawn objects (404) are deleted. |
| Images | `metsearch/images.py` | `primaryImageSmall`, resized to ≤512px, cached on disk, skipped if present, 5 req/s. |
| Embed | `metsearch/embed.py` | open_clip ViT-B/32 on CPU. One image vector and one text vector per object (title, object name, culture, period, classification, medium, artist, tags). |
| Index | `sql/schema.sql` | HNSW (`vector_cosine_ops`) on both columns. |
| Search | `metsearch/search.py` | Visual kNN, hybrid (reciprocal rank fusion of image kNN, metadata-text kNN and Postgres full-text rank), filters as SQL `WHERE`. |
| NL filters | `metsearch/nl_filter.py` | Claude fills a strict tool schema; nothing else. |
| Schedule | `.github/workflows/weekly-ingest.yml` | `metadataDate` endpoint → changed objects we hold, new accessions above a per-department high-water mark, and objects that vanished. |
| Eval | `metsearch/evaluate.py` | Neighbour-match vs random baseline, per-department breakdown, worst queries. |

### Data-cleaning decisions worth knowing

- **Dates:** `objectDate` is free text ("ca. 1850", "late 12th century", "1550–1295 B.C."). Filters use
  the integer `objectBeginDate` / `objectEndDate` instead, with an overlap test (a work dated 1790–1810
  matches "1800s"). The API reports unknown dates as `0`/`0`; those become `NULL`.
- **Images:** most of the collection has no image or is not public domain, and the ID list doesn't say
  which. (The Met retired the `/search?hasImages=true` endpoint in October 2026.) So discovery fetches in
  rounds, measures the usable share, and queues just enough more to reach the cap; the transform only
  promotes records with `isPublicDomain` and a `primaryImageSmall`.
- **Tags:** missing on many records, so they enrich the text embedding and keyword search but are
  never required.
- **Sampling:** object IDs cluster by accession batch (one donor, one excavation), so the first N IDs
  of a department are not representative. Discovery takes a seeded random sample.

### Why Claude only parses

"Japanese woodblock prints of birds from the 1800s" becomes
`cultures=["Japan"], classifications=["Prints"], years 1800–1899` plus the free text `"birds"`, which
is CLIP-embedded. SQL applies the filters; vectors rank within them. Keeping the LLM out of ranking
makes it testable (assert on the parsed filters) and it cannot invent an artwork. Without an API key
the app falls back to plain text search.

## Running it

```bash
cp .env.example .env                 # DEPARTMENTS=6 (Asian Art) and a 5,000 cap to start
docker compose up -d                 # pgvector/pgvector:pg16
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m metsearch run --limit 50   # smoke test: 50 objects through every step
python -m metsearch run              # one department, ~5K usable objects (fetch ~10-15 min, CLIP ~10-20 min on CPU)
python -m metsearch stats
python -m metsearch eval --n 500     # writes eval/results_image.{md,json}
python -m metsearch eval --embedding text
streamlit run app/streamlit_app.py
```

Then widen: `DEPARTMENTS=6,9,10,11 MAX_OBJECTS_PER_DEPT=5000 python -m metsearch run`. Each step only
does outstanding work, so re-running after a crash or a change of scope picks up where it left off.
Individual steps are also commands: `discover`, `fetch`, `transform`, `images`, `embed`.

Weekly job: add a hosted Postgres with pgvector (Neon and Supabase both have it) as the `DATABASE_URL`
repository secret, optionally set the `DEPARTMENTS` repository variable, and the workflow runs
Mondays at 06:00 UTC (or on demand with an optional `since` date).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests run the real pipeline against `tests/fake_met.py`, a local server that speaks the Met API's
JSON shapes and serves generated images, with an embedded Postgres + pgvector (`pgserver`) and a
deterministic stand-in embedder. They cover the cap, filtering to public-domain-with-image,
transform idempotency, resuming a crashed fetch, retrying transient errors, incremental pick-up of new
and withdrawn objects, filtered and hybrid search, the eval, and the Claude filter parser (with a
mocked client). CI runs them against a `pgvector/pgvector:pg16` service container.

You can also run the whole app offline on fake data:

```bash
python -m tests.fake_met --port 8765 &
MET_API_BASE=http://localhost:8765/v1 EMBEDDER=fake DEPARTMENTS=6,9,10,11 python -m metsearch run
EMBEDDER=fake streamlit run app/streamlit_app.py
```

## Schema

`objects` (object_id, title, artist, department_id, object_date, begin_year, end_year, culture, period,
medium, classification, object_name, is_public_domain, image_url, image_small_url, object_url,
metadata_date, fetched_at, search_tsv) · `departments` · `tags` + `object_tags` ·
`object_images` (local cache bookkeeping) · `object_embeddings` (image_embedding vector(512),
text_embedding vector(512), text_source, model_name, created_at) · `ingest_runs` · staging:
`stg_departments`, `stg_object_ids`, `stg_objects_raw` · `department_state` (incremental high-water marks).

## Known limitations

- With only one department loaded, department match is trivially 100%; culture, classification and
  period are the informative columns until you widen.
- pgvector < 0.8 applies `WHERE` filters after the HNSW scan, so very selective filters can return
  fewer than 10 results; the code raises `ef_search` and enables iterative scans where available.
