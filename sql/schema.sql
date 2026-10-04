-- Met Museum visual search schema.
-- Staging tables hold raw API responses (ELT); core tables are rebuilt from staging
-- by sql/transform.sql with idempotent upserts.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- run log
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id          BIGSERIAL PRIMARY KEY,
    kind            TEXT NOT NULL,                 -- 'full' | 'incremental'
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    since_date      DATE,                          -- for incremental runs
    ids_discovered  INT NOT NULL DEFAULT 0,
    objects_fetched INT NOT NULL DEFAULT 0,
    images_downloaded INT NOT NULL DEFAULT 0,
    objects_embedded INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running', -- running | succeeded | failed
    error           TEXT
);

-- ---------------------------------------------------------------- staging
CREATE TABLE IF NOT EXISTS stg_departments (
    department_id   INT PRIMARY KEY,
    payload         JSONB NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Object IDs to fetch. The fetch step's resumable cursor is simply
-- "rows here whose object has no fresh row in stg_objects_raw".
CREATE TABLE IF NOT EXISTS stg_object_ids (
    object_id       INT PRIMARY KEY,
    department_id   INT NOT NULL,
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id          BIGINT REFERENCES ingest_runs(run_id),
    needs_fetch     BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS stg_object_ids_needs_fetch ON stg_object_ids (object_id) WHERE needs_fetch;

CREATE TABLE IF NOT EXISTS stg_objects_raw (
    object_id       INT PRIMARY KEY,
    http_status     INT NOT NULL,
    payload         JSONB,                         -- NULL for 404s
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id          BIGINT REFERENCES ingest_runs(run_id)
);

-- ---------------------------------------------------------------- core
CREATE TABLE IF NOT EXISTS departments (
    department_id   INT PRIMARY KEY,
    name            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    object_id        INT PRIMARY KEY,
    title            TEXT,
    artist           TEXT,
    department_id    INT REFERENCES departments(department_id),
    object_date      TEXT,          -- free text ("ca. 1850"), display only
    begin_year       INT,           -- clean integer years: used for filters
    end_year         INT,
    culture          TEXT,
    period           TEXT,
    medium           TEXT,
    classification   TEXT,
    object_name      TEXT,
    is_public_domain BOOLEAN NOT NULL,
    image_url        TEXT,
    image_small_url  TEXT,
    object_url       TEXT,
    metadata_date    TIMESTAMPTZ,   -- Met's own last-modified timestamp
    fetched_at       TIMESTAMPTZ NOT NULL,
    search_tsv       TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(title,'') || ' ' || coalesce(artist,'') || ' ' ||
            coalesce(culture,'') || ' ' || coalesce(medium,'') || ' ' ||
            coalesce(classification,'') || ' ' || coalesce(object_name,''))
    ) STORED
);
CREATE INDEX IF NOT EXISTS objects_department ON objects (department_id);
CREATE INDEX IF NOT EXISTS objects_years ON objects (begin_year, end_year);
CREATE INDEX IF NOT EXISTS objects_tsv ON objects USING GIN (search_tsv);

CREATE TABLE IF NOT EXISTS tags (
    tag_id          SERIAL PRIMARY KEY,
    term            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS object_tags (
    object_id       INT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    tag_id          INT NOT NULL REFERENCES tags(tag_id),
    PRIMARY KEY (object_id, tag_id)
);

-- Local image cache bookkeeping (files live on disk under IMAGE_DIR).
CREATE TABLE IF NOT EXISTS object_images (
    object_id       INT PRIMARY KEY REFERENCES objects(object_id) ON DELETE CASCADE,
    local_path      TEXT,
    source_url      TEXT NOT NULL,
    status          TEXT NOT NULL,               -- ok | failed
    error           TEXT,
    downloaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS object_embeddings (
    object_id       INT PRIMARY KEY REFERENCES objects(object_id) ON DELETE CASCADE,
    image_embedding vector(512),
    text_embedding  vector(512),
    text_source     TEXT,          -- the exact string that was text-embedded
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW indexes for cosine distance (<=>). Embeddings are L2-normalised, so cosine
-- and inner-product orderings agree.
CREATE INDEX IF NOT EXISTS object_embeddings_image_hnsw
    ON object_embeddings USING hnsw (image_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS object_embeddings_text_hnsw
    ON object_embeddings USING hnsw (text_embedding vector_cosine_ops);
