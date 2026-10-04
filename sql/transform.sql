-- Staging JSON -> core tables. Idempotent; run inside one transaction by
-- metsearch.transform. Only objects that are public domain AND have an image
-- make it into `objects` (everything else stays in staging for auditing).

INSERT INTO departments (department_id, name)
SELECT department_id, payload->>'displayName'
FROM stg_departments
ON CONFLICT (department_id) DO UPDATE SET name = EXCLUDED.name;

-- Usable rows from staging, normalised once.
DROP TABLE IF EXISTS _usable, _obj_tags;
CREATE TEMP TABLE _usable ON COMMIT DROP AS
SELECT
    r.object_id,
    nullif(btrim(p->>'title'), '')                 AS title,
    nullif(btrim(p->>'artistDisplayName'), '')     AS artist,
    d.department_id,
    nullif(btrim(p->>'objectDate'), '')            AS object_date,
    -- objectDate is free text ("ca. 1850", "late 12th century"); the integer
    -- objectBeginDate/objectEndDate fields are the ones we filter on. The API
    -- returns 0/0 when a date is unknown, so treat that pair as NULL.
    CASE WHEN (p->>'objectBeginDate')::int = 0 AND (p->>'objectEndDate')::int = 0
         THEN NULL ELSE (p->>'objectBeginDate')::int END AS begin_year,
    CASE WHEN (p->>'objectBeginDate')::int = 0 AND (p->>'objectEndDate')::int = 0
         THEN NULL ELSE (p->>'objectEndDate')::int END   AS end_year,
    nullif(btrim(p->>'culture'), '')               AS culture,
    nullif(btrim(p->>'period'), '')                AS period,
    nullif(btrim(p->>'medium'), '')                AS medium,
    nullif(btrim(p->>'classification'), '')        AS classification,
    nullif(btrim(p->>'objectName'), '')            AS object_name,
    (p->>'isPublicDomain')::boolean                AS is_public_domain,
    nullif(p->>'primaryImage', '')                 AS image_url,
    nullif(p->>'primaryImageSmall', '')            AS image_small_url,
    nullif(p->>'objectURL', '')                    AS object_url,
    nullif(p->>'metadataDate', '')::timestamptz    AS metadata_date,
    r.fetched_at,
    p->'tags'                                      AS tags
FROM stg_objects_raw r
CROSS JOIN LATERAL (SELECT r.payload AS p) x
LEFT JOIN departments d ON d.name = p->>'department'
WHERE r.http_status = 200
  AND (p->>'isPublicDomain')::boolean
  AND coalesce(p->>'primaryImageSmall', '') <> '';

INSERT INTO objects (object_id, title, artist, department_id, object_date, begin_year,
                     end_year, culture, period, medium, classification, object_name,
                     is_public_domain, image_url, image_small_url, object_url,
                     metadata_date, fetched_at)
SELECT object_id, title, artist, department_id, object_date, begin_year,
       end_year, culture, period, medium, classification, object_name,
       is_public_domain, image_url, image_small_url, object_url,
       metadata_date, fetched_at
FROM _usable
ON CONFLICT (object_id) DO UPDATE SET
    title = EXCLUDED.title, artist = EXCLUDED.artist,
    department_id = EXCLUDED.department_id, object_date = EXCLUDED.object_date,
    begin_year = EXCLUDED.begin_year, end_year = EXCLUDED.end_year,
    culture = EXCLUDED.culture, period = EXCLUDED.period, medium = EXCLUDED.medium,
    classification = EXCLUDED.classification, object_name = EXCLUDED.object_name,
    is_public_domain = EXCLUDED.is_public_domain, image_url = EXCLUDED.image_url,
    image_small_url = EXCLUDED.image_small_url, object_url = EXCLUDED.object_url,
    metadata_date = EXCLUDED.metadata_date, fetched_at = EXCLUDED.fetched_at;

-- Objects that went private / lost their image since the last load.
DELETE FROM objects o
USING stg_objects_raw r
WHERE r.object_id = o.object_id
  AND NOT EXISTS (SELECT 1 FROM _usable u WHERE u.object_id = o.object_id);

-- Tags: `tags` is null on many records, so this is best-effort enrichment.
CREATE TEMP TABLE _obj_tags ON COMMIT DROP AS
SELECT DISTINCT u.object_id, btrim(t->>'term') AS term
FROM _usable u
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(u.tags) = 'array' THEN u.tags ELSE '[]'::jsonb END) t
WHERE coalesce(btrim(t->>'term'), '') <> '';

INSERT INTO tags (term)
SELECT DISTINCT term FROM _obj_tags
ON CONFLICT (term) DO NOTHING;

DELETE FROM object_tags ot USING _usable u WHERE ot.object_id = u.object_id;

INSERT INTO object_tags (object_id, tag_id)
SELECT ot.object_id, t.tag_id
FROM _obj_tags ot JOIN tags t USING (term)
ON CONFLICT DO NOTHING;

-- The Met edited this record after we embedded it: re-embed on the next run.
DELETE FROM object_embeddings e
USING objects o
WHERE e.object_id = o.object_id AND e.created_at < o.metadata_date;

-- Image URL changed: drop the cached-image row so it is re-downloaded.
DELETE FROM object_images i
USING objects o
WHERE i.object_id = o.object_id AND i.source_url IS DISTINCT FROM o.image_small_url;
