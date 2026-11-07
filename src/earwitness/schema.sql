-- Earwitness schema. Applied by earwitness.db.migrate() inside the schema named
-- by EARWITNESS_DB_SCHEMA (search_path is set first), so nothing here is
-- schema-qualified. Every statement must be idempotent.

-- GB grid frequency, one row per sample, from every source we have fetched.
CREATE TABLE IF NOT EXISTS grid_frequency (
    time   TIMESTAMPTZ      NOT NULL,
    source TEXT             NOT NULL,
    hz     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (source, time)
) WITH (
    tsdb.hypertable,
    tsdb.segmentby = 'source',
    tsdb.orderby = 'time DESC'
);

-- Windows already fetched from a source: tells "never fetched" apart from
-- "fetched and complete", so a covered window needs no HTTP at all.
CREATE TABLE IF NOT EXISTS grid_coverage (
    source     TEXT        NOT NULL,
    start_ts   TIMESTAMPTZ NOT NULL,
    end_ts     TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS grid_coverage_span ON grid_coverage (source, start_ts, end_ts);

-- Minute summary of the grid record, kept fresh by TimescaleDB.
CREATE MATERIALIZED VIEW IF NOT EXISTS grid_frequency_1m
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       source,
       avg(hz) AS avg_hz,
       min(hz) AS min_hz,
       max(hz) AS max_hz,
       count(*) AS samples
FROM grid_frequency
GROUP BY bucket, source
WITH NO DATA;

-- Real-time: windows stored since the last refresh are included straight away.
ALTER MATERIALIZED VIEW grid_frequency_1m SET (timescaledb.materialized_only = false);

SELECT add_continuous_aggregate_policy(
    'grid_frequency_1m',
    start_offset => NULL,
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '10 minutes',
    if_not_exists => true
);

-- Index of every analysed clip. events.jsonl stays the replay source of truth.
CREATE TABLE IF NOT EXISTS cases (
    id            TEXT PRIMARY KEY,
    filename      TEXT        NOT NULL,
    created       TIMESTAMPTZ NOT NULL,
    status        TEXT        NOT NULL,
    sha256        TEXT,
    duration_s    DOUBLE PRECISION,
    verdict       TEXT,
    summary_plain TEXT,
    dated_start   TIMESTAMPTZ,
    options       JSONB       NOT NULL DEFAULT '{}',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cases_dated_start ON cases (dated_start);

-- Every finding of every case, on the time it was reported.
CREATE TABLE IF NOT EXISTS findings (
    time       TIMESTAMPTZ NOT NULL,
    case_id    TEXT        NOT NULL,
    finding_id TEXT        NOT NULL,
    module     TEXT        NOT NULL,
    severity   TEXT        NOT NULL,
    title      TEXT        NOT NULL,
    detail     TEXT,
    tags       TEXT[]      NOT NULL DEFAULT '{}',
    data       JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (case_id, finding_id, time)
) WITH (
    tsdb.hypertable,
    tsdb.segmentby = 'case_id',
    tsdb.orderby = 'time DESC'
);
