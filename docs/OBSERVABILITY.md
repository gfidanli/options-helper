# Observability: Run Ledger + Watermarks + Data Checks (DuckDB `meta.*`)

Status: Draft  
Purpose: define an ingestion visibility “control plane” stored in DuckDB.

This enables:
- a Streamlit “Health” dashboard (primary UX)
- CLI health commands (optional)
- optional Dagster integration (secondary UX)

---

## Why DuckDB for observability?

`options-helper` is intentionally single-node and offline-friendly.
Putting operational metadata into the same DuckDB database as analytical tables means:

- zero additional services
- simple querying (`SELECT ... FROM meta.*`)
- portable state (copy the DB file and you copy history + watermarks)

Dagster can still be used, but it should not be the only place where history lives.

---

## Core concepts

### Run
A single execution of a top-level pipeline step:
- `ingest_candles`
- `ingest_options_bars`
- `snapshot_options`
- `compute_flow`
- `compute_derived`
- `build_briefing`
- `build_dashboard`

(Exact job names are a convention; pick stable names and don’t change them casually.)

### Asset record
Within a run, one or more assets may be created/updated.
Examples:
- a DuckDB table partition
- a snapshot file written to disk
- a derived metrics table append

### Watermark (freshness)
A “last seen max timestamp” per asset and scope (e.g., per symbol).
Used to quickly compute staleness and drive coverage dashboards.

### Check
A persisted data-quality result:
- pass / fail / warn
- metrics payload (JSON)
- linked to a run and (optionally) partition key

---

## DuckDB schema (DDL)

Create schema:

```sql
CREATE SCHEMA IF NOT EXISTS meta;
```

### `meta.ingestion_runs`

```sql
CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
  run_id           VARCHAR PRIMARY KEY,         -- uuid4
  parent_run_id    VARCHAR,

  job_name         VARCHAR NOT NULL,            -- stable identifier
  triggered_by     VARCHAR NOT NULL,            -- "cli" | "dagster" | "cron" | "manual"
  status           VARCHAR NOT NULL,            -- "started" | "success" | "failed"

  started_at       TIMESTAMP NOT NULL,
  ended_at         TIMESTAMP,
  duration_ms      BIGINT,

  -- execution context
  provider         VARCHAR,                     -- e.g. "alpaca" | "yahoo"
  storage_backend  VARCHAR,                     -- "duckdb" | "filesystem"
  args_json        VARCHAR,                     -- JSON string (keep it simple)
  git_sha          VARCHAR,
  app_version      VARCHAR,

  -- error info (only populated on failure)
  error_type       VARCHAR,
  error_message    VARCHAR,
  error_stack_hash VARCHAR
);
```

### `meta.ingestion_run_assets`

```sql
CREATE TABLE IF NOT EXISTS meta.ingestion_run_assets (
  run_id           VARCHAR NOT NULL,
  asset_key        VARCHAR NOT NULL,            -- e.g. "candles_daily"
  asset_kind       VARCHAR NOT NULL,            -- "table" | "file" | "view"
  partition_key    VARCHAR,                     -- e.g. "2026-02-05" or "AAPL|2026-02-05"
  status           VARCHAR NOT NULL,            -- "success" | "failed" | "skipped"

  -- metrics (best effort)
  rows_inserted    BIGINT,
  rows_updated     BIGINT,
  rows_deleted     BIGINT,
  bytes_written    BIGINT,

  -- touched event-time range (best effort)
  min_event_ts     TIMESTAMP,
  max_event_ts     TIMESTAMP,

  extra_json       VARCHAR,                     -- JSON string for asset-specific metadata

  started_at       TIMESTAMP,
  ended_at         TIMESTAMP,
  duration_ms      BIGINT,

  PRIMARY KEY (run_id, asset_key, partition_key)
);
```

### `meta.asset_watermarks`

Fast “freshness” queries.

```sql
CREATE TABLE IF NOT EXISTS meta.asset_watermarks (
  asset_key        VARCHAR NOT NULL,
  scope_key        VARCHAR NOT NULL,            -- e.g. symbol or "ALL"
  watermark_ts     TIMESTAMP NOT NULL,          -- last ingested event timestamp
  updated_at       TIMESTAMP NOT NULL,
  last_run_id      VARCHAR,

  PRIMARY KEY (asset_key, scope_key)
);
```

### `meta.asset_checks`

```sql
CREATE TABLE IF NOT EXISTS meta.asset_checks (
  check_id         VARCHAR PRIMARY KEY,         -- uuid4
  asset_key        VARCHAR NOT NULL,
  partition_key    VARCHAR,
  check_name       VARCHAR NOT NULL,            -- stable identifier
  severity         VARCHAR NOT NULL,            -- "error" | "warn" | "info"
  status           VARCHAR NOT NULL,            -- "pass" | "fail" | "skip"

  checked_at       TIMESTAMP NOT NULL,
  metrics_json     VARCHAR,                     -- JSON
  message          VARCHAR,

  run_id           VARCHAR
);
```

---

## Canonical asset keys

Pick stable names; dashboards depend on these.

Recommended set:

- `candles_daily` (DuckDB table; ingested by `ingest candles`)
- `options_bars` (DuckDB table; ingested by `ingest options-bars`)
- `options_snapshot_file` (snapshot JSON files under `data/options_snapshots/`)
- `options_flow` (DuckDB table; computed by `flow`)
- `derived_metrics` (DuckDB table; computed by `derived`)
- `briefing_markdown` (Markdown artifact produced by `briefing`)
- `dashboard_views` (tables/views produced for UI use)

Operational:
- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

---

## Required behaviors (contract)

### Run lifecycle
1) Insert a row in `meta.ingestion_runs` with:
   - `status='started'`
   - `started_at=NOW()`
2) Record one or more rows in `meta.ingestion_run_assets` as work is performed
3) Finish by updating `meta.ingestion_runs`:
   - on success: `status='success'`, `ended_at`, `duration_ms`
   - on failure: `status='failed'`, populate error fields, `ended_at`, `duration_ms`

### Asset rows: best-effort metrics
Do not block ingestion just to compute perfect metrics.
- If `rows_inserted` is cheap to compute, record it.
- Otherwise record `NULL` and rely on `min/max ts` + watermarks.

### Watermark updates
When you successfully ingest/append time-series data:
- compute `watermark_ts = max(event_ts)` for the write scope (symbol or ALL)
- upsert into `meta.asset_watermarks`

### Checks
- Persist every check result (pass and fail) so you can view trends.
- Severity guidelines:
  - **error**: indicates broken pipeline or corrupt dataset
  - **warn**: indicates suspicious data but maybe “normal” in illiquid options
  - **info**: informative metrics only

---

## Practical checks to start with

### Candles
- `candles_unique_symbol_date` (error)
- `candles_monotonic_date` (error)
- `candles_no_negative_prices` (error)
- `candles_gap_days_last_30` (warn; output count)

### Options bars
- `options_bars_monotonic_ts` (error)
- `options_bars_no_negative_prices` (error)
- `options_bars_duplicate_pk` (error)

### Snapshots / flow / derived
- `snapshot_parseable_contract_symbol` (warn/error depending on frequency)
- `flow_no_null_primary_keys` (error)
- `derived_no_duplicate_keys` (error)

---

## Health queries (for Streamlit)

### Latest run per job
```sql
SELECT job_name, status, started_at, ended_at, duration_ms
FROM meta.ingestion_runs
QUALIFY ROW_NUMBER() OVER (PARTITION BY job_name ORDER BY started_at DESC) = 1
ORDER BY job_name;
```

### Failures last 7 days
```sql
SELECT started_at, job_name, error_type, error_message
FROM meta.ingestion_runs
WHERE status = 'failed'
  AND started_at >= (NOW() - INTERVAL '7 days')
ORDER BY started_at DESC;
```

### Watermarks / freshness
```sql
SELECT asset_key, scope_key, watermark_ts, updated_at,
       DATE_DIFF('day', watermark_ts, NOW()) AS staleness_days
FROM meta.asset_watermarks
ORDER BY asset_key, scope_key;
```

### Recent check failures
```sql
SELECT checked_at, asset_key, partition_key, check_name, severity, message
FROM meta.asset_checks
WHERE status = 'fail'
ORDER BY checked_at DESC
LIMIT 200;
```

---

## Concurrency note (Streamlit reading while ingestion writes)

DuckDB supports many readers + a single writer. In practice:

- Streamlit should open the DuckDB connection in **read-only** mode if possible.
- Ingestion should:
  - commit frequently
  - avoid extremely long write transactions (break into partitions)

If you later move to a multi-machine setup, you can revisit this, but it’s fine for “one laptop + cron”.

---

## Instrumentation: where to hook in the code

Wrap the top-level CLI handlers for:
- `ingest candles`
- `ingest options-bars`
- `snapshot-options`
- `flow`
- `derived`
- `briefing`
- `dashboard`

Recommended pattern:
- `RunLogger` context manager:
  - creates `run_id`
  - writes the started row
  - exposes `log_asset_*()` methods
  - catches exceptions and closes run status properly

Dagster assets (if enabled) should reuse the same `RunLogger` helper.
