-- options-helper DuckDB schema v3
-- Adds observability ledger tables and persisted options flow rows.

CREATE SCHEMA IF NOT EXISTS meta;

-- Ingestion run ledger (one row per top-level pipeline execution)
CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
  run_id VARCHAR PRIMARY KEY,
  parent_run_id VARCHAR,

  job_name VARCHAR NOT NULL,
  triggered_by VARCHAR NOT NULL,
  status VARCHAR NOT NULL,

  started_at TIMESTAMP NOT NULL,
  ended_at TIMESTAMP,
  duration_ms BIGINT,

  provider VARCHAR,
  storage_backend VARCHAR,
  args_json JSON,
  git_sha VARCHAR,
  app_version VARCHAR,

  error_type VARCHAR,
  error_message VARCHAR,
  error_stack_hash VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_job_started
  ON meta.ingestion_runs(job_name, started_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status_started
  ON meta.ingestion_runs(status, started_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
  ON meta.ingestion_runs(started_at);

-- Per-asset events within a run
CREATE TABLE IF NOT EXISTS meta.ingestion_run_assets (
  run_id VARCHAR NOT NULL,
  asset_key VARCHAR NOT NULL,
  asset_kind VARCHAR NOT NULL,
  partition_key VARCHAR NOT NULL DEFAULT 'ALL',
  status VARCHAR NOT NULL,

  rows_inserted BIGINT,
  rows_updated BIGINT,
  rows_deleted BIGINT,
  bytes_written BIGINT,

  min_event_ts TIMESTAMP,
  max_event_ts TIMESTAMP,

  extra_json JSON,
  started_at TIMESTAMP,
  ended_at TIMESTAMP,
  duration_ms BIGINT,

  PRIMARY KEY(run_id, asset_key, partition_key)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_run_assets_run_id
  ON meta.ingestion_run_assets(run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_run_assets_asset_partition
  ON meta.ingestion_run_assets(asset_key, partition_key);
CREATE INDEX IF NOT EXISTS idx_ingestion_run_assets_asset_status_started
  ON meta.ingestion_run_assets(asset_key, status, started_at);

-- Asset-level freshness
CREATE TABLE IF NOT EXISTS meta.asset_watermarks (
  asset_key VARCHAR NOT NULL,
  scope_key VARCHAR NOT NULL,
  watermark_ts TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
  last_run_id VARCHAR,

  PRIMARY KEY(asset_key, scope_key)
);

CREATE INDEX IF NOT EXISTS idx_asset_watermarks_updated_at
  ON meta.asset_watermarks(updated_at);
CREATE INDEX IF NOT EXISTS idx_asset_watermarks_last_run_id
  ON meta.asset_watermarks(last_run_id);

-- Persisted data quality checks
CREATE TABLE IF NOT EXISTS meta.asset_checks (
  check_id VARCHAR PRIMARY KEY,
  asset_key VARCHAR NOT NULL,
  partition_key VARCHAR,
  check_name VARCHAR NOT NULL,
  severity VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  checked_at TIMESTAMP NOT NULL,
  metrics_json JSON,
  message VARCHAR,
  run_id VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_asset_checks_status_checked_at
  ON meta.asset_checks(status, checked_at);
CREATE INDEX IF NOT EXISTS idx_asset_checks_asset_checked_at
  ON meta.asset_checks(asset_key, checked_at);
CREATE INDEX IF NOT EXISTS idx_asset_checks_run_id
  ON meta.asset_checks(run_id);

-- Persisted options flow rows (portal/query friendly)
CREATE TABLE IF NOT EXISTS options_flow (
  symbol VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  from_date DATE NOT NULL,
  to_date DATE NOT NULL,
  window_size INTEGER NOT NULL,
  group_by VARCHAR NOT NULL,
  row_key VARCHAR NOT NULL,

  contract_symbol VARCHAR,
  expiry DATE,
  option_type VARCHAR,
  strike DOUBLE,

  delta_oi DOUBLE,
  delta_oi_notional DOUBLE,
  volume_notional DOUBLE,
  delta_notional DOUBLE,
  n_pairs BIGINT,

  snapshot_dates_json JSON,
  generated_at TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  PRIMARY KEY(symbol, from_date, to_date, window_size, group_by, row_key)
);

CREATE INDEX IF NOT EXISTS idx_options_flow_symbol_dates
  ON options_flow(symbol, from_date, to_date);
CREATE INDEX IF NOT EXISTS idx_options_flow_as_of
  ON options_flow(as_of);
CREATE INDEX IF NOT EXISTS idx_options_flow_group_by
  ON options_flow(group_by);
CREATE INDEX IF NOT EXISTS idx_options_flow_expiry
  ON options_flow(expiry);
