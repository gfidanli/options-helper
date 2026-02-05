-- options-helper DuckDB schema v1
-- Local-first warehouse tables used by DuckDB-backed stores.

-- Derived daily metrics (per symbol/day)
CREATE TABLE IF NOT EXISTS derived_daily (
  symbol VARCHAR NOT NULL,
  date DATE NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  -- Canonical derived columns (match options_helper.data.derived.DERIVED_COLUMNS)
  spot DOUBLE NOT NULL,
  pc_oi DOUBLE,
  pc_vol DOUBLE,
  call_wall DOUBLE,
  put_wall DOUBLE,
  gamma_peak_strike DOUBLE,
  atm_iv_near DOUBLE,
  em_near_pct DOUBLE,
  skew_near_pp DOUBLE,
  rv_20d DOUBLE,
  rv_60d DOUBLE,
  iv_rv_20d DOUBLE,
  atm_iv_near_percentile DOUBLE,
  iv_term_slope DOUBLE,

  PRIMARY KEY(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_derived_daily_symbol_date ON derived_daily(symbol, date);

-- Journal signal events
CREATE SEQUENCE IF NOT EXISTS signal_events_id_seq START 1;

CREATE TABLE IF NOT EXISTS signal_events (
  id BIGINT PRIMARY KEY DEFAULT nextval('signal_events_id_seq'),
  created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  date DATE NOT NULL,
  symbol VARCHAR NOT NULL,
  context VARCHAR NOT NULL,

  snapshot_date DATE,
  contract_symbol VARCHAR,

  payload_json JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signal_events_symbol_date ON signal_events(symbol, date);
CREATE INDEX IF NOT EXISTS idx_signal_events_context_date ON signal_events(context, date);

-- Daily candles cache (settings-aware)
CREATE TABLE IF NOT EXISTS candles_daily (
  symbol VARCHAR NOT NULL,
  interval VARCHAR NOT NULL,
  auto_adjust BOOLEAN NOT NULL,
  back_adjust BOOLEAN NOT NULL,

  ts TIMESTAMP NOT NULL,

  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  dividends DOUBLE,
  splits DOUBLE,
  capital_gains DOUBLE,

  PRIMARY KEY(symbol, interval, auto_adjust, back_adjust, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_daily_symbol_ts ON candles_daily(symbol, ts);

CREATE TABLE IF NOT EXISTS candles_meta (
  symbol VARCHAR NOT NULL,
  interval VARCHAR NOT NULL,
  auto_adjust BOOLEAN NOT NULL,
  back_adjust BOOLEAN NOT NULL,

  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
  rows BIGINT NOT NULL,
  start_ts TIMESTAMP,
  end_ts TIMESTAMP,

  PRIMARY KEY(symbol, interval, auto_adjust, back_adjust)
);

-- Options snapshot headers (inventory / metadata)
CREATE TABLE IF NOT EXISTS options_snapshot_headers (
  symbol VARCHAR NOT NULL,
  snapshot_date DATE NOT NULL,
  provider VARCHAR NOT NULL,

  chain_path VARCHAR NOT NULL,
  meta_path VARCHAR,
  raw_path VARCHAR,

  spot DOUBLE,
  risk_free_rate DOUBLE,
  contracts BIGINT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
  meta_json JSON,

  PRIMARY KEY(symbol, snapshot_date, provider)
);

CREATE INDEX IF NOT EXISTS idx_options_snapshot_headers_symbol_date
  ON options_snapshot_headers(symbol, snapshot_date);
