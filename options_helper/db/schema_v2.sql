-- options-helper DuckDB schema v2
-- Adds Alpaca candle fields + options contracts/bars storage.

ALTER TABLE candles_daily ADD COLUMN vwap DOUBLE;
ALTER TABLE candles_daily ADD COLUMN trade_count BIGINT;

-- Option contracts dimension (per contract symbol)
CREATE TABLE IF NOT EXISTS option_contracts (
  contract_symbol VARCHAR PRIMARY KEY,
  underlying VARCHAR,
  expiry DATE,
  option_type VARCHAR,
  strike DECIMAL(18,3),
  multiplier INTEGER,
  provider VARCHAR,
  updated_at TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_option_contracts_underlying_expiry
  ON option_contracts(underlying, expiry);

-- Option contract snapshots (per day/provider)
CREATE TABLE IF NOT EXISTS option_contract_snapshots (
  contract_symbol VARCHAR NOT NULL,
  as_of_date DATE NOT NULL,
  open_interest BIGINT,
  open_interest_date DATE,
  close_price DOUBLE,
  close_price_date DATE,
  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP DEFAULT current_timestamp,
  raw_json JSON,

  PRIMARY KEY(contract_symbol, as_of_date, provider)
);

-- Daily option bars (future-proof interval)
CREATE TABLE IF NOT EXISTS option_bars (
  contract_symbol VARCHAR NOT NULL,
  interval VARCHAR NOT NULL,
  ts TIMESTAMP NOT NULL,

  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  vwap DOUBLE,
  trade_count BIGINT,

  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP DEFAULT current_timestamp,

  PRIMARY KEY(contract_symbol, interval, ts, provider)
);

CREATE INDEX IF NOT EXISTS idx_option_bars_symbol_ts
  ON option_bars(contract_symbol, ts);

-- Ingestion meta/status for option bars
CREATE TABLE IF NOT EXISTS option_bars_meta (
  contract_symbol VARCHAR NOT NULL,
  interval VARCHAR NOT NULL,
  provider VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  rows BIGINT NOT NULL DEFAULT 0,
  start_ts TIMESTAMP,
  end_ts TIMESTAMP,
  last_success_at TIMESTAMP,
  last_attempt_at TIMESTAMP,
  last_error VARCHAR,
  error_count INTEGER NOT NULL DEFAULT 0,

  PRIMARY KEY(contract_symbol, interval, provider)
);
