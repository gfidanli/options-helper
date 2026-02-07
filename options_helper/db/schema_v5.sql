-- options-helper DuckDB schema v5
-- Adds persisted research metrics tables for IV surface, dealer exposure, and intraday flow.

CREATE TABLE IF NOT EXISTS iv_surface_tenor (
  symbol VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  tenor_target_dte INTEGER NOT NULL,
  expiry DATE,
  dte INTEGER,
  tenor_gap_dte INTEGER,
  atm_strike DOUBLE,
  atm_iv DOUBLE,
  atm_mark DOUBLE,
  straddle_mark DOUBLE,
  expected_move_pct DOUBLE,
  skew_25d_pp DOUBLE,
  skew_10d_pp DOUBLE,
  contracts_used INTEGER,
  warnings_json JSON,
  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  PRIMARY KEY(symbol, as_of, tenor_target_dte, provider)
);

CREATE INDEX IF NOT EXISTS idx_iv_surface_tenor_symbol_as_of
  ON iv_surface_tenor(symbol, as_of);
CREATE INDEX IF NOT EXISTS idx_iv_surface_tenor_provider_as_of
  ON iv_surface_tenor(provider, as_of);

CREATE TABLE IF NOT EXISTS iv_surface_delta_buckets (
  symbol VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  tenor_target_dte INTEGER NOT NULL,
  expiry DATE,
  option_type VARCHAR NOT NULL,
  delta_bucket VARCHAR NOT NULL,
  avg_iv DOUBLE,
  median_iv DOUBLE,
  n_contracts INTEGER,
  warnings_json JSON,
  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  PRIMARY KEY(symbol, as_of, tenor_target_dte, option_type, delta_bucket, provider)
);

CREATE INDEX IF NOT EXISTS idx_iv_surface_delta_symbol_as_of
  ON iv_surface_delta_buckets(symbol, as_of);
CREATE INDEX IF NOT EXISTS idx_iv_surface_delta_provider_as_of
  ON iv_surface_delta_buckets(provider, as_of);

CREATE TABLE IF NOT EXISTS dealer_exposure_strikes (
  symbol VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  expiry DATE NOT NULL,
  strike DOUBLE NOT NULL,
  call_oi DOUBLE,
  put_oi DOUBLE,
  call_gex DOUBLE,
  put_gex DOUBLE,
  net_gex DOUBLE,
  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  PRIMARY KEY(symbol, as_of, expiry, strike, provider)
);

CREATE INDEX IF NOT EXISTS idx_dealer_exposure_symbol_as_of
  ON dealer_exposure_strikes(symbol, as_of);
CREATE INDEX IF NOT EXISTS idx_dealer_exposure_symbol_strike
  ON dealer_exposure_strikes(symbol, strike);

CREATE TABLE IF NOT EXISTS intraday_option_flow (
  symbol VARCHAR NOT NULL,
  market_date DATE NOT NULL,
  source VARCHAR NOT NULL,
  contract_symbol VARCHAR NOT NULL,
  expiry DATE,
  option_type VARCHAR,
  strike DOUBLE,
  delta_bucket VARCHAR,
  buy_volume DOUBLE,
  sell_volume DOUBLE,
  unknown_volume DOUBLE,
  buy_notional DOUBLE,
  sell_notional DOUBLE,
  net_notional DOUBLE,
  trade_count BIGINT,
  unknown_trade_share DOUBLE,
  quote_coverage_pct DOUBLE,
  warnings_json JSON,
  provider VARCHAR NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,

  PRIMARY KEY(symbol, market_date, source, contract_symbol, provider)
);

CREATE INDEX IF NOT EXISTS idx_intraday_option_flow_symbol_day
  ON intraday_option_flow(symbol, market_date);
CREATE INDEX IF NOT EXISTS idx_intraday_option_flow_contract_day
  ON intraday_option_flow(contract_symbol, market_date);
