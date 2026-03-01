# Data Contracts

Technical backtesting data contracts support research/decision workflows only. They are not trade recommendations or financial advice.

## 1) Canonical CandleFrame Contract
Required columns:
- `Open`, `High`, `Low`, `Close`

Optional columns:
- `Volume`

Index requirements:
- `DatetimeIndex`
- sorted ascending
- unique timestamps

Normalization behavior:
- Input can be index-based or include a date-like column (`date`/`datetime`/`timestamp`).
- Runtime parses timestamps with UTC semantics.
- tz-aware timestamps are converted to UTC and stored as tz-naive.
- tz-naive timestamps are treated as UTC.

## 2) Daily Cache Input Contract
`technicals` daily workflows use candle cache/yfinance-backed OHLC data.

Adjustment policy:
- `data.candles.price_adjustment.auto_adjust` and `back_adjust` must be explicit.
- Both cannot be `true` simultaneously.
- Adjustments are persisted in artifact metadata.

## 3) Intraday Partition Contract
Intraday mode reads `IntradayStore` stock bar partitions:
- `kind="stocks"`
- `dataset="bars"`
- `timeframe in {"1Min", "5Min"}`
- per-day partitions over `[intraday_start, intraday_end]`

Accepted timestamp columns in partitions:
- `timestamp` or `ts` or `time`

Accepted OHLCV columns (case-insensitive):
- `open`, `high`, `low`, `close`, optional `volume`

Missing/empty partitions:
- warning + continue
- recorded in intraday coverage metadata

## 4) Intraday Resample Contract
Resampling target interval:
- accepted forms: e.g. `1Min`, `5Min`, `15m`, `30m`, `1h`
- must be `>=` base timeframe
- must be an integer multiple of base timeframe

Resample semantics:
- UTC-normalized index
- `label="left"`, `closed="left"`
- OHLC aggregation: first/max/min/last
- Volume aggregation: sum (`min_count=1`)

## 5) FeatureFrame Contract
`compute_features` output is CandleFrame plus feature columns, including:
- ATR/ATRP (`atr_*`, `atrp_*`)
- SMA (`sma_*`)
- z-score (`zscore_*`)
- Bollinger (`bb_*`)
- extension (`extension_atr_*_*`)
- optional RSI (`rsi_*`)
- weekly columns: `weekly_sma_*`, `weekly_trend_up`

## 6) No-Lookahead Timing Contract
- Weekly regime values are shifted by one completed weekly bar before forward-fill.
- Strategy logic that is close-confirmed at bar `t` must anchor execution from `t+1` open when `trade_on_close=false`.
- `CvdDivergenceMSB` pivot signals are only consumed after pivot confirmation lag (`pivot_right`).

## 7) Artifact Data Meta Contract
Technical backtesting params artifacts include `data` payload with:
- `start`, `end`, `bars`, `warmup_bars`, `interval`
- optional `intraday_coverage`:
  - `symbol`, `base_timeframe`, `target_interval`
  - `requested_days`, `loaded_days`, `missing_days`, `empty_days`
  - day counts and row counts

This metadata is part of auditability for best-effort data quality and anti-lookahead interpretation.
