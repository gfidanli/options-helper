# Candle Cache (Daily OHLCV) — Design & Usage

## Goal
Make technical analysis efficient and consistent by:

1) Downloading historical candles **once** per symbol (daily OHLCV from the selected provider; default: Alpaca)
2) Persisting those candles locally
3) On every analysis run, downloading **only missing / recent candles** (incremental update)
4) Reshaping (resampling) daily candles into higher timeframes (3‑business‑day and weekly) for indicators

This avoids repeatedly fetching large history windows and enables computing higher-timeframe indicators that require
a deeper candle set (e.g., weekly EMA50, weekly RSI, breakout lookbacks).

## Data source / caveats
- Candles come from the active **market data provider** (`--provider`, default: `alpaca`).
- `yahoo` uses `yfinance` (Yahoo Finance). Data quality and timeliness are “best effort”.
- Daily candles may update after the close; the cache update strategy intentionally re-fetches a small number of recent
  candles to pick up revisions.

## Storage layout
By default the cache is stored relative to the current working directory:

- `data/candles/`
  - `{SYMBOL}.csv`
  - `{SYMBOL}.meta.json` (cache metadata: interval + adjustment settings)

Each file contains a date-time index and the OHLCV columns returned by the provider (commonly: `Open`, `High`, `Low`,
`Close`, `Volume`). Some providers may include additional columns (e.g. `Dividends`/`Stock Splits` from Yahoo, or
`trade_count`/`vwap` from Alpaca).

By default this project uses **adjusted OHLC**:
- `auto_adjust=True`
- `back_adjust=False`

In this mode, Yahoo/`yfinance` typically omits the `Adj Close` column because `Close` is already the adjusted series.
(Legacy caches may still include `Adj Close`; the cache can be upgraded on refresh.)

You can override the cache directory with the CLI option:

- `options-helper analyze portfolio.json --cache-dir path/to/cache`

## Update strategy (incremental fetch)
On every `analyze` run for each unique symbol:

1) Load cached daily candles (if present).
2) Compute the minimum start date required for the requested history window (`--period`, default `2y`).
3) Fetch missing candles:
   - If the cache is missing early history, backfill from the required start date up to the first cached candle.
   - Always refresh the most recent candles by re-fetching from `(last_cached_date - backfill_days)` to “now”.
4) Merge, de-duplicate by timestamp, sort, and write back to the cache file.

Defaults:
- `period`: `2y` (configurable)
- `backfill_days`: 5 (re-fetches the last few trading days to capture late corrections)

## Resampling (high timeframes)
All technical indicators are computed from the cached daily candles.

- Daily: compute on daily closes directly
- 3-business-day: `resample("3B").last()` from daily close series
- Weekly: `resample("W-FRI").last()` for prices and `resample("W-FRI").sum()` for volume

This ensures consistent indicators between runs and avoids extra Yahoo calls for higher timeframe data.

## Clearing the cache
To force a full re-download:

- delete files under `data/candles/` (or your chosen `--cache-dir`)

## Keeping candles fresh (cron)
The default cron job (`scripts/cron_daily_options_snapshot.sh`) runs:

- `options-helper refresh-candles portfolio.json` (portfolio symbols + watchlists)
- `options-helper snapshot-options portfolio.json --windowed --position-expiries` (options chain snapshots for position expiries)

This keeps the candle cache up to date daily so `research` can compute higher-timeframe technicals efficiently.

Related cron scripts (optional):
- Daily monitor watchlist option snapshots: `scripts/cron_daily_monitor_options_snapshot.sh`
- Daily briefing report: `scripts/cron_daily_briefing.sh`
- Weekly earnings refresh: `scripts/cron_weekly_refresh_earnings.sh`
