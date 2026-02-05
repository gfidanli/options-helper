# Ingestion (DuckDB)

Bulk backfills for daily candles and Alpaca option bars. DuckDB is the default storage backend
(`--storage filesystem` opts out). **Not financial advice.**

## Prerequisites
- Install Alpaca extras: `pip install -e ".[dev,alpaca]"`
- Set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` (or use `config/alpaca.env`)
- Ensure your Alpaca account has the required market data entitlements (options feeds often require upgrades).

## Commands

### Candles (daily OHLCV)
Backfills daily candles for watchlist symbols (default watchlists: `positions`, `monitor`).

```bash
options-helper ingest candles
options-helper ingest candles --watchlist positions --watchlist monitor
options-helper ingest candles --symbol AAPL --symbol MSFT
```

Common flags:
- `--watchlists-path PATH`
- `--watchlist NAME` (repeatable)
- `--symbol TICKER` (repeatable; overrides watchlists)
- `--candle-cache-dir PATH`

### Options bars (daily OHLCV + vwap + trade_count)
Discovers Alpaca option contracts (expired + active) and backfills daily bars. Bars are fetched
per-contract symbol to maximize coverage (slower but more complete).

```bash
options-helper ingest options-bars --watchlist positions --watchlist monitor
```

Common flags:
- `--contracts-exp-start YYYY-MM-DD`
- `--contracts-exp-end YYYY-MM-DD` (default: today + 5y)
- `--lookback-years N` (default: 10)
- `--page-limit N` (default: 200)
- `--max-underlyings N`, `--max-contracts N`, `--max-expiries N`
- `--resume/--no-resume` (uses `option_bars_meta` coverage)
- `--dry-run` (no writes)
- `--fail-fast/--best-effort`

## What gets stored
All ingestion writes to the DuckDB warehouse (default: `data/warehouse/options.duckdb`):

- `candles_daily`
- `option_contracts`
- `option_contract_snapshots`
- `option_bars`
- `option_bars_meta` (coverage / resume metadata)

## Operational notes
- Large universes can take time; start with `--max-underlyings` or `--dry-run`.
- If you see 403/402 errors, your data entitlement likely needs adjustment.
- Avoid running multiple ingestion jobs concurrently (DuckDB is single-writer).
- `ingest options-bars` is resumable: it records per-contract attempts in `option_bars_meta`, skips contracts already attempted today, and avoids refetching historical data for expired expiries.
