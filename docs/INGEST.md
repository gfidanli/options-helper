# Ingestion (DuckDB)

Bulk backfills for daily candles and Alpaca option bars. DuckDB is the default storage backend
(`--storage filesystem` opts out). **Not financial advice.**

## Prerequisites
- Install Alpaca extras: `pip install -e ".[dev,alpaca]"`
- Set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` (or use `config/alpaca.env`)
- Ensure your Alpaca account has the required market data entitlements (options feeds often require upgrades).

## Pipeline diagram

![Ingestion pipeline](assets/diagrams/generated/ingest_pipeline.svg)

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
- `--candles-concurrency N` (stock-bars fetch workers; default `1`)
- `--candles-max-rps FLOAT` (soft throttle for stock-bars requests/sec; default `8.0`)
- `--alpaca-http-pool-maxsize N`, `--alpaca-http-pool-connections N`
- `--log-rate-limits/--no-log-rate-limits`
- `--auto-tune/--no-auto-tune`
- `--tune-config PATH` (default: `config/ingest_tuning.json`, local state)

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
- `--page-limit N` (default: 200, contracts pagination safety cap)
- `--contracts-page-size N` (default: `10000`)
- `--max-underlyings N`, `--max-contracts N`, `--max-expiries N`
- `--contracts-max-rps FLOAT` (default: `2.5`, soft throttle for options-contracts requests/sec)
- `--bars-concurrency N` (default: `8`; forced to `1` with `--fail-fast`)
- `--bars-max-rps FLOAT` (default: `30.0`, soft throttle for options-bars requests/sec)
- `--bars-batch-mode adaptive|per-contract` (default: `adaptive`)
- `--bars-batch-size N` (default: `8`)
- `--bars-write-batch-size N` (default: `200`, batches bars/meta writes to DuckDB)
- `--alpaca-http-pool-maxsize N` (override Alpaca `requests` pool max size for this run)
- `--alpaca-http-pool-connections N` (override Alpaca `requests` pool connection pools for this run)
- `--log-rate-limits/--no-log-rate-limits` (override per-request Alpaca rate-limit logging)
- `--auto-tune/--no-auto-tune`
- `--tune-config PATH` (default: `config/ingest_tuning.json`, local state)
- `--resume/--no-resume` (uses `option_bars_meta` coverage)
- `--dry-run` (no writes)
- `--fetch-only` (benchmark mode: fetches contracts/bars, skips warehouse writes)
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
- Endpoint tuning baseline (Alpaca): keep `--contracts-max-rps` around `2.5` and start bars at `--bars-concurrency 8 --bars-max-rps 30.0`.
- Throughput tuning loop: enable `--log-rate-limits`, increase bars knobs until you first observe bars `status=429`, then back off to ~80% of that setting.
- If bars throughput plateaus before any 429s, raise Alpaca HTTP pool sizes (for example `--alpaca-http-pool-maxsize 256 --alpaca-http-pool-connections 256`) before pushing concurrency/RPS higher.
- Use `--fetch-only` for benchmarking raw fetch throughput without DuckDB write overhead.
- `--auto-tune` updates `config/ingest_tuning.json` from observed endpoint stats (429s/timeouts/latency/splits).
- For endpoint-by-endpoint tuning status and DuckDB optimization details, see [Ingestion optimization playbook](INGEST_OPTIMIZATION.md).
