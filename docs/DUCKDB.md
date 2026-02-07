# DuckDB storage backend

This repo is **local-first**: it should work great for a single trader on a single machine.

Historically, most data was persisted as many CSV/JSON files under `data/…`. This works early,
but becomes painful when you start storing:

- full-chain option snapshots across many symbols/dates
- multi-year daily candles for many symbols
- derived metrics histories and journal events

DuckDB mode keeps the CLI UX the same while giving you a “real database” under the hood.
It is now the **default** storage backend (opt out with `--storage filesystem`).

## What gets stored where

### Warehouse (DuckDB file)
Default: `data/warehouse/options.duckdb`

DuckDB stores:
- schema migrations + versioning
- candle cache tables
- derived metrics table
- journal events table
- option contracts + option bars tables
- research metrics tables (`iv_surface_tenor`, `iv_surface_delta_buckets`,
  `dealer_exposure_strikes`, `intraday_option_flow`)
- option snapshot *headers/index* (what exists, where it lives)

### Lake (Parquet partitions)
Default: under the existing cache dirs (e.g., `data/options_snapshots/...`)

Parquet stores:
- large fact tables, especially full-chain snapshots:
  - `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/chain.parquet`

The warehouse stores the **path** and metadata so it can find and load Parquet quickly.

## How to use

Initialize the DB (optional but recommended):

```bash
options-helper db init
```

Run any existing command (DuckDB is default):

```bash
options-helper snapshot-options --watchlist core
options-helper analyze --symbol AAPL
```

Opt out (legacy filesystem storage):

```bash
options-helper --storage filesystem analyze --symbol AAPL
```

## CLI flags

- `--storage filesystem|duckdb` (default: `duckdb`)
- `--duckdb-path PATH` (only used in duckdb mode)
- `--duckdb-snapshot-legacy-files/--no-duckdb-snapshot-legacy-files`
  (default: `--duckdb-snapshot-legacy-files`; disable for faster snapshot writes with no CSV/raw sidecars)

## Operational notes

- Back up the DB file: copy `data/warehouse/options.duckdb`
- Parquet partitions are plain files; back up `data/options_snapshots/` too if you want snapshots.
- Schema changes are applied via migrations (see `options_helper/db/migrations.py`).

## Why DuckDB
DuckDB is an embedded analytical database designed for fast local analytics and integrates extremely well with Python/pandas.
It also supports storing and querying Parquet directly, which makes it a strong fit for a single-machine “quant research lakehouse.”
