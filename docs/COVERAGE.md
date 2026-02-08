# Coverage

This feature reports local data coverage for one symbol from DuckDB and suggests incremental repair commands.

Use this for diagnostics and operations planning only. It is **not financial advice**.

## Design

- DuckDB is the source of truth for coverage metrics.
- Legacy filesystem CSV/JSON files may still exist, but they are not authoritative for this view.
- Default deep lookback is 60 business days.
- Coverage is read-only: no network calls and no writes.

## What it measures

For a symbol, coverage reports:

- Candle coverage: row count, date range, missing business days in lookback, missing OHLCV cells.
- Options snapshot header coverage: days captured and contracts-per-day stats.
- Contract/OI snapshot coverage: contract/day coverage, OI presence, and OI delta coverage (1d/3d/5d).
- Option bars coverage from `option_bars_meta`: status counts and lookback-end coverage.
- Repair suggestions: copy/paste commands only.

## CLI

```bash
./.venv/bin/options-helper coverage SPY --days 60
```

JSON output:

```bash
./.venv/bin/options-helper coverage SPY --days 60 --json --out data/reports/coverage_spy.json
```

Useful flags:

- `--days`: business-day lookback window.
- `--json`: emit machine-readable JSON.
- `--out`: save JSON payload to a file.
- `--duckdb-path`: override DuckDB file path.

## Repair command semantics

Coverage suggestions are deterministic and local-data-only:

- Candle gaps: suggest `ingest candles --symbol SYMBOL`.
- Contract/OI day gaps: suggest `ingest options-bars --symbol SYMBOL --contracts-only ...` for forward capture.
- Option bars gaps: suggest `ingest options-bars --symbol SYMBOL --lookback-years N --resume`.
- Snapshot header gaps: suggest watchlist + `snapshot-options` commands.

Notes:

- OI history is point-in-time; many missing historical OI days cannot be backfilled.
- Sparse options snapshot-day coverage often reflects live-only chain snapshot behavior; use bars backfills for historical studies and snapshots for forward daily capture.
- Weekends/holidays are treated as non-business days and are not counted as gaps.

## Streamlit

The portal includes a read-only Coverage page:

- `apps/streamlit/pages/08_Coverage.py`
- backed by `apps/streamlit/components/coverage_page.py`

It uses cached reads and does not trigger ingestion writes on page load.
