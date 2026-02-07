# Alpaca: Ingesting `SPXW*` Option Contracts (Root Symbol)

`SPXW` contracts are commonly identified by an OCC/OSI-style `contractSymbol` that begins with `SPXW`. In Alpaca, these may not be discoverable via the usual `underlying_symbol=SPX` filters, so this repo supports discovering contracts by **root symbol**.

This tool is for information/decision support only — **not financial advice**.

## Requirements

- Alpaca credentials in the environment:
  - `APCA_API_KEY_ID`
  - `APCA_API_SECRET_KEY`
- Optional:
  - `OH_ALPACA_OPTIONS_FEED` (e.g. `opra`)

## Discover contracts (metadata only)

Persists contract metadata + contract snapshot rows (OI/close when available) into DuckDB:

```bash
./.venv/bin/options-helper --provider alpaca ingest options-bars \
  --contracts-root-symbol SPXW \
  --contracts-symbol-prefix SPXW \
  --contracts-exp-start 2016-01-01 \
  --contracts-exp-end 2030-12-31 \
  --contracts-only
```

## Discover + backfill daily option bars

Backfills daily bars for the discovered contracts:

```bash
./.venv/bin/options-helper --provider alpaca ingest options-bars \
  --contracts-root-symbol SPXW \
  --contracts-symbol-prefix SPXW \
  --contracts-exp-start 2016-01-01 \
  --contracts-exp-end 2030-12-31 \
  --lookback-years 10
```

## Safety knobs (recommended for large runs)

- `--max-contracts N`: cap total contracts discovered/ingested
- `--max-expiries N`: only ingest most-recent expiries
- `--contracts-page-size N`: page size for `/v2/options/contracts`
- `--contracts-max-rps X`: throttle contract discovery
- `--bars-concurrency N` / `--bars-max-rps X`: throttle bars backfills

## Where the data goes

By default, DuckDB writes to `data/warehouse/options.duckdb` (override with `--duckdb-path`).

- `option_contracts`: contract dimension rows
- `option_contract_snapshots`: per-run-day snapshot rows (OI/close + raw JSON payload)
- `option_bars`: daily bars for contracts
- `option_bars_meta`: ingestion status/coverage per contract

## Notes / limitations

- Alpaca options coverage varies by subscription and asset class. If Alpaca does not support `SPXW` in your account/feed, discovery may return zero contracts.
- `--contracts-symbol-prefix` is applied **after** discovery as a guardrail (useful if Alpaca returns mixed roots for a query).

