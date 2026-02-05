# Plan: DuckDB-by-default + Alpaca ingestion backfills (candles + options daily bars)

**Generated**: 2026-02-05

## Overview
Make DuckDB the default storage backend across the CLI, then add robust Alpaca-based ingestion commands to:
1) backfill **daily underlying candles** for watchlists `positions` + `monitor`, stored in DuckDB, and
2) backfill **options contract metadata + daily option bars** (all expiries/strikes per ticker, expired + active) as far back as Alpaca provides, stored in DuckDB.

This plan explicitly does **not** attempt “historical option chain snapshots” (providers generally only provide *current* chain snapshots); instead, it ingests **daily option bars** + **contract metadata**, and relies on existing `snapshot-options` to collect once-daily chain snapshots going forward.

## Decisions (locked)
- Storage default: **DuckDB for all commands** (filesystem remains opt-out via `--storage filesystem`).
- Candle backfill symbols: **watchlists `positions` + `monitor`**.
- Options history: **daily option bars**, **all expiries + all strikes** per underlying, **include expired + active**, **max available**.

## Prerequisites
- Runtime ingestion requires Alpaca SDK: `pip install -e ".[alpaca]"` (recommend updating README to `.[dev,alpaca]`).
- Alpaca credentials via env or `config/alpaca.env` (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`).
- Appropriate Alpaca market data entitlements / feeds (options feed commonly fails with 403/402 if not enabled).

## Public API / Interface changes
### CLI
- Change global default: `--storage` defaults to `duckdb` (still supports `filesystem`).
- Add new command group: `options-helper ingest`
  - `options-helper ingest candles`
  - `options-helper ingest options-bars`

### DuckDB schema (new v2 migration)
- Extend `candles_daily` to store Alpaca fields currently dropped:
  - `vwap DOUBLE`, `trade_count BIGINT`
- Add new options warehouse tables (DuckDB):
  - `option_contracts` (dimension keyed by `contract_symbol`)
  - `option_contract_snapshots` (time-varying fields like open interest / close price, keyed by `contract_symbol` + `as_of_date`)
  - `option_bars` (daily bars keyed by `contract_symbol` + `interval` + `ts`)
  - `option_bars_meta` (checkpoint + status per contract+interval for resumable ingestion)

## Dependency graph (high level)
```text
T1 ──┬── T3 ──┐
     │        ├── T6 ── T7 ── T8
     ├── T4 ──┤
     │        └── T5 ──┘
T2 ───────────┘
```

## Tasks

### T1: DuckDB schema v2 + migrations
- **depends_on**: []
- **location**:
  - `options_helper/db/migrations.py`
  - `options_helper/db/schema_v2.sql` (new)
- **description**:
  - Add schema version 2 migration support to `ensure_schema(...)`:
    - Apply v1 if missing (existing).
    - Apply v2 if current version < 2.
  - Create `schema_v2.sql` that:
    - `ALTER TABLE candles_daily ADD COLUMN vwap DOUBLE;`
    - `ALTER TABLE candles_daily ADD COLUMN trade_count BIGINT;`
    - Create new tables + indexes (all `CREATE TABLE IF NOT EXISTS`):
      - `option_contracts`:
        - `contract_symbol VARCHAR PRIMARY KEY`
        - `underlying VARCHAR`
        - `expiry DATE`
        - `option_type VARCHAR`
        - `strike DECIMAL(18,3)` (avoid float-equality footguns)
        - `multiplier INTEGER`
        - `provider VARCHAR`
        - `updated_at TIMESTAMP DEFAULT current_timestamp`
      - `option_contract_snapshots`:
        - `contract_symbol VARCHAR NOT NULL`
        - `as_of_date DATE NOT NULL`
        - `open_interest BIGINT`
        - `open_interest_date DATE`
        - `close_price DOUBLE`
        - `close_price_date DATE`
        - `provider VARCHAR NOT NULL`
        - `updated_at TIMESTAMP DEFAULT current_timestamp`
        - `raw_json JSON` (optional, best-effort; keep nullable)
        - PRIMARY KEY (`contract_symbol`, `as_of_date`, `provider`)
      - `option_bars` (store daily bars; future-proof interval):
        - `contract_symbol VARCHAR NOT NULL`
        - `interval VARCHAR NOT NULL` (e.g. `1d`)
        - `ts TIMESTAMP NOT NULL`
        - `open DOUBLE`, `high DOUBLE`, `low DOUBLE`, `close DOUBLE`
        - `volume DOUBLE`
        - `vwap DOUBLE`
        - `trade_count BIGINT`
        - `provider VARCHAR NOT NULL`
        - `updated_at TIMESTAMP DEFAULT current_timestamp`
        - PRIMARY KEY (`contract_symbol`, `interval`, `ts`, `provider`)
      - `option_bars_meta` (resumable checkpoint + error tracking):
        - `contract_symbol VARCHAR NOT NULL`
        - `interval VARCHAR NOT NULL`
        - `provider VARCHAR NOT NULL`
        - `status VARCHAR NOT NULL` (`pending|ok|partial|error|forbidden|not_found`)
        - `rows BIGINT NOT NULL DEFAULT 0`
        - `start_ts TIMESTAMP`
        - `end_ts TIMESTAMP`
        - `last_success_at TIMESTAMP`
        - `last_attempt_at TIMESTAMP`
        - `last_error VARCHAR`
        - `error_count INTEGER NOT NULL DEFAULT 0`
        - PRIMARY KEY (`contract_symbol`, `interval`, `provider`)
    - Indexes:
      - `idx_option_contracts_underlying_expiry`
      - `idx_option_bars_symbol_ts` (contract_symbol, ts)
  - Record `schema_version=2` in `schema_migrations`.
- **validation**:
  - Add test `tests/test_duckdb_migrations_v2.py`:
    - initialize empty DB, run `ensure_schema`, assert version 2 and tables/columns exist.
- **status**: Completed
- **log**: Added v2 migration + schema file, updated migrations to apply v1→v2, and added v2 migration tests.
- **files**: `options_helper/db/migrations.py`, `options_helper/db/schema_v2.sql`, `tests/test_duckdb_migrations.py`, `tests/test_duckdb_migrations_v2.py`

### T2: Flip default storage backend to DuckDB
- **depends_on**: [T1]
- **location**:
  - `options_helper/cli.py`
  - `options_helper/data/storage_runtime.py`
  - docs updates in T8
- **description**:
  - Change CLI default `--storage` from `filesystem` → `duckdb`.
  - Change `_DEFAULT_STORAGE_BACKEND` default from `"filesystem"` → `"duckdb"`.
  - Ensure `--storage filesystem` still behaves as before.
- **validation**:
  - Update/add tests asserting default backend is duckdb and contextvars reset after CLI run (adapt `tests/test_duckdb_cli_db.py`).
- **status**: Completed
- **log**: Switched CLI/runtime defaults to duckdb, aligned storage runtime fallback behavior, and updated CLI tests for the new default/reset flow.
- **files**: `options_helper/cli.py`, `options_helper/data/storage_runtime.py`, `tests/test_duckdb_cli_db.py`

### T3: Preserve + persist Alpaca `vwap` / `trade_count` in candle ingestion
- **depends_on**: [T1]
- **location**:
  - `options_helper/data/alpaca_client.py`
  - `options_helper/data/stores_duckdb.py`
- **description**:
  - Update Alpaca stock bars normalization to retain vwap/trade count:
    - rename Alpaca columns (`vw|vwap`) → `VWAP`
    - rename (`n|trade_count|tradeCount`) → `Trade Count`
    - keep nullable; do not coerce missing to 0.
  - Extend `DuckDBCandleStore`:
    - Save: map `VWAP` → `vwap`, `Trade Count` → `trade_count`.
    - Load: include those columns in SELECT (as `VWAP` / `Trade Count`).
- **validation**:
  - Add test `tests/test_duckdb_candle_store_vwap_trade_count.py` verifying roundtrip.
- **status**: Completed
- **log**: Normalized Alpaca stock bars to keep VWAP/trade count, persisted in DuckDB, and added a roundtrip test.
- **files**: `options_helper/data/alpaca_client.py`, `options_helper/data/stores_duckdb.py`, `tests/test_duckdb_candle_store_vwap_trade_count.py`

### T4: DuckDB option contracts store (metadata + daily snapshots)
- **depends_on**: [T1]
- **location**:
  - `options_helper/data/stores_duckdb.py` (or new module `options_helper/data/option_contracts_duckdb.py`)
  - `options_helper/data/store_factory.py`
  - `options_helper/cli_deps.py`
- **description**:
  - Implement `DuckDBOptionContractsStore` with:
    - `upsert_contracts(df_contracts, *, provider, as_of_date, raw_by_contract_symbol=None, meta=None)`:
      - Upsert dimension fields into `option_contracts`.
      - Upsert per-day `option_contract_snapshots` for fields present (OI/close price + dates).
    - `list_contracts(underlying, *, exp_gte=None, exp_lte=None)` returning DataFrame.
  - Wire into `store_factory` + `cli_deps` as a stable seam for tests.
  - For new ingestion paths: fail fast in filesystem mode with a clear error (“requires DuckDB backend”).
- **validation**:
  - Test `tests/test_duckdb_option_contracts_store.py`: upsert + query + snapshot upsert idempotency.
- **status**: Completed
- **log**: Added DuckDB option contracts store (dimension + snapshots), wired factory/CLI deps with duckdb-only guard, and added coverage for upsert/list/idempotency.
- **files**: `options_helper/data/stores_duckdb.py`, `options_helper/data/store_factory.py`, `options_helper/cli_deps.py`, `tests/test_duckdb_option_contracts_store.py`

### T5: Alpaca option daily bars fetcher (full OHLCV, paginated, resilient)
- **depends_on**: [T1]
- **location**:
  - `options_helper/data/alpaca_client.py`
- **description**:
  - Add `AlpacaClient.get_option_bars_daily_full(...)` that returns a flat DataFrame of **all bars** (not just latest-per-contract):
    - Inputs: `symbols: list[str]`, `start`, `end`, `interval="1d"`, `feed`, `chunk_size`, `max_retries`, `page_limit`.
    - Output columns: `contractSymbol`, `ts`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `trade_count`.
  - Implement pagination defensively:
    - Pass `limit` and `page_token`/equivalent if supported (use `_filter_kwargs` and “inspect response for next_page_token” pattern).
    - Enforce `page_limit` safety guard.
  - Rate-limit handling:
    - exponential backoff + jitter on 429; honor `Retry-After` if exposed by exception.
    - retry transient timeouts/5xx; do not retry other 4xx except 408/429.
- **validation**:
  - Unit tests with a fake payload object are sufficient (offline). Add normalization tests ensuring MultiIndex → flat rows works.
- **status**: Completed
- **log**: Added paginated daily option bars fetcher with retry/backoff handling and full OHLCV normalization, plus tests for pagination/page limits.
- **files**: `options_helper/data/alpaca_client.py`, `tests/test_alpaca_option_bars_daily_full.py`

### T6: DuckDB option bars store (upsert + resume metadata)
- **depends_on**: [T1, T5]
- **location**:
  - `options_helper/data/stores_duckdb.py` (or new module `options_helper/data/option_bars_duckdb.py`)
  - `options_helper/data/store_factory.py`
  - `options_helper/cli_deps.py`
- **description**:
  - Implement `DuckDBOptionBarsStore`:
    - `upsert_bars(df, *, interval="1d", provider, updated_at=now)`:
      - de-dupe rows by (`contractSymbol`, `ts`) before insert.
      - transactional delete+insert using a temp table join (idempotent chunk replay).
    - `mark_meta_success(contract_symbols, interval, provider, stats...)`
    - `mark_meta_error(contract_symbols, interval, provider, error, status)` increment error_count.
    - `coverage(contract_symbol, ...)` helper to decide whether to skip/resume.
  - DuckDB single-writer mitigation:
    - catch DuckDB lock errors; emit a clear message instructing user to avoid concurrent ingestion.
- **validation**:
  - Test `tests/test_duckdb_option_bars_store.py` verifies idempotent upsert + meta updates.
- **status**: Completed
- **log**: Added DuckDB option bars store with transactional upsert, meta success/error helpers, and lock guard; wired store factory/CLI deps and added store tests.
- **files**: `options_helper/data/option_bars.py`, `options_helper/data/stores_duckdb.py`, `options_helper/data/store_factory.py`, `options_helper/cli_deps.py`, `tests/test_duckdb_option_bars_store.py`
### T7: Add `ingest` CLI group with candle + option-bars backfills
- **depends_on**: [T2, T3, T4, T6]
- **location**:
  - `options_helper/commands/ingest.py` (new)
  - `options_helper/cli.py` (register new typer app)
  - `options_helper/data/ingestion/` (new helper modules to keep CLI thin and testable)
- **description**:
  - `ingest candles`:
    - Default `--watchlist positions --watchlist monitor` using `data/watchlists.json`.
    - For each symbol: `candle_store.get_daily_history(period="max")`.
    - Output per-symbol status + summary; best-effort continue on failures.
  - `ingest options-bars`:
    - Enforce provider == `alpaca` (fail fast otherwise).
    - Contract discovery (expired + active, “max available”):
      - Use **expiration-year windows** from `(today.year + 5)` down to `2000`, querying Alpaca contracts per-year.
      - Stop early after **3 consecutive empty years**.
      - Persist:
        - `option_contracts` dimension
        - `option_contract_snapshots` for `as_of=today`
    - Bars backfill (daily):
      - Group discovered contracts by `expiry`.
      - For each expiry group:
        - `end = min(today, expiry)`
        - `start = max(2000-01-01, min(today, expiry - lookback_years))` with `lookback_years` default **10**.
      - Chunk symbols (default 200).
      - On chunk failure: **bisect chunk** until isolating bad contract(s); record per-contract errors in `option_bars_meta` and continue.
      - Store bars to DuckDB via `DuckDBOptionBarsStore`.
    - Flags to keep the job controllable:
      - `--watchlists-path`, `--watchlist`, `--symbol` (override)
      - `--contracts-exp-start`, `--contracts-exp-end` (defaults: 2000-01-01 → today+5y)
      - `--lookback-years` (default 10)
      - `--chunk-size` (default 200)
      - `--page-limit` (default 200)
      - `--max-underlyings`, `--max-contracts`, `--max-expiries` (safety caps; default none)
      - `--resume/--no-resume` (default resume)
      - `--dry-run` (no writes; prints what would be fetched)
      - `--fail-fast/--best-effort` (default best-effort)
- **validation**:
  - CLI tests with injected fakes:
    - `tests/test_ingest_candles_command.py` (MockProvider + DuckDBCandleStore)
    - `tests/test_ingest_options_bars_command.py` (FakeAlpacaClient returning deterministic contracts + bars)
- **status**: Completed
- **log**: Added ingest CLI group for candles/options bars with helper modules for watchlist resolution, contract discovery, resumable bars backfill (chunking + bisection), and CLI tests using fake providers/clients.
- **files**: `options_helper/commands/ingest.py`, `options_helper/cli.py`, `options_helper/data/ingestion/__init__.py`, `options_helper/data/ingestion/common.py`, `options_helper/data/ingestion/candles.py`, `options_helper/data/ingestion/options_bars.py`, `tests/test_ingest_candles_command.py`, `tests/test_ingest_options_bars_command.py`

### T8: Documentation updates + runbook
- **depends_on**: [T2, T7]
- **location**:
  - `README.md`
  - `docs/DUCKDB.md`
  - `docs/INGEST.md` (new)
- **description**:
  - Update docs to reflect DuckDB is now default; filesystem is opt-out.
  - Add ingest runbook:
    - recommended install (`.[dev,alpaca]`)
    - required env vars + feed notes
    - example commands for:
      - candles backfill
      - options contracts + bars backfill
    - expected runtime/data volume caveats + “not financial advice”.
- **validation**:
  - `mkdocs serve` renders without broken links (manual).
- **status**: Completed
- **log**: Documented DuckDB as the default backend and added ingestion runbook with prerequisites, commands, flags, and operational notes.
- **files**: `README.md`, `docs/DUCKDB.md`, `docs/INGEST.md`

## Parallel execution groups

| Wave | Tasks | Can start when |
|------|-------|----------------|
| 1 | T1 | Immediately |
| 2 | T3, T4, T5 | T1 complete |
| 3 | T6 | T5 + T1 complete |
| 4 | T2 | T1 complete (but land after tests updated) |
| 5 | T7 | T2 + T3 + T4 + T6 complete |
| 6 | T8 | T2 + T7 complete |

## Testing strategy
- Unit-level DuckDB tests using temp DB files (`tmp_path / "options.duckdb"`).
- No network calls in tests:
  - Mock contracts + bars payloads.
  - Mock provider for candles.
- CLI tests via Typer’s runner, asserting:
  - correct tables are populated
  - failures are recorded in `option_bars_meta`
  - default backend is duckdb and context resets correctly after command exit.

## Risks & mitigations
- **Massive data volume / runtime**: chunking + safety caps + resumable `option_bars_meta`.
- **Rate limits / entitlements**: backoff + fail-fast on 403/402 with actionable message.
- **Poisoned batch requests**: bisect chunks to isolate bad contract symbols.
- **DuckDB single-writer locking**: keep transactions small; detect lock errors and exit with clear instructions.
- **Timezone / session differences**: normalize all timestamps to UTC-naive in storage; store `provider_params`/feed choice in meta (where applicable).

## Acceptance criteria
- Running `options-helper` with no `--storage` uses DuckDB and continues to function for existing commands.
- `options-helper ingest candles --watchlist positions --watchlist monitor` backfills candles into DuckDB.
- `options-helper ingest options-bars --watchlist positions --watchlist monitor` stores:
  - option contracts in `option_contracts` + `option_contract_snapshots`
  - daily option bars in `option_bars`
  - progress + errors in `option_bars_meta`
- All tests pass offline via `python -m pytest`.
