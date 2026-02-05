You are working in the repo https://github.com/gfidanli/options-helper (already checked out locally).

Goal: implement the DuckDB storage backend epic (local-first “warehouse + lake”) using **PR-sized commits**
that match these plan IDs exactly: **IMP-040, IMP-041, IMP-042, IMP-043**.

Before coding:
1) Read these plan docs in `docs/plans/`:
   - `EPIC-DUCKDB.md`
   - `IMP-040.md`
   - `IMP-041.md`
   - `IMP-042.md`
   - `IMP-043.md`
2) Keep **filesystem** behavior as the default. DuckDB must be opt-in via `--storage duckdb`.
3) All tests must be **offline + deterministic**. Do not add network calls.

Repo constraints:
- Python >= 3.10
- Use `duckdb` Python dependency (add to `pyproject.toml`)
- Avoid heavy deps; do NOT add `pyarrow` as a required dependency.
- Prefer additive changes; refactor call sites minimally.

--------------------------------------------------------------------
PR/Commit plan (DO NOT COMBINE THESE):

========================
COMMIT / PR: IMP-040
========================
Title: "IMP-040: DuckDB scaffold + migrations + CLI toggles"

Implement the DuckDB foundation:
- Add `duckdb` to `[project].dependencies` in `pyproject.toml`
- Create new package `options_helper/db/`:
  - `__init__.py`
  - `warehouse.py` (DuckDBWarehouse wrapper)
  - `migrations.py` with `ensure_schema()` + `current_schema_version()`
  - `schema_v1.sql` defining v1 tables (derived_daily, signal_events, candles_daily/meta, options_snapshot_headers)
    *Use a SEQUENCE for signal_events id (DuckDB doesn’t support MySQL AUTO_INCREMENT).
- Create storage runtime + store factory:
  - `options_helper/data/storage_runtime.py` (contextvars: backend + duckdb_path)
  - `options_helper/data/store_factory.py` (get_warehouse + close_warehouses + get_*_store functions; filesystem-only routing is OK for now)
- Wire CLI:
  - Add global options: `--storage {filesystem|duckdb}` and `--duckdb-path PATH`
  - Add `db` Typer sub-app with:
    - `options-helper ... db init` (calls ensure_schema)
    - `options-helper ... db info` (prints schema version + path)
  - On CLI shutdown, call `close_warehouses()` and reset contextvars.
- Add docs:
  - `docs/DUCKDB.md` (how to use, what lives in warehouse vs lake)
- Add tests:
  - `tests/test_duckdb_migrations.py` verifying ensure_schema creates v1.

Acceptance for IMP-040:
- `options-helper db init` works in duckdb mode.
- `pytest` passes.

========================
COMMIT / PR: IMP-041
========================
Title: "IMP-041: Derived metrics + Journal → DuckDB"

Add DuckDB store implementations for derived + journal:
- Create `options_helper/data/stores_duckdb.py` with:
  - `DuckDBDerivedStore` (load/upsert)
  - `DuckDBJournalStore` (append/read/query + path())
- Update `options_helper/data/store_factory.py` to route derived + journal in duckdb mode.
- Update **CLI call sites** to use the factory instead of direct store constructors for:
  - Derived commands (`derived update/show/stats/...`)
  - Journal commands (`journal log/evaluate/...`)
- Add tests:
  - `tests/test_duckdb_derived_store.py`
  - `tests/test_duckdb_journal_store.py`

Acceptance for IMP-041:
- All existing derived/journal CLI commands work with both storage backends.
- `pytest` passes.

========================
COMMIT / PR: IMP-042
========================
Title: "IMP-042: Candle cache → DuckDB"

Add DuckDB-backed candle cache:
- Extend `options_helper/data/stores_duckdb.py` with `DuckDBCandleStore`:
  - Must be settings-aware (interval, auto_adjust, back_adjust)
  - Must remain API-compatible with `CandleStore` (load/save/load_meta)
- Update `options_helper/data/store_factory.py` to route candle store in duckdb mode.
- Update `options_helper/data/technical_backtesting_io.py` to use `get_candle_store()`.
- Update CLI call sites to use `get_candle_store()` everywhere candles are created.
- Add test:
  - `tests/test_duckdb_candle_store.py`

Acceptance for IMP-042:
- Offline and backtesting flows that rely on candle cache still work.
- `pytest` passes.

========================
COMMIT / PR: IMP-043
========================
Title: "IMP-043: Options snapshot lake (Parquet) + DuckDB index"

Add Parquet snapshot storage + DuckDB index:
- Add `save_day_snapshot()` to the filesystem `OptionsSnapshotStore` in `options_helper/data/options_snapshots.py`.
  - It should preserve the existing on-disk layout (one CSV per expiry + meta.json).
- Extend `options_helper/data/stores_duckdb.py` with `DuckDBOptionsSnapshotStore`:
  - Writes full-day chain to `.../{SYMBOL}/{YYYY-MM-DD}/chain.parquet` via DuckDB COPY
  - Writes compressed meta/raw JSON files
  - Upserts a header row into `options_snapshot_headers` keyed by (symbol, snapshot_date, provider)
  - `list_dates()` and `resolve_date()` must work without filesystem scanning
  - `load_day()` reads Parquet via DuckDB `read_parquet()`
- Update `options_helper/data/store_factory.py` to route options snapshot store in duckdb mode.
- Refactor `options_helper/data/options_snapshotter.py`:
  - Instead of writing per-expiry files in a loop, build a single `chain_df` and call `save_day_snapshot()` once per symbol/day.
- Update CLI call sites to use the factory for `OptionsSnapshotStore(...)`.
- Add tests:
  - `tests/test_duckdb_options_snapshot_store.py`
  - Extend/adjust existing snapshot store tests to cover filesystem `save_day_snapshot()`.

Acceptance for IMP-043:
- Snapshotting many symbols in duckdb mode produces Parquet partitions + DuckDB header rows.
- Existing filesystem snapshots remain readable.
- `pytest` passes.

--------------------------------------------------------------------
General implementation rules:
- Keep changes minimal and incremental; do NOT refactor unrelated code.
- Ensure imports are clean and avoid circular imports.
- Update docstrings where it helps future contributors.
- After each commit, run unit tests and fix failures before moving on.
