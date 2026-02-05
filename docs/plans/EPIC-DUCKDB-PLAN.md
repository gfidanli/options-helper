# Plan: DuckDB storage backend epic (IMP-040 → IMP-043)

**Generated**: 2026-02-05

## Overview
Implement the remaining wiring and snapshot API changes needed to make the already-present DuckDB “warehouse + lake” code reachable via the CLI, while keeping filesystem as the default backend and keeping all tests offline/deterministic.

This plan delivers **four PRs** (each PR contains only its IMP scope) with titles **exactly**:
- `IMP-040: DuckDB scaffold + migrations + CLI toggles`
- `IMP-041: Derived metrics + Journal → DuckDB`
- `IMP-042: Candle cache → DuckDB`
- `IMP-043: Options snapshot lake (Parquet) + DuckDB index`

## Public API / Interface changes
### CLI (global)
- Add `--storage {filesystem|duckdb}` (default `filesystem`)
- Add `--duckdb-path PATH` (default `data/warehouse/options.duckdb`)
- Add new command group: `db`
  - `options-helper ... db init`
  - `options-helper ... db info`

### Python APIs
- Add `OptionsSnapshotStore.save_day_snapshot(...)` (filesystem backend) with the same signature as DuckDB’s `save_day_snapshot`.

### Behavioral guarantees
- Filesystem remains default; DuckDB is **opt-in** via `--storage duckdb`.
- Tests remain **offline and deterministic** (stub providers and candle history where needed).
- Avoid heavy imports during `options-helper --help` (keep DuckDB imports lazy where possible).

## Dependency graph (high level)
IMP-040 ──► IMP-041 ──► IMP-042 ──► IMP-043
│ │ │ │
└─ CLI flags └─ derived/journal routing└─ candle routing└─ snapshot save_day + parquet/index


## Tasks

### T040-1: Add DuckDB dependency
- **depends_on**: []
- **location**: `pyproject.toml`
- **description**: Add `duckdb` to `[project].dependencies` (runtime dependency; no `pyarrow`).
- **validation**: `python -c "import duckdb; print(duckdb.__version__)"`
- **status**: completed
- **log**: Added `duckdb` to runtime dependencies.
- **files**: `pyproject.toml`
- **notes**: Import check optional; not required for plan completion.

### T040-2: Add `db` command group (lazy imports)
- **depends_on**: []
- **location**: `options_helper/commands/db.py` (new)
- **description**:
  - Create `typer.Typer()` app with `init` and `info`.
  - Ensure imports of `duckdb`/warehouse/migrations/store_factory happen **inside command functions** so `--help` stays light.
  - `db init`: ensure schema exists and print `{path} + schema version`.
  - `db info`: print `{path} + current schema version (0 if uninitialized)`.
- **validation**: `options-helper db info` and `options-helper db init` (with a temp `--duckdb-path`)
- **status**: completed
- **log**: Added `db` Typer app with `init`/`info` and lazy imports.
- **files**: `options_helper/commands/db.py`
- **notes**: Command wiring occurs in T040-3.

### T040-3: Wire storage runtime + shutdown cleanup in CLI
- **depends_on**: [T040-2]
- **location**: `options_helper/cli.py`
- **description**:
  - Add global options to root callback:
    - `--storage` validated via a small `Enum`/choice (fail fast on invalid).
    - `--duckdb-path` as `Path`.
  - In callback: set storage runtime contextvars via tokens (`set_default_storage_backend`, `set_default_duckdb_path`).
  - On close (`ctx.call_on_close`): reset both tokens and call `close_warehouses()`; keep imports lazy (import `close_warehouses` inside callback/on_close).
  - Register `db` sub-app (`app.add_typer(db_app, name="db")`).
- **validation**:
  - `pytest -q tests/test_cli_contract.py` (ensures `--help` still passes)
  - Manual: `options-helper --storage duckdb --duckdb-path /tmp/x.duckdb db init`
- **status**: completed
- **log**: Added `--storage`/`--duckdb-path`, set/reset storage contextvars, registered `db` sub-app, and closed warehouses on shutdown.
- **files**: `options_helper/cli.py`
- **validation_log**: `pytest -q tests/test_cli_contract.py`

### T040-4: Add CLI regression tests for DB commands + contextvar reset
- **depends_on**: [T040-3]
- **location**: `tests/test_duckdb_cli_db.py` (new) or extend `tests/test_cli_contract.py`
- **description**:
  - Use `typer.testing.CliRunner` to run `db info` / `db init` with `--duckdb-path tmp_path/...`.
  - Assert exit codes, and that the DB file exists after `db init`.
  - Add regression test that storage contextvars do **not** leak across invocations:
    - invoke once with `--storage duckdb`, then assert `get_default_storage_backend()` is back to `filesystem` afterward.
- **validation**: `pytest -q tests/test_duckdb_cli_db.py`
- **status**: completed
- **log**: Added CLI tests for `db info`, `db init`, and storage contextvar reset behavior.
- **files**: `tests/test_duckdb_cli_db.py`
- **validation_log**: `pytest -q tests/test_duckdb_cli_db.py`

### T040-5: Expose DuckDB doc in MkDocs nav (small doc wiring)
- **depends_on**: []
- **location**: `mkdocs.yml`
- **description**: Add `docs/DUCKDB.md` to `nav` (e.g., under “Data & Artifacts”).
- **validation**: `mkdocs build` (optional) and/or verify nav entry exists.
- **status**: completed
- **log**: Added DuckDB doc to MkDocs nav under Data & Artifacts.
- **files**: `mkdocs.yml`

### T040-6: PR/commit assembly + full test run
- **depends_on**: [T040-1, T040-2, T040-3, T040-4, T040-5]
- **location**: repo-wide
- **description**:
  - Open PR titled **exactly** `IMP-040: DuckDB scaffold + migrations + CLI toggles`.
  - Ensure `pytest` passes.
- **validation**: `./.venv/bin/python -m pytest`
- **status**: completed
- **log**: Committed IMP-040 and validated with full pytest run.
- **notes**: Full pytest run completed at epic wrap-up.

---

### T041-1: Route derived + journal via store factory through `cli_deps`
- **depends_on**: [T040-6]
- **location**: `options_helper/cli_deps.py`
- **description**:
  - Change `build_derived_store` to call `options_helper.data.store_factory.get_derived_store`.
  - Change `build_journal_store` to call `options_helper.data.store_factory.get_journal_store`.
  - Keep imports inside functions (avoid DuckDB import during `--help`).
- **validation**:
  - `options-helper --storage duckdb derived show --help` (smoke)
  - `pytest` full suite
- **status**: completed
- **log**: Routed `build_derived_store` and `build_journal_store` through the store factory with lazy imports.
- **files**: `options_helper/cli_deps.py`
- **validation_log**: `./.venv/bin/options-helper --storage duckdb derived show --help`

### T041-2: PR/commit assembly + full test run
- **depends_on**: [T041-1]
- **location**: repo-wide
- **description**:
  - Open PR titled **exactly** `IMP-041: Derived metrics + Journal → DuckDB`.
  - Ensure `pytest` passes.
- **validation**: `./.venv/bin/python -m pytest`
- **status**: completed
- **log**: Committed IMP-041 and validated with full pytest run.

---

### T042-1: Route candle store via factory through `cli_deps`
- **depends_on**: [T041-2]
- **location**: `options_helper/cli_deps.py`
- **description**:
  - Change `build_candle_store` to call `options_helper.data.store_factory.get_candle_store`.
  - Preserve passthrough kwargs exactly (`provider`, `fetcher`, `auto_adjust`, `back_adjust`).
  - Keep imports inside function.
- **validation**: `pytest -q tests/test_snapshot_options_full.py` (ensures stubbed candles still work)
- **status**: completed
- **log**: Routed `build_candle_store` through store factory while preserving kwargs passthrough.
- **files**: `options_helper/cli_deps.py`

### T042-2: Make technical backtesting IO respect storage backend
- **depends_on**: [T042-1]
- **location**: `options_helper/data/technical_backtesting_io.py`
- **description**: Replace direct `CandleStore(cache_dir)` with `get_candle_store(cache_dir)` in `load_ohlc_from_cache`.
- **validation**: `pytest -q tests/test_extension_stats_cli.py` (and full suite)
- **status**: completed
- **log**: Swapped direct CandleStore instantiation for factory-backed store in `load_ohlc_from_cache`.
- **files**: `options_helper/data/technical_backtesting_io.py`
- **validation_log**: `./.venv/bin/python -m pytest -q tests/test_extension_stats_cli.py`

### T042-3: Update candle construction in snapshotter data module
- **depends_on**: [T042-1]
- **location**: `options_helper/data/options_snapshotter.py`
- **description**: Replace `CandleStore(...)` construction with `get_candle_store(..., provider=provider)` (leave snapshot store construction for IMP-043).
- **validation**: `pytest -q tests/test_scanner_cli.py` (and full suite)
- **status**: completed
- **log**: Snapshotter now constructs candle store via `get_candle_store` for DuckDB awareness.
- **files**: `options_helper/data/options_snapshotter.py`

### T042-4: PR/commit assembly + full test run
- **depends_on**: [T042-1, T042-2, T042-3]
- **location**: repo-wide
- **description**:
  - Open PR titled **exactly** `IMP-042: Candle cache → DuckDB`.
  - Ensure `pytest` passes.
- **validation**: `./.venv/bin/python -m pytest`
- **status**: completed
- **log**: Committed IMP-042 and validated with full pytest run.

---

### T043-1: Add filesystem `save_day_snapshot()` API (unified writer)
- **depends_on**: [T042-4]
- **location**: `options_helper/data/options_snapshots.py`
- **description**:
  - Implement `save_day_snapshot(symbol, snapshot_date, *, chain, expiries, raw_by_expiry=None, meta=None)`.
  - Preserve layout:
    - `{day_dir}/{expiry}.csv`
    - `{day_dir}/{expiry}.raw.json` (when `raw_by_expiry` present)
    - `{day_dir}/meta.json`
  - Idempotence: prune *out-of-scope* expiry artifacts in the day dir (`*.csv` + matching `*.raw.json`) when their expiry is **not** in the provided `expiries` list.
  - Use `_upsert_meta` once with `meta` (caller supplies `underlying`, quote quality, etc).
  - Return `day_dir` (primary artifact is a directory in filesystem mode).
- **validation**: new unit tests (T043-6)
- **status**: completed
- **log**: Added filesystem `save_day_snapshot` with per-expiry CSV/raw writes, stale expiry pruning, and meta upsert.
- **files**: `options_helper/data/options_snapshots.py`

### T043-2: Refactor `snapshot-options` CLI workflow to use `save_day_snapshot()`
- **depends_on**: [T043-1]
- **location**: `options_helper/commands/workflows.py`
- **description**:
  - Replace per-expiry `save_expiry_snapshot*` + `_upsert_meta` usage with:
    - accumulate per-expiry DataFrames into one `chain_df`
    - accumulate `raw_by_expiry` in full-chain mode
    - compute `quote_quality` summary and include in `meta`
    - include `underlying` in `meta` in full-chain mode (use last/first successful expiry’s `raw["underlying"]`)
    - call `store.save_day_snapshot(...)` once per symbol/day
  - Keep filesystem behavior identical (tests in `tests/test_snapshot_options_full.py` must keep passing).
- **validation**: `pytest -q tests/test_snapshot_options_full.py`
- **status**: completed
- **log**: Snapshot-options now aggregates per-expiry frames/raw, computes quote quality once, and writes via `save_day_snapshot` per symbol/day.
- **files**: `options_helper/commands/workflows.py`
- **validation_log**: `./.venv/bin/python -m pytest -q tests/test_snapshot_options_full.py`

### T043-3: Refactor scanner snapshotter to use `save_day_snapshot()` once/day
- **depends_on**: [T043-1]
- **location**: `options_helper/data/options_snapshotter.py`
- **description**:
  - Stop using `OptionsSnapshotStore.save_expiry_snapshot*` and private helpers.
  - Build `chain_df` and `raw_by_expiry`, compute quote quality, and call `store.save_day_snapshot(...)`.
  - Switch snapshot store construction to factory: `get_options_snapshot_store(cache_dir)` (so duckdb mode writes Parquet + headers).
- **validation**: unit/integration test in T043-6 (stub provider)
- **status**: completed
- **log**: Snapshotter now builds a combined chain, computes quote quality, and saves via `save_day_snapshot` using the snapshot store factory.
- **files**: `options_helper/data/options_snapshotter.py`

### T043-4: Route snapshot store via factory through `cli_deps`
- **depends_on**: [T043-2, T043-3]
- **location**: `options_helper/cli_deps.py`
- **description**: Change `build_snapshot_store` to call `options_helper.data.store_factory.get_options_snapshot_store`.
- **validation**: `pytest -q tests/test_derived_cli.py tests/test_journal_cli.py` (and full suite)
- **status**: completed
- **log**: Routed `build_snapshot_store` through `get_options_snapshot_store` with lazy imports.
- **files**: `options_helper/cli_deps.py`

### T043-5: Replace remaining direct `OptionsSnapshotStore(...)` constructions on hot paths
- **depends_on**: [T043-4]
- **location**: repo-wide search; expected targets:
  - `options_helper/data/options_snapshotter.py` (covered in T043-3)
  - any other new occurrences (guard with `rg`)
- **description**: Ensure no CLI/runtime path bypasses the factory in duckdb mode.
- **validation**: `rg -n "\\bOptionsSnapshotStore\\(" options_helper`
- **status**: completed
- **log**: Verified no remaining `OptionsSnapshotStore(...)` instantiations in runtime code beyond factory.
- **validation_log**: `rg -n "\\bOptionsSnapshotStore\\(" options_helper`

### T043-6: Tests for filesystem `save_day_snapshot` + optional duckdb snapshotter integration
- **depends_on**: [T043-1, T043-2, T043-3, T043-4]
- **location**:
  - new: `tests/test_options_snapshots_save_day.py` (recommended)
  - optional: `tests/test_duckdb_snapshotter_integration.py`
- **description**:
  - Filesystem tests:
    - `save_day_snapshot` writes expected files for multiple expiries + raw payloads + meta.json.
    - pruning removes expiry files not in provided expiries list.
    - `load_day` still dedupes as expected.
  - Optional integration test (no CLI) in duckdb mode:
    - set storage backend contextvar to `duckdb` and provider name contextvar
    - run `snapshot_full_chain_for_symbols(...)` with a stub provider
    - assert `chain.parquet` exists and `store.list_dates(symbol)` returns the snapshot date (verifies header insert).
- **validation**: `./.venv/bin/python -m pytest`
- **status**: completed
- **log**: Added filesystem `save_day_snapshot` tests covering file layout and pruning.
- **files**: `tests/test_options_snapshots_save_day.py`
- **notes**: Optional DuckDB snapshotter integration test not added.
- **validation_log**: `./.venv/bin/python -m pytest -q tests/test_options_snapshots_save_day.py`

### T043-7: PR/commit assembly + full test run
- **depends_on**: [T043-5, T043-6]
- **location**: repo-wide
- **description**:
  - Open PR titled **exactly** `IMP-043: Options snapshot lake (Parquet) + DuckDB index`.
  - Ensure `pytest` passes.
- **validation**: `./.venv/bin/python -m pytest`
- **status**: completed
- **log**: Committed IMP-043 and validated with full pytest run.

## Parallel execution groups
| Wave | Tasks | Can start when |
|------|-------|----------------|
| 1 (IMP-040) | T040-1, T040-2, T040-5 | Immediately |
| 1b (IMP-040) | T040-3 | After T040-2 |
| 1c (IMP-040) | T040-4 | After T040-3 |
| 2 (IMP-041) | T041-1 | After T040-6 |
| 3 (IMP-042) | T042-1 | After T041-2 |
| 3b (IMP-042) | T042-2, T042-3 | After T042-1 |
| 4 (IMP-043) | T043-1 | After T042-4 |
| 4b (IMP-043) | T043-2, T043-3 | After T043-1 |
| 4c (IMP-043) | T043-4 | After T043-2 + T043-3 |
| 4d (IMP-043) | T043-6 | After T043-4 |
| 4e (IMP-043) | T043-7 | After T043-6 |

## Testing strategy
- Run `pytest` after each IMP PR (gate merges on passing tests).
- Keep all snapshotting/candle tests offline:
  - stub provider via `options_helper.cli_deps.build_provider`
  - stub candle history via `options_helper.data.candles.CandleStore.get_daily_history`
- Add targeted regression tests for contextvar cleanup (prevents flaky cross-test contamination).

## Assumptions / defaults
- DuckDB remains a **required** dependency (added to base deps), but `duckdb` imports are kept **lazy** so `--help` stays fast.
- We route storage backend selection primarily through `cli_deps` (stable test seam) rather than refactoring every command to import store_factory directly.
- DuckDB snapshot store does **not** scan filesystem as a fallback (per design: list/resolve come from DuckDB headers); importing existing filesystem snapshots into the DuckDB index is out of scope.
- Filesystem `save_day_snapshot` prunes expiry artifacts not in the passed `expiries` list to improve idempotence when run settings change.
