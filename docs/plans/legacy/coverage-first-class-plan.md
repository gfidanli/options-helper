# Plan: First-Class Ticker Coverage (CLI + Streamlit)

**Generated**: 2026-02-07

## Overview
Add a first-class “coverage” feature that answers, for a given underlying symbol:

- Candle coverage (rows, start/end, missing business days, missing values)
- Options snapshot coverage (days captured, contracts per day, basic OI/volume presence)
- Alpaca `/v2/options/contracts` coverage (days captured, contracts listed, OI presence per contract/day)
- DuckDB option-bars coverage (when available) using `option_bars_meta` and related tables
- Actionable “repair” suggestions (copy/paste commands) to improve coverage incrementally

Deliverables:
- CLI: `options-helper coverage SYMBOL ...`
- Portal: Streamlit “Coverage” page (read-only, no writes on load)
- Docs: `docs/COVERAGE.md`
- Tests: deterministic, offline

This tool is informational/educational only and **not financial advice**.

## Prerequisites
- Working venv: `python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
- DuckDB default path: `data/warehouse/options.duckdb` (or set `--duckdb-path` / `OPTIONS_HELPER_DUCKDB_PATH`)
- Storage backend: `--storage duckdb` (DuckDB is the source of truth; legacy CSV/lake files may exist but are not authoritative).
- Optional UI deps for portal work: `pip install -e ".[dev,ui]"`

## Decisions
1. **Source of truth**: DuckDB is the canonical store (filesystem `.csv` snapshots are legacy and may be stale/out of date).
   - Contract/OI coverage reads DuckDB `option_contracts` + `option_contract_snapshots`.
2. **Default lookback**: 60d for “deep” per-contract OI stats.
   - Emphasize recent OI deltas (1d/3d/5d) when computing/printing.
3. **Portal UX**: create a new Streamlit page (`apps/streamlit/pages/08_Coverage.py`).

The plan below treats DuckDB-first as MVP. An optional one-time import of legacy filesystem contract snapshots can seed historical OI if desired.

## Dependency Graph

```
T1 ─┬─ T2 ─┬─ T5 ── T6 ─┬─ T9
    │      │            └─ T10
    │      ├──────── T7 ─┘
    │      └─────┬─ T8 ──┘
    ├─ T4 ───────┘
    └─ T3 ───────┘

T11 (optional) can run anytime after T1.
```

## Tasks

### T1: Coverage Spec (Decisions Locked)
- **depends_on**: []
- **location**: `docs/COVERAGE.md` (draft outline), `coverage-first-class-plan.md`
- **description**:
  - Record the decisions above (DuckDB source of truth, 60d deep lookback, new portal page).
  - Define MVP metrics and “deep” metrics, and set performance guardrails (default lookback=60d, caching).
- **validation**:
  - Decisions recorded in `docs/COVERAGE.md` “Design” section.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T2: Pure Coverage Model + Metrics (No I/O)
- **depends_on**: [T1]
- **location**: `options_helper/analysis/coverage.py` (new)
- **description**:
  - Define dataclasses/TypedDicts for coverage summaries:
    - candle coverage (row counts, start/end, missing business days last N, missing cells)
    - snapshot coverage (days present, per-day contract counts, OI presence ratios)
    - contract/OI coverage (per-day counts, per-contract OI-days/missing-days within lookback, 1d/3d/5d OI-delta coverage)
  - Implement pure functions that accept DataFrames / lists of rows and return metrics.
- **validation**:
  - Unit tests pass for metric functions using small fixtures (`pytest -q` specific tests added in T9).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3: DuckDB Option-Contracts Snapshot Ingest (Daily OI Capture)
- **depends_on**: [T1]
- **location**: `options_helper/commands/ingest.py` (extend) or `options_helper/commands/ingest_contracts.py` (new), `options_helper/pipelines/visibility_jobs.py` (extend)
- **description**:
  - Ensure we persist a *daily snapshot* of Alpaca `/v2/options/contracts` results into DuckDB:
    - dimension: `option_contracts`
    - daily snapshots: `option_contract_snapshots` (with `as_of_date` = run day)
  - Implement a lightweight ingest mode that discovers contracts and writes to DuckDB but does **not** fetch option bars.
    - Option A (preferred): add `--contracts-only` to `options-helper ingest options-bars`.
    - Option B: add a dedicated command (e.g. `options-helper ingest option-contracts`).
  - This is the mechanism that lets Coverage compute OI history from DuckDB (and is what daily cron should run).
- **validation**:
  - Deterministic tests by injecting a fake Alpaca client factory (no network) and asserting rows land in `option_contract_snapshots`.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: DuckDB Coverage Query Helpers (Read-Only)
- **depends_on**: [T1]
- **location**: `options_helper/data/coverage_duckdb.py` (new)
- **description**:
  - Add small query helpers (read-only) for:
    - `candles_daily` / `candles_meta` per symbol (counts, start/end, last N rows)
    - `options_snapshot_headers` per symbol when present
    - `option_contracts`, `option_contract_snapshots` per underlying (when present)
    - `option_bars_meta` coverage summary (when present)
  - Keep functions resilient to missing tables (return empty frames + notes).
- **validation**:
  - Tests with a temporary DuckDB file seeded with minimal schema + rows.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Coverage Service Orchestrator
- **depends_on**: [T2, T4]
- **location**: `options_helper/data/coverage_service.py` (new)
- **description**:
  - Build a single entrypoint `build_symbol_coverage(symbol, *, days, duckdb_path, …)` that:
    - loads required data from DuckDB (candles, bars meta, snapshot headers, contracts + OI snapshots when present)
    - calls pure metric functions from `options_helper/analysis/coverage.py`
    - returns a structured payload for both CLI + portal rendering
  - Ensure no network calls; this is coverage of local data only.
- **validation**:
  - Unit tests: orchestrator returns expected payload from seeded temp DuckDB.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6: CLI Command `options-helper coverage`
- **depends_on**: [T5]
- **location**: `options_helper/commands/coverage.py` (new), `options_helper/cli.py`
- **description**:
  - Add `coverage` command with flags:
    - `--days` (lookback), `--json`, `--out` (optional file), `--duckdb-path` passthrough
  - Render concise Rich tables (candles, contract snapshots, option bars meta, snapshot coverage).
  - Include a small “OI delta coverage” section (1d/3d/5d) when `option_contract_snapshots` data exists.
  - Print a “repair suggestions” section (copy/paste commands only; no execution).
- **validation**:
  - `./.venv/bin/options-helper coverage SPY --days 30 --json` runs in a seeded test environment.
  - CLI tests via Typer `CliRunner`.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: Streamlit Coverage Page (Read-Only)
- **depends_on**: [T5]
- **location**: `apps/streamlit/pages/08_Coverage.py` (new), `apps/streamlit/components/coverage_page.py` (new)
- **description**:
  - Add a dedicated Coverage page with:
    - symbol selection + lookback slider
    - coverage summary metrics + tables
    - “OI delta coverage” callouts (1d/3d/5d) when `option_contract_snapshots` exists
    - “repair suggestions” commands (copy/paste)
  - Keep page thin; move query/transform logic into `components/*`.
  - Use Streamlit caching (`st.cache_data`) and handle missing DB/table states gracefully.
  - Do not run ingestion writes from page load/render.
- **validation**:
  - Import smoke tests for page/modules.
  - Manual: `./.venv/bin/options-helper ui` shows Coverage page without triggering writes.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Repair Suggestions / Incremental Load Planner
- **depends_on**: [T3, T5]
- **location**: `options_helper/analysis/coverage_repair.py` (new) or within `options_helper/analysis/coverage.py`
- **description**:
  - Convert coverage deficits into prioritized commands:
    - candles gaps → `ingest candles --symbol SYMBOL`
    - missing contract/OI snapshots days → run the DuckDB contracts-snapshot ingest (T3); note historical OI backfill is generally not possible
    - missing option bars coverage (DuckDB) → `ingest options-bars --symbol SYMBOL --lookback-years N --resume`
    - missing snapshot coverage → `snapshot-options ... --symbol SYMBOL --all-expiries --full-chain`
  - Keep this deterministic; do not inspect network.
- **validation**:
  - Unit tests for rule outputs given synthetic coverage inputs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Tests (Offline, Deterministic)
- **depends_on**: [T2, T3, T4, T6, T7, T8]
- **location**: `tests/test_coverage_analysis.py` (new), `tests/test_coverage_cli.py` (new), `tests/portal/test_coverage_page.py` (new)
- **description**:
  - Add focused tests for:
    - pure metric computations
    - orchestrator wiring with a seeded temp DuckDB
    - CLI output contract
    - Streamlit helper module import + core helper functions (use `pytest.importorskip("streamlit")`)
- **validation**:
  - `./.venv/bin/python -m pytest -q`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T10: Documentation
- **depends_on**: [T6, T7]
- **location**: `docs/COVERAGE.md` (new), `mkdocs.yml` (nav update if needed)
- **description**:
  - Document:
    - what “coverage” means per data source
    - where data lives (DuckDB tables as source of truth; legacy filesystem caches may exist)
    - how to run CLI + portal
    - interpretation caveats (weekends/holidays, symbol skips, OI semantics)
  - Include “not financial advice” disclaimer.
- **validation**:
  - `mkdocs build` (optional) or basic link sanity checks.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11 (Optional): Import Legacy Filesystem Contract Snapshots into DuckDB
- **depends_on**: [T1]
- **location**: `options_helper/commands/ingest_contracts.py` (new) or `options_helper/commands/ingest.py` (extend), `options_helper/data/stores_duckdb.py` (existing `DuckDBOptionContractsStore`)
- **description**:
  - Optional one-time migration utility: read legacy `data/option_contracts/**/contracts.csv` for a symbol (and lookback window) and upsert:
    - `option_contracts` dimension
    - `option_contract_snapshots` daily OI rows
  - Enables fast DuckDB-only coverage queries and portal views without filesystem scans.
- **validation**:
  - Tests with temp dirs + temp DuckDB; verify rows land in `option_contract_snapshots`.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1 | Immediately |
| 2 | T2, T3, T4, T11 | T1 complete |
| 3 | T5 | T2 + T4 complete |
| 4 | T6, T7 | T5 complete |
| 5 | T8 | T3 + T5 complete |
| 6 | T9, T10 | T6 + T7 + T8 complete |

## Testing Strategy
- Unit tests for `options_helper/analysis/coverage.py` (pure functions).
- Integration-ish tests using a temp DuckDB file seeded with minimal rows.
- CLI smoke via Typer `CliRunner` and import smoke for portal modules.

## Risks & Mitigations
- **Performance scanning many days/contracts**: default 60d cap + Streamlit caching + “deep” mode flag.
- **Missing/partial DuckDB tables**: query helpers return empty + notes; UI renders warnings, not crashes.
- **Date semantics**: treat “day” as `as_of_date` in DuckDB for contract/OI snapshots; surface `open_interest_date` lag/staleness when present.
- **OI semantics**: OI is point-in-time; historical backfill is generally not possible. Planner should suggest forward daily capture (T3), not impossible backfills.
