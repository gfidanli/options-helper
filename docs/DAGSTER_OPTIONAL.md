# Dagster orchestration (Optional)

Status: Draft  
Goal: add scheduling/backfills/lineage **without** rewriting the CLI or making Dagster mandatory.

---

## Why Dagster is a good fit (but optional)

Your workflows already form a pipeline graph:

- ingest candles / options bars (raw)
- snapshot options (offline chain state)
- flow (ΔOI / volume deltas)
- derived (history + compact signals)
- briefing/dashboard artifacts

Dagster adds:
- an explicit asset graph (lineage)
- daily partitions and backfills
- a UI for runs, logs, and asset checks

But we keep it optional because:
- the project is CLI-first
- many users will run cron + local DuckDB only
- the portal should not depend on Dagster being present

---

## Non-negotiable integration rule

Even when Dagster is enabled, the run ledger **must** still be written into DuckDB:

- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

Reason:
- Streamlit reads DuckDB
- cron/CLI runs need to show up too
- a single “health source of truth” is simpler

---

## Dagster code layout (suggested)

Keep Dagster isolated from core library dependencies:

```
apps/dagster/
  defs/
    __init__.py
    assets.py
    checks.py
    jobs.py
    schedules.py
    resources.py
```

---

## Asset mapping (suggested)

### Raw market data
- `candles_daily` — from `ingest candles`
- `options_bars` — from `ingest options-bars`

### Snapshot artifacts
- `options_snapshot_file` — from `snapshot-options`
  - treat as file asset (observation/materialization metadata contains path + size)

### Derived
- `options_flow` — from `flow` (depends on snapshots)
- `derived_metrics` — from `derived` (depends on snapshots/flow)
- `briefing_markdown` — from `briefing` (depends on derived)
- (optional) `dashboard_views` — from `dashboard`

---

## Partitions (start simple)

Use **daily partitions** for everything that is “as-of date” based.

- partition key: `YYYY-MM-DD`

Avoid `symbol × date` partitions at first (too many partitions).
Instead:
- store symbol-level watermarks in DuckDB (`meta.asset_watermarks`)
- compute symbol-level coverage in Streamlit

---

## Dagster resources

### DuckDB resource
Dagster resources should reuse the same DuckDB path/config as the CLI.

Prefer:
- a small `get_duckdb_path()` function in `options_helper` core config module
- Dagster calls it rather than re-inventing config

### Provider resource
Dagster should reuse provider setup already present:
- `--provider alpaca` semantics (keys from env/config files)

---

## Asset checks (Dagster + persisted checks)

Dagster “asset checks” are a natural home for:
- duplicates
- nulls
- monotonic timestamps
- freshness bounds

But you still want the check result persisted for Streamlit.

Pattern:
- Dagster runs the check
- check function writes to `meta.asset_checks`
- check returns pass/fail to Dagster

This yields:
- Dagster UI visibility
- Streamlit portal visibility

---

## Scheduling (local)

A daily schedule order that matches your existing automation docs:

1. `ingest candles` (for watchlists + portfolio underlyings)
2. `ingest options-bars` (if used)
3. `snapshot-options`
4. `flow`
5. `derived`
6. `briefing` (and optionally `dashboard`)

---

## Backfills

Backfills should be date-driven:

- fill missing days for candles / bars
- re-run derived assets for a past date range after logic changes

Dagster’s partition backfills are a strong fit here, but your “gap planner”
should still live in DuckDB/Streamlit so it works without Dagster too.

---

## Dependency management

Keep Dagster out of the default install. Recommended optional extras:

- `orchestrator`:
  - `dagster`
  - `dagster-webserver`

Install:
```bash
pip install -e ".[orchestrator]"
```

---

## When to NOT use Dagster

If your workflow is purely:
- manual CLI runs
- occasional cron

…and you don’t need partition backfills or lineage, you can skip Dagster and
still have a professional visibility tool via DuckDB `meta.*` + Streamlit Health.
