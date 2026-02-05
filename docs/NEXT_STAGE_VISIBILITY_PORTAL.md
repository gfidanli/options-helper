# Next Stage: Visibility + Portal + Orchestration (Design Context)

Status: Draft (working doc)  
Owner: `options-helper` maintainers  
Audience: contributors implementing ingestion visibility, Streamlit portal dashboards, and optional Dagster orchestration.

---

## Current state (baseline)

`options-helper` is already shaped like a small “local data platform”:

- **CLI-first workflows** for portfolio analysis and research.
- **DuckDB is the default storage backend** (filesystem remains supported).
- **Ingestion commands**:
  - `ingest candles`
  - `ingest options-bars`
- **Snapshot-driven analysis**:
  - `snapshot-options` → saves option-chain snapshots (offline / repeatable)
  - `flow` → computes ΔOI/volume-based positioning proxies
  - `derived` → maintains compact per-symbol derived-metrics history
  - `briefing` → daily Markdown artifact
  - `dashboard` → current read-only dashboard output
- **Provider abstraction**:
  - Global `--provider` (default: `alpaca`, `yahoo` fallback)

This next stage does **not** replace these; it makes them observable and easier to use.

---

## What “professionalize” means here

### 1) Visibility / Observability
You should be able to answer, instantly:

- What ran? When? With what parameters?
- What failed, and what’s the recurring failure mode?
- Which datasets are **stale** relative to expected cadence?
- Where are the **gaps** (missing days, missing symbols)?
- Are there **quality issues** (duplicates, invalid timestamps, negative prices, suspicious null rates)?

### 2) Portal UI (Streamlit)
A product-like UI that:
- keeps the project **one-laptop friendly**
- provides the best parts of the existing CLI outputs as interactive dashboards
- includes an “Ops / Health” page (ingestion visibility tool)

### 3) Optional orchestration (Dagster)
Dagster is a good fit because the CLI steps already form an asset graph:
- ingest → snapshot → flow → derived → briefing

But Dagster should remain **optional**:
- the CLI remains the canonical interface
- metadata must still be written to DuckDB so that:
  - Streamlit can read it
  - CLI-only runs (cron/manual) show up in the same health UI

---

## Key decisions

### D1 — “Hybrid visibility”
- Dagster UI (if enabled) is helpful for deep debugging and backfills.
- But the **source of truth** for run history and health is persisted in DuckDB (`meta.*`).
- Streamlit reads from DuckDB; it should not scrape Dagster.

### D2 — Store operational metadata next to analytical data
- Add a `meta` schema inside the same DuckDB file.
- Benefits:
  - portable (copy DB = copy state)
  - no extra infrastructure
  - easy SQL queries for health dashboards

### D3 — Partitioning: date-first
Most assets have a natural daily cadence.
- Start with daily partitions: `YYYY-MM-DD`
- Add symbol scoping only where you need it (watermarks, coverage heatmaps)

---

## Asset model (conceptual)

Think in “assets” (tables/files) even if you don’t use Dagster yet.

### Raw market data assets
- `candles_daily` (DuckDB table)
- `options_bars` (DuckDB table)

### Snapshot artifacts (offline-first)
- `options_snapshot_file` (files under `data/options_snapshots/`)
- optionally also: `options_snapshot_table` (if/when mirrored into DuckDB)

### Derived assets (from snapshots)
- `options_flow` (DuckDB table)
- `derived_metrics` (DuckDB table)
- `briefing_markdown` (file artifact)
- `dashboard_views` (views or precomputed tables for UI)

### Operational assets (new)
- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

See: `docs/OBSERVABILITY.md`

---

## New user-facing surfaces

### A) Streamlit “Portal”
Multipage app with:
- **Health**: runs, freshness, gaps, quality checks
- Portfolio: positions + Greeks/stress + daily performance
- Symbol explorer: candles + snapshot + derived signals
- Flow: ΔOI/volume positioning proxy views
- Data explorer: DuckDB schema/table browser

See: `docs/PORTAL_STREAMLIT.md` and `docs/PORTAL_STOCKPEERS_STYLE.md`

### B) Dagster UI (optional)
Used for:
- scheduled runs
- partition backfills
- asset graph visualization
- asset checks as first-class citizens

See: `docs/DAGSTER_OPTIONAL.md`

---

## Rollout plan (high-level)

1) Add `meta.*` schema + write run ledger from existing CLI commands  
2) Build Streamlit MVP with Health + Portfolio + Symbol Explorer  
3) Add persisted data-quality checks and display in UI  
4) Add Dagster assets/schedules/backfills (optional)  
5) Expand dashboards and “gap backfill planner” UX  

Detailed tasks: `docs/plans/VISIBILITY_PORTAL_IMPLEMENTATION_PLAN.md`
