# Plan: Visibility + Portal + (Optional) Dagster

Status: Draft  
Target: next stage milestone for `options-helper`  
Scope: add ingestion visibility tooling + Streamlit portal + optional orchestration.

---

## Guiding principles

1) **Do not break the CLI**  
   The CLI is the primary interface.

2) **DuckDB is the “control plane”**  
   Persist run ledger + checks in DuckDB (`meta.*`).

3) **Read-only portal first**  
   Ship dashboards before adding “actions” in the UI.

4) **Orchestration is optional**  
   Dagster should be additive, not required.

---

## Milestone 1 — Add DuckDB operational metadata schema

### Tasks
- [ ] Add a `meta` schema initializer (`ensure_meta_schema(conn)`)
- [ ] Implement `RunLogger` context manager:
  - run_id generation
  - insert started row
  - log asset rows (success/fail)
  - finalize run (success/fail)
- [ ] Add watermark upsert helper
- [ ] Add a minimal JSON helper to store args/extra payloads safely

### Acceptance criteria
- Running `options-helper ingest candles ...` inserts:
  - one `meta.ingestion_runs` row
  - one+ `meta.ingestion_run_assets` row(s)
- Failures populate `error_type` + `error_message`
- Watermarks update for `candles_daily`

---

## Milestone 2 — Instrument existing CLI commands

### Target commands
- [ ] `ingest candles`
- [ ] `ingest options-bars`
- [ ] `snapshot-options`
- [ ] `flow`
- [ ] `derived`
- [ ] `briefing`
- [ ] `dashboard`

### Tasks
- [ ] Wrap top-level handler with `RunLogger`
- [ ] Record:
  - provider, storage backend
  - args_json (symbols, dates, flags)
  - per-asset min/max timestamps written (best effort)
- [ ] Record file assets:
  - snapshot file path(s), bytes written
  - briefing Markdown path, bytes written

### Acceptance criteria
- Each command appears in `meta.ingestion_runs` with stable `job_name`
- One day of “automation order” produces a clear run sequence

---

## Milestone 3 — Implement data-quality checks (persisted)

### Tasks
- [ ] Implement a small check library (callable from CLI and/or Dagster)
- [ ] Write results to `meta.asset_checks` (pass/fail + metrics)
- [ ] Add checks for:
  - candles duplicates/gaps
  - options bars duplicates
  - snapshot parseability / null-IV ratio

### Acceptance criteria
- At least 6 checks exist, and failures show up in DuckDB.

---

## Milestone 4 — Streamlit portal scaffolding + theme

### Tasks
- [ ] Create `apps/streamlit` multipage scaffolding
- [ ] Add Stockpeers theme file:
  - `.streamlit/config.toml` (already provided in this pack)
- [ ] Implement shared DuckDB connection helper with `st.cache_resource`
- [ ] Implement shared query helper with `st.cache_data`

### Acceptance criteria
- `streamlit run apps/streamlit/streamlit_app.py` boots locally
- Theme matches Stockpeers vibe (dark + Space Grotesk + bordered widgets)

---

## Milestone 5 — Streamlit MVP pages

### Page: Health
- [ ] Latest run per job
- [ ] Failures last N days
- [ ] Watermarks/freshness table
- [ ] Check failures table

### Page: Portfolio
- [ ] Load portfolio JSON + render positions
- [ ] Show portfolio-level Greeks/stress summary (even if computed on the fly initially)

### Page: Symbol Explorer
- [ ] Symbol selector (query params)
- [ ] Candles chart + basic indicators
- [ ] Latest snapshot summary (simple tables first)

### Acceptance criteria
- Health page is useful as an ingestion visibility tool.
- Symbol explorer supports “bookmarkable research” via URL query params.

---

## Milestone 6 — Dagster optional integration

### Tasks
- [ ] Add `apps/dagster/defs` with assets + daily partitions
- [ ] Reuse the same underlying functions used by CLI
- [ ] Add asset checks (reuse same check functions)
- [ ] Persist checks into DuckDB still

### Acceptance criteria
- Dagster can materialize at least `candles_daily` and `options_snapshot_file`
- Dagster runs show up in DuckDB run ledger as `triggered_by='dagster'`

---

## Milestone 7 — Expand analytics dashboards

### Tasks
- [ ] Flow dashboard (top flows, expiry buckets, drill-down)
- [ ] Derived history explorer (regime/trend over time)
- [ ] Briefing viewer (render Markdown artifacts)
- [ ] Gap backfill planner UX (detect missing partitions, propose backfill commands)

### Acceptance criteria
- Portal becomes the primary “browse + explore” interface.
- CLI remains the primary “ingest + compute” interface.

