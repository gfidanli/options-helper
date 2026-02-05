# apps/streamlit/ — Portal Conventions

## Product constraints
- Portal is read-only by default.
- Show explicit disclaimer text that this is not financial advice.
- Do not trigger ingestion writes from page load/render paths.

## Architecture
- Keep pages thin and move query/transform logic into `components/*`.
- Use cached helpers (`st.cache_resource`, `st.cache_data`) through shared modules.
- Use query-parameter sync for shareable state where relevant.
- For observability views, consume centralized query helpers from `options_helper.data.observability_meta` through a component adapter module.
- Keep gap/backfill planners display-only (show copy/paste commands, do not execute writes from the page).

## Resilience
- Handle missing DuckDB/database/table states gracefully (warnings/info, not crashes).
- Prefer best-effort rendering when a single section fails.

## Tests
- Add deterministic helper/query tests under `tests/portal/`.
- Keep tests offline with temporary DuckDB files and fixtures.
