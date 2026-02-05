# apps/ — Optional Runtime Surfaces

This directory holds optional runtime surfaces that sit on top of the core CLI:
- `apps/streamlit/` (read-only portal)
- `apps/dagster/` (optional orchestration)

## Rules
- Keep these surfaces optional; core CLI startup must not depend on them.
- Preserve lazy imports for optional dependencies (`streamlit`, `dagster`).
- Reuse `options_helper` service/runtime seams instead of duplicating business logic.
- Treat this project as informational/educational tooling only (not financial advice).

## Testing expectations
- Add deterministic tests for helpers and module import smoke.
- Use `pytest.importorskip(...)` for optional dependency coverage.
