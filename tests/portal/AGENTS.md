# tests/portal/ — Streamlit Portal Test Conventions

- Keep tests offline and deterministic.
- Prefer testing query/helper modules over page rendering internals.
- Use temporary DuckDB files (`tmp_path`) for portal data setup.
- Use `pytest.importorskip("streamlit")` for import/runtime smoke checks.
- Validate graceful behavior when DB/table inputs are missing.
