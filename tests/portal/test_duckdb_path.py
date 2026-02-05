from __future__ import annotations

from pathlib import Path

from apps.streamlit.components import (
    data_explorer_page,
    derived_history_page,
    flow_page,
    health_page,
    symbol_explorer_page,
)
from apps.streamlit.components.duckdb_path import (
    DEFAULT_DUCKDB_PATH,
    DUCKDB_PATH_ENV_VAR,
    resolve_duckdb_path,
)


def test_resolve_duckdb_path_defaults_when_env_not_set(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(DUCKDB_PATH_ENV_VAR, raising=False)
    assert resolve_duckdb_path(None) == DEFAULT_DUCKDB_PATH.expanduser().resolve()


def test_resolve_duckdb_path_uses_env_var_when_input_blank(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path: Path
) -> None:
    configured = tmp_path / "custom.duckdb"
    monkeypatch.setenv(DUCKDB_PATH_ENV_VAR, str(configured))
    expected = configured.resolve()

    assert resolve_duckdb_path(None) == expected
    assert health_page.resolve_duckdb_path(None) == expected
    assert flow_page.resolve_duckdb_path("   ") == expected
    assert symbol_explorer_page.resolve_duckdb_path(None) == expected
    assert derived_history_page.resolve_duckdb_path(None) == expected
    assert data_explorer_page.resolve_duckdb_path(None) == expected


def test_resolve_duckdb_path_explicit_arg_overrides_env(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DUCKDB_PATH_ENV_VAR, str(tmp_path / "env.duckdb"))
    explicit = tmp_path / "explicit.duckdb"
    assert resolve_duckdb_path(str(explicit)) == explicit.resolve()
