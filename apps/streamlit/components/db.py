from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


@st.cache_resource(show_spinner=False)
def get_read_only_connection(database_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = resolve_duckdb_path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {db_path}. Run `options-helper db init` first."
        )
    return duckdb.connect(str(db_path), read_only=True)
