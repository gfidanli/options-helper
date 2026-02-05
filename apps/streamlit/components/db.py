from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st

DEFAULT_DUCKDB_PATH = Path("data/options_helper.duckdb")


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    candidate = DEFAULT_DUCKDB_PATH if database_path is None else Path(database_path)
    return candidate.expanduser().resolve()


@st.cache_resource(show_spinner=False)
def get_read_only_connection(database_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = resolve_duckdb_path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {db_path}. Run `options-helper db init` first."
        )
    return duckdb.connect(str(db_path), read_only=True)
