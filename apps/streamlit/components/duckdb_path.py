from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DUCKDB_PATH = Path("data/warehouse/options.duckdb")
DUCKDB_PATH_ENV_VAR = "OPTIONS_HELPER_DUCKDB_PATH"


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    raw_override = "" if database_path is None else str(database_path).strip()
    if raw_override:
        return Path(raw_override).expanduser().resolve()

    raw_env = str(os.getenv(DUCKDB_PATH_ENV_VAR) or "").strip()
    if raw_env:
        return Path(raw_env).expanduser().resolve()

    return DEFAULT_DUCKDB_PATH.expanduser().resolve()
