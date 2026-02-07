from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path
from options_helper.data.coverage_duckdb import list_candidate_symbols
from options_helper.data.coverage_service import build_symbol_coverage


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})
    return cleaned or default.strip().upper()


def list_coverage_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    path = resolve_duckdb_path(database_path)
    return list_candidate_symbols(duckdb_path=path)


@st.cache_data(ttl=60, show_spinner=False)
def _load_coverage_payload_cached(
    *,
    symbol: str,
    lookback_days: int,
    database_path: str,
) -> dict[str, Any]:
    return build_symbol_coverage(
        symbol,
        days=max(1, int(lookback_days)),
        duckdb_path=Path(database_path),
    )


def load_coverage_payload(
    *,
    symbol: str,
    lookback_days: int,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    resolved = resolve_duckdb_path(database_path)
    try:
        payload = _load_coverage_payload_cached(
            symbol=normalize_symbol(symbol),
            lookback_days=max(1, int(lookback_days)),
            database_path=str(resolved),
        )
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
