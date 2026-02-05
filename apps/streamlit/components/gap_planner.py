from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import streamlit as st


def _query_params_store(
    query_params: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    return st.query_params if query_params is None else query_params


def read_query_param(
    name: str,
    default: str | None = None,
    *,
    query_params: MutableMapping[str, Any] | None = None,
) -> str | None:
    params = _query_params_store(query_params)
    raw_value = params.get(name, default)
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else default
    if raw_value in (None, ""):
        return default
    return str(raw_value)


def sync_query_param(
    name: str,
    value: str | None,
    *,
    default: str | None = None,
    query_params: MutableMapping[str, Any] | None = None,
) -> str | None:
    params = _query_params_store(query_params)
    normalized = value if value not in (None, "") else default
    if normalized in (None, ""):
        params.pop(name, None)
        return None
    current = read_query_param(name=name, default=default, query_params=params)
    if current != normalized:
        params[name] = normalized
    return normalized


def sync_csv_query_param(
    name: str,
    values: Sequence[str],
    *,
    query_params: MutableMapping[str, Any] | None = None,
) -> list[str]:
    cleaned_values = [value.strip() for value in values if value and value.strip()]
    serialized = ",".join(cleaned_values) if cleaned_values else None
    sync_query_param(name=name, value=serialized, query_params=query_params)
    return cleaned_values
