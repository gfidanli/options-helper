from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st

from apps.streamlit.components.db import get_read_only_connection

DEFAULT_QUERY_TTL_SECONDS = 60


def _normalize_params(params: Sequence[Any] | None) -> tuple[Any, ...]:
    if params is None:
        return ()
    return tuple(params)


@st.cache_data(ttl=DEFAULT_QUERY_TTL_SECONDS, show_spinner=False)
def run_cached_query(
    sql: str,
    params: tuple[Any, ...] = (),
    database_path: str | None = None,
) -> pd.DataFrame:
    conn = get_read_only_connection(database_path=database_path)
    return conn.execute(sql, params).df()


def run_query(
    sql: str,
    params: Sequence[Any] | None = None,
    database_path: str | None = None,
) -> pd.DataFrame:
    return run_cached_query(
        sql=sql,
        params=_normalize_params(params),
        database_path=database_path,
    )
