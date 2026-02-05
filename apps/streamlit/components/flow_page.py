from __future__ import annotations

from collections.abc import MutableMapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path

DEFAULT_FLOW_GROUP = "expiry-strike"
FLOW_GROUP_VALUES = ("contract", "strike", "expiry", "expiry-strike")

_PARTITION_COLUMNS = [
    "as_of",
    "from_date",
    "to_date",
    "window",
    "group_by",
    "rows",
    "net_delta_oi_notional",
    "total_volume_notional",
    "net_delta_notional",
    "updated_at",
]
_ROW_COLUMNS = [
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "delta_oi",
    "delta_oi_notional",
    "volume_notional",
    "delta_notional",
    "n_pairs",
]


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def normalize_flow_group(value: Any, *, default: str = DEFAULT_FLOW_GROUP) -> str:
    raw = str(value or "").strip().lower()
    if raw in FLOW_GROUP_VALUES:
        return raw
    return default


def resolve_flow_group_selection(
    available_groups: list[str],
    *,
    query_group: str | None = None,
    user_group: str | None = None,
    default_group: str = DEFAULT_FLOW_GROUP,
) -> str:
    normalized_available = [normalize_flow_group(group, default=default_group) for group in available_groups]
    normalized_available = [group for group in normalized_available if group]
    desired = normalize_flow_group(user_group or query_group or default_group, default=default_group)
    if desired in normalized_available:
        return desired
    if normalized_available:
        return normalized_available[0]
    return desired


def sync_flow_group_query_param(
    *,
    group_by: str | None,
    query_params: MutableMapping[str, Any],
    name: str = "group_by",
    default_group: str = DEFAULT_FLOW_GROUP,
) -> str:
    normalized_default = normalize_flow_group(default_group, default=default_group)
    normalized = normalize_flow_group(group_by, default=normalized_default)
    current = _read_query_param(query_params, name=name)
    if normalized == normalized_default:
        query_params.pop(name, None)
    elif current != normalized:
        query_params[name] = normalized
    return normalized


def list_flow_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    df, note = _run_query_safe(
        """
        SELECT DISTINCT UPPER(symbol) AS symbol
        FROM options_flow
        WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
        ORDER BY symbol ASC
        """,
        database_path=database_path,
    )
    if note:
        return [], note
    if df.empty:
        return [], None
    symbols = [normalize_symbol(value) for value in df["symbol"].tolist()]
    return sorted(symbol for symbol in symbols if symbol), None


def list_flow_groups(
    symbol: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[list[str], str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        SELECT DISTINCT LOWER(group_by) AS group_by
        FROM options_flow
        WHERE UPPER(symbol) = ?
        ORDER BY group_by ASC
        """,
        params=[sym],
        database_path=database_path,
    )
    if note:
        return [], note
    if df.empty:
        return [], None

    allowed = [group for group in FLOW_GROUP_VALUES if group in set(df["group_by"].astype("string"))]
    if allowed:
        return allowed, None
    return sorted(
        {
            normalize_flow_group(value, default="")
            for value in df["group_by"].tolist()
            if normalize_flow_group(value, default="")
        }
    ), None


def load_flow_asof_bounds(
    symbol: str,
    *,
    group_by: str | None = None,
    database_path: str | Path | None = None,
) -> tuple[date | None, date | None, str | None]:
    where_clauses = ["UPPER(symbol) = ?"]
    params: list[Any] = [normalize_symbol(symbol)]

    if group_by:
        where_clauses.append("LOWER(group_by) = ?")
        params.append(normalize_flow_group(group_by))

    where_sql = " AND ".join(where_clauses)
    df, note = _run_query_safe(
        f"""
        SELECT
          MIN(as_of) AS min_as_of,
          MAX(as_of) AS max_as_of
        FROM options_flow
        WHERE {where_sql}
        """,
        params=params,
        database_path=database_path,
    )
    if note:
        return None, None, note
    if df.empty:
        return None, None, None
    row = df.iloc[0].to_dict()
    return _coerce_date(row.get("min_as_of")), _coerce_date(row.get("max_as_of")), None


def load_flow_partition_summaries(
    symbol: str,
    *,
    group_by: str | None = None,
    as_of_start: date | str | None = None,
    as_of_end: date | str | None = None,
    limit: int = 300,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    where_clauses = ["UPPER(symbol) = ?"]
    params: list[Any] = [normalize_symbol(symbol)]

    if group_by:
        where_clauses.append("LOWER(group_by) = ?")
        params.append(normalize_flow_group(group_by))

    start_date = _coerce_date(as_of_start)
    if start_date is not None:
        where_clauses.append("as_of >= ?")
        params.append(start_date)

    end_date = _coerce_date(as_of_end)
    if end_date is not None:
        where_clauses.append("as_of <= ?")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
    params.append(max(1, int(limit)))
    df, note = _run_query_safe(
        f"""
        SELECT
          as_of,
          from_date,
          to_date,
          window_size AS window,
          group_by,
          COUNT(*)::BIGINT AS rows,
          SUM(COALESCE(delta_oi_notional, 0.0)) AS net_delta_oi_notional,
          SUM(COALESCE(volume_notional, 0.0)) AS total_volume_notional,
          SUM(COALESCE(delta_notional, 0.0)) AS net_delta_notional,
          MAX(updated_at) AS updated_at
        FROM options_flow
        WHERE {where_sql}
        GROUP BY as_of, from_date, to_date, window_size, group_by
        ORDER BY as_of DESC, to_date DESC, from_date DESC, window_size ASC
        LIMIT ?
        """,
        params=params,
        database_path=database_path,
    )
    if note:
        return _empty_partitions(), note
    if df.empty:
        return _empty_partitions(), None

    out = df.copy()
    for column in ("as_of", "from_date", "to_date", "updated_at"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in ("rows",):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("net_delta_oi_notional", "total_volume_notional", "net_delta_notional"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(
        by=["as_of", "to_date", "from_date", "window"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return out, None


def load_flow_rows_for_partition(
    symbol: str,
    *,
    as_of: date | str,
    from_date: date | str,
    to_date: date | str,
    window: int,
    group_by: str,
    top_n: int = 200,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    as_of_date = _coerce_date(as_of)
    from_dt = _coerce_date(from_date)
    to_dt = _coerce_date(to_date)
    if as_of_date is None or from_dt is None or to_dt is None:
        return _empty_rows(), "Invalid flow partition date input."

    df, note = _run_query_safe(
        """
        SELECT
          contract_symbol,
          expiry,
          option_type,
          strike,
          delta_oi,
          delta_oi_notional,
          volume_notional,
          delta_notional,
          n_pairs
        FROM options_flow
        WHERE UPPER(symbol) = ?
          AND as_of = ?
          AND from_date = ?
          AND to_date = ?
          AND window_size = ?
          AND LOWER(group_by) = ?
        ORDER BY
          ABS(COALESCE(delta_oi_notional, 0.0)) DESC,
          ABS(COALESCE(volume_notional, 0.0)) DESC,
          row_key ASC
        LIMIT ?
        """,
        params=[
            normalize_symbol(symbol),
            as_of_date,
            from_dt,
            to_dt,
            int(window),
            normalize_flow_group(group_by),
            max(1, int(top_n)),
        ],
        database_path=database_path,
    )
    if note:
        return _empty_rows(), note
    if df.empty:
        return _empty_rows(), None

    out = df.copy()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    for column in ("strike", "delta_oi", "delta_oi_notional", "volume_notional", "delta_notional", "n_pairs"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.reset_index(drop=True), None


def build_flow_timeseries(partitions_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "as_of",
        "partition_count",
        "net_delta_oi_notional",
        "total_volume_notional",
        "net_delta_notional",
    ]
    if partitions_df.empty:
        return pd.DataFrame(columns=columns)

    temp = partitions_df.copy()
    temp["as_of"] = pd.to_datetime(temp["as_of"], errors="coerce")
    temp = temp.dropna(subset=["as_of"])
    if temp.empty:
        return pd.DataFrame(columns=columns)

    grouped = temp.groupby("as_of", as_index=False).agg(
        partition_count=("as_of", "size"),
        net_delta_oi_notional=("net_delta_oi_notional", "sum"),
        total_volume_notional=("total_volume_notional", "sum"),
        net_delta_notional=("net_delta_notional", "sum"),
    )
    for column in ("partition_count", "net_delta_oi_notional", "total_volume_notional", "net_delta_notional"):
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce")
    return grouped.sort_values(by="as_of", kind="stable").reset_index(drop=True)


def build_flow_option_type_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["option_type", "rows", "net_delta_oi_notional", "total_volume_notional", "net_delta_notional"]
    if rows_df.empty:
        return pd.DataFrame(columns=columns)

    temp = rows_df.copy()
    option_series = temp["option_type"].astype("string").str.lower().fillna("")
    temp["option_type"] = option_series.map(_normalize_option_type)
    for column in ("delta_oi_notional", "volume_notional", "delta_notional"):
        temp[column] = pd.to_numeric(temp[column], errors="coerce").fillna(0.0)

    grouped = temp.groupby("option_type", as_index=False).agg(
        rows=("option_type", "size"),
        net_delta_oi_notional=("delta_oi_notional", "sum"),
        total_volume_notional=("volume_notional", "sum"),
        net_delta_notional=("delta_notional", "sum"),
    )
    order = {"call": 0, "put": 1, "other": 2}
    grouped["sort_order"] = grouped["option_type"].map(lambda value: order.get(str(value), 99))
    grouped = grouped.sort_values(by=["sort_order", "option_type"], kind="stable").drop(columns=["sort_order"])
    return grouped.reset_index(drop=True)


def _empty_partitions() -> pd.DataFrame:
    return pd.DataFrame(columns=_PARTITION_COLUMNS)


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=_ROW_COLUMNS)


def _normalize_option_type(value: str) -> str:
    if value.startswith("c"):
        return "call"
    if value.startswith("p"):
        return "put"
    return "other"


def _run_query_safe(
    sql: str,
    *,
    params: list[Any] | None = None,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    path = resolve_duckdb_path(database_path)
    if not path.exists():
        return pd.DataFrame(), f"DuckDB database not found: {path}"
    try:
        conn = duckdb.connect(str(path), read_only=True)
        try:
            frame = conn.execute(sql, params or []).df()
        finally:
            conn.close()
        return frame, None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), _friendly_error(str(exc))


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if "options_flow" in lowered and "does not exist" in lowered:
        return "options_flow table not found. Run `options-helper flow ... --persist` first."
    return message


def _coerce_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _read_query_param(query_params: MutableMapping[str, Any], *, name: str) -> str | None:
    value = query_params.get(name)
    if isinstance(value, list):
        if not value:
            return None
        return str(value[0])
    if value is None:
        return None
    return str(value)
