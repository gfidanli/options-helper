from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from apps.streamlit.components.flow_page import (
    DEFAULT_FLOW_GROUP,
    FLOW_GROUP_VALUES,
    build_flow_option_type_summary,
    build_flow_timeseries,
    list_flow_groups,
    list_flow_symbols,
    load_flow_asof_bounds,
    load_flow_partition_summaries,
    load_flow_rows_for_partition,
    normalize_flow_group,
    normalize_symbol,
    resolve_flow_group_selection,
    sync_flow_group_query_param,
)
from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param


def _fmt_date(value: object) -> str:
    if value is None:
        return "-"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "-"
    return parsed.date().isoformat()


def _fmt_number(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number:,.{digits}f}"


def _normalize_date_range(value: object, *, fallback_start: date, fallback_end: date) -> tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        start_raw, end_raw = value
    elif isinstance(value, list) and len(value) == 2:
        start_raw, end_raw = value
    else:
        start_raw, end_raw = value, value

    start = _coerce_date(start_raw) or fallback_start
    end = _coerce_date(end_raw) or fallback_end
    if start > end:
        start, end = end, start
    return start, end


def _coerce_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _partition_label(row: pd.Series) -> str:
    return (
        f"as_of={_fmt_date(row.get('as_of'))} | "
        f"from={_fmt_date(row.get('from_date'))} | "
        f"to={_fmt_date(row.get('to_date'))} | "
        f"window={int(float(row.get('window') or 0))} | "
        f"rows={int(float(row.get('rows') or 0))}"
    )


def _flow_row_label(row: pd.Series) -> str:
    contract_symbol = str(row.get("contract_symbol") or "").strip()
    if contract_symbol:
        return contract_symbol
    option_type = str(row.get("option_type") or "").strip().upper()
    expiry = _fmt_date(row.get("expiry"))
    strike = _fmt_number(row.get("strike"), digits=2)
    return f"{option_type} {strike} {expiry}".strip()


st.title("Flow")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only view. This page does not trigger ingestion or write to DuckDB.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    top_n = st.slider("Top rows in partition", min_value=10, max_value=200, value=60, step=10)

database_arg = database_path or None
flow_symbols, symbols_note = list_flow_symbols(database_path=database_arg)
if symbols_note:
    st.warning(f"Flow symbols unavailable: {symbols_note}")

query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
if flow_symbols:
    default_symbol = query_symbol if query_symbol in flow_symbols else flow_symbols[0]
    selected_symbol = st.selectbox(
        "Symbol",
        options=flow_symbols,
        index=flow_symbols.index(default_symbol),
        help="Symbols detected from persisted options_flow partitions.",
    )
else:
    selected_symbol = st.text_input("Symbol", value=query_symbol, max_chars=16)

active_symbol = sync_symbol_query_param(
    symbol=selected_symbol,
    query_params=st.query_params,
    default_symbol="SPY",
)

group_options, groups_note = list_flow_groups(active_symbol, database_path=database_arg)
if groups_note:
    st.warning(f"Flow groups unavailable: {groups_note}")

available_groups = group_options if group_options else list(FLOW_GROUP_VALUES)
query_group = normalize_flow_group(st.query_params.get("group_by"), default=DEFAULT_FLOW_GROUP)
default_group = resolve_flow_group_selection(
    available_groups,
    query_group=query_group,
    user_group=None,
    default_group=DEFAULT_FLOW_GROUP,
)
selected_group = st.selectbox(
    "Group",
    options=available_groups,
    index=available_groups.index(default_group) if default_group in available_groups else 0,
    help="Flow partition aggregation level persisted by `options-helper flow --group-by`.",
)
active_group = sync_flow_group_query_param(
    group_by=selected_group,
    query_params=st.query_params,
    default_group=DEFAULT_FLOW_GROUP,
)

min_as_of, max_as_of, bounds_note = load_flow_asof_bounds(
    active_symbol,
    group_by=active_group,
    database_path=database_arg,
)
if bounds_note:
    st.warning(f"Flow partitions unavailable: {bounds_note}")
if min_as_of is None or max_as_of is None:
    st.info(f"No flow partitions found for {active_symbol} ({active_group}).")
    st.stop()
    # `st.stop()` is a no-op in bare script execution (used by import smoke tests).
    # Keep fallback values so the module remains import-safe in that mode.
    today = date.today()
    min_as_of = today
    max_as_of = today

default_start = max(min_as_of, max_as_of - timedelta(days=30))
range_value = st.date_input(
    "As-of date range",
    value=(default_start, max_as_of),
    min_value=min_as_of,
    max_value=max_as_of,
    help="Filter persisted flow partitions by as_of date.",
)
as_of_start, as_of_end = _normalize_date_range(
    range_value,
    fallback_start=default_start,
    fallback_end=max_as_of,
)

partitions_df, partitions_note = load_flow_partition_summaries(
    active_symbol,
    group_by=active_group,
    as_of_start=as_of_start,
    as_of_end=as_of_end,
    limit=400,
    database_path=database_arg,
)
if partitions_note:
    st.warning(f"Flow partition query unavailable: {partitions_note}")
if partitions_df.empty:
    st.info("No persisted flow partitions match the selected filters.")
    st.stop()
else:
    latest_partition = partitions_df.iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Partitions", value=f"{len(partitions_df)}")
    metric_cols[1].metric("Latest As-of", value=_fmt_date(latest_partition.get("as_of")))
    metric_cols[2].metric(
        "Latest Net ΔOI Notional",
        value=_fmt_number(latest_partition.get("net_delta_oi_notional"), digits=0),
    )
    metric_cols[3].metric(
        "Latest Volume Notional",
        value=_fmt_number(latest_partition.get("total_volume_notional"), digits=0),
    )

    st.subheader("Partition Timeseries")
    timeseries_df = build_flow_timeseries(partitions_df)
    if timeseries_df.empty:
        st.info("Timeseries summary unavailable for selected filters.")
    else:
        st.line_chart(
            timeseries_df.set_index("as_of")[["net_delta_oi_notional", "total_volume_notional"]],
            use_container_width=True,
        )

    st.subheader("Partition Summary Table")
    display_partitions = partitions_df.copy()
    for column in ("as_of", "from_date", "to_date", "updated_at"):
        display_partitions[column] = pd.to_datetime(display_partitions[column], errors="coerce").dt.date.astype(str)
    st.dataframe(display_partitions, hide_index=True, use_container_width=True)

    partition_labels = [_partition_label(row) for _, row in partitions_df.iterrows()]
    selected_partition = st.selectbox(
        "Inspect Partition Rows",
        options=partition_labels,
        index=0,
        help="Loads top rows by absolute ΔOI notional from a single persisted partition.",
    )
    selected_row = partitions_df.iloc[partition_labels.index(selected_partition)]

    rows_df, rows_note = load_flow_rows_for_partition(
        active_symbol,
        as_of=selected_row.get("as_of"),
        from_date=selected_row.get("from_date"),
        to_date=selected_row.get("to_date"),
        window=int(float(selected_row.get("window") or 1)),
        group_by=active_group,
        top_n=top_n,
        database_path=database_arg,
    )
    if rows_note:
        st.warning(f"Partition row query unavailable: {rows_note}")
    elif rows_df.empty:
        st.info("No flow rows found for the selected partition.")
    else:
        row_metrics = st.columns(4)
        row_metrics[0].metric("Rows Loaded", value=f"{len(rows_df)}")
        row_metrics[1].metric(
            "Net ΔOI Notional",
            value=_fmt_number(rows_df["delta_oi_notional"].sum(), digits=0),
        )
        row_metrics[2].metric(
            "Net Delta Notional",
            value=_fmt_number(rows_df["delta_notional"].sum(), digits=0),
        )
        row_metrics[3].metric(
            "Total Volume Notional",
            value=_fmt_number(rows_df["volume_notional"].sum(), digits=0),
        )

        option_summary = build_flow_option_type_summary(rows_df)
        if not option_summary.empty:
            st.markdown("**Call/Put Summary**")
            st.dataframe(option_summary, hide_index=True, use_container_width=True)

        chart_rows = rows_df.head(min(20, len(rows_df))).copy()
        chart_rows["label"] = chart_rows.apply(_flow_row_label, axis=1)
        st.markdown("**Top Rows by ΔOI Notional**")
        st.bar_chart(
            chart_rows.set_index("label")[["delta_oi_notional"]],
            use_container_width=True,
        )

        display_rows = rows_df.copy()
        display_rows["expiry"] = pd.to_datetime(display_rows["expiry"], errors="coerce").dt.date.astype(str)
        st.dataframe(display_rows, hide_index=True, use_container_width=True)
