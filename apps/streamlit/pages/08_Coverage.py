from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.coverage_page import (
    list_coverage_symbols,
    load_coverage_payload,
    normalize_symbol,
)
from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param


def _fmt_int(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{int(value):,d}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object, *, digits: int = 1) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number * 100.0:.{digits}f}%"


st.title("Coverage")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only coverage diagnostics from local DuckDB data.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    lookback_days = st.slider(
        "Lookback (business days)",
        min_value=20,
        max_value=260,
        value=60,
        step=5,
    )

database_arg: str | Path | None = database_path or None
symbols, symbols_note = list_coverage_symbols(database_path=database_arg)
if symbols_note:
    st.warning(f"Symbol discovery note: {symbols_note}")

query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
if symbols:
    default_symbol = query_symbol if query_symbol in symbols else symbols[0]
    selected_symbol = st.selectbox(
        "Symbol",
        options=symbols,
        index=symbols.index(default_symbol),
    )
else:
    selected_symbol = st.text_input("Symbol", value=query_symbol).strip().upper()

selected_symbol = sync_symbol_query_param(symbol=selected_symbol, query_params=st.query_params)

payload, payload_note = load_coverage_payload(
    symbol=selected_symbol,
    lookback_days=lookback_days,
    database_path=database_arg,
)
if payload_note:
    st.warning(f"Coverage unavailable: {payload_note}")
    st.stop()

if not payload:
    st.info("No coverage payload available.")
    st.stop()

for note in payload.get("notes") or []:
    st.warning(note)

candles = dict(payload.get("candles") or {})
snapshots = dict(payload.get("snapshots") or {})
contracts_oi = dict(payload.get("contracts_oi") or {})
option_bars = dict(payload.get("option_bars") or {})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candle rows", _fmt_int(candles.get("rows_total")))
col2.metric("Candle gaps", _fmt_int(candles.get("missing_business_days")))
col3.metric("Contracts", _fmt_int(contracts_oi.get("contracts_total")))
col4.metric("OI day coverage", _fmt_pct(contracts_oi.get("oi_day_coverage_ratio")))

st.subheader("Candles")
st.dataframe(
    pd.DataFrame(
        [
            {
                "rows_total": candles.get("rows_total"),
                "rows_lookback": candles.get("rows_lookback"),
                "start_date": candles.get("start_date"),
                "end_date": candles.get("end_date"),
                "missing_business_days": candles.get("missing_business_days"),
                "missing_value_cells": candles.get("missing_value_cells"),
            }
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Options Snapshot Headers")
st.dataframe(
    pd.DataFrame(
        [
            {
                "days_present_total": snapshots.get("days_present_total"),
                "days_present_lookback": snapshots.get("days_present_lookback"),
                "start_date": snapshots.get("start_date"),
                "end_date": snapshots.get("end_date"),
                "missing_business_days": snapshots.get("missing_business_days"),
                "avg_contracts_per_day": snapshots.get("avg_contracts_per_day"),
            }
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Contract + OI Coverage")
contracts_table = pd.DataFrame(
    [
        {
            "contracts_total": contracts_oi.get("contracts_total"),
            "contracts_with_snapshots": contracts_oi.get("contracts_with_snapshots"),
            "contracts_with_oi": contracts_oi.get("contracts_with_oi"),
            "snapshot_day_coverage_ratio": contracts_oi.get("snapshot_day_coverage_ratio"),
            "oi_day_coverage_ratio": contracts_oi.get("oi_day_coverage_ratio"),
            "snapshot_days_missing": contracts_oi.get("snapshot_days_missing"),
        }
    ]
)
st.dataframe(contracts_table, hide_index=True, use_container_width=True)

oi_delta_rows = contracts_oi.get("oi_delta_coverage") or []
if oi_delta_rows:
    st.markdown("#### OI Delta Coverage")
    st.dataframe(pd.DataFrame(oi_delta_rows), hide_index=True, use_container_width=True)

st.subheader("Option Bars Meta")
st.dataframe(
    pd.DataFrame(
        [
            {
                "contracts_total": option_bars.get("contracts_total"),
                "contracts_with_rows": option_bars.get("contracts_with_rows"),
                "rows_total": option_bars.get("rows_total"),
                "start_date": option_bars.get("start_date"),
                "end_date": option_bars.get("end_date"),
                "contracts_covering_lookback_end": option_bars.get("contracts_covering_lookback_end"),
                "covering_lookback_end_ratio": option_bars.get("covering_lookback_end_ratio"),
            }
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

status_counts = option_bars.get("status_counts") or {}
if status_counts:
    st.markdown("#### Option Bars Status")
    st.dataframe(
        pd.DataFrame(
            [{"status": key, "count": value} for key, value in sorted(status_counts.items())]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Repair Suggestions")
repair_rows = payload.get("repair_suggestions") or []
if not repair_rows:
    st.success("No repair commands suggested.")
else:
    for row in repair_rows:
        reason = str((row or {}).get("reason") or "")
        command = str((row or {}).get("command") or "")
        note = str((row or {}).get("note") or "").strip()
        st.markdown(f"**{reason}**")
        st.code(command, language="bash")
        if note:
            st.caption(note)
