from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.streamlit.components.derived_history_page import (
    build_derived_latest_summary,
    list_derived_symbols,
    load_derived_history,
    normalize_symbol,
    slice_derived_history_window,
)
from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param


def _fmt_currency(value: object) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"${number:,.2f}"


def _fmt_float(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number:.{digits}f}"


def _fmt_pct(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number * 100.0:.{digits}f}%"


st.title("Derived History")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only view. This page does not run derived jobs or modify stored rows.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    window_days = st.select_slider("Time window (days)", options=[30, 60, 90, 180, 365, 730], value=180)
    preview_rows = st.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)

database_arg = database_path or None
derived_symbols, symbols_note = list_derived_symbols(database_path=database_arg)
if symbols_note:
    st.warning(f"Derived symbols unavailable: {symbols_note}")

query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
if derived_symbols:
    default_symbol = query_symbol if query_symbol in derived_symbols else derived_symbols[0]
    selected_symbol = st.selectbox(
        "Symbol",
        options=derived_symbols,
        index=derived_symbols.index(default_symbol),
        help="Symbols detected from persisted derived_daily rows.",
    )
else:
    selected_symbol = st.text_input("Symbol", value=query_symbol, max_chars=16)

active_symbol = sync_symbol_query_param(
    symbol=selected_symbol,
    query_params=st.query_params,
    default_symbol="SPY",
)

derived_df, derived_note = load_derived_history(
    active_symbol,
    database_path=database_arg,
    limit=3000,
)
if derived_note:
    st.warning(f"Derived history unavailable: {derived_note}")
if derived_df.empty:
    st.info(f"No derived history found for {active_symbol}.")
    st.stop()

window_df = slice_derived_history_window(derived_df, window_days=window_days)
summary = build_derived_latest_summary(window_df)

if summary is not None:
    st.subheader("Latest Snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("As-of", value=str(summary.get("as_of") or "-"))
    metric_cols[1].metric("Spot", value=_fmt_currency(summary.get("spot")))
    metric_cols[2].metric("P/C OI", value=_fmt_float(summary.get("pc_oi"), digits=2))
    metric_cols[3].metric("P/C Vol", value=_fmt_float(summary.get("pc_vol"), digits=2))

    metric_cols2 = st.columns(4)
    metric_cols2[0].metric("ATM IV", value=_fmt_pct(summary.get("atm_iv_near")))
    metric_cols2[1].metric("IV/RV20", value=_fmt_float(summary.get("iv_rv_20d"), digits=2))
    metric_cols2[2].metric(
        "IV Percentile",
        value=_fmt_pct(summary.get("atm_iv_near_percentile")),
    )
    metric_cols2[3].metric("IV Term Slope", value=_fmt_float(summary.get("iv_term_slope"), digits=3))

    metric_cols3 = st.columns(3)
    metric_cols3[0].metric("Spot 1D", value=_fmt_pct(summary.get("spot_change_1d")))
    metric_cols3[1].metric("ATM IV 1D", value=_fmt_pct(summary.get("atm_iv_change_1d")))
    metric_cols3[2].metric("Rows in Window", value=f"{int(summary.get('sample_rows') or 0)}")

st.subheader(f"Timeseries ({window_days}D Window)")
spot_cols = [column for column in ("spot", "gamma_peak_strike", "call_wall", "put_wall") if column in window_df.columns]
if spot_cols:
    st.markdown("**Price / Strikes**")
    st.line_chart(window_df.set_index("date")[spot_cols], use_container_width=True)

vol_cols = [column for column in ("atm_iv_near", "rv_20d", "rv_60d", "iv_rv_20d") if column in window_df.columns]
if vol_cols:
    st.markdown("**Volatility Regime**")
    st.line_chart(window_df.set_index("date")[vol_cols], use_container_width=True)

ratio_cols = [column for column in ("pc_oi", "pc_vol", "atm_iv_near_percentile", "iv_term_slope") if column in window_df.columns]
if ratio_cols:
    st.markdown("**Ratios / Positioning**")
    st.line_chart(window_df.set_index("date")[ratio_cols], use_container_width=True)

st.subheader("Recent Rows")
preview = window_df.tail(preview_rows).copy()
preview["date"] = pd.to_datetime(preview["date"], errors="coerce").dt.date.astype(str)
st.dataframe(preview, hide_index=True, use_container_width=True)
