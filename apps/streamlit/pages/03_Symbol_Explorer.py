from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.streamlit.components.symbol_explorer_page import (
    build_derived_snippet,
    build_snapshot_strike_table,
    list_available_symbols,
    load_candles_history,
    load_derived_history,
    load_latest_snapshot_header,
    load_snapshot_chain,
    normalize_symbol,
    resolve_symbol_selection,
    sync_symbol_query_param,
    summarize_snapshot,
)


def _fmt_currency(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{int(float(value)):,d}"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "-"


st.title("Symbol Explorer")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only view. This page does not trigger ingestion or modify stored artifacts.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )

available_symbols, symbol_notes = list_available_symbols(database_path=database_path or None)
query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
default_symbol = resolve_symbol_selection(
    available_symbols,
    query_symbol=query_symbol,
    user_symbol=None,
    default_symbol="SPY",
)

if available_symbols:
    selection_index = available_symbols.index(default_symbol) if default_symbol in available_symbols else 0
    selected_symbol = st.selectbox(
        "Symbol",
        options=available_symbols,
        index=selection_index,
        help="Symbols detected from candles, snapshot headers, or derived history.",
    )
else:
    selected_symbol = st.text_input("Symbol", value=default_symbol, max_chars=16)

active_symbol = sync_symbol_query_param(
    symbol=selected_symbol,
    query_params=st.query_params,
    default_symbol="SPY",
)

for note in symbol_notes:
    st.caption(f"Best effort note: {note}")

st.subheader(f"{active_symbol} Candles")
candles_df, candles_note = load_candles_history(
    active_symbol,
    database_path=database_path or None,
    limit=365,
)
if candles_note:
    st.warning(f"Candles unavailable: {candles_note}")
elif candles_df.empty:
    st.info(f"No candles history found for {active_symbol}.")
else:
    chart_df = candles_df.set_index("ts")[["close"]]
    st.line_chart(chart_df, use_container_width=True)
    latest_candle = candles_df.iloc[-1]
    candle_cols = st.columns(4)
    candle_cols[0].metric("Close", value=_fmt_currency(latest_candle.get("close")))
    candle_cols[1].metric("Open", value=_fmt_currency(latest_candle.get("open")))
    candle_cols[2].metric("High", value=_fmt_currency(latest_candle.get("high")))
    candle_cols[3].metric("Low", value=_fmt_currency(latest_candle.get("low")))

st.subheader("Latest Snapshot")
snapshot_header, snapshot_header_note = load_latest_snapshot_header(
    active_symbol,
    database_path=database_path or None,
)
if snapshot_header_note:
    st.warning(f"Snapshot headers unavailable: {snapshot_header_note}")
elif snapshot_header is None:
    st.info(f"No snapshot header found for {active_symbol}.")
else:
    st.caption(
        "Snapshot date: "
        f"{snapshot_header.get('snapshot_date')} | provider: {snapshot_header.get('provider')}"
    )
    chain_df, chain_note = load_snapshot_chain(
        snapshot_header.get("chain_path"),
        database_path=database_path or None,
    )
    if chain_note:
        st.warning(f"Snapshot chain unavailable: {chain_note}")
    else:
        snapshot_summary = summarize_snapshot(snapshot_header=snapshot_header, chain_df=chain_df)
        sum_cols = st.columns(4)
        sum_cols[0].metric("Contracts", value=_fmt_int(snapshot_summary.get("contracts")))
        sum_cols[1].metric("Spot", value=_fmt_currency(snapshot_summary.get("spot")))
        sum_cols[2].metric("Total OI", value=_fmt_int(snapshot_summary.get("total_open_interest")))
        sum_cols[3].metric("Total Volume", value=_fmt_int(snapshot_summary.get("total_volume")))

        ratio_cols = st.columns(4)
        ratio_cols[0].metric("Call OI", value=_fmt_int(snapshot_summary.get("call_open_interest")))
        ratio_cols[1].metric("Put OI", value=_fmt_int(snapshot_summary.get("put_open_interest")))
        ratio_cols[2].metric("Put/Call OI", value=_fmt_float(snapshot_summary.get("put_call_oi_ratio"), 2))
        ratio_cols[3].metric(
            "ATM IV",
            value=_fmt_pct(snapshot_summary.get("atm_implied_volatility")),
        )

        strike_df = build_snapshot_strike_table(chain_df, top_n=15)
        if strike_df.empty:
            st.info("Snapshot chain loaded, but strike-level OI summary is unavailable.")
        else:
            display_strikes = strike_df.copy()
            for column in ("call_oi", "put_oi", "total_oi", "total_volume"):
                display_strikes[column] = pd.to_numeric(display_strikes[column], errors="coerce")
            st.markdown("**Top Strikes by Open Interest**")
            st.dataframe(display_strikes, hide_index=True, use_container_width=True)

st.subheader("Derived Snippet")
derived_df, derived_note = load_derived_history(
    active_symbol,
    database_path=database_path or None,
    limit=120,
)
if derived_note:
    st.warning(f"Derived history unavailable: {derived_note}")
elif derived_df.empty:
    st.info(f"No derived history found for {active_symbol}.")
else:
    derived_snippet = build_derived_snippet(derived_df)
    if derived_snippet is not None:
        snippet_cols = st.columns(4)
        snippet_cols[0].metric("As-of", value=str(derived_snippet.get("as_of") or "-"))
        snippet_cols[1].metric("Spot", value=_fmt_currency(derived_snippet.get("spot")))
        snippet_cols[2].metric("P/C OI", value=_fmt_float(derived_snippet.get("pc_oi"), 2))
        snippet_cols[3].metric("P/C Vol", value=_fmt_float(derived_snippet.get("pc_vol"), 2))

        snippet_cols2 = st.columns(4)
        snippet_cols2[0].metric("ATM IV", value=_fmt_pct(derived_snippet.get("atm_iv_near")))
        snippet_cols2[1].metric("IV/RV20", value=_fmt_float(derived_snippet.get("iv_rv_20d"), 2))
        snippet_cols2[2].metric(
            "IV Percentile",
            value=_fmt_pct(derived_snippet.get("atm_iv_near_percentile")),
        )
        snippet_cols2[3].metric("IV Term Slope", value=_fmt_float(derived_snippet.get("iv_term_slope"), 3))

    st.markdown("**Recent Derived Rows**")
    preview = derived_df.tail(10).copy()
    preview["date"] = preview["date"].dt.date.astype(str)
    st.dataframe(preview, hide_index=True, use_container_width=True)
