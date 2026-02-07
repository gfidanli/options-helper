from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from apps.streamlit.components.sfp_page import (
    list_sfp_symbols,
    load_sfp_payload,
    normalize_symbol,
)
from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param


def _fmt_pct(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _render_events_table(df: pd.DataFrame, *, title: str, tail_low_pct: float, tail_high_pct: float) -> None:
    st.subheader(title)
    if df.empty:
        st.info("No events for the selected filters.")
        return

    display = df.copy()
    display = display.rename(
        columns={
            "event_ts": "date",
            "swept_swing_ts": "swept_swing_date",
            "bars_since_swing": "bars_from_swing",
            "candle_close": "close",
            "sweep_level": "sweep",
            "extension_percentile": "ext_pct",
            "forward_1d_pct": "max_1d_pct",
            "forward_5d_pct": "max_5d_pct",
            "forward_10d_pct": "max_10d_pct",
            "week_has_daily_extension_extreme": "week_has_daily_extreme",
        }
    )
    if "week_has_daily_extreme" in display.columns:
        display["week_has_daily_extreme"] = display["week_has_daily_extreme"].map(lambda x: "yes" if bool(x) else "no")

    ordered_cols = [
        "date",
        "timeframe",
        "direction",
        "close",
        "sweep",
        "ext_pct",
        "extension_atr",
        "rsi",
        "rsi_regime",
        "max_1d_pct",
        "max_5d_pct",
        "max_10d_pct",
        "swept_swing_date",
        "bars_from_swing",
    ]
    if "week_has_daily_extreme" in display.columns:
        ordered_cols.insert(10, "week_has_daily_extreme")

    display = display.reindex(columns=[col for col in ordered_cols if col in display.columns])
    for column in ("close", "sweep", "ext_pct", "extension_atr", "rsi", "max_1d_pct", "max_5d_pct", "max_10d_pct"):
        if column in display.columns:
            display[column] = display[column].map(_fmt_float)

    st.caption(
        f"Extension extremes currently defined as <= {tail_low_pct:.1f} or >= {tail_high_pct:.1f} percentile."
    )
    st.dataframe(display, hide_index=True, use_container_width=True)


st.title("SFP Research")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only SFP research view from persisted daily candles.")

with st.sidebar:
    st.markdown("### Data Source")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    lookback_days = st.slider(
        "Events lookback (calendar days)",
        min_value=30,
        max_value=2000,
        value=365,
        step=30,
    )
    tail_mode = st.select_slider(
        "Daily extension extreme threshold",
        options=["5/95", "2.5/97.5"],
        value="5/95",
    )
    show_only_extreme = st.checkbox("Show only events at extension extremes", value=False)
    with st.expander("Advanced Settings", expanded=False):
        swing_left = st.slider("Swing left bars", min_value=1, max_value=5, value=2, step=1)
        swing_right = st.slider("Swing right bars", min_value=1, max_value=5, value=2, step=1)
        min_swing_distance = st.slider("Min bars from swept swing", min_value=1, max_value=10, value=1, step=1)
        rsi_extremes = st.slider("RSI extreme band", min_value=5, max_value=45, value=(30, 70), step=1)

tail_low_pct, tail_high_pct = (5.0, 95.0) if tail_mode == "5/95" else (2.5, 97.5)
rsi_oversold, rsi_overbought = int(rsi_extremes[0]), int(rsi_extremes[1])

database_arg: str | Path | None = database_path or None
symbols, symbols_note = list_sfp_symbols(database_path=database_arg)
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
    selected_symbol = st.text_input("Symbol", value=query_symbol, max_chars=16)

active_symbol = sync_symbol_query_param(symbol=selected_symbol, query_params=st.query_params)

payload, payload_note = load_sfp_payload(
    symbol=active_symbol,
    lookback_days=lookback_days,
    tail_low_pct=tail_low_pct,
    tail_high_pct=tail_high_pct,
    rsi_overbought=float(rsi_overbought),
    rsi_oversold=float(rsi_oversold),
    swing_left_bars=swing_left,
    swing_right_bars=swing_right,
    min_swing_distance_bars=min_swing_distance,
    database_path=database_arg,
)
if payload_note:
    st.warning(f"SFP payload unavailable: {payload_note}")
    st.stop()
if payload is None:
    st.info("No SFP payload available.")
    st.stop()

for note in payload.get("notes") or []:
    st.warning(note)

counts = dict(payload.get("counts") or {})
metrics = st.columns(4)
metrics[0].metric("As-of", payload.get("asof") or "-")
metrics[1].metric("Daily SFP Events", str(int(counts.get("daily_events", 0))))
metrics[2].metric("Weekly SFP Events", str(int(counts.get("weekly_events", 0))))
metrics[3].metric(
    "Daily Bullish / Bearish",
    f"{int(counts.get('daily_bullish', 0))} / {int(counts.get('daily_bearish', 0))}",
)

summary_rows = payload.get("summary_rows") or []
summary_df = pd.DataFrame(summary_rows)
st.subheader("Key Insights")
if summary_df.empty:
    st.info("No summary metrics available.")
else:
    summary_display = summary_df.copy()
    summary_display = summary_display.rename(
        columns={
            "group": "group",
            "count": "count",
            "median_1d_pct": "median_max_1d_pct",
            "median_5d_pct": "median_max_5d_pct",
            "median_10d_pct": "median_max_10d_pct",
        }
    )
    for column in ("median_max_1d_pct", "median_max_5d_pct", "median_max_10d_pct"):
        if column in summary_display.columns:
            summary_display[column] = summary_display[column].map(_fmt_pct)
    st.dataframe(summary_display, hide_index=True, use_container_width=True)

daily_events_df = pd.DataFrame(payload.get("daily_events") or [])
weekly_events_df = pd.DataFrame(payload.get("weekly_events") or [])

if show_only_extreme:
    if not daily_events_df.empty and "extension_percentile" in daily_events_df.columns:
        daily_events_df = daily_events_df[
            (pd.to_numeric(daily_events_df["extension_percentile"], errors="coerce") <= tail_low_pct)
            | (pd.to_numeric(daily_events_df["extension_percentile"], errors="coerce") >= tail_high_pct)
        ]
    if not weekly_events_df.empty and "extension_percentile" in weekly_events_df.columns:
        weekly_events_df = weekly_events_df[
            (pd.to_numeric(weekly_events_df["extension_percentile"], errors="coerce") <= tail_low_pct)
            | (pd.to_numeric(weekly_events_df["extension_percentile"], errors="coerce") >= tail_high_pct)
        ]

_render_events_table(
    daily_events_df,
    title="Daily SFP Events",
    tail_low_pct=tail_low_pct,
    tail_high_pct=tail_high_pct,
)
_render_events_table(
    weekly_events_df,
    title="Weekly SFP Events (Week-Start Labels)",
    tail_low_pct=tail_low_pct,
    tail_high_pct=tail_high_pct,
)
