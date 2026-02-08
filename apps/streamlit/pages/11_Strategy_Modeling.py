from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

import options_helper.cli_deps as cli_deps
from apps.streamlit.components.strategy_modeling_page import (
    list_strategy_modeling_symbols,
    load_strategy_modeling_data_payload,
)
from options_helper.analysis.strategy_modeling import StrategyModelingRequest

DISCLAIMER_TEXT = "Informational and educational use only. Not financial advice."
_SEGMENT_DIMENSIONS = [
    "symbol",
    "direction",
    "extension_bucket",
    "rsi_regime",
    "rsi_divergence",
    "volatility_regime",
    "bars_since_swing_bucket",
]
_RESULT_STATE_KEY = "strategy_modeling_last_result"


def _to_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def _rows_to_df(rows: object) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()

    normalized: list[dict[str, Any]] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            mapped = _to_dict(row)
            if mapped:
                normalized.append(mapped)
    return pd.DataFrame(normalized)


def _as_symbol_filter(raw: str) -> list[str]:
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return sorted(set(values))


def _filter_segments(
    segment_df: pd.DataFrame,
    *,
    dimensions: Sequence[str],
    segment_values: Sequence[str],
) -> pd.DataFrame:
    if segment_df.empty:
        return segment_df

    out = segment_df.copy()
    if dimensions and "segment_dimension" in out.columns:
        out = out[out["segment_dimension"].astype(str).isin(dimensions)]

    if segment_values and "segment_value" in out.columns:
        wanted = {value.upper() for value in segment_values}
        out = out[out["segment_value"].astype(str).str.upper().isin(wanted)]

    return out


def _format_metric(value: object, *, pct: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pct:
        return f"{number:.2f}%"
    return f"{number:,.2f}"


st.title("Strategy Modeling")
st.caption(DISCLAIMER_TEXT)
st.info(
    "Read-only strategy modeling dashboard. This page runs deterministic local analysis only "
    "(no ingestion writes from portal interactions)."
)

symbols, symbol_notes = list_strategy_modeling_symbols(database_path=None)
default_end = date.today()
default_start = default_end - timedelta(days=365)

with st.sidebar:
    st.markdown("### Modeling Inputs")

    strategy = st.selectbox("Strategy", options=["sfp", "msb"], index=0)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    intraday_timeframe = st.selectbox("Intraday timeframe", options=["1Min", "5Min", "15Min", "30Min", "60Min"])
    intraday_source = st.selectbox(
        "Intraday source",
        options=["stocks_bars_local"],
        help="Current modeling service consumes persisted stock bars partitions.",
    )

    st.markdown("### Portfolio Policy")
    starting_capital = st.number_input("Starting capital", min_value=1_000.0, value=10_000.0, step=500.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    gap_policy = st.selectbox("Gap policy", options=["fill_at_open", "strict_touch"], index=0)
    max_hold_bars = st.number_input("Max hold (bars)", min_value=1, max_value=100, value=20, step=1)
    one_open_per_symbol = st.checkbox("One open trade per symbol", value=True)

    st.markdown("### Filters")
    if symbols:
        selected_symbols = st.multiselect("Tickers", options=symbols, default=symbols[: min(8, len(symbols))])
        symbol_filter_text = ""
    else:
        symbol_filter_text = st.text_input("Tickers (comma-separated)", value="SPY")
        selected_symbols = _as_symbol_filter(symbol_filter_text)

    segment_dimensions = st.multiselect(
        "Segment dimensions",
        options=_SEGMENT_DIMENSIONS,
        default=["symbol", "direction"],
    )
    segment_values_raw = st.text_input("Segment values (comma-separated)", value="")

for note in symbol_notes:
    st.warning(note)

payload = load_strategy_modeling_data_payload(
    symbols=selected_symbols,
    start_date=start_date,
    end_date=end_date,
    intraday_timeframe=intraday_timeframe,
    require_intraday_bars=True,
)

for item in payload.get("notes") or []:
    st.warning(str(item))
for item in payload.get("errors") or []:
    st.error(str(item))

blocking = dict(payload.get("blocking") or {})
coverage_rows = pd.DataFrame(blocking.get("coverage_rows") or [])
run_is_blocked = bool(blocking.get("is_blocked", False))

if run_is_blocked:
    blocked_symbols = blocking.get("blocked_symbols") or []
    missing_sessions_total = int(blocking.get("missing_sessions_total", 0) or 0)
    blocked_text = ", ".join(str(value) for value in blocked_symbols) if blocked_symbols else "selected symbols"
    st.warning(
        "Missing required intraday coverage for requested scope. "
        f"Blocked symbols: {blocked_text}. Missing sessions: {missing_sessions_total}."
    )
    if not coverage_rows.empty:
        details = coverage_rows[[col for col in ["symbol", "required_count", "covered_count", "missing_count", "missing_days"] if col in coverage_rows.columns]]
        st.dataframe(details, hide_index=True, use_container_width=True)

run_clicked = st.button(
    "Run Strategy Modeling",
    type="primary",
    use_container_width=True,
    disabled=run_is_blocked,
    help="Runs local deterministic strategy modeling. Disabled when required intraday coverage is missing.",
)
clear_clicked = st.button("Clear Results", use_container_width=True)

if clear_clicked:
    st.session_state.pop(_RESULT_STATE_KEY, None)

if run_clicked and not run_is_blocked:
    service = cli_deps.build_strategy_modeling_service()
    request = StrategyModelingRequest(
        strategy=strategy,
        symbols=tuple(selected_symbols),
        start_date=start_date,
        end_date=end_date,
        intraday_dir=Path("data/intraday"),
        intraday_timeframe=intraday_timeframe,
        starting_capital=float(starting_capital),
        max_hold_bars=int(max_hold_bars),
        policy={
            "require_intraday_bars": True,
            "risk_per_trade_pct": float(risk_pct),
            "gap_fill_policy": gap_policy,
            "one_open_per_symbol": bool(one_open_per_symbol),
        },
        block_on_missing_intraday_coverage=True,
    )
    st.session_state[_RESULT_STATE_KEY] = service.run(request)

result = st.session_state.get(_RESULT_STATE_KEY)

metrics_payload = _to_dict(getattr(result, "portfolio_metrics", None))
r_ladder_df = _rows_to_df(getattr(result, "target_hit_rates", None))
equity_df = _rows_to_df(getattr(result, "equity_curve", None))
segment_df = _rows_to_df(getattr(result, "segment_records", None))
trade_df = _rows_to_df(getattr(result, "trade_simulations", None))

segment_values = _as_symbol_filter(segment_values_raw)
if not segment_df.empty:
    segment_df = _filter_segments(segment_df, dimensions=segment_dimensions, segment_values=segment_values)

st.subheader("Key Metrics")
if not metrics_payload:
    st.info("Run modeling to view portfolio metrics.")
else:
    metric_cols = st.columns(6)
    metric_cols[0].metric("Starting Capital", _format_metric(metrics_payload.get("starting_capital")))
    metric_cols[1].metric("Ending Capital", _format_metric(metrics_payload.get("ending_capital")))
    metric_cols[2].metric("Total Return", _format_metric(metrics_payload.get("total_return_pct"), pct=True))
    metric_cols[3].metric("Trade Count", str(int(metrics_payload.get("trade_count") or 0)))
    metric_cols[4].metric("Win Rate", _format_metric(metrics_payload.get("win_rate"), pct=True))
    metric_cols[5].metric("Expectancy (R)", _format_metric(metrics_payload.get("expectancy_r")))

st.subheader("R-Ladder")
if r_ladder_df.empty:
    st.info("No R-ladder rows available yet.")
else:
    ladder = r_ladder_df.copy()
    if "target_r" in ladder.columns:
        ladder["target_r"] = pd.to_numeric(ladder["target_r"], errors="coerce")
        ladder = ladder.sort_values(by=["target_r", "target_label"], kind="stable")
    if "target_label" in ladder.columns and "hit_rate" in ladder.columns:
        chart_data = ladder[["target_label", "hit_rate"]].set_index("target_label")
        st.bar_chart(chart_data)
    st.dataframe(ladder, hide_index=True, use_container_width=True)

st.subheader("Equity Curve")
if equity_df.empty:
    st.info("No equity-curve rows available yet.")
else:
    curve = equity_df.copy()
    if "ts" in curve.columns:
        curve["ts"] = pd.to_datetime(curve["ts"], errors="coerce", utc=True)
        curve = curve.sort_values(by="ts", kind="stable")
    if {"ts", "equity"}.issubset(curve.columns):
        st.line_chart(curve.set_index("ts")["equity"])
    else:
        st.dataframe(curve, hide_index=True, use_container_width=True)

st.subheader("Segmented Breakdowns")
if segment_df.empty:
    st.info("No segment records available for the selected filters.")
else:
    st.dataframe(segment_df, hide_index=True, use_container_width=True)

st.subheader("Trade Log")
st.caption(
    "Realized R includes gap-through outcomes. Trade rows can show losses below -1.0R when stop fills occur at open."
)
if trade_df.empty:
    st.info("No trade simulations available yet.")
else:
    trades = trade_df.copy()
    if "entry_ts" in trades.columns:
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], errors="coerce", utc=True)
        trades = trades.sort_values(by="entry_ts", ascending=False, kind="stable")

    realized = pd.to_numeric(trades.get("realized_r"), errors="coerce")
    below_one_r = trades.loc[realized < -1.0].copy()
    if not below_one_r.empty:
        st.warning(
            f"{len(below_one_r)} trade(s) realized below -1.0R under current gap-policy assumptions."
        )

    display_cols = [
        col
        for col in [
            "trade_id",
            "symbol",
            "direction",
            "entry_ts",
            "entry_price",
            "stop_price",
            "target_price",
            "exit_ts",
            "exit_price",
            "exit_reason",
            "status",
            "realized_r",
            "mae_r",
            "mfe_r",
            "gap_fill_applied",
            "reject_code",
        ]
        if col in trades.columns
    ]
    st.dataframe(trades[display_cols], hide_index=True, use_container_width=True)

st.caption(
    f"Intraday source: `{intraday_source}`. Page behavior is read-only and does not execute ingest/backfill writes."
)
