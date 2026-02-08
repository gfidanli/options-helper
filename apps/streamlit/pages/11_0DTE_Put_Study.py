from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param
from apps.streamlit.components.zero_dte_put_page import (
    load_calibration_curves,
    load_forward_snapshots,
    load_probability_surface,
    load_strike_table,
    load_walk_forward_summary,
    list_zero_dte_symbols,
    normalize_symbol,
)


def _fmt_pct_unit(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, *, digits: int = 3) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "-"
    return parsed.date().isoformat()


def _dedupe_notes(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        note = str(value or "").strip()
        if not note or note in seen:
            continue
        seen.add(note)
        out.append(note)
    return out


st.title("0DTE Put Study (SPY Proxy)")
st.caption("Informational and educational use only. Not financial advice.")
st.info(
    "Read-only portal view over persisted 0DTE study artifacts and forward snapshots "
    "(no writes from this page)."
)

with st.sidebar:
    st.markdown("### Data Source")
    reports_root = st.text_input(
        "Reports root",
        value="data/reports",
        help="Root path containing zero_dte_put_study artifacts.",
    )

symbols, symbols_note = list_zero_dte_symbols(reports_root=reports_root or None)
if symbols_note:
    st.warning(symbols_note)

query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
if symbols:
    default_symbol = query_symbol if query_symbol in symbols else symbols[0]
    selected_symbol = st.selectbox("Symbol", options=symbols, index=symbols.index(default_symbol))
else:
    selected_symbol = st.text_input("Symbol", value=query_symbol, max_chars=16)
active_symbol = sync_symbol_query_param(
    symbol=selected_symbol,
    query_params=st.query_params,
    default_symbol="SPY",
)

base_probabilities, base_probability_notes = load_probability_surface(
    active_symbol,
    reports_root=reports_root or None,
)
base_ladder, base_ladder_notes = load_strike_table(
    active_symbol,
    reports_root=reports_root or None,
)

decision_modes = sorted(
    set(base_probabilities["decision_mode"].dropna().astype(str).str.lower().tolist())
    | set(base_ladder["decision_mode"].dropna().astype(str).str.lower().tolist())
)
decision_times = sorted(
    set(base_probabilities["decision_time_et"].dropna().astype(str).tolist())
    | set(base_ladder["decision_time_et"].dropna().astype(str).tolist())
)
risk_tiers = sorted(
    set(pd.to_numeric(base_probabilities["risk_tier"], errors="coerce").dropna().tolist())
    | set(pd.to_numeric(base_ladder["risk_tier"], errors="coerce").dropna().tolist())
)
fill_models = sorted(set(base_ladder["fill_model"].dropna().astype(str).str.lower().tolist()))
strike_returns = pd.to_numeric(base_probabilities["strike_return"], errors="coerce").dropna().abs()
if strike_returns.empty:
    strike_returns = pd.to_numeric(base_ladder["strike_return"], errors="coerce").dropna().abs()
strike_distance_max = float(max(strike_returns.max() * 100.0, 1.0)) if not strike_returns.empty else 3.0

with st.sidebar:
    st.markdown("### Filters")
    decision_mode_filter = st.selectbox(
        "Decision mode",
        options=["all", *decision_modes] if decision_modes else ["all"],
        index=0,
    )
    decision_time_filter = st.selectbox(
        "Decision time (ET)",
        options=["all", *decision_times] if decision_times else ["all"],
        index=0,
    )
    risk_tier_labels = ["all", *[f"{float(value):.3%}" for value in risk_tiers]]
    risk_tier_choice = st.selectbox("Risk tier", options=risk_tier_labels, index=0)
    risk_tier_value = None if risk_tier_choice == "all" else float(risk_tier_choice.replace("%", "")) / 100.0
    strike_distance_filter = st.slider(
        "Max strike distance (%)",
        min_value=0.10,
        max_value=round(max(strike_distance_max, 0.10), 2),
        value=round(max(strike_distance_max, 0.10), 2),
        step=0.10,
    )
    fill_model_filter = st.selectbox(
        "Fill assumption",
        options=["all", *fill_models] if fill_models else ["all"],
        index=0,
    )

probability_df, probability_notes = load_probability_surface(
    active_symbol,
    reports_root=reports_root or None,
    decision_mode=decision_mode_filter,
    decision_time_et=decision_time_filter,
    risk_tier=risk_tier_value,
    max_strike_distance_pct=strike_distance_filter,
)

ladder_df, ladder_notes = load_strike_table(
    active_symbol,
    reports_root=reports_root or None,
    decision_mode=decision_mode_filter,
    decision_time_et=decision_time_filter,
    risk_tier=risk_tier_value,
    max_strike_distance_pct=strike_distance_filter,
    fill_model=fill_model_filter,
)

walk_summary_df, walk_notes = load_walk_forward_summary(
    active_symbol,
    reports_root=reports_root or None,
    decision_mode=decision_mode_filter,
    risk_tier=risk_tier_value,
)

calibration_df, calibration_notes = load_calibration_curves(
    active_symbol,
    reports_root=reports_root or None,
    risk_tier=risk_tier_value,
)

forward_df, forward_notes = load_forward_snapshots(
    active_symbol,
    reports_root=reports_root or None,
    decision_mode=decision_mode_filter,
    decision_time_et=decision_time_filter,
    risk_tier=risk_tier_value,
    max_strike_distance_pct=strike_distance_filter,
)

for note in _dedupe_notes(
    [
        *base_probability_notes,
        *base_ladder_notes,
        *probability_notes,
        *ladder_notes,
        *walk_notes,
        *calibration_notes,
        *forward_notes,
    ]
):
    st.warning(note)

metric_cols = st.columns(4)
metric_cols[0].metric("Symbol", active_symbol)
metric_cols[1].metric("Probability Rows", str(len(probability_df)))
metric_cols[2].metric("Ladder Rows", str(len(ladder_df)))
metric_cols[3].metric("Forward Rows", str(len(forward_df)))

tabs = st.tabs(
    [
        "Current State Probability",
        "Recommended Ladder",
        "Calibration / Reliability",
        "Walk-Forward + Forward-Test",
    ]
)

with tabs[0]:
    st.subheader("Current-State Probability Surface")
    if probability_df.empty:
        st.info("No probability surface rows match the selected filters.")
    else:
        latest_ts = probability_df["decision_ts"].max()
        current_df = probability_df.loc[probability_df["decision_ts"] == latest_ts].copy()
        current_df = current_df.sort_values(by=["risk_tier", "strike_return"], kind="stable")
        current_df["session_date"] = current_df["session_date"].map(_fmt_date)
        current_df["risk_tier"] = current_df["risk_tier"].map(_fmt_pct_unit)
        current_df["strike_return"] = current_df["strike_return"].map(_fmt_pct_unit)
        current_df["breach_probability"] = current_df["breach_probability"].map(_fmt_pct_unit)
        current_df["breach_probability_ci_low"] = current_df["breach_probability_ci_low"].map(_fmt_pct_unit)
        current_df["breach_probability_ci_high"] = current_df["breach_probability_ci_high"].map(_fmt_pct_unit)
        st.caption(f"Latest decision timestamp: {latest_ts}")
        st.dataframe(
            current_df[
                [
                    "session_date",
                    "decision_time_et",
                    "decision_mode",
                    "risk_tier",
                    "strike_return",
                    "breach_probability",
                    "breach_probability_ci_low",
                    "breach_probability_ci_high",
                    "sample_size",
                    "quote_quality_status",
                    "skip_reason",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

with tabs[1]:
    st.subheader("Recommended Ladder")
    if ladder_df.empty:
        st.info("No strike ladder rows match the selected filters.")
    else:
        latest_ts = ladder_df["decision_ts"].max()
        current_ladder = ladder_df.loc[ladder_df["decision_ts"] == latest_ts].copy()
        current_ladder = current_ladder.sort_values(by=["risk_tier", "ladder_rank"], kind="stable")
        current_ladder["session_date"] = current_ladder["session_date"].map(_fmt_date)
        current_ladder["risk_tier"] = current_ladder["risk_tier"].map(_fmt_pct_unit)
        current_ladder["strike_return"] = current_ladder["strike_return"].map(_fmt_pct_unit)
        current_ladder["breach_probability"] = current_ladder["breach_probability"].map(_fmt_pct_unit)
        current_ladder["strike_price"] = current_ladder["strike_price"].map(_fmt_float)
        current_ladder["premium_estimate"] = current_ladder["premium_estimate"].map(_fmt_float)
        st.caption(f"Latest decision timestamp: {latest_ts}")
        st.dataframe(
            current_ladder[
                [
                    "session_date",
                    "decision_time_et",
                    "decision_mode",
                    "risk_tier",
                    "ladder_rank",
                    "strike_price",
                    "strike_return",
                    "breach_probability",
                    "premium_estimate",
                    "fill_model",
                    "quote_quality_status",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("Calibration / Reliability")
    if calibration_df.empty:
        st.info("No calibration rows are available yet. Finalized forward-test outcomes are required.")
    else:
        display = calibration_df.copy()
        display["risk_tier"] = display["risk_tier"].map(_fmt_pct_unit)
        display["predicted_mean"] = display["predicted_mean"].map(_fmt_pct_unit)
        display["observed_rate"] = display["observed_rate"].map(_fmt_pct_unit)
        display["abs_gap"] = display["abs_gap"].map(_fmt_pct_unit)
        st.dataframe(display, hide_index=True, use_container_width=True)

        curve = calibration_df.pivot_table(
            index="bin_index",
            columns="risk_tier",
            values="abs_gap",
            aggfunc="mean",
        )
        if not curve.empty:
            st.caption("Reliability gap by probability bin and risk tier (forward test).")
            st.line_chart(curve, use_container_width=True)

with tabs[3]:
    st.subheader("Walk-Forward Summary")
    hold_col, adaptive_col = st.columns(2)
    hold_summary = walk_summary_df.loc[walk_summary_df["exit_mode"] == "hold_to_close"].copy()
    adaptive_summary = walk_summary_df.loc[walk_summary_df["exit_mode"] == "adaptive_exit"].copy()

    with hold_col:
        st.markdown("**Hold to Close**")
        if hold_summary.empty:
            st.info("No hold_to_close walk-forward rows available for the selected filters.")
        else:
            show = hold_summary.copy()
            show["risk_tier"] = show["risk_tier"].map(_fmt_pct_unit)
            show["avg_pnl_per_contract"] = show["avg_pnl_per_contract"].map(_fmt_float)
            show["median_pnl_per_contract"] = show["median_pnl_per_contract"].map(_fmt_float)
            show["win_rate"] = show["win_rate"].map(_fmt_pct_unit)
            st.dataframe(show, hide_index=True, use_container_width=True)

    with adaptive_col:
        st.markdown("**Adaptive Exit**")
        if adaptive_summary.empty:
            st.info("No adaptive_exit walk-forward rows available for the selected filters.")
        else:
            show = adaptive_summary.copy()
            show["risk_tier"] = show["risk_tier"].map(_fmt_pct_unit)
            show["avg_pnl_per_contract"] = show["avg_pnl_per_contract"].map(_fmt_float)
            show["median_pnl_per_contract"] = show["median_pnl_per_contract"].map(_fmt_float)
            show["win_rate"] = show["win_rate"].map(_fmt_pct_unit)
            st.dataframe(show, hide_index=True, use_container_width=True)

    st.subheader("Forward Test Snapshots")
    if forward_df.empty:
        st.info("No forward snapshot rows are available yet for this symbol/filters.")
    else:
        latest_session = forward_df["session_date"].max()
        st.caption(f"Latest forward session: {latest_session}")
        show = forward_df.copy()
        show["session_date"] = show["session_date"].map(_fmt_date)
        show["risk_tier"] = show["risk_tier"].map(_fmt_pct_unit)
        show["strike_return"] = show["strike_return"].map(_fmt_pct_unit)
        show["breach_probability"] = show["breach_probability"].map(_fmt_pct_unit)
        show["breach_probability_ci_low"] = show["breach_probability_ci_low"].map(_fmt_pct_unit)
        show["breach_probability_ci_high"] = show["breach_probability_ci_high"].map(_fmt_pct_unit)
        show["realized_close_return_from_entry"] = show["realized_close_return_from_entry"].map(_fmt_pct_unit)
        show["strike_price"] = show["strike_price"].map(_fmt_float)
        show["premium_estimate"] = show["premium_estimate"].map(_fmt_float)
        st.dataframe(show, hide_index=True, use_container_width=True)
