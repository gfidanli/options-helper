from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.portfolio_page import (
    build_portfolio_risk_summary,
    build_positions_dataframe,
    load_portfolio_safe,
)


def _fmt_int(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{int(value)}"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_currency(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "-"

st.title("Portfolio")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only view. This page does not modify portfolio files or market data.")

with st.sidebar:
    st.markdown("### Data Sources")
    portfolio_path = st.text_input("Portfolio JSON", value="portfolio.json")
    reports_path = st.text_input("Reports directory (briefing fallback)", value="reports")

portfolio, resolved_portfolio_path, portfolio_error = load_portfolio_safe(portfolio_path)
if portfolio_error:
    st.error(portfolio_error)
    st.stop()
if portfolio is None:
    st.warning("No portfolio data loaded.")
    st.stop()

st.caption(f"Loaded portfolio from `{resolved_portfolio_path}`.")

st.subheader("Positions")
positions_df = build_positions_dataframe(portfolio)
if positions_df.empty:
    st.info("No positions in portfolio.")
else:
    display_df = positions_df.copy()
    for column in ("cost_basis", "premium_at_risk", "strike"):
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

st.subheader("Risk Summary (best effort)")
risk_summary, risk_note = build_portfolio_risk_summary(
    portfolio,
    reports_path=reports_path,
)

if risk_summary.get("source") == "briefing":
    report_date = risk_summary.get("report_date") or "-"
    as_of = risk_summary.get("as_of") or "-"
    st.caption(f"Source: latest briefing artifact (report date {report_date}, as-of {as_of}).")
else:
    st.caption("Source: computed from portfolio metadata (fallback).")

if risk_note:
    st.info(risk_note)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Positions", value=_fmt_int(risk_summary.get("position_count")))
col2.metric("Symbols", value=_fmt_int(risk_summary.get("symbol_count")))
col3.metric("Premium At Risk", value=_fmt_currency(risk_summary.get("premium_at_risk")))
col4.metric("Capital Basis", value=_fmt_currency(risk_summary.get("capital_cost_basis")))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Portfolio Delta", value=_fmt_float(risk_summary.get("total_delta_shares"), 1))
col6.metric("Theta / Day ($)", value=_fmt_float(risk_summary.get("total_theta_dollars_per_day"), 1))
col7.metric("Vega / 1.00 IV ($)", value=_fmt_float(risk_summary.get("total_vega_dollars_per_iv"), 1))
col8.metric("Missing Greeks", value=_fmt_int(risk_summary.get("missing_greeks")))

col9, col10, col11, col12 = st.columns(4)
col9.metric("Cash", value=_fmt_currency(risk_summary.get("cash")))
col10.metric("Next Expiry", value=str(risk_summary.get("next_expiry") or "-"))
col11.metric("Take Profit", value=_fmt_pct(risk_summary.get("take_profit_pct")))
col12.metric("Stop Loss", value=_fmt_pct(risk_summary.get("stop_loss_pct")))

stress_rows = risk_summary.get("stress") or []
if isinstance(stress_rows, list) and stress_rows:
    st.markdown("**Stress Scenarios**")
    stress_df = pd.DataFrame(stress_rows)
    st.dataframe(stress_df, hide_index=True, use_container_width=True)

warnings = risk_summary.get("warnings") or []
if isinstance(warnings, list) and warnings:
    st.markdown("**Risk Warnings**")
    for warning in warnings:
        st.write(f"- {warning}")
