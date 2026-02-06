from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from apps.streamlit.components.market_analysis_page import (
    compute_end_return_percentile,
    compute_tail_risk_for_symbol,
    list_candle_symbols,
    load_candles_close,
    load_latest_derived_row,
    normalize_symbol,
)
from apps.streamlit.components.symbol_explorer_page import sync_symbol_query_param
from options_helper.analysis.iv_context import classify_iv_rv
from options_helper.analysis.tail_risk import TailRiskConfig
from options_helper.data.confluence_config import ConfigError, load_confluence_config


def _fmt_currency(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct_100(value: object, *, digits: int = 0) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, *, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _load_iv_thresholds() -> tuple[float, float]:
    default_low = 0.8
    default_high = 1.2
    try:
        cfg = load_confluence_config()
    except (ConfigError, OSError, ValueError):
        return default_low, default_high
    iv_cfg = (cfg.get("iv_regime") or {}) if isinstance(cfg, dict) else {}
    try:
        low = float(iv_cfg.get("low", default_low))
        high = float(iv_cfg.get("high", default_high))
    except Exception:  # noqa: BLE001
        return default_low, default_high
    if low >= high:
        return default_low, default_high
    return low, high


st.title("Market Analysis")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only Monte Carlo tail-risk view from persisted candles and derived metrics.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    st.markdown("### Simulation")
    lookback_years = st.slider("Lookback (years)", min_value=1, max_value=10, value=6, step=1)
    horizon_days = st.slider("Horizon (days)", min_value=10, max_value=120, value=60, step=5)
    num_simulations = st.slider("Simulations", min_value=5000, max_value=100000, value=25000, step=5000)
    seed = int(st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1))
    var_confidence = st.slider("VaR confidence", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    show_sample_paths = st.checkbox("Show sample paths", value=False)
    sample_paths = st.slider("Sample paths count", min_value=20, max_value=200, value=80, step=20)

database_arg: str | Path | None = database_path or None
symbols, symbols_note = list_candle_symbols(database_path=database_arg)
if symbols_note:
    st.warning(f"Candle symbols unavailable: {symbols_note}")

query_symbol = normalize_symbol(st.query_params.get("symbol"), default="SPY")
if symbols:
    default_symbol = query_symbol if query_symbol in symbols else symbols[0]
    selected_symbol = st.selectbox(
        "Symbol",
        options=symbols,
        index=symbols.index(default_symbol),
        help="Symbols detected from persisted candles_daily rows.",
    )
else:
    selected_symbol = st.text_input("Symbol", value=query_symbol, max_chars=16)

active_symbol = sync_symbol_query_param(
    symbol=selected_symbol,
    query_params=st.query_params,
    default_symbol="SPY",
)

config = TailRiskConfig(
    lookback_days=lookback_years * 252,
    horizon_days=horizon_days,
    num_simulations=num_simulations,
    seed=seed,
    var_confidence=var_confidence,
    sample_paths=sample_paths,
)
result, result_note = compute_tail_risk_for_symbol(
    active_symbol,
    database_path=database_arg,
    config=config,
)
if result_note:
    st.warning(f"Tail risk unavailable: {result_note}")
if result is None:
    st.info("No tail-risk output available for this symbol and settings.")
else:
    st.subheader("Tail Risk (Monte Carlo)")
    candles_df, candles_note = load_candles_close(active_symbol, database_path=database_arg, limit=365)
    if candles_note:
        st.warning(f"Price chart unavailable: {candles_note}")
    elif candles_df.empty:
        st.info("No candle data found for recent price chart.")
    else:
        price_chart = candles_df.set_index("ts")[["close"]]
        st.markdown("**Last ~1Y Close**")
        st.line_chart(price_chart, use_container_width=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Spot", value=_fmt_currency(result.spot))
    metric_cols[1].metric("RV (Annualized)", value=_fmt_pct(result.realized_vol_annual))
    metric_cols[2].metric(f"VaR ({var_confidence:.0%})", value=_fmt_pct(result.var_return))
    metric_cols[3].metric(f"CVaR ({var_confidence:.0%})", value=_fmt_pct(result.cvar_return))

    metric_cols2 = st.columns(2)
    metric_cols2[0].metric("Expected Return (Annualized)", value=_fmt_pct(result.expected_return_annual))
    metric_cols2[1].metric("Warnings", value=str(len(result.warnings)))
    st.caption("Model assumption note: Monte Carlo paths use normally-distributed daily log-return shocks.")

    band_chart = result.daily_price_bands.copy()
    if not band_chart.empty:
        st.markdown("**Forecast Fan (Percentile Bands)**")
        st.line_chart(band_chart, use_container_width=True)

    if show_sample_paths and not result.sample_price_paths.empty:
        sample_df = result.sample_price_paths.copy()
        if sample_df.shape[1] > 60:
            sample_df = sample_df.iloc[:, :60]
        st.markdown("**Sample Paths (subset)**")
        st.line_chart(sample_df, use_container_width=True)

    rows = []
    for pct in sorted(result.end_price_percentiles):
        rows.append(
            {
                "percentile": f"{pct:.0f}",
                "end_price": result.end_price_percentiles.get(pct),
                "end_return": result.end_return_percentiles.get(pct),
            }
        )
    end_percentiles_df = pd.DataFrame(rows)
    if not end_percentiles_df.empty:
        end_percentiles_df["end_price"] = pd.to_numeric(end_percentiles_df["end_price"], errors="coerce")
        end_percentiles_df["end_return"] = pd.to_numeric(end_percentiles_df["end_return"], errors="coerce")
        st.markdown("**End-Horizon Percentiles**")
        st.dataframe(end_percentiles_df, hide_index=True, use_container_width=True)

    st.subheader("Move Percentile (Backtest Mode)")
    start_col, end_col = st.columns(2)
    default_spot = float(result.spot) if np.isfinite(result.spot) else 0.0
    start_price = float(
        start_col.number_input("Start Price", min_value=0.0, value=max(default_spot, 0.0), step=0.01)
    )
    end_price = float(
        end_col.number_input("End Price", min_value=0.0, value=max(default_spot, 0.0), step=0.01)
    )

    if start_price <= 0.0:
        st.warning("Start price must be greater than zero to compute move percentile.")
    else:
        realized_return = (end_price / start_price) - 1.0
        move_percentile = compute_end_return_percentile(pd.Series(result.end_returns), realized_return)
        percentile_cols = st.columns(3)
        percentile_cols[0].metric("Realized Return", value=_fmt_pct(realized_return))
        percentile_cols[1].metric(
            "Simulated Percentile",
            value="-" if move_percentile is None else f"{move_percentile:.1f}%",
        )
        percentile_cols[2].metric("Horizon Days", value=str(config.horizon_days))

    st.subheader("IV Context")
    derived_row, derived_note = load_latest_derived_row(active_symbol, database_path=database_arg)
    if derived_note:
        st.warning(f"IV context unavailable: {derived_note}")
    elif derived_row is None:
        st.info(
            "No derived_daily row found for this symbol. Run "
            f"`options-helper snapshot-options --symbol {active_symbol} ...` "
            "and `options-helper derived update --symbol {active_symbol}`."
        )
    else:
        low, high = _load_iv_thresholds()
        iv_rv = derived_row.get("iv_rv_20d")
        regime = classify_iv_rv(iv_rv, low=low, high=high)
        if regime is not None:
            st.metric("IV Regime", value=regime.label.upper())
            st.caption(f"{regime.reason} (low={regime.low:.2f}, high={regime.high:.2f})")

        iv_cols = st.columns(3)
        iv_cols[0].metric("ATM IV (Near)", value=_fmt_pct(derived_row.get("atm_iv_near")))
        iv_cols[1].metric("RV20", value=_fmt_pct(derived_row.get("rv_20d")))
        iv_cols[2].metric("RV60", value=_fmt_pct(derived_row.get("rv_60d")))

        iv_cols2 = st.columns(3)
        iv_cols2[0].metric("IV/RV20", value=_fmt_float(derived_row.get("iv_rv_20d"), digits=2))
        iv_cols2[1].metric(
            "IV Percentile",
            value=_fmt_pct_100(derived_row.get("atm_iv_near_percentile"), digits=0),
        )
        iv_cols2[2].metric("IV Term Slope", value=_fmt_float(derived_row.get("iv_term_slope"), digits=3))

    if result.warnings:
        st.markdown("**Model Warnings**")
        for warning in result.warnings:
            st.caption(f"- {warning}")
