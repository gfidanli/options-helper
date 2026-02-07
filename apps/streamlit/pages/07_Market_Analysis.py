from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.market_analysis_page import (
    compute_end_return_percentile,
    compute_tail_risk_for_symbol,
    list_candle_symbols,
    load_candles_close,
    load_latest_derived_row,
    normalize_symbol,
)
from apps.streamlit.components.research_metrics_page import (
    build_exposure_top_strikes,
    build_iv_delta_bucket_sparkline,
    build_iv_tenor_sparkline,
    build_latest_iv_delta_table,
    build_latest_iv_tenor_table,
    compute_exposure_flip_strike,
    load_exposure_by_strike_latest,
    load_intraday_flow_summary,
    load_intraday_flow_top_contracts,
    load_intraday_flow_top_strikes,
    load_iv_surface_delta_history,
    load_iv_surface_tenor_history,
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


def _fmt_number(value: object, *, digits: int = 0) -> str:
    try:
        if value is None:
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number:,.{digits}f}"


def _fmt_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "-"
    return parsed.date().isoformat()


def _render_exposure_chart(exposure_df: pd.DataFrame, *, flip_strike: float | None) -> None:
    chart_rows = exposure_df.copy()
    chart_rows["strike"] = pd.to_numeric(chart_rows["strike"], errors="coerce")
    chart_rows["net_gex"] = pd.to_numeric(chart_rows["net_gex"], errors="coerce")
    chart_rows = chart_rows.dropna(subset=["strike", "net_gex"])
    chart_rows = chart_rows.sort_values(by="strike", kind="stable")
    if chart_rows.empty:
        st.info("Exposure chart unavailable for selected symbol.")
        return

    try:
        import altair as alt
    except Exception:  # noqa: BLE001
        st.bar_chart(
            chart_rows.set_index("strike")[["net_gex"]],
            use_container_width=True,
        )
        if flip_strike is not None:
            st.caption(f"Flip marker (cumulative net GEX): {flip_strike:,.2f}")
        return

    bars = alt.Chart(chart_rows).mark_bar().encode(
        x=alt.X("strike:Q", title="Strike"),
        y=alt.Y("net_gex:Q", title="Net GEX"),
        color=alt.condition(
            alt.datum.net_gex >= 0,
            alt.value("#2ca02c"),
            alt.value("#d62728"),
        ),
        tooltip=[
            alt.Tooltip("strike:Q", format=",.2f"),
            alt.Tooltip("net_gex:Q", format=",.2f"),
        ],
    )

    layers: list[object] = [bars]
    if flip_strike is not None:
        flip_df = pd.DataFrame([{"flip_strike": float(flip_strike)}])
        marker = alt.Chart(flip_df).mark_rule(color="#1f77b4", strokeDash=[8, 4], size=2).encode(
            x=alt.X("flip_strike:Q", title="Strike")
        )
        layers.append(marker)

    st.altair_chart(
        alt.layer(*layers).properties(height=320),
        use_container_width=True,
    )


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
    st.markdown("### Research Panels")
    research_lookback_days = st.slider(
        "Research lookback (days)",
        min_value=14,
        max_value=365,
        value=90,
        step=7,
    )
    intraday_top_n = st.slider("Intraday top rows", min_value=5, max_value=50, value=15, step=5)

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

st.subheader("Persisted Research Metrics")
st.caption("Read-only DuckDB panels. Informational only; no ingestion or writes are triggered from this page.")
tabs = st.tabs(
    [
        "IV Surface",
        "Exposure by Strike",
        "Intraday Flow",
    ]
)

with tabs[0]:
    tenor_df, tenor_note = load_iv_surface_tenor_history(
        active_symbol,
        limit_dates=research_lookback_days,
        database_path=database_arg,
    )
    if tenor_note:
        st.warning(f"IV tenor surface unavailable: {tenor_note}")

    delta_df, delta_note = load_iv_surface_delta_history(
        active_symbol,
        limit_dates=research_lookback_days,
        database_path=database_arg,
    )
    if delta_note:
        st.warning(f"IV delta-bucket surface unavailable: {delta_note}")

    latest_tenor = build_latest_iv_tenor_table(tenor_df)
    latest_delta = build_latest_iv_delta_table(delta_df, top_n=60)

    if latest_tenor.empty and latest_delta.empty:
        st.info(f"No persisted IV surface rows found for {active_symbol}.")
    else:
        latest_as_of = max(
            pd.to_datetime(latest_tenor.get("as_of"), errors="coerce").max()
            if not latest_tenor.empty
            else pd.NaT,
            pd.to_datetime(latest_delta.get("as_of"), errors="coerce").max()
            if not latest_delta.empty
            else pd.NaT,
        )
        iv_metrics = st.columns(3)
        iv_metrics[0].metric("Latest As-of", value=_fmt_date(latest_as_of))
        iv_metrics[1].metric("Tenor Rows", value=f"{len(latest_tenor)}")
        iv_metrics[2].metric("Delta Rows", value=f"{len(latest_delta)}")

        tenor_spark = build_iv_tenor_sparkline(tenor_df)
        st.markdown("**ATM IV by Tenor (Sparkline)**")
        if tenor_spark.empty:
            st.info("No tenor sparkline data available.")
        else:
            st.line_chart(tenor_spark, use_container_width=True)

        delta_spark = build_iv_delta_bucket_sparkline(delta_df, max_series=8)
        st.markdown("**Delta Buckets (Sparkline, Top Series by Contracts)**")
        if delta_spark.empty:
            st.info("No delta-bucket sparkline data available.")
        else:
            st.line_chart(delta_spark, use_container_width=True)

        if not latest_tenor.empty:
            st.markdown("**Latest Tenor Surface Rows**")
            latest_tenor_view = latest_tenor.copy()
            latest_tenor_view["as_of"] = pd.to_datetime(latest_tenor_view["as_of"], errors="coerce").dt.date.astype(str)
            latest_tenor_view["updated_at"] = (
                pd.to_datetime(latest_tenor_view["updated_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            st.dataframe(latest_tenor_view, hide_index=True, use_container_width=True)

        if not latest_delta.empty:
            st.markdown("**Latest Delta-Bucket Rows**")
            latest_delta_view = latest_delta.copy()
            latest_delta_view["as_of"] = pd.to_datetime(latest_delta_view["as_of"], errors="coerce").dt.date.astype(str)
            latest_delta_view["updated_at"] = (
                pd.to_datetime(latest_delta_view["updated_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            st.dataframe(latest_delta_view, hide_index=True, use_container_width=True)

with tabs[1]:
    exposure_df, exposure_note = load_exposure_by_strike_latest(
        active_symbol,
        database_path=database_arg,
    )
    if exposure_note:
        st.warning(f"Exposure data unavailable: {exposure_note}")

    if exposure_df.empty:
        st.info(f"No persisted exposure rows found for {active_symbol}.")
    else:
        flip_strike = compute_exposure_flip_strike(exposure_df)
        latest_as_of = pd.to_datetime(exposure_df["as_of"], errors="coerce").max()
        top_exposure = build_exposure_top_strikes(exposure_df, top_n=20)

        exposure_metrics = st.columns(4)
        exposure_metrics[0].metric("Latest As-of", value=_fmt_date(latest_as_of))
        exposure_metrics[1].metric("Strikes", value=f"{len(exposure_df)}")
        exposure_metrics[2].metric("Net GEX", value=_fmt_number(exposure_df["net_gex"].sum(), digits=0))
        exposure_metrics[3].metric(
            "Flip Marker",
            value="-" if flip_strike is None else f"{float(flip_strike):,.2f}",
        )

        st.markdown("**Net GEX by Strike**")
        _render_exposure_chart(exposure_df, flip_strike=flip_strike)
        if flip_strike is None:
            st.caption("No cumulative net-GEX zero-crossing found for the latest snapshot.")

        if not top_exposure.empty:
            st.markdown("**Top Strikes by |Net GEX|**")
            top_view = top_exposure.copy()
            top_view["as_of"] = pd.to_datetime(top_view["as_of"], errors="coerce").dt.date.astype(str)
            st.dataframe(top_view, hide_index=True, use_container_width=True)

with tabs[2]:
    intraday_summary, summary_note = load_intraday_flow_summary(
        active_symbol,
        database_path=database_arg,
    )
    if summary_note:
        st.warning(f"Intraday flow summary unavailable: {summary_note}")

    top_strikes_df, top_strikes_note = load_intraday_flow_top_strikes(
        active_symbol,
        top_n=intraday_top_n,
        database_path=database_arg,
    )
    if top_strikes_note:
        st.warning(f"Intraday top strikes unavailable: {top_strikes_note}")

    top_contracts_df, top_contracts_note = load_intraday_flow_top_contracts(
        active_symbol,
        top_n=intraday_top_n,
        database_path=database_arg,
    )
    if top_contracts_note:
        st.warning(f"Intraday top contracts unavailable: {top_contracts_note}")

    if intraday_summary is None and top_strikes_df.empty and top_contracts_df.empty:
        st.info(f"No persisted intraday flow rows found for {active_symbol}.")
    else:
        if intraday_summary is not None:
            summary_cols = st.columns(4)
            summary_cols[0].metric("Market Date", value=str(intraday_summary.get("market_date") or "-"))
            summary_cols[1].metric("Contracts", value=str(intraday_summary.get("contracts") or 0))
            summary_cols[2].metric("Trade Count", value=str(intraday_summary.get("trade_count") or 0))
            summary_cols[3].metric("Net Notional", value=_fmt_currency(intraday_summary.get("net_notional")))

            summary_cols2 = st.columns(4)
            summary_cols2[0].metric("Buy Notional", value=_fmt_currency(intraday_summary.get("buy_notional")))
            summary_cols2[1].metric("Sell Notional", value=_fmt_currency(intraday_summary.get("sell_notional")))
            summary_cols2[2].metric(
                "Unknown Trade Share",
                value=_fmt_pct(intraday_summary.get("avg_unknown_trade_share")),
            )
            summary_cols2[3].metric(
                "Quote Coverage",
                value=_fmt_pct(intraday_summary.get("avg_quote_coverage_pct")),
            )

        if not top_strikes_df.empty:
            st.markdown("**Top Strikes by |Net Notional|**")
            strike_chart = top_strikes_df.copy()
            strike_chart["label"] = strike_chart.apply(
                lambda row: (
                    f"{_fmt_number(row.get('strike'), digits=2)} "
                    f"{str(row.get('option_type') or '').upper()} "
                    f"{str(row.get('delta_bucket') or '')}"
                ).strip(),
                axis=1,
            )
            st.bar_chart(
                strike_chart.set_index("label")[["net_notional"]],
                use_container_width=True,
            )
            strike_view = top_strikes_df.copy()
            strike_view["market_date"] = pd.to_datetime(strike_view["market_date"], errors="coerce").dt.date.astype(str)
            st.dataframe(strike_view, hide_index=True, use_container_width=True)

        if not top_contracts_df.empty:
            st.markdown("**Top Contracts by |Net Notional|**")
            contract_view = top_contracts_df.copy()
            contract_view["market_date"] = pd.to_datetime(
                contract_view["market_date"],
                errors="coerce",
            ).dt.date.astype(str)
            contract_view["expiry"] = pd.to_datetime(contract_view["expiry"], errors="coerce").dt.date.astype(str)
            st.dataframe(contract_view, hide_index=True, use_container_width=True)
