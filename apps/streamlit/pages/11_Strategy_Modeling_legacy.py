from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

import options_helper.cli_deps as cli_deps
from apps.streamlit.components.strategy_modeling_page import (
    list_strategy_modeling_symbols,
    load_strategy_modeling_data_payload,
)
from options_helper.analysis.strategy_modeling import StrategyModelingRequest
from options_helper.analysis.strategy_simulator import build_r_target_ladder
from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts
from options_helper.data.strategy_modeling_profiles import (
    list_strategy_modeling_profiles,
    load_strategy_modeling_profile,
    save_strategy_modeling_profile,
)
from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig
from options_helper.schemas.strategy_modeling_profile import StrategyModelingProfile
from options_helper.schemas.strategy_modeling_policy import normalize_max_hold_timeframe

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
_VOLATILITY_REGIMES = ["low", "normal", "high"]
_RESULT_STATE_KEY = "strategy_modeling_last_result"
_DEFAULT_TICKERS = ("SPY", "AAPL", "AMZN", "NVDA")
_SEGMENT_LOW_COLOR = (254, 215, 206)
_SEGMENT_NEUTRAL_COLOR = (255, 251, 235)
_SEGMENT_HIGH_COLOR = (187, 247, 208)
_SEGMENT_TEXT_COLOR = "#111827"
_DEFAULT_EXPORT_REPORTS_DIR = "data/reports/technicals/strategy_modeling"
_DEFAULT_EXPORT_TIMEZONE = "America/Chicago"
_REQUEST_STATE_KEY = "strategy_modeling_last_request"
_PROFILE_PATH_KEY = "strategy_modeling_profile_path"
_PROFILE_SELECTED_KEY = "strategy_modeling_profile_selected"
_PROFILE_SAVE_NAME_KEY = "strategy_modeling_profile_save_name"
_PROFILE_OVERWRITE_KEY = "strategy_modeling_profile_overwrite"
_PROFILE_PENDING_SELECTED_KEY = "strategy_modeling_profile_pending_selected"
_PROFILE_PENDING_SAVE_NAME_KEY = "strategy_modeling_profile_pending_save_name"
_PROFILE_PENDING_LOAD_KEY = "strategy_modeling_profile_pending_load"
_PROFILE_FEEDBACK_KEY = "strategy_modeling_profile_feedback"
_IMPORT_SUMMARY_PATH_KEY = "strategy_modeling_import_summary_path"
_IMPORT_FEEDBACK_KEY = "strategy_modeling_import_feedback"
_TRADE_REVIEW_BEST_KEY = "strategy_modeling_trade_review_top_best"
_TRADE_REVIEW_WORST_KEY = "strategy_modeling_trade_review_top_worst"
_TRADE_REVIEW_LOG_KEY = "strategy_modeling_trade_review_full_log"
_TRADE_DRILLDOWN_TIMEFRAME_KEY = "strategy_modeling_trade_drilldown_timeframe"
_TRADE_DRILLDOWN_PRE_BARS_KEY = "strategy_modeling_trade_drilldown_pre_context_bars"
_TRADE_DRILLDOWN_POST_BARS_KEY = "strategy_modeling_trade_drilldown_post_context_bars"
_TRADE_DRILLDOWN_X_RANGE_KEY = "strategy_modeling_trade_drilldown_x_range"
_TRADE_DRILLDOWN_Y_RANGE_KEY = "strategy_modeling_trade_drilldown_y_range"

_WIDGET_KEYS: dict[str, str] = {
    "strategy": "strategy_modeling_strategy",
    "start_date": "strategy_modeling_start_date",
    "end_date": "strategy_modeling_end_date",
    "intraday_timeframe": "strategy_modeling_intraday_timeframe",
    "intraday_source": "strategy_modeling_intraday_source",
    "starting_capital": "strategy_modeling_starting_capital",
    "risk_per_trade_pct": "strategy_modeling_risk_per_trade_pct",
    "fib_retracement_pct": "strategy_modeling_fib_retracement_pct",
    "gap_fill_policy": "strategy_modeling_gap_fill_policy",
    "max_hold_enabled": "strategy_modeling_max_hold_enabled",
    "max_hold_bars": "strategy_modeling_max_hold_bars",
    "max_hold_timeframe": "strategy_modeling_max_hold_timeframe",
    "one_open_per_symbol": "strategy_modeling_one_open_per_symbol",
    "symbols": "strategy_modeling_symbols",
    "symbols_text": "strategy_modeling_symbols_text",
    "segment_dimensions": "strategy_modeling_segment_dimensions",
    "segment_values_text": "strategy_modeling_segment_values_text",
    "ma_fast_window": "strategy_modeling_ma_fast_window",
    "ma_slow_window": "strategy_modeling_ma_slow_window",
    "ma_trend_window": "strategy_modeling_ma_trend_window",
    "ma_fast_type": "strategy_modeling_ma_fast_type",
    "ma_slow_type": "strategy_modeling_ma_slow_type",
    "ma_trend_type": "strategy_modeling_ma_trend_type",
    "trend_slope_lookback_bars": "strategy_modeling_trend_slope_lookback_bars",
    "atr_window": "strategy_modeling_atr_window",
    "atr_stop_multiple": "strategy_modeling_atr_stop_multiple",
    "allow_shorts": "strategy_modeling_allow_shorts",
    "enable_orb_confirmation": "strategy_modeling_enable_orb_confirmation",
    "orb_range_minutes": "strategy_modeling_orb_range_minutes",
    "orb_confirmation_cutoff_et": "strategy_modeling_orb_confirmation_cutoff_et",
    "orb_stop_policy": "strategy_modeling_orb_stop_policy",
    "enable_atr_stop_floor": "strategy_modeling_enable_atr_stop_floor",
    "atr_stop_floor_multiple": "strategy_modeling_atr_stop_floor_multiple",
    "enable_rsi_extremes": "strategy_modeling_enable_rsi_extremes",
    "enable_ema9_regime": "strategy_modeling_enable_ema9_regime",
    "ema9_slope_lookback_bars": "strategy_modeling_ema9_slope_lookback_bars",
    "enable_volatility_regime": "strategy_modeling_enable_volatility_regime",
    "allowed_volatility_regimes": "strategy_modeling_allowed_volatility_regimes",
    "r_ladder_min_tenths": "strategy_modeling_r_ladder_min_tenths",
    "r_ladder_max_tenths": "strategy_modeling_r_ladder_max_tenths",
    "r_ladder_step_tenths": "strategy_modeling_r_ladder_step_tenths",
    "export_reports_dir": "strategy_modeling_export_reports_dir",
    "export_output_timezone": "strategy_modeling_export_output_timezone",
}



from apps.streamlit.pages.strategy_modeling_page_helpers_legacy import (
    _apply_pending_profile_updates,
    _as_symbol_filter,
    _build_export_request,
    _build_profile_from_inputs,
    _build_signal_kwargs,
    _coerce_float,
    _coerce_int,
    _ensure_sidebar_state_defaults,
    _filter_segments,
    _format_metric,
    _load_imported_strategy_modeling_summary,
    _normalize_volatility_regimes,
    _rows_to_df,
    _styled_segment_breakdown,
    _to_dict,
)
from apps.streamlit.pages.strategy_modeling_trade_log_legacy import render_trade_log_section

st.title("Strategy Modeling")
st.caption(DISCLAIMER_TEXT)
st.info(
    "Read-only strategy modeling dashboard. This page runs deterministic local analysis only "
    "(no ingestion writes from portal interactions)."
)

symbols, symbol_notes = list_strategy_modeling_symbols(database_path=None)
default_end = date.today()
default_start = default_end - timedelta(days=365)
_STRATEGY_OPTIONS = ["sfp", "msb", "orb", "ma_crossover", "trend_following", "fib_retracement"]
_INTRADAY_TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "60Min"]
_INTRADAY_SOURCES = ["stocks_bars_local", "alpaca"]
_MAX_HOLD_TIMEFRAME_OPTIONS = ["entry", "1Min", "5Min", "10Min", "15Min", "30Min", "60Min", "1H", "1D", "1W"]

_ensure_sidebar_state_defaults(default_start=default_start, default_end=default_end, symbols=symbols)

profile_store_path = Path(str(st.session_state.get(_PROFILE_PATH_KEY) or "").strip() or "config/strategy_modeling_profiles.json")
profile_list_error: str | None = None
available_profile_names: list[str] = []
try:
    available_profile_names = list_strategy_modeling_profiles(profile_store_path)
except ValueError as exc:
    profile_list_error = str(exc)
    available_profile_names = []
profile_options = available_profile_names if available_profile_names else [""]
_apply_pending_profile_updates(
    profile_options=profile_options,
    available_symbols=symbols,
    available_intraday_sources=_INTRADAY_SOURCES,
    available_timeframes=_INTRADAY_TIMEFRAMES,
)

with st.sidebar:
    st.markdown("### Profiles")
    profile_path_input = st.text_input(
        "Profile store path",
        value=str(st.session_state.get(_PROFILE_PATH_KEY, "config/strategy_modeling_profiles.json")),
        key=_PROFILE_PATH_KEY,
    )
    profile_store_path = Path(str(profile_path_input or "").strip() or "config/strategy_modeling_profiles.json")
    selected_profile_name = st.selectbox(
        "Saved profiles",
        options=profile_options,
        key=_PROFILE_SELECTED_KEY,
        format_func=lambda value: value if value else "(none)",
    )
    profile_save_name = st.text_input(
        "Profile name",
        value=str(st.session_state.get(_PROFILE_SAVE_NAME_KEY, "")),
        key=_PROFILE_SAVE_NAME_KEY,
    )
    overwrite_profile = st.checkbox(
        "Overwrite existing profile",
        value=bool(st.session_state.get(_PROFILE_OVERWRITE_KEY, False)),
        key=_PROFILE_OVERWRITE_KEY,
    )
    profile_load_clicked = st.button(
        "Load Profile",
        use_container_width=True,
        disabled=not bool(available_profile_names),
    )
    profile_save_clicked = st.button("Save Profile", use_container_width=True)

    st.markdown("### Import Results")
    import_summary_path = st.text_input(
        "Import summary path",
        value=str(st.session_state.get(_IMPORT_SUMMARY_PATH_KEY, "")),
        key=_IMPORT_SUMMARY_PATH_KEY,
        help="Path to strategy-modeling summary.json or a run directory containing summary.json.",
    )
    import_clicked = st.button("Import Results", use_container_width=True)

    st.markdown("### Modeling Inputs")
    strategy = st.selectbox("Strategy", options=_STRATEGY_OPTIONS, key=_WIDGET_KEYS["strategy"])
    start_date = st.date_input(
        "Start date",
        value=st.session_state.get(_WIDGET_KEYS["start_date"], default_start),
        key=_WIDGET_KEYS["start_date"],
    )
    end_date = st.date_input(
        "End date",
        value=st.session_state.get(_WIDGET_KEYS["end_date"], default_end),
        key=_WIDGET_KEYS["end_date"],
    )
    intraday_timeframe = st.selectbox(
        "Intraday timeframe",
        options=_INTRADAY_TIMEFRAMES,
        key=_WIDGET_KEYS["intraday_timeframe"],
    )
    intraday_source = st.selectbox(
        "Intraday source",
        options=_INTRADAY_SOURCES,
        key=_WIDGET_KEYS["intraday_source"],
        help="Current modeling service consumes persisted stock bars partitions.",
    )
    try:
        normalized_max_hold_timeframe = normalize_max_hold_timeframe(
            st.session_state.get(_WIDGET_KEYS["max_hold_timeframe"], "entry")
        )
    except ValueError:
        normalized_max_hold_timeframe = "entry"
    if normalized_max_hold_timeframe not in set(_MAX_HOLD_TIMEFRAME_OPTIONS):
        normalized_max_hold_timeframe = "entry"
    st.session_state[_WIDGET_KEYS["max_hold_timeframe"]] = normalized_max_hold_timeframe

    st.markdown("### Portfolio Policy")
    starting_capital = st.number_input(
        "Starting capital",
        min_value=1_000.0,
        value=float(st.session_state.get(_WIDGET_KEYS["starting_capital"], 10_000.0)),
        step=500.0,
        key=_WIDGET_KEYS["starting_capital"],
    )
    risk_pct = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=10.0,
        value=float(st.session_state.get(_WIDGET_KEYS["risk_per_trade_pct"], 1.0)),
        step=0.1,
        key=_WIDGET_KEYS["risk_per_trade_pct"],
    )
    gap_policy = st.selectbox(
        "Gap policy",
        options=["fill_at_open", "strict_touch"],
        key=_WIDGET_KEYS["gap_fill_policy"],
    )
    max_hold_enabled = st.checkbox(
        "Enable max hold time stop",
        value=bool(st.session_state.get(_WIDGET_KEYS["max_hold_enabled"], True)),
        key=_WIDGET_KEYS["max_hold_enabled"],
        help="When disabled, trades hold through the entry session close unless stop/target exits earlier.",
    )
    max_hold_timeframe = st.selectbox(
        "Max hold timeframe",
        options=_MAX_HOLD_TIMEFRAME_OPTIONS,
        key=_WIDGET_KEYS["max_hold_timeframe"],
        format_func=lambda value: "Entry timeframe" if str(value) == "entry" else str(value),
        help="Bar unit used by Max hold (bars). Example: 6 bars at 10Min = 60 minutes.",
    )
    max_hold_bars_value = int(
        st.number_input(
            "Max hold (bars)",
            min_value=1,
            max_value=500,
            value=int(st.session_state.get(_WIDGET_KEYS["max_hold_bars"], 20)),
            step=1,
            key=_WIDGET_KEYS["max_hold_bars"],
            disabled=not bool(max_hold_enabled),
        )
    )
    max_hold_bars: int | None = int(max_hold_bars_value) if max_hold_enabled else None
    one_open_per_symbol = st.checkbox(
        "One open trade per symbol",
        value=bool(st.session_state.get(_WIDGET_KEYS["one_open_per_symbol"], True)),
        key=_WIDGET_KEYS["one_open_per_symbol"],
    )
    r_ladder_min_tenths = int(
        st.number_input(
            "R ladder min (tenths)",
            min_value=1,
            max_value=200,
            value=int(st.session_state.get(_WIDGET_KEYS["r_ladder_min_tenths"], 10)),
            step=1,
            key=_WIDGET_KEYS["r_ladder_min_tenths"],
        )
    )
    r_ladder_max_tenths = int(
        st.number_input(
            "R ladder max (tenths)",
            min_value=1,
            max_value=400,
            value=int(st.session_state.get(_WIDGET_KEYS["r_ladder_max_tenths"], 20)),
            step=1,
            key=_WIDGET_KEYS["r_ladder_max_tenths"],
        )
    )
    r_ladder_step_tenths = int(
        st.number_input(
            "R ladder step (tenths)",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get(_WIDGET_KEYS["r_ladder_step_tenths"], 1)),
            step=1,
            key=_WIDGET_KEYS["r_ladder_step_tenths"],
        )
    )

    st.markdown("### Filters")
    if symbols:
        selected_symbols = st.multiselect(
            "Tickers",
            options=symbols,
            default=list(st.session_state.get(_WIDGET_KEYS["symbols"]) or []),
            key=_WIDGET_KEYS["symbols"],
        )
        symbol_filter_text = ""
    else:
        symbol_filter_text = st.text_input(
            "Tickers (comma-separated)",
            value=str(st.session_state.get(_WIDGET_KEYS["symbols_text"], "SPY")),
            key=_WIDGET_KEYS["symbols_text"],
        )
        selected_symbols = _as_symbol_filter(symbol_filter_text)

    segment_dimensions = st.multiselect(
        "Segment dimensions",
        options=_SEGMENT_DIMENSIONS,
        default=list(st.session_state.get(_WIDGET_KEYS["segment_dimensions"]) or []),
        key=_WIDGET_KEYS["segment_dimensions"],
    )
    segment_values_raw = st.text_input(
        "Segment values (comma-separated)",
        value=str(st.session_state.get(_WIDGET_KEYS["segment_values_text"], "")),
        key=_WIDGET_KEYS["segment_values_text"],
    )

    st.markdown("### Strategy Signal Parameters")
    fib_retracement_pct = float(st.session_state.get(_WIDGET_KEYS["fib_retracement_pct"], 61.8))
    if str(strategy) == "fib_retracement":
        fib_retracement_pct = float(
            st.number_input(
                "Fib retracement (%)",
                min_value=0.001,
                max_value=100.0,
                value=float(st.session_state.get(_WIDGET_KEYS["fib_retracement_pct"], 61.8)),
                step=0.1,
                key=_WIDGET_KEYS["fib_retracement_pct"],
            )
        )
    ma_fast_window = int(
        st.number_input(
            "MA fast window",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get(_WIDGET_KEYS["ma_fast_window"], 20)),
            step=1,
            key=_WIDGET_KEYS["ma_fast_window"],
        )
    )
    ma_slow_window = int(
        st.number_input(
            "MA slow window",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get(_WIDGET_KEYS["ma_slow_window"], 50)),
            step=1,
            key=_WIDGET_KEYS["ma_slow_window"],
        )
    )
    ma_trend_window = int(
        st.number_input(
            "MA trend window",
            min_value=1,
            max_value=2000,
            value=int(st.session_state.get(_WIDGET_KEYS["ma_trend_window"], 200)),
            step=1,
            key=_WIDGET_KEYS["ma_trend_window"],
        )
    )
    ma_fast_type = st.selectbox("MA fast type", options=["sma", "ema"], key=_WIDGET_KEYS["ma_fast_type"])
    ma_slow_type = st.selectbox("MA slow type", options=["sma", "ema"], key=_WIDGET_KEYS["ma_slow_type"])
    ma_trend_type = st.selectbox("MA trend type", options=["sma", "ema"], key=_WIDGET_KEYS["ma_trend_type"])
    trend_slope_lookback_bars = int(
        st.number_input(
            "Trend slope lookback (bars)",
            min_value=1,
            max_value=200,
            value=int(st.session_state.get(_WIDGET_KEYS["trend_slope_lookback_bars"], 3)),
            step=1,
            key=_WIDGET_KEYS["trend_slope_lookback_bars"],
        )
    )
    atr_window = int(
        st.number_input(
            "ATR window",
            min_value=1,
            max_value=500,
            value=int(st.session_state.get(_WIDGET_KEYS["atr_window"], 14)),
            step=1,
            key=_WIDGET_KEYS["atr_window"],
        )
    )
    atr_stop_multiple = float(
        st.number_input(
            "ATR stop multiple",
            min_value=0.1,
            max_value=20.0,
            value=float(st.session_state.get(_WIDGET_KEYS["atr_stop_multiple"], 2.0)),
            step=0.1,
            key=_WIDGET_KEYS["atr_stop_multiple"],
        )
    )

    st.markdown("### Entry Filters")
    allow_shorts = st.checkbox(
        "Allow short-direction entries",
        value=bool(st.session_state.get(_WIDGET_KEYS["allow_shorts"], True)),
        key=_WIDGET_KEYS["allow_shorts"],
    )
    enable_orb_confirmation = st.checkbox(
        "Enable ORB confirmation gate",
        value=bool(st.session_state.get(_WIDGET_KEYS["enable_orb_confirmation"], False)),
        key=_WIDGET_KEYS["enable_orb_confirmation"],
    )
    orb_range_minutes = int(
        st.number_input(
            "ORB range (minutes)",
            min_value=1,
            max_value=120,
            value=int(st.session_state.get(_WIDGET_KEYS["orb_range_minutes"], 15)),
            step=1,
            key=_WIDGET_KEYS["orb_range_minutes"],
        )
    )
    orb_confirmation_cutoff_et = st.text_input(
        "ORB confirmation cutoff ET (HH:MM)",
        value=str(st.session_state.get(_WIDGET_KEYS["orb_confirmation_cutoff_et"], "10:30")),
        key=_WIDGET_KEYS["orb_confirmation_cutoff_et"],
    )
    orb_stop_policy = st.selectbox(
        "ORB stop policy",
        options=["base", "orb_range", "tighten"],
        key=_WIDGET_KEYS["orb_stop_policy"],
    )
    enable_atr_stop_floor = st.checkbox(
        "Enable ATR stop floor",
        value=bool(st.session_state.get(_WIDGET_KEYS["enable_atr_stop_floor"], False)),
        key=_WIDGET_KEYS["enable_atr_stop_floor"],
    )
    atr_stop_floor_multiple = float(
        st.number_input(
            "ATR stop floor multiple",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get(_WIDGET_KEYS["atr_stop_floor_multiple"], 0.5)),
            step=0.1,
            key=_WIDGET_KEYS["atr_stop_floor_multiple"],
        )
    )
    enable_rsi_extremes = st.checkbox(
        "Enable RSI extremes gate",
        value=bool(st.session_state.get(_WIDGET_KEYS["enable_rsi_extremes"], False)),
        key=_WIDGET_KEYS["enable_rsi_extremes"],
    )
    enable_ema9_regime = st.checkbox(
        "Enable EMA9 regime gate",
        value=bool(st.session_state.get(_WIDGET_KEYS["enable_ema9_regime"], False)),
        key=_WIDGET_KEYS["enable_ema9_regime"],
    )
    ema9_slope_lookback_bars = int(
        st.number_input(
            "EMA9 slope lookback (bars)",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get(_WIDGET_KEYS["ema9_slope_lookback_bars"], 3)),
            step=1,
            key=_WIDGET_KEYS["ema9_slope_lookback_bars"],
        )
    )
    enable_volatility_regime = st.checkbox(
        "Enable volatility regime gate",
        value=bool(st.session_state.get(_WIDGET_KEYS["enable_volatility_regime"], False)),
        key=_WIDGET_KEYS["enable_volatility_regime"],
    )
    allowed_volatility_regimes = st.multiselect(
        "Allowed volatility regimes",
        options=_VOLATILITY_REGIMES,
        default=list(st.session_state.get(_WIDGET_KEYS["allowed_volatility_regimes"]) or []),
        key=_WIDGET_KEYS["allowed_volatility_regimes"],
    )
    st.markdown("### Export")
    export_reports_dir = st.text_input(
        "Export reports dir",
        value=str(st.session_state.get(_WIDGET_KEYS["export_reports_dir"], _DEFAULT_EXPORT_REPORTS_DIR)),
        key=_WIDGET_KEYS["export_reports_dir"],
    )
    export_output_timezone = st.text_input(
        "Export output timezone",
        value=str(st.session_state.get(_WIDGET_KEYS["export_output_timezone"], _DEFAULT_EXPORT_TIMEZONE)),
        key=_WIDGET_KEYS["export_output_timezone"],
    )

if profile_list_error:
    st.error(f"Profiles: {profile_list_error}")
profile_feedback = str(st.session_state.pop(_PROFILE_FEEDBACK_KEY, "") or "").strip()
if profile_feedback:
    st.success(profile_feedback)
import_feedback = str(st.session_state.pop(_IMPORT_FEEDBACK_KEY, "") or "").strip()
if import_feedback:
    st.success(import_feedback)

if profile_load_clicked:
    if not selected_profile_name:
        st.error("Choose a saved profile to load.")
    else:
        try:
            loaded_profile = load_strategy_modeling_profile(profile_store_path, selected_profile_name)
        except ValueError as exc:
            st.error(f"Failed to load profile: {exc}")
        else:
            st.session_state[_PROFILE_PENDING_LOAD_KEY] = loaded_profile.model_dump(mode="python")
            st.session_state[_PROFILE_PENDING_SELECTED_KEY] = selected_profile_name
            st.session_state[_PROFILE_PENDING_SAVE_NAME_KEY] = selected_profile_name
            st.session_state[_PROFILE_FEEDBACK_KEY] = f"Loaded profile '{selected_profile_name}'"
            st.rerun()

if import_clicked:
    try:
        imported_result, imported_request, imported_path = _load_imported_strategy_modeling_summary(
            import_summary_path
        )
    except ValueError as exc:
        st.error(f"Import failed: {exc}")
    else:
        st.session_state[_RESULT_STATE_KEY] = imported_result
        st.session_state[_REQUEST_STATE_KEY] = imported_request
        st.session_state[_IMPORT_FEEDBACK_KEY] = f"Imported results from {imported_path}"
        st.rerun()

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

filter_config_payload = {
    "allow_shorts": bool(allow_shorts),
    "enable_orb_confirmation": bool(enable_orb_confirmation),
    "orb_range_minutes": int(orb_range_minutes),
    "orb_confirmation_cutoff_et": str(orb_confirmation_cutoff_et),
    "orb_stop_policy": str(orb_stop_policy),
    "enable_atr_stop_floor": bool(enable_atr_stop_floor),
    "atr_stop_floor_multiple": float(atr_stop_floor_multiple),
    "enable_rsi_extremes": bool(enable_rsi_extremes),
    "enable_ema9_regime": bool(enable_ema9_regime),
    "ema9_slope_lookback_bars": int(ema9_slope_lookback_bars),
    "enable_volatility_regime": bool(enable_volatility_regime),
    "allowed_volatility_regimes": _normalize_volatility_regimes(allowed_volatility_regimes),
}
filter_config: StrategyEntryFilterConfig | None = None
filter_config_error: str | None = None
try:
    filter_config = StrategyEntryFilterConfig.model_validate(filter_config_payload)
except ValidationError as exc:
    detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
    filter_config_error = f"Invalid entry filter configuration: {detail}"

signal_kwargs: dict[str, object] = {}
signal_kwargs_error: str | None = None
try:
    signal_kwargs = _build_signal_kwargs(
        strategy=str(strategy),
        ma_fast_window=int(ma_fast_window),
        ma_slow_window=int(ma_slow_window),
        ma_trend_window=int(ma_trend_window),
        ma_fast_type=str(ma_fast_type),
        ma_slow_type=str(ma_slow_type),
        ma_trend_type=str(ma_trend_type),
        trend_slope_lookback_bars=int(trend_slope_lookback_bars),
        atr_window=int(atr_window),
        atr_stop_multiple=float(atr_stop_multiple),
        fib_retracement_pct=float(fib_retracement_pct),
    )
except ValueError as exc:
    signal_kwargs_error = f"Invalid strategy signal parameters: {exc}"

target_ladder = ()
target_ladder_error: str | None = None
try:
    target_ladder = build_r_target_ladder(
        min_target_tenths=int(r_ladder_min_tenths),
        max_target_tenths=int(r_ladder_max_tenths),
        step_tenths=int(r_ladder_step_tenths),
    )
except ValueError as exc:
    target_ladder_error = f"Invalid R-ladder configuration: {exc}"

profile_model: StrategyModelingProfile | None = None
profile_validation_error: str | None = None
try:
    profile_model = _build_profile_from_inputs(
        strategy=str(strategy),
        symbols=tuple(selected_symbols),
        start_date=start_date,
        end_date=end_date,
        intraday_timeframe=str(intraday_timeframe),
        intraday_source=str(intraday_source),
        starting_capital=float(starting_capital),
        risk_per_trade_pct=float(risk_pct),
        fib_retracement_pct=float(fib_retracement_pct),
        gap_fill_policy=str(gap_policy),
        max_hold_bars=max_hold_bars,
        max_hold_timeframe=str(max_hold_timeframe),
        one_open_per_symbol=bool(one_open_per_symbol),
        r_ladder_min_tenths=int(r_ladder_min_tenths),
        r_ladder_max_tenths=int(r_ladder_max_tenths),
        r_ladder_step_tenths=int(r_ladder_step_tenths),
        allow_shorts=bool(allow_shorts),
        enable_orb_confirmation=bool(enable_orb_confirmation),
        orb_range_minutes=int(orb_range_minutes),
        orb_confirmation_cutoff_et=str(orb_confirmation_cutoff_et),
        orb_stop_policy=str(orb_stop_policy),
        enable_atr_stop_floor=bool(enable_atr_stop_floor),
        atr_stop_floor_multiple=float(atr_stop_floor_multiple),
        enable_rsi_extremes=bool(enable_rsi_extremes),
        enable_ema9_regime=bool(enable_ema9_regime),
        ema9_slope_lookback_bars=int(ema9_slope_lookback_bars),
        enable_volatility_regime=bool(enable_volatility_regime),
        allowed_volatility_regimes=tuple(allowed_volatility_regimes),
        ma_fast_window=int(ma_fast_window),
        ma_slow_window=int(ma_slow_window),
        ma_trend_window=int(ma_trend_window),
        ma_fast_type=str(ma_fast_type),
        ma_slow_type=str(ma_slow_type),
        ma_trend_type=str(ma_trend_type),
        trend_slope_lookback_bars=int(trend_slope_lookback_bars),
        atr_window=int(atr_window),
        atr_stop_multiple=float(atr_stop_multiple),
    )
except ValidationError as exc:
    detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
    profile_validation_error = f"Invalid profile values: {detail}"

if profile_save_clicked:
    save_name = str(profile_save_name or "").strip()
    if not save_name:
        st.error("Enter a profile name before saving.")
    elif profile_model is None:
        st.error(profile_validation_error or "Current inputs are invalid and cannot be saved as a profile.")
    else:
        try:
            save_strategy_modeling_profile(
                profile_store_path,
                save_name,
                profile_model,
                overwrite=bool(overwrite_profile),
            )
        except ValueError as exc:
            st.error(f"Failed to save profile: {exc}")
        else:
            st.session_state[_PROFILE_PENDING_SELECTED_KEY] = save_name
            st.session_state[_PROFILE_PENDING_SAVE_NAME_KEY] = save_name
            st.session_state[_PROFILE_FEEDBACK_KEY] = f"Saved profile '{save_name}'"
            st.rerun()

if filter_config_error:
    st.error(filter_config_error)
if signal_kwargs_error:
    st.error(signal_kwargs_error)
if target_ladder_error:
    st.error(target_ladder_error)
if profile_validation_error:
    st.error(profile_validation_error)

run_disabled = (
    run_is_blocked
    or filter_config is None
    or signal_kwargs_error is not None
    or target_ladder_error is not None
    or profile_validation_error is not None
)
run_help = "Runs local deterministic strategy modeling. Disabled when required intraday coverage is missing."
if filter_config is None:
    run_help = f"{run_help} Fix invalid entry-filter settings first."
if signal_kwargs_error is not None:
    run_help = f"{run_help} Fix invalid strategy signal parameters first."
if target_ladder_error is not None:
    run_help = f"{run_help} Fix invalid R-ladder settings first."
if profile_validation_error is not None:
    run_help = f"{run_help} Fix invalid profile/run input values first."

run_clicked = st.button(
    "Run Strategy Modeling",
    type="primary",
    use_container_width=True,
    disabled=run_disabled,
    help=run_help,
)
clear_clicked = st.button("Clear Results", use_container_width=True)

if clear_clicked:
    st.session_state.pop(_RESULT_STATE_KEY, None)
    st.session_state.pop(_REQUEST_STATE_KEY, None)

if (
    run_clicked
    and not run_is_blocked
    and filter_config is not None
    and signal_kwargs_error is None
    and target_ladder_error is None
    and profile_validation_error is None
):
    service = cli_deps.build_strategy_modeling_service()
    request = StrategyModelingRequest(
        strategy=strategy,
        symbols=tuple(selected_symbols),
        start_date=start_date,
        end_date=end_date,
        intraday_dir=Path("data/intraday"),
        intraday_timeframe=intraday_timeframe,
        target_ladder=target_ladder,
        signal_kwargs=signal_kwargs,
        starting_capital=float(starting_capital),
        max_hold_bars=max_hold_bars,
        filter_config=filter_config,
        policy={
            "require_intraday_bars": True,
            "risk_per_trade_pct": float(risk_pct),
            "gap_fill_policy": gap_policy,
            "max_hold_bars": max_hold_bars,
            "max_hold_timeframe": str(max_hold_timeframe),
            "one_open_per_symbol": bool(one_open_per_symbol),
        },
        block_on_missing_intraday_coverage=True,
    )
    st.session_state[_RESULT_STATE_KEY] = service.run(request)
    st.session_state[_REQUEST_STATE_KEY] = request

result = st.session_state.get(_RESULT_STATE_KEY)
request_state = st.session_state.get(_REQUEST_STATE_KEY)

export_clicked = st.button(
    "Export Reports",
    use_container_width=True,
    disabled=result is None,
    help=(
        "Write summary.json/summary.md/trades.csv/r_ladder.csv/segments.csv/"
        "top_20_best_trades.csv/top_20_worst_trades.csv for the latest run."
    ),
)
if export_clicked:
    if result is None:
        st.error("Run modeling first, then export reports.")
    elif request_state is None:
        st.error("Request context is missing. Run modeling again before exporting.")
    else:
        strategy_label = str(getattr(request_state, "strategy", "")).strip().lower()
        if not strategy_label:
            st.error("Strategy value is missing from request context; rerun modeling before exporting.")
        else:
            export_request = _build_export_request(
                request_state,
                intraday_source=intraday_source,
                gap_fill_policy=gap_policy,
                output_timezone=export_output_timezone,
            )
            paths = write_strategy_modeling_artifacts(
                out_dir=Path(export_reports_dir),
                strategy=strategy_label,
                request=export_request,
                run_result=result,
                write_json=True,
                write_csv=True,
                write_md=True,
            )
            st.success(f"Exported reports: {paths.run_dir}")

metrics_payload = _to_dict(getattr(result, "portfolio_metrics", None))
r_ladder_df = _rows_to_df(getattr(result, "target_hit_rates", None))
equity_df = _rows_to_df(getattr(result, "equity_curve", None))
segment_df = _rows_to_df(getattr(result, "segment_records", None))
trade_df = _rows_to_df(getattr(result, "trade_simulations", None))
filter_summary_payload = _to_dict(getattr(result, "filter_summary", None))
directional_payload = _to_dict(getattr(result, "directional_metrics", None))

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

st.subheader("Filter Summary")
if not filter_summary_payload:
    st.info("No filter summary returned for this run.")
else:
    summary_rows = []
    for field in ("base_event_count", "kept_event_count", "rejected_event_count"):
        parsed = _coerce_int(filter_summary_payload.get(field))
        if parsed is None:
            continue
        summary_rows.append({"metric": field, "value": parsed})
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    reject_counts = _to_dict(filter_summary_payload.get("reject_counts"))
    reject_rows = []
    for reason in sorted(reject_counts):
        count = _coerce_int(reject_counts.get(reason))
        if count is None or count <= 0:
            continue
        reject_rows.append({"reject_reason": reason, "count": count})
    if reject_rows:
        st.dataframe(pd.DataFrame(reject_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No non-zero filter reject counts.")

st.subheader("Directional Results")
if not directional_payload:
    st.info("No directional results returned for this run.")
else:
    directional_rows: list[dict[str, Any]] = []
    for label in ("combined", "long_only", "short_only"):
        bucket = _to_dict(directional_payload.get(label))
        if not bucket:
            continue
        trade_count = _coerce_int(bucket.get("trade_count"))
        total_return_pct = _coerce_float(bucket.get("total_return_pct"))
        if total_return_pct is None:
            total_return_pct = _coerce_float(_to_dict(bucket.get("portfolio_metrics")).get("total_return_pct"))
        directional_rows.append(
            {
                "directional_bucket": label,
                "trade_count": trade_count,
                "total_return_pct": total_return_pct,
            }
        )

    if directional_rows:
        st.dataframe(pd.DataFrame(directional_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Directional payload is present but empty.")

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
    st.dataframe(_styled_segment_breakdown(segment_df), hide_index=True, use_container_width=True)

render_trade_log_section(
    trade_df=trade_df,
    result=result,
    request_state=request_state,
    intraday_timeframe=str(intraday_timeframe),
    trade_review_best_key=_TRADE_REVIEW_BEST_KEY,
    trade_review_worst_key=_TRADE_REVIEW_WORST_KEY,
    trade_review_log_key=_TRADE_REVIEW_LOG_KEY,
    trade_drilldown_timeframe_key=_TRADE_DRILLDOWN_TIMEFRAME_KEY,
    trade_drilldown_pre_bars_key=_TRADE_DRILLDOWN_PRE_BARS_KEY,
    trade_drilldown_post_bars_key=_TRADE_DRILLDOWN_POST_BARS_KEY,
    trade_drilldown_x_range_key=_TRADE_DRILLDOWN_X_RANGE_KEY,
    trade_drilldown_y_range_key=_TRADE_DRILLDOWN_Y_RANGE_KEY,
)

st.caption(
    f"Intraday source: `{intraday_source}`. Page behavior is read-only and does not execute ingest/backfill writes."
)
