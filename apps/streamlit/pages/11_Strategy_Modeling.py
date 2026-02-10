from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
from pandas.io.formats.style import Styler
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

_WIDGET_KEYS: dict[str, str] = {
    "strategy": "strategy_modeling_strategy",
    "start_date": "strategy_modeling_start_date",
    "end_date": "strategy_modeling_end_date",
    "intraday_timeframe": "strategy_modeling_intraday_timeframe",
    "intraday_source": "strategy_modeling_intraday_source",
    "starting_capital": "strategy_modeling_starting_capital",
    "risk_per_trade_pct": "strategy_modeling_risk_per_trade_pct",
    "gap_fill_policy": "strategy_modeling_gap_fill_policy",
    "max_hold_bars": "strategy_modeling_max_hold_bars",
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
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except TypeError:
            return {}
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


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_volatility_regimes(raw_values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _normalize_ma_type(value: str, *, field_name: str) -> str:
    token = str(value or "").strip().lower()
    if token not in {"sma", "ema"}:
        raise ValueError(f"{field_name} must be one of: sma, ema.")
    return token


def _build_signal_kwargs(
    *,
    strategy: str,
    ma_fast_window: int,
    ma_slow_window: int,
    ma_trend_window: int,
    ma_fast_type: str,
    ma_slow_type: str,
    ma_trend_type: str,
    trend_slope_lookback_bars: int,
    atr_window: int,
    atr_stop_multiple: float,
) -> dict[str, object]:
    if ma_fast_window < 1:
        raise ValueError("MA fast window must be >= 1.")
    if ma_slow_window < 1:
        raise ValueError("MA slow window must be >= 1.")
    if ma_trend_window < 1:
        raise ValueError("MA trend window must be >= 1.")
    if trend_slope_lookback_bars < 1:
        raise ValueError("Trend slope lookback bars must be >= 1.")
    if atr_window < 1:
        raise ValueError("ATR window must be >= 1.")
    if atr_stop_multiple <= 0.0:
        raise ValueError("ATR stop multiple must be > 0.")

    fast_type = _normalize_ma_type(ma_fast_type, field_name="MA fast type")
    slow_type = _normalize_ma_type(ma_slow_type, field_name="MA slow type")
    trend_type = _normalize_ma_type(ma_trend_type, field_name="MA trend type")

    if strategy == "ma_crossover":
        if ma_fast_window >= ma_slow_window:
            raise ValueError("For ma_crossover, MA fast window must be smaller than MA slow window.")
        return {
            "fast_window": int(ma_fast_window),
            "slow_window": int(ma_slow_window),
            "fast_type": fast_type,
            "slow_type": slow_type,
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    if strategy == "trend_following":
        return {
            "trend_window": int(ma_trend_window),
            "trend_type": trend_type,
            "fast_window": int(ma_fast_window),
            "fast_type": fast_type,
            "slope_lookback_bars": int(trend_slope_lookback_bars),
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    return {}


def _default_ticker_selection(symbols: Sequence[str]) -> list[str]:
    preferred = [ticker for ticker in _DEFAULT_TICKERS if ticker in symbols]
    if preferred:
        return preferred
    return list(symbols[: min(8, len(symbols))])


def _build_export_request(
    request: object,
    *,
    intraday_source: str,
    gap_fill_policy: str,
    output_timezone: str,
) -> SimpleNamespace:
    payload = _to_dict(request)
    payload["intraday_source"] = str(intraday_source)
    payload["gap_fill_policy"] = str(gap_fill_policy)
    payload["intra_bar_tie_break_rule"] = "stop_first"
    payload["output_timezone"] = str(output_timezone).strip() or _DEFAULT_EXPORT_TIMEZONE
    return SimpleNamespace(**payload)


def _ensure_sidebar_state_defaults(
    *,
    default_start: date,
    default_end: date,
    symbols: Sequence[str],
) -> None:
    defaults: dict[str, object] = {
        _PROFILE_PATH_KEY: "config/strategy_modeling_profiles.json",
        _PROFILE_SELECTED_KEY: "",
        _PROFILE_SAVE_NAME_KEY: "",
        _PROFILE_OVERWRITE_KEY: False,
        _WIDGET_KEYS["strategy"]: "sfp",
        _WIDGET_KEYS["start_date"]: default_start,
        _WIDGET_KEYS["end_date"]: default_end,
        _WIDGET_KEYS["intraday_timeframe"]: "1Min",
        _WIDGET_KEYS["intraday_source"]: "stocks_bars_local",
        _WIDGET_KEYS["starting_capital"]: 10_000.0,
        _WIDGET_KEYS["risk_per_trade_pct"]: 1.0,
        _WIDGET_KEYS["gap_fill_policy"]: "fill_at_open",
        _WIDGET_KEYS["max_hold_bars"]: 20,
        _WIDGET_KEYS["one_open_per_symbol"]: True,
        _WIDGET_KEYS["symbols_text"]: "SPY",
        _WIDGET_KEYS["segment_dimensions"]: ["symbol", "direction"],
        _WIDGET_KEYS["segment_values_text"]: "",
        _WIDGET_KEYS["ma_fast_window"]: 20,
        _WIDGET_KEYS["ma_slow_window"]: 50,
        _WIDGET_KEYS["ma_trend_window"]: 200,
        _WIDGET_KEYS["ma_fast_type"]: "sma",
        _WIDGET_KEYS["ma_slow_type"]: "sma",
        _WIDGET_KEYS["ma_trend_type"]: "sma",
        _WIDGET_KEYS["trend_slope_lookback_bars"]: 3,
        _WIDGET_KEYS["atr_window"]: 14,
        _WIDGET_KEYS["atr_stop_multiple"]: 2.0,
        _WIDGET_KEYS["allow_shorts"]: True,
        _WIDGET_KEYS["enable_orb_confirmation"]: False,
        _WIDGET_KEYS["orb_range_minutes"]: 15,
        _WIDGET_KEYS["orb_confirmation_cutoff_et"]: "10:30",
        _WIDGET_KEYS["orb_stop_policy"]: "base",
        _WIDGET_KEYS["enable_atr_stop_floor"]: False,
        _WIDGET_KEYS["atr_stop_floor_multiple"]: 0.5,
        _WIDGET_KEYS["enable_rsi_extremes"]: False,
        _WIDGET_KEYS["enable_ema9_regime"]: False,
        _WIDGET_KEYS["ema9_slope_lookback_bars"]: 3,
        _WIDGET_KEYS["enable_volatility_regime"]: False,
        _WIDGET_KEYS["allowed_volatility_regimes"]: list(_VOLATILITY_REGIMES),
        _WIDGET_KEYS["r_ladder_min_tenths"]: 10,
        _WIDGET_KEYS["r_ladder_max_tenths"]: 20,
        _WIDGET_KEYS["r_ladder_step_tenths"]: 1,
        _WIDGET_KEYS["export_reports_dir"]: _DEFAULT_EXPORT_REPORTS_DIR,
        _WIDGET_KEYS["export_output_timezone"]: _DEFAULT_EXPORT_TIMEZONE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if symbols:
        selected = st.session_state.get(_WIDGET_KEYS["symbols"])
        selected_values = _sanitize_symbol_selection(selected, options=symbols)
        if not selected_values:
            selected_values = _default_ticker_selection(symbols)
        st.session_state[_WIDGET_KEYS["symbols"]] = selected_values
    else:
        st.session_state.setdefault(_WIDGET_KEYS["symbols"], [])


def _sanitize_symbol_selection(raw_values: object, *, options: Sequence[str]) -> list[str]:
    available = {str(item).upper() for item in options}
    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
        values = list(raw_values)
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().upper()
        if not token or token in seen or token not in available:
            continue
        seen.add(token)
        out.append(token)
    return out


def _build_profile_from_inputs(
    *,
    strategy: str,
    symbols: Sequence[str],
    start_date: date | None,
    end_date: date | None,
    intraday_timeframe: str,
    intraday_source: str,
    starting_capital: float,
    risk_per_trade_pct: float,
    gap_fill_policy: str,
    max_hold_bars: int | None,
    one_open_per_symbol: bool,
    r_ladder_min_tenths: int,
    r_ladder_max_tenths: int,
    r_ladder_step_tenths: int,
    allow_shorts: bool,
    enable_orb_confirmation: bool,
    orb_range_minutes: int,
    orb_confirmation_cutoff_et: str,
    orb_stop_policy: str,
    enable_atr_stop_floor: bool,
    atr_stop_floor_multiple: float,
    enable_rsi_extremes: bool,
    enable_ema9_regime: bool,
    ema9_slope_lookback_bars: int,
    enable_volatility_regime: bool,
    allowed_volatility_regimes: Sequence[str],
    ma_fast_window: int,
    ma_slow_window: int,
    ma_trend_window: int,
    ma_fast_type: str,
    ma_slow_type: str,
    ma_trend_type: str,
    trend_slope_lookback_bars: int,
    atr_window: int,
    atr_stop_multiple: float,
) -> StrategyModelingProfile:
    payload = {
        "strategy": strategy,
        "symbols": tuple(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "intraday_timeframe": intraday_timeframe,
        "intraday_source": intraday_source,
        "starting_capital": float(starting_capital),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "gap_fill_policy": gap_fill_policy,
        "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        "one_open_per_symbol": bool(one_open_per_symbol),
        "r_ladder_min_tenths": int(r_ladder_min_tenths),
        "r_ladder_max_tenths": int(r_ladder_max_tenths),
        "r_ladder_step_tenths": int(r_ladder_step_tenths),
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
        "allowed_volatility_regimes": tuple(_normalize_volatility_regimes(allowed_volatility_regimes)),
        "ma_fast_window": int(ma_fast_window),
        "ma_slow_window": int(ma_slow_window),
        "ma_trend_window": int(ma_trend_window),
        "ma_fast_type": str(ma_fast_type),
        "ma_slow_type": str(ma_slow_type),
        "ma_trend_type": str(ma_trend_type),
        "trend_slope_lookback_bars": int(trend_slope_lookback_bars),
        "atr_window": int(atr_window),
        "atr_stop_multiple": float(atr_stop_multiple),
    }
    return StrategyModelingProfile.model_validate(payload)


def _apply_loaded_profile_to_state(
    *,
    profile: StrategyModelingProfile,
    available_symbols: Sequence[str],
    available_intraday_sources: Sequence[str],
    available_timeframes: Sequence[str],
) -> None:
    st.session_state[_WIDGET_KEYS["strategy"]] = profile.strategy
    if profile.start_date is not None:
        st.session_state[_WIDGET_KEYS["start_date"]] = profile.start_date
    if profile.end_date is not None:
        st.session_state[_WIDGET_KEYS["end_date"]] = profile.end_date
    st.session_state[_WIDGET_KEYS["intraday_timeframe"]] = (
        profile.intraday_timeframe
        if profile.intraday_timeframe in set(available_timeframes)
        else available_timeframes[0]
    )
    st.session_state[_WIDGET_KEYS["intraday_source"]] = (
        profile.intraday_source
        if profile.intraday_source in set(available_intraday_sources)
        else available_intraday_sources[0]
    )
    st.session_state[_WIDGET_KEYS["starting_capital"]] = float(profile.starting_capital)
    st.session_state[_WIDGET_KEYS["risk_per_trade_pct"]] = float(profile.risk_per_trade_pct)
    st.session_state[_WIDGET_KEYS["gap_fill_policy"]] = profile.gap_fill_policy
    st.session_state[_WIDGET_KEYS["max_hold_bars"]] = (
        int(profile.max_hold_bars) if profile.max_hold_bars is not None else 20
    )
    st.session_state[_WIDGET_KEYS["one_open_per_symbol"]] = bool(profile.one_open_per_symbol)
    st.session_state[_WIDGET_KEYS["segment_values_text"]] = st.session_state.get(
        _WIDGET_KEYS["segment_values_text"],
        "",
    )
    st.session_state[_WIDGET_KEYS["ma_fast_window"]] = int(profile.ma_fast_window)
    st.session_state[_WIDGET_KEYS["ma_slow_window"]] = int(profile.ma_slow_window)
    st.session_state[_WIDGET_KEYS["ma_trend_window"]] = int(profile.ma_trend_window)
    st.session_state[_WIDGET_KEYS["ma_fast_type"]] = profile.ma_fast_type
    st.session_state[_WIDGET_KEYS["ma_slow_type"]] = profile.ma_slow_type
    st.session_state[_WIDGET_KEYS["ma_trend_type"]] = profile.ma_trend_type
    st.session_state[_WIDGET_KEYS["trend_slope_lookback_bars"]] = int(profile.trend_slope_lookback_bars)
    st.session_state[_WIDGET_KEYS["atr_window"]] = int(profile.atr_window)
    st.session_state[_WIDGET_KEYS["atr_stop_multiple"]] = float(profile.atr_stop_multiple)
    st.session_state[_WIDGET_KEYS["allow_shorts"]] = bool(profile.allow_shorts)
    st.session_state[_WIDGET_KEYS["enable_orb_confirmation"]] = bool(profile.enable_orb_confirmation)
    st.session_state[_WIDGET_KEYS["orb_range_minutes"]] = int(profile.orb_range_minutes)
    st.session_state[_WIDGET_KEYS["orb_confirmation_cutoff_et"]] = profile.orb_confirmation_cutoff_et
    st.session_state[_WIDGET_KEYS["orb_stop_policy"]] = profile.orb_stop_policy
    st.session_state[_WIDGET_KEYS["enable_atr_stop_floor"]] = bool(profile.enable_atr_stop_floor)
    st.session_state[_WIDGET_KEYS["atr_stop_floor_multiple"]] = float(profile.atr_stop_floor_multiple)
    st.session_state[_WIDGET_KEYS["enable_rsi_extremes"]] = bool(profile.enable_rsi_extremes)
    st.session_state[_WIDGET_KEYS["enable_ema9_regime"]] = bool(profile.enable_ema9_regime)
    st.session_state[_WIDGET_KEYS["ema9_slope_lookback_bars"]] = int(profile.ema9_slope_lookback_bars)
    st.session_state[_WIDGET_KEYS["enable_volatility_regime"]] = bool(profile.enable_volatility_regime)
    st.session_state[_WIDGET_KEYS["allowed_volatility_regimes"]] = list(profile.allowed_volatility_regimes)
    st.session_state[_WIDGET_KEYS["r_ladder_min_tenths"]] = int(profile.r_ladder_min_tenths)
    st.session_state[_WIDGET_KEYS["r_ladder_max_tenths"]] = int(profile.r_ladder_max_tenths)
    st.session_state[_WIDGET_KEYS["r_ladder_step_tenths"]] = int(profile.r_ladder_step_tenths)

    profile_symbols = list(profile.symbols)
    st.session_state[_WIDGET_KEYS["symbols_text"]] = ",".join(profile_symbols)
    if available_symbols:
        st.session_state[_WIDGET_KEYS["symbols"]] = _sanitize_symbol_selection(
            profile_symbols,
            options=available_symbols,
        )
    else:
        st.session_state[_WIDGET_KEYS["symbols"]] = profile_symbols


def _interpolate_rgb(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, float(ratio)))
    return tuple(
        int(round(base + ((target - base) * clamped)))
        for base, target in zip(start, end, strict=True)
    )


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, float(score)))
    if clamped <= 0.5:
        return _interpolate_rgb(_SEGMENT_LOW_COLOR, _SEGMENT_NEUTRAL_COLOR, clamped / 0.5)
    return _interpolate_rgb(_SEGMENT_NEUTRAL_COLOR, _SEGMENT_HIGH_COLOR, (clamped - 0.5) / 0.5)


def _column_heat_styles(column: pd.Series) -> list[str]:
    numeric = pd.to_numeric(column, errors="coerce")
    finite = numeric[numeric.notna()]
    if finite.empty:
        return [""] * len(column)

    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    scores = pd.Series(0.5, index=numeric.index, dtype=float) if span == 0.0 else (numeric - lo) / span

    styles: list[str] = []
    for idx, score in scores.items():
        if pd.isna(score):
            styles.append("")
            continue
        red, green, blue = _score_to_rgb(float(score))
        declarations = [
            f"background-color: rgb({red}, {green}, {blue})",
            f"color: {_SEGMENT_TEXT_COLOR}",
            "border: 1px solid rgba(17, 24, 39, 0.25)",
        ]
        value = numeric.loc[idx]
        if pd.notna(value) and value == hi:
            declarations.append("font-weight: 700")
            declarations.append("border: 2px solid #14532d")
            declarations.append("outline: 2px solid #14532d")
        elif pd.notna(value) and value == lo:
            declarations.append("font-weight: 700")
            declarations.append("border: 2px solid #7f1d1d")
            declarations.append("outline: 2px solid #7f1d1d")
        styles.append("; ".join(declarations))
    return styles


def _styled_segment_breakdown(segment_df: pd.DataFrame) -> pd.DataFrame | Styler:
    if segment_df.empty:
        return segment_df

    styled = segment_df.copy()
    numeric_columns: list[str] = []
    for column in styled.columns:
        numeric = pd.to_numeric(styled[column], errors="coerce")
        if numeric.notna().any():
            styled[column] = numeric
            numeric_columns.append(column)

    if not numeric_columns:
        return styled

    styler = styled.style
    for column in numeric_columns:
        styler = styler.apply(_column_heat_styles, subset=[column], axis=0)
    return styler


st.title("Strategy Modeling")
st.caption(DISCLAIMER_TEXT)
st.info(
    "Read-only strategy modeling dashboard. This page runs deterministic local analysis only "
    "(no ingestion writes from portal interactions)."
)

symbols, symbol_notes = list_strategy_modeling_symbols(database_path=None)
default_end = date.today()
default_start = default_end - timedelta(days=365)
_STRATEGY_OPTIONS = ["sfp", "msb", "orb", "ma_crossover", "trend_following"]
_INTRADAY_TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "60Min"]
_INTRADAY_SOURCES = ["stocks_bars_local", "alpaca"]

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
if st.session_state.get(_PROFILE_SELECTED_KEY) not in profile_options:
    st.session_state[_PROFILE_SELECTED_KEY] = profile_options[0]

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
    max_hold_bars = int(
        st.number_input(
            "Max hold (bars)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get(_WIDGET_KEYS["max_hold_bars"], 20)),
            step=1,
            key=_WIDGET_KEYS["max_hold_bars"],
        )
    )
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

if profile_load_clicked:
    if not selected_profile_name:
        st.error("Choose a saved profile to load.")
    else:
        try:
            loaded_profile = load_strategy_modeling_profile(profile_store_path, selected_profile_name)
        except ValueError as exc:
            st.error(f"Failed to load profile: {exc}")
        else:
            _apply_loaded_profile_to_state(
                profile=loaded_profile,
                available_symbols=symbols,
                available_intraday_sources=_INTRADAY_SOURCES,
                available_timeframes=_INTRADAY_TIMEFRAMES,
            )
            st.session_state[_PROFILE_SELECTED_KEY] = selected_profile_name
            st.session_state[_PROFILE_SAVE_NAME_KEY] = selected_profile_name
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
        gap_fill_policy=str(gap_policy),
        max_hold_bars=int(max_hold_bars),
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
            st.session_state[_PROFILE_SELECTED_KEY] = save_name
            st.session_state[_PROFILE_SAVE_NAME_KEY] = save_name
            st.success(f"Saved profile '{save_name}'")
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
        max_hold_bars=int(max_hold_bars),
        filter_config=filter_config,
        policy={
            "require_intraday_bars": True,
            "risk_per_trade_pct": float(risk_pct),
            "gap_fill_policy": gap_policy,
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
    help="Write summary.json/summary.md/trades.csv/r_ladder.csv/segments.csv for the latest run.",
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
