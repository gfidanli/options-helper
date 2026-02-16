from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
from pandas.io.formats.style import Styler
from pydantic import ValidationError
import streamlit as st

from options_helper.analysis.fib_retracement import normalize_fib_retracement_pct
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


def _coerce_finite_float(value: object) -> float | None:
    number = _coerce_float(value)
    if number is None or not math.isfinite(number):
        return None
    return number

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
    fib_retracement_pct: float,
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
    if strategy == "fib_retracement":
        return {"fib_retracement_pct": float(normalize_fib_retracement_pct(fib_retracement_pct))}
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


def _normalize_string_list(value: object, *, uppercase: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = [str(item or "").strip() for item in value if str(item or "").strip()]
    else:
        token = str(value or "").strip()
        raw_items = [token] if token else []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = item.upper() if uppercase else item
        dedupe_key = token.upper() if uppercase else token.lower()
        if not token or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(token)
    return normalized


def _resolve_import_summary_path(raw_path: str) -> Path:
    token = str(raw_path or "").strip()
    if not token:
        raise ValueError("Enter a summary path to import.")

    resolved = Path(token).expanduser()
    if resolved.is_dir():
        resolved = resolved / "summary.json"

    if not resolved.exists():
        raise ValueError(f"Import path not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Import path must point to a file: {resolved}")
    return resolved


def _load_import_summary_payload(summary_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Import file is not valid JSON: {summary_path}") from exc
    except OSError as exc:
        raise ValueError(f"Failed reading import file: {summary_path} ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Import file must contain a JSON object.")
    return payload


def _resolve_import_symbols(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    requested_symbols = _normalize_string_list(payload.get("requested_symbols"), uppercase=True)
    modeled_symbols = _normalize_string_list(payload.get("modeled_symbols"), uppercase=True)
    if not modeled_symbols:
        modeled_symbols = requested_symbols
    if not requested_symbols:
        requested_symbols = modeled_symbols
    return requested_symbols, modeled_symbols


def _extract_import_rows(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], ...]:
    trade_rows = _rows_to_df(payload.get("trade_log")).to_dict(orient="records")
    r_ladder_rows = _rows_to_df(payload.get("r_ladder")).to_dict(orient="records")
    segment_rows = _rows_to_df(payload.get("segments")).to_dict(orient="records")
    equity_rows = _rows_to_df(payload.get("equity_curve") or payload.get("equity")).to_dict(orient="records")
    return trade_rows, r_ladder_rows, segment_rows, equity_rows


def _pad_trade_ids(ids: list[str], *, prefix: str, target_count: int | None) -> list[str]:
    if target_count is None or target_count <= len(ids):
        return ids
    out = list(ids)
    for index in range(len(out) + 1, target_count + 1):
        out.append(f"imported-{prefix}-{index}")
    return out


def _resolve_import_trade_ids(
    *,
    trade_rows: Sequence[Mapping[str, Any]],
    summary_payload: Mapping[str, Any],
) -> tuple[list[str], list[str], tuple[int, ...]]:
    accepted_trade_ids = [
        str(row.get("trade_id") or "").strip()
        for row in trade_rows
        if str(row.get("trade_id") or "").strip() and not str(row.get("reject_code") or "").strip()
    ]
    skipped_trade_ids = [
        str(row.get("trade_id") or "").strip()
        for row in trade_rows
        if str(row.get("trade_id") or "").strip() and str(row.get("reject_code") or "").strip()
    ]
    accepted_trade_ids = _pad_trade_ids(
        accepted_trade_ids,
        prefix="accepted",
        target_count=_coerce_int(summary_payload.get("accepted_trade_count")),
    )
    skipped_trade_ids = _pad_trade_ids(
        skipped_trade_ids,
        prefix="skipped",
        target_count=_coerce_int(summary_payload.get("skipped_trade_count")),
    )
    signal_event_count = _coerce_int(summary_payload.get("signal_event_count")) or 0
    signal_events = tuple(range(signal_event_count)) if signal_event_count > 0 else ()
    return accepted_trade_ids, skipped_trade_ids, signal_events


def _resolve_import_end_date(*, summary_path: Path, payload: Mapping[str, Any]) -> date | None:
    try:
        return date.fromisoformat(summary_path.parent.name)
    except ValueError:
        generated_ts = pd.to_datetime(payload.get("generated_at"), errors="coerce")
        if isinstance(generated_ts, pd.Timestamp) and not pd.isna(generated_ts):
            return generated_ts.date()
    return None


def _resolve_max_hold_timeframe(policy_metadata: Mapping[str, Any]) -> str:
    raw_max_hold_timeframe = str(policy_metadata.get("max_hold_timeframe") or "entry")
    try:
        return normalize_max_hold_timeframe(raw_max_hold_timeframe)
    except ValueError:
        return "entry"


def _build_import_request_state(
    *,
    strategy: str,
    requested_symbols: Sequence[str],
    modeled_symbols: Sequence[str],
    policy_metadata: Mapping[str, Any],
    end_date: date | None,
) -> SimpleNamespace:
    max_hold_bars = _coerce_int(policy_metadata.get("max_hold_bars"))
    if max_hold_bars is not None and max_hold_bars < 1:
        max_hold_bars = None
    normalized_max_hold_timeframe = _resolve_max_hold_timeframe(policy_metadata)
    risk_per_trade_pct = _coerce_float(policy_metadata.get("risk_per_trade_pct"))
    if risk_per_trade_pct is None:
        risk_per_trade_pct = 1.0
    output_timezone = str(policy_metadata.get("output_timezone") or _DEFAULT_EXPORT_TIMEZONE).strip()
    if not output_timezone:
        output_timezone = _DEFAULT_EXPORT_TIMEZONE
    request_symbols = tuple(requested_symbols or modeled_symbols)
    intraday_timeframe = str(policy_metadata.get("intraday_timeframe") or "1Min").strip() or "1Min"
    intraday_source = str(policy_metadata.get("intraday_source") or "imported_summary").strip() or "imported_summary"
    gap_fill_policy = str(policy_metadata.get("gap_fill_policy") or "fill_at_open")
    return SimpleNamespace(
        strategy=strategy,
        symbols=request_symbols,
        start_date=None,
        end_date=end_date,
        intraday_dir=Path("data/intraday"),
        intraday_timeframe=intraday_timeframe,
        intraday_source=intraday_source,
        gap_fill_policy=gap_fill_policy,
        max_hold_bars=max_hold_bars,
        max_hold_timeframe=normalized_max_hold_timeframe,
        output_timezone=output_timezone,
        policy={
            "require_intraday_bars": bool(policy_metadata.get("require_intraday_bars", True)),
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "gap_fill_policy": gap_fill_policy,
            "max_hold_bars": max_hold_bars,
            "max_hold_timeframe": normalized_max_hold_timeframe,
            "one_open_per_symbol": bool(policy_metadata.get("one_open_per_symbol", True)),
        },
    )


def _build_import_run_result(
    *,
    strategy: str,
    requested_symbols: Sequence[str],
    modeled_symbols: Sequence[str],
    signal_events: Sequence[int],
    accepted_trade_ids: Sequence[str],
    skipped_trade_ids: Sequence[str],
    rows: tuple[list[dict[str, Any]], ...],
    metrics_payload: Mapping[str, Any],
    filter_metadata: Mapping[str, Any],
    filter_summary: Mapping[str, Any],
    directional_metrics: Mapping[str, Any],
) -> SimpleNamespace:
    trade_rows, r_ladder_rows, segment_rows, equity_rows = rows
    return SimpleNamespace(
        strategy=strategy,
        requested_symbols=tuple(requested_symbols),
        modeled_symbols=tuple(modeled_symbols),
        signal_events=tuple(signal_events),
        accepted_trade_ids=tuple(accepted_trade_ids),
        skipped_trade_ids=tuple(skipped_trade_ids),
        portfolio_metrics=metrics_payload,
        target_hit_rates=tuple(r_ladder_rows),
        equity_curve=tuple(equity_rows),
        segment_records=tuple(segment_rows),
        trade_simulations=tuple(trade_rows),
        filter_metadata=filter_metadata,
        filter_summary=filter_summary,
        directional_metrics=directional_metrics,
    )


def _load_imported_strategy_modeling_summary(raw_path: str) -> tuple[SimpleNamespace, SimpleNamespace, Path]:
    summary_path = _resolve_import_summary_path(raw_path)
    payload = _load_import_summary_payload(summary_path)
    strategy = str(payload.get("strategy") or "").strip().lower()
    if not strategy:
        raise ValueError("Import file is missing required `strategy` value.")

    requested_symbols, modeled_symbols = _resolve_import_symbols(payload)
    rows = _extract_import_rows(payload)
    trade_rows = rows[0]
    metrics_payload = _to_dict(payload.get("metrics"))
    filter_metadata = _to_dict(payload.get("filter_metadata"))
    filter_summary = _to_dict(payload.get("filter_summary"))
    directional_metrics = _to_dict(payload.get("directional_metrics"))
    summary_payload = _to_dict(payload.get("summary"))
    policy_metadata = _to_dict(payload.get("policy_metadata"))
    accepted_trade_ids, skipped_trade_ids, signal_events = _resolve_import_trade_ids(
        trade_rows=trade_rows,
        summary_payload=summary_payload,
    )
    request_state = _build_import_request_state(
        strategy=strategy,
        requested_symbols=requested_symbols,
        modeled_symbols=modeled_symbols,
        policy_metadata=policy_metadata,
        end_date=_resolve_import_end_date(summary_path=summary_path, payload=payload),
    )
    run_result = _build_import_run_result(
        strategy=strategy,
        requested_symbols=requested_symbols,
        modeled_symbols=modeled_symbols,
        signal_events=signal_events,
        accepted_trade_ids=accepted_trade_ids,
        skipped_trade_ids=skipped_trade_ids,
        rows=rows,
        metrics_payload=metrics_payload,
        filter_metadata=filter_metadata,
        filter_summary=filter_summary,
        directional_metrics=directional_metrics,
    )
    return run_result, request_state, summary_path


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
        _PROFILE_PENDING_SELECTED_KEY: "",
        _PROFILE_PENDING_SAVE_NAME_KEY: "",
        _PROFILE_PENDING_LOAD_KEY: None,
        _PROFILE_FEEDBACK_KEY: "",
        _IMPORT_SUMMARY_PATH_KEY: "",
        _IMPORT_FEEDBACK_KEY: "",
        _WIDGET_KEYS["strategy"]: "sfp",
        _WIDGET_KEYS["start_date"]: default_start,
        _WIDGET_KEYS["end_date"]: default_end,
        _WIDGET_KEYS["intraday_timeframe"]: "1Min",
        _WIDGET_KEYS["intraday_source"]: "stocks_bars_local",
        _WIDGET_KEYS["starting_capital"]: 10_000.0,
        _WIDGET_KEYS["risk_per_trade_pct"]: 1.0,
        _WIDGET_KEYS["fib_retracement_pct"]: 61.8,
        _WIDGET_KEYS["gap_fill_policy"]: "fill_at_open",
        _WIDGET_KEYS["max_hold_enabled"]: True,
        _WIDGET_KEYS["max_hold_bars"]: 20,
        _WIDGET_KEYS["max_hold_timeframe"]: "entry",
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


def _apply_pending_profile_updates(
    *,
    profile_options: Sequence[str],
    available_symbols: Sequence[str],
    available_intraday_sources: Sequence[str],
    available_timeframes: Sequence[str],
) -> None:
    pending_profile = st.session_state.pop(_PROFILE_PENDING_LOAD_KEY, None)
    if pending_profile is not None:
        try:
            loaded_profile = StrategyModelingProfile.model_validate(pending_profile)
        except ValidationError:
            st.error("Failed to apply loaded profile state.")
        else:
            _apply_loaded_profile_to_state(
                profile=loaded_profile,
                available_symbols=available_symbols,
                available_intraday_sources=available_intraday_sources,
                available_timeframes=available_timeframes,
            )

    pending_selected_name = str(st.session_state.pop(_PROFILE_PENDING_SELECTED_KEY, "") or "").strip()
    if pending_selected_name:
        st.session_state[_PROFILE_SELECTED_KEY] = pending_selected_name
    pending_save_name = str(st.session_state.pop(_PROFILE_PENDING_SAVE_NAME_KEY, "") or "").strip()
    if pending_save_name:
        st.session_state[_PROFILE_SAVE_NAME_KEY] = pending_save_name

    if st.session_state.get(_PROFILE_SELECTED_KEY) not in profile_options:
        st.session_state[_PROFILE_SELECTED_KEY] = profile_options[0]


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
    fib_retracement_pct: float,
    gap_fill_policy: str,
    max_hold_bars: int | None,
    max_hold_timeframe: str,
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
        "fib_retracement_pct": float(fib_retracement_pct),
        "gap_fill_policy": gap_fill_policy,
        "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        "max_hold_timeframe": str(max_hold_timeframe),
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
    st.session_state[_WIDGET_KEYS["fib_retracement_pct"]] = float(profile.fib_retracement_pct)
    st.session_state[_WIDGET_KEYS["gap_fill_policy"]] = profile.gap_fill_policy
    st.session_state[_WIDGET_KEYS["max_hold_enabled"]] = profile.max_hold_bars is not None
    st.session_state[_WIDGET_KEYS["max_hold_bars"]] = (
        int(profile.max_hold_bars) if profile.max_hold_bars is not None else 20
    )
    try:
        normalized_max_hold_timeframe = normalize_max_hold_timeframe(profile.max_hold_timeframe)
    except ValueError:
        normalized_max_hold_timeframe = "entry"
    st.session_state[_WIDGET_KEYS["max_hold_timeframe"]] = normalized_max_hold_timeframe
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

__all__ = [
    "_apply_loaded_profile_to_state",
    "_apply_pending_profile_updates",
    "_as_symbol_filter",
    "_build_export_request",
    "_build_profile_from_inputs",
    "_build_signal_kwargs",
    "_coerce_finite_float",
    "_coerce_float",
    "_coerce_int",
    "_column_heat_styles",
    "_default_ticker_selection",
    "_ensure_sidebar_state_defaults",
    "_filter_segments",
    "_format_metric",
    "_load_imported_strategy_modeling_summary",
    "_normalize_ma_type",
    "_normalize_string_list",
    "_normalize_volatility_regimes",
    "_resolve_import_summary_path",
    "_rows_to_df",
    "_sanitize_symbol_selection",
    "_score_to_rgb",
    "_styled_segment_breakdown",
    "_to_dict",
]
