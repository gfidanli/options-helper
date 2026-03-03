from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame

PriceRegimeTag = Literal["trend_up", "trend_down", "choppy", "mixed", "insufficient_data"]

DEFAULT_MIN_HISTORY_BARS = 60
DEFAULT_CHOP_WINDOW = 14
DEFAULT_EMA_SLOPE_LOOKBACK_BARS = 5
DEFAULT_EMA_CROSS_COUNT_WINDOW = 20
DEFAULT_CHOP_TREND_MAX = 45.0
DEFAULT_CHOP_CHOPPY_MIN = 60.0
DEFAULT_MAX_TREND_CROSSES = 2
DEFAULT_MIN_CHOPPY_CROSSES = 4
DEFAULT_MIN_TREND_SPACING_PCT = 0.001


def classify_price_regime(
    ohlc: pd.DataFrame,
    *,
    ema9: pd.Series | None = None,
    ema21: pd.Series | None = None,
    chop14: pd.Series | None = None,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
    chop_window: int = DEFAULT_CHOP_WINDOW,
    ema_slope_lookback_bars: int = DEFAULT_EMA_SLOPE_LOOKBACK_BARS,
    ema_cross_count_window: int = DEFAULT_EMA_CROSS_COUNT_WINDOW,
    chop_trend_max: float = DEFAULT_CHOP_TREND_MAX,
    chop_choppy_min: float = DEFAULT_CHOP_CHOPPY_MIN,
) -> tuple[PriceRegimeTag, dict[str, Any]]:
    frame, close, high, low = _prepare_ohlc_inputs(ohlc)
    diagnostics = _base_diagnostics(
        history_bars=len(frame),
        min_history_bars=min_history_bars,
        chop_window=chop_window,
        ema_slope_lookback_bars=ema_slope_lookback_bars,
        ema_cross_count_window=ema_cross_count_window,
        chop_trend_max=chop_trend_max,
        chop_choppy_min=chop_choppy_min,
    )
    if frame.empty:
        diagnostics["reason"] = "empty_ohlc"
        return "insufficient_data", diagnostics
    if len(frame) < int(min_history_bars):
        diagnostics["reason"] = "insufficient_history"
        return "insufficient_data", diagnostics

    metrics = _compute_regime_metrics(
        close=close,
        ema9=_resolve_indicator(ema9, fallback=_ema_series(close, span=9), index=frame.index),
        ema21=_resolve_indicator(ema21, fallback=_ema_series(close, span=21), index=frame.index),
        chop14=chop14,
        high=high,
        low=low,
        index=frame.index,
        chop_window=chop_window,
        ema_slope_lookback_bars=ema_slope_lookback_bars,
        ema_cross_count_window=ema_cross_count_window,
    )
    diagnostics.update(_metrics_diagnostics(metrics, ema_cross_count_window=ema_cross_count_window))
    return _classify_from_metrics(
        metrics=metrics,
        diagnostics=diagnostics,
        chop_trend_max=chop_trend_max,
        chop_choppy_min=chop_choppy_min,
    )


def compute_choppiness_index(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_CHOP_WINDOW,
) -> pd.Series:
    lookback = int(window)
    if lookback <= 1:
        raise ValueError("window must be >= 2 for CHOP")
    if len(close) == 0:
        return pd.Series([], index=close.index, dtype="float64")

    high_series = pd.to_numeric(high, errors="coerce").astype("float64")
    low_series = pd.to_numeric(low, errors="coerce").astype("float64")
    close_series = pd.to_numeric(close, errors="coerce").astype("float64")
    prev_close = close_series.shift(1)
    tr = pd.concat(
        [
            high_series - low_series,
            (high_series - prev_close).abs(),
            (low_series - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(window=lookback, min_periods=lookback).sum()
    range_high = high_series.rolling(window=lookback, min_periods=lookback).max()
    range_low = low_series.rolling(window=lookback, min_periods=lookback).min()
    span = range_high - range_low
    ratio = (tr_sum / span).where((tr_sum > 0.0) & (span > 0.0))
    chop = 100.0 * (np.log10(ratio) / np.log10(float(lookback)))
    return chop.astype("float64")


def _prepare_ohlc_inputs(ohlc: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    try:
        frame = normalize_ohlc_frame(ohlc)
    except Exception:  # noqa: BLE001
        frame = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").astype("float64")
    high = pd.to_numeric(frame.get("High", pd.Series(dtype="float64")), errors="coerce").astype("float64")
    low = pd.to_numeric(frame.get("Low", pd.Series(dtype="float64")), errors="coerce").astype("float64")
    return frame, close, high, low


def _base_diagnostics(
    *,
    history_bars: int,
    min_history_bars: int,
    chop_window: int,
    ema_slope_lookback_bars: int,
    ema_cross_count_window: int,
    chop_trend_max: float,
    chop_choppy_min: float,
) -> dict[str, Any]:
    return {
        "history_bars": int(history_bars),
        "min_history_bars": int(min_history_bars),
        "chop_window": int(chop_window),
        "ema_slope_lookback_bars": int(ema_slope_lookback_bars),
        "ema_cross_count_window": int(ema_cross_count_window),
        "chop_trend_max": float(chop_trend_max),
        "chop_choppy_min": float(chop_choppy_min),
        "reason": None,
    }


def _resolve_indicator(series: pd.Series | None, *, fallback: pd.Series, index: pd.Index) -> pd.Series:
    if series is None:
        return fallback.reindex(index).astype("float64")
    return pd.to_numeric(series.reindex(index), errors="coerce").astype("float64")


def _compute_regime_metrics(
    *,
    close: pd.Series,
    ema9: pd.Series,
    ema21: pd.Series,
    chop14: pd.Series | None,
    high: pd.Series,
    low: pd.Series,
    index: pd.Index,
    chop_window: int,
    ema_slope_lookback_bars: int,
    ema_cross_count_window: int,
) -> dict[str, float | int | None]:
    chop_series = _resolve_indicator(
        chop14,
        fallback=compute_choppiness_index(high=high, low=low, close=close, window=chop_window),
        index=index,
    )
    ema9_last = _last_finite(ema9)
    ema21_last = _last_finite(ema21)
    spacing_numerator = None
    spacing_denominator = None
    if ema9_last is not None and ema21_last is not None:
        spacing_numerator = ema9_last - ema21_last
        spacing_denominator = abs(ema21_last)
    return {
        "ema9_last": ema9_last,
        "ema21_last": ema21_last,
        "ema9_slope": _slope_last_value(ema9, lookback_bars=ema_slope_lookback_bars),
        "ema21_slope": _slope_last_value(ema21, lookback_bars=ema_slope_lookback_bars),
        "chop_last": _last_finite(chop_series),
        "spacing_pct": _safe_divide(spacing_numerator, spacing_denominator),
        "cross_count": _count_ema_crosses(close=close, ema=ema21, window=ema_cross_count_window),
    }


def _metrics_diagnostics(
    metrics: dict[str, float | int | None],
    *,
    ema_cross_count_window: int,
) -> dict[str, Any]:
    cross_count = int(metrics["cross_count"] or 0)
    diagnostics: dict[str, Any] = {
        "ema9": _round_or_none(_to_float_or_none(metrics["ema9_last"])),
        "ema21": _round_or_none(_to_float_or_none(metrics["ema21_last"])),
        "ema9_slope": _round_or_none(_to_float_or_none(metrics["ema9_slope"])),
        "ema21_slope": _round_or_none(_to_float_or_none(metrics["ema21_slope"])),
        "ema_spacing_pct": _round_or_none(_to_float_or_none(metrics["spacing_pct"])),
        "chop14": _round_or_none(_to_float_or_none(metrics["chop_last"])),
        "ema21_cross_count": cross_count,
    }
    diagnostics[f"ema21_cross_count_{int(ema_cross_count_window)}"] = cross_count
    return diagnostics


def _classify_from_metrics(
    *,
    metrics: dict[str, float | int | None],
    diagnostics: dict[str, Any],
    chop_trend_max: float,
    chop_choppy_min: float,
) -> tuple[PriceRegimeTag, dict[str, Any]]:
    ema9_last = _to_float_or_none(metrics["ema9_last"])
    ema21_last = _to_float_or_none(metrics["ema21_last"])
    ema9_slope = _to_float_or_none(metrics["ema9_slope"])
    ema21_slope = _to_float_or_none(metrics["ema21_slope"])
    spacing_pct = _to_float_or_none(metrics["spacing_pct"])
    chop_last = _to_float_or_none(metrics["chop_last"])
    cross_count = int(metrics["cross_count"] or 0)

    up_alignment = _is_up_alignment(
        ema9_last=ema9_last,
        ema21_last=ema21_last,
        ema9_slope=ema9_slope,
        ema21_slope=ema21_slope,
        spacing_pct=spacing_pct,
    )
    down_alignment = _is_down_alignment(
        ema9_last=ema9_last,
        ema21_last=ema21_last,
        ema9_slope=ema9_slope,
        ema21_slope=ema21_slope,
        spacing_pct=spacing_pct,
    )
    trend_context = chop_last is not None and chop_last <= float(chop_trend_max)
    trend_context = trend_context and cross_count <= DEFAULT_MAX_TREND_CROSSES
    choppy_context = chop_last is not None and chop_last >= float(chop_choppy_min)
    choppy_context = choppy_context and cross_count >= DEFAULT_MIN_CHOPPY_CROSSES

    if up_alignment and trend_context:
        return "trend_up", diagnostics
    if down_alignment and trend_context:
        return "trend_down", diagnostics
    if choppy_context:
        return "choppy", diagnostics
    return "mixed", diagnostics


def _ema_series(close: pd.Series, *, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().astype("float64")


def _slope_last_value(series: pd.Series, *, lookback_bars: int) -> float | None:
    lookback = int(lookback_bars)
    if lookback < 1:
        raise ValueError("lookback_bars must be >= 1")
    slope_series = ((series - series.shift(lookback)) / float(lookback)).astype("float64")
    return _last_finite(slope_series)


def _count_ema_crosses(*, close: pd.Series, ema: pd.Series, window: int) -> int:
    lookback = int(window)
    if lookback < 2:
        return 0
    relation = np.sign((close - ema).to_numpy(dtype=float))
    relation_series = pd.Series(relation, index=close.index, dtype="float64")
    recent = relation_series.tail(lookback).replace(0.0, np.nan).dropna()
    if len(recent) < 2:
        return 0
    changes = recent.ne(recent.shift(1))
    return int(changes.iloc[1:].sum())


def _is_up_alignment(
    *,
    ema9_last: float | None,
    ema21_last: float | None,
    ema9_slope: float | None,
    ema21_slope: float | None,
    spacing_pct: float | None,
) -> bool:
    return (
        ema9_last is not None
        and ema21_last is not None
        and ema9_last > ema21_last
        and ema9_slope is not None
        and ema21_slope is not None
        and ema9_slope > 0.0
        and ema21_slope > 0.0
        and spacing_pct is not None
        and spacing_pct >= DEFAULT_MIN_TREND_SPACING_PCT
    )


def _is_down_alignment(
    *,
    ema9_last: float | None,
    ema21_last: float | None,
    ema9_slope: float | None,
    ema21_slope: float | None,
    spacing_pct: float | None,
) -> bool:
    return (
        ema9_last is not None
        and ema21_last is not None
        and ema9_last < ema21_last
        and ema9_slope is not None
        and ema21_slope is not None
        and ema9_slope < 0.0
        and ema21_slope < 0.0
        and spacing_pct is not None
        and spacing_pct <= -DEFAULT_MIN_TREND_SPACING_PCT
    )


def _last_finite(series: pd.Series) -> float | None:
    finite = series[np.isfinite(series.to_numpy(dtype=float))]
    if finite.empty:
        return None
    return float(finite.iloc[-1])


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    ratio = float(numerator) / float(denominator)
    if not np.isfinite(ratio):
        return None
    return ratio


def _to_float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    as_float = float(value)
    if not np.isfinite(as_float):
        return None
    return as_float


def _round_or_none(value: float | None, *, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), int(digits))


__all__ = [
    "PriceRegimeTag",
    "classify_price_regime",
    "compute_choppiness_index",
]
