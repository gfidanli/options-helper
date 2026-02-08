from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame
from options_helper.analysis.volatility import realized_vol
from options_helper.technicals_backtesting.extension_percentiles import rolling_percentile_rank
from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags, rsi_regime_tag


@dataclass(frozen=True)
class StrategyFeatureConfig:
    extension_sma_window: int = 20
    extension_atr_window: int = 14
    extension_percentile_window: int = 252
    extension_bucket_low_pct: float = 5.0
    extension_bucket_high_pct: float = 95.0
    ema9_slope_lookback_bars: int = 3
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_divergence_left_window_bars: int = 14
    rsi_divergence_min_separation_bars: int = 2
    rsi_divergence_min_price_delta_pct: float = 0.0
    rsi_divergence_min_rsi_delta: float = 0.0
    realized_vol_window: int = 20
    realized_vol_percentile_window: int = 252
    realized_vol_low_pct: float = 33.0
    realized_vol_high_pct: float = 67.0
    bars_since_swing_boundaries: tuple[int, int, int] = (3, 10, 20)

    def __post_init__(self) -> None:
        _validate_positive_window(self.extension_sma_window, "extension_sma_window")
        _validate_positive_window(self.extension_atr_window, "extension_atr_window")
        _validate_min_window(self.extension_percentile_window, "extension_percentile_window", minimum=2)
        _validate_positive_window(self.ema9_slope_lookback_bars, "ema9_slope_lookback_bars")
        _validate_positive_window(self.rsi_window, "rsi_window")
        _validate_min_window(self.rsi_divergence_left_window_bars, "rsi_divergence_left_window_bars", minimum=2)
        _validate_min_window(
            self.rsi_divergence_min_separation_bars,
            "rsi_divergence_min_separation_bars",
            minimum=1,
        )
        _validate_positive_window(self.realized_vol_window, "realized_vol_window")
        _validate_min_window(
            self.realized_vol_percentile_window,
            "realized_vol_percentile_window",
            minimum=2,
        )
        _validate_threshold_pair(
            low=self.extension_bucket_low_pct,
            high=self.extension_bucket_high_pct,
            low_name="extension_bucket_low_pct",
            high_name="extension_bucket_high_pct",
        )
        _validate_threshold_pair(
            low=self.realized_vol_low_pct,
            high=self.realized_vol_high_pct,
            low_name="realized_vol_low_pct",
            high_name="realized_vol_high_pct",
        )
        _validate_threshold_pair(
            low=self.rsi_oversold,
            high=self.rsi_overbought,
            low_name="rsi_oversold",
            high_name="rsi_overbought",
        )
        _validate_bars_since_swing_boundaries(self.bars_since_swing_boundaries)


def parse_strategy_feature_config(overrides: Mapping[str, Any] | None = None) -> StrategyFeatureConfig:
    payload = dict(overrides or {})
    boundaries = payload.get("bars_since_swing_boundaries")
    if boundaries is not None and not isinstance(boundaries, tuple):
        payload["bars_since_swing_boundaries"] = tuple(int(value) for value in boundaries)
    return StrategyFeatureConfig(**payload)


def compute_strategy_features(
    ohlc: pd.DataFrame,
    *,
    config: StrategyFeatureConfig | None = None,
    rsi_series: pd.Series | None = None,
    bars_since_swing: pd.Series | None = None,
) -> pd.DataFrame:
    cfg = config or StrategyFeatureConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return _empty_strategy_features(index=frame.index)

    index = frame.index
    close = pd.to_numeric(frame["Close"], errors="coerce").astype("float64")
    high = pd.to_numeric(frame["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(frame["Low"], errors="coerce").astype("float64")

    atr = _atr_series(high=high, low=low, close=close, window=cfg.extension_atr_window)
    ema9 = _ema_series(close=close, span=9)
    ema9_slope = compute_ema_slope(ema9, lookback_bars=cfg.ema9_slope_lookback_bars)
    sma = close.rolling(window=cfg.extension_sma_window, min_periods=cfg.extension_sma_window).mean()
    extension_atr = (close - sma) / atr.replace(0.0, np.nan)
    extension_atr = extension_atr.astype("float64")

    extension_percentile = _rolling_percentile_rank_clean(
        extension_atr,
        window=cfg.extension_percentile_window,
    )
    extension_bucket = _map_labels(
        extension_percentile,
        lambda value: label_percentile_bucket(
            value,
            low=cfg.extension_bucket_low_pct,
            high=cfg.extension_bucket_high_pct,
        ),
    )

    if rsi_series is None:
        rsi_values = _rsi_series(close=close, window=cfg.rsi_window)
    else:
        rsi_values = pd.to_numeric(rsi_series.reindex(index), errors="coerce").astype("float64")

    rsi_regime = _map_labels(
        rsi_values,
        lambda value: rsi_regime_tag(
            rsi_value=value,
            rsi_overbought=cfg.rsi_overbought,
            rsi_oversold=cfg.rsi_oversold,
        )
        if value == value
        else None,
    )

    divergence, bearish_divergence, bullish_divergence = _compute_divergence_columns(
        close=close,
        rsi_values=rsi_values,
        cfg=cfg,
    )

    rv = realized_vol(close, window=cfg.realized_vol_window).reindex(index).astype("float64")
    rv_percentile = _rolling_percentile_rank_clean(rv, window=cfg.realized_vol_percentile_window)
    rv_regime = _map_labels(
        rv_percentile,
        lambda value: label_percentile_bucket(
            value,
            low=cfg.realized_vol_low_pct,
            high=cfg.realized_vol_high_pct,
        ),
    )

    if bars_since_swing is None:
        bars_since_swing_values = pd.Series(np.nan, index=index, dtype="float64")
        bars_since_swing_bucket = pd.Series([None] * len(index), index=index, dtype="object")
    else:
        bars_since_swing_values = pd.to_numeric(bars_since_swing.reindex(index), errors="coerce").astype("float64")
        bars_since_swing_bucket = _map_labels(
            bars_since_swing_values,
            lambda value: label_bars_since_swing_bucket(value, boundaries=cfg.bars_since_swing_boundaries),
        )

    out = pd.DataFrame(index=index)
    out["atr"] = atr
    out["ema9"] = ema9
    out["ema9_slope"] = ema9_slope
    out["extension_atr"] = extension_atr
    out["extension_percentile"] = extension_percentile
    out["extension_bucket"] = extension_bucket
    out["rsi"] = rsi_values
    out["rsi_regime"] = rsi_regime
    out["rsi_divergence"] = divergence
    out["bearish_rsi_divergence"] = bearish_divergence
    out["bullish_rsi_divergence"] = bullish_divergence
    out["realized_vol"] = rv
    out["realized_vol_percentile"] = rv_percentile
    out["realized_vol_regime"] = rv_regime
    out["volatility_regime"] = rv_regime.copy()
    out["bars_since_swing"] = bars_since_swing_values
    out["bars_since_swing_bucket"] = bars_since_swing_bucket
    return out


def label_percentile_bucket(
    value: float | int | None,
    *,
    low: float,
    high: float,
    low_label: str = "low",
    mid_label: str = "normal",
    high_label: str = "high",
) -> str | None:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    if value_f <= float(low):
        return low_label
    if value_f >= float(high):
        return high_label
    return mid_label


def label_bars_since_swing_bucket(
    value: float | int | None,
    *,
    boundaries: Sequence[int] = (3, 10, 20),
) -> str | None:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f) or value_f < 0:
        return None

    b0, b1, b2 = _normalize_bars_since_swing_boundaries(boundaries)
    if value_f < float(b0):
        return f"0-{b0 - 1}"
    if value_f < float(b1):
        return f"{b0}-{b1 - 1}"
    if value_f < float(b2):
        return f"{b1}-{b2 - 1}"
    return f"{b2}+"


def compute_ema_slope(
    ema_series: pd.Series,
    *,
    lookback_bars: int,
) -> pd.Series:
    if ema_series.empty:
        return pd.Series([], index=ema_series.index, dtype="float64")
    lookback = int(lookback_bars)
    _validate_min_window(lookback, "lookback_bars", minimum=1)
    ema_values = pd.to_numeric(ema_series, errors="coerce").astype("float64")
    return ((ema_values - ema_values.shift(lookback)) / float(lookback)).astype("float64")


def _compute_divergence_columns(
    *,
    close: pd.Series,
    rsi_values: pd.Series,
    cfg: StrategyFeatureConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    divergence = pd.Series([None] * len(close), index=close.index, dtype="object")
    bearish_divergence = pd.Series(False, index=close.index, dtype=bool)
    bullish_divergence = pd.Series(False, index=close.index, dtype=bool)

    flags = compute_rsi_divergence_flags(
        close_series=close,
        rsi_series=rsi_values,
        extension_percentile_series=None,
        window_days=cfg.rsi_divergence_left_window_bars,
        min_extension_days=0,
        min_price_delta_pct=cfg.rsi_divergence_min_price_delta_pct,
        min_rsi_delta=cfg.rsi_divergence_min_rsi_delta,
        rsi_overbought=cfg.rsi_overbought,
        rsi_oversold=cfg.rsi_oversold,
        require_rsi_extreme=False,
        min_separation_bars=cfg.rsi_divergence_min_separation_bars,
    )
    if flags.empty:
        return divergence, bearish_divergence, bullish_divergence

    if "divergence" in flags.columns:
        divergence_raw = flags["divergence"].reindex(close.index)
        divergence = _map_labels(divergence_raw, _normalize_divergence_label)
    if "bearish_divergence" in flags.columns:
        bearish_divergence = flags["bearish_divergence"].reindex(close.index).eq(True)
    if "bullish_divergence" in flags.columns:
        bullish_divergence = flags["bullish_divergence"].reindex(close.index).eq(True)

    return divergence, bearish_divergence, bullish_divergence


def _normalize_divergence_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"bearish", "bullish"}:
        return text
    return None


def _rolling_percentile_rank_clean(series: pd.Series, *, window: int) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="float64")
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    ranked = rolling_percentile_rank(clean.astype("float64"), window=int(window))
    return ranked.reindex(series.index).astype("float64")


def _atr_series(*, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _ema_series(*, close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().astype("float64")


def _rsi_series(*, close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return (100.0 - (100.0 / (1.0 + rs))).astype("float64")


def _map_labels(series: pd.Series, mapper: Any) -> pd.Series:
    return pd.Series([mapper(value) for value in series], index=series.index, dtype="object")


def _empty_strategy_features(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["atr"] = pd.Series([], index=index, dtype="float64")
    out["ema9"] = pd.Series([], index=index, dtype="float64")
    out["ema9_slope"] = pd.Series([], index=index, dtype="float64")
    out["extension_atr"] = pd.Series([], index=index, dtype="float64")
    out["extension_percentile"] = pd.Series([], index=index, dtype="float64")
    out["extension_bucket"] = pd.Series([], index=index, dtype="object")
    out["rsi"] = pd.Series([], index=index, dtype="float64")
    out["rsi_regime"] = pd.Series([], index=index, dtype="object")
    out["rsi_divergence"] = pd.Series([], index=index, dtype="object")
    out["bearish_rsi_divergence"] = pd.Series([], index=index, dtype=bool)
    out["bullish_rsi_divergence"] = pd.Series([], index=index, dtype=bool)
    out["realized_vol"] = pd.Series([], index=index, dtype="float64")
    out["realized_vol_percentile"] = pd.Series([], index=index, dtype="float64")
    out["realized_vol_regime"] = pd.Series([], index=index, dtype="object")
    out["volatility_regime"] = pd.Series([], index=index, dtype="object")
    out["bars_since_swing"] = pd.Series([], index=index, dtype="float64")
    out["bars_since_swing_bucket"] = pd.Series([], index=index, dtype="object")
    return out


def _validate_positive_window(value: int, name: str) -> None:
    _validate_min_window(value, name, minimum=1)


def _validate_min_window(value: int, name: str, *, minimum: int) -> None:
    value_i = int(value)
    if value_i < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_threshold_pair(*, low: float, high: float, low_name: str, high_name: str) -> None:
    low_f = float(low)
    high_f = float(high)
    if low_f >= high_f:
        raise ValueError(f"{low_name} must be < {high_name}")


def _normalize_bars_since_swing_boundaries(boundaries: Sequence[int]) -> tuple[int, int, int]:
    if len(boundaries) != 3:
        raise ValueError("bars_since_swing_boundaries must contain exactly three ascending integers")
    values = tuple(int(value) for value in boundaries)
    if values[0] < 1 or values[0] >= values[1] or values[1] >= values[2]:
        raise ValueError("bars_since_swing_boundaries must be strictly ascending and start at >= 1")
    return values


def _validate_bars_since_swing_boundaries(boundaries: Sequence[int]) -> None:
    _normalize_bars_since_swing_boundaries(boundaries)


__all__ = [
    "StrategyFeatureConfig",
    "compute_ema_slope",
    "compute_strategy_features",
    "label_bars_since_swing_bucket",
    "label_percentile_bucket",
    "parse_strategy_feature_config",
]
