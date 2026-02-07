from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from options_helper.analysis.strategy_features import (
    StrategyFeatureConfig,
    compute_strategy_features,
    label_bars_since_swing_bucket,
    label_percentile_bucket,
)


def _build_ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        },
        index=close.index,
    )


def test_compute_strategy_features_short_history_is_stable() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    bars_since_swing = pd.Series([0.0, 1.0, 2.0, 3.0, 9.0], index=idx)

    features = compute_strategy_features(
        _build_ohlc(close),
        config=StrategyFeatureConfig(
            extension_sma_window=10,
            extension_atr_window=10,
            extension_percentile_window=10,
            rsi_window=14,
            realized_vol_window=20,
            realized_vol_percentile_window=20,
        ),
        bars_since_swing=bars_since_swing,
    )

    assert list(features.index) == list(idx)
    assert features["extension_atr"].isna().all()
    assert features["extension_percentile"].isna().all()
    assert features["rsi"].isna().all()
    assert features["realized_vol"].isna().all()
    assert features["realized_vol_percentile"].isna().all()
    assert features["rsi_regime"].tolist() == [None, None, None, None, None]
    assert features["rsi_divergence"].tolist() == [None, None, None, None, None]
    assert features["realized_vol_regime"].tolist() == [None, None, None, None, None]
    assert features["bars_since_swing_bucket"].tolist() == ["0-2", "0-2", "0-2", "3-9", "3-9"]


def test_compute_strategy_features_is_deterministic_with_nans() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx, dtype="float64")
    close.iloc[5] = np.nan
    close.iloc[8] = 0.0
    close.iloc[14] = np.nan

    frame = _build_ohlc(close)
    frame.loc[idx[20], "High"] = np.nan
    frame.loc[idx[21], "Low"] = np.nan

    cfg = StrategyFeatureConfig(
        extension_sma_window=5,
        extension_atr_window=5,
        extension_percentile_window=10,
        rsi_window=4,
        realized_vol_window=5,
        realized_vol_percentile_window=10,
    )

    first = compute_strategy_features(frame, config=cfg)
    second = compute_strategy_features(frame, config=cfg)

    pdt.assert_frame_equal(first, second)
    assert set(first["rsi_regime"].dropna().unique()).issubset({"overbought", "oversold", "neutral"})
    assert set(first["rsi_divergence"].dropna().unique()).issubset({"bearish", "bullish"})
    assert set(first["realized_vol_regime"].dropna().unique()).issubset({"low", "normal", "high"})


def test_rsi_divergence_uses_configurable_left_window() -> None:
    idx = pd.date_range("2026-01-01", periods=12, freq="B")
    close = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0, 102.0, 101.0, 100.0, 101.0, 103.0, 102.0, 101.0],
        index=idx,
    )
    custom_rsi = pd.Series([50.0] * len(idx), index=idx)
    custom_rsi.iloc[5] = 70.0
    custom_rsi.iloc[9] = 65.0

    narrow = compute_strategy_features(
        _build_ohlc(close),
        config=StrategyFeatureConfig(
            extension_sma_window=3,
            extension_atr_window=3,
            extension_percentile_window=5,
            rsi_window=2,
            rsi_divergence_left_window_bars=3,
            realized_vol_window=3,
            realized_vol_percentile_window=5,
        ),
        rsi_series=custom_rsi,
    )
    wide = compute_strategy_features(
        _build_ohlc(close),
        config=StrategyFeatureConfig(
            extension_sma_window=3,
            extension_atr_window=3,
            extension_percentile_window=5,
            rsi_window=2,
            rsi_divergence_left_window_bars=6,
            realized_vol_window=3,
            realized_vol_percentile_window=5,
        ),
        rsi_series=custom_rsi,
    )

    assert narrow.loc[idx[9], "rsi_divergence"] is None
    assert wide.loc[idx[9], "rsi_divergence"] == "bearish"
    assert bool(wide.loc[idx[9], "bearish_rsi_divergence"]) is True
    assert wide.loc[idx[9], "rsi_regime"] == "neutral"


def test_bucket_helpers_are_consistent() -> None:
    assert label_percentile_bucket(5.0, low=5.0, high=95.0) == "low"
    assert label_percentile_bucket(50.0, low=5.0, high=95.0) == "normal"
    assert label_percentile_bucket(95.0, low=5.0, high=95.0) == "high"
    assert label_percentile_bucket(float("nan"), low=5.0, high=95.0) is None
    assert label_percentile_bucket(None, low=5.0, high=95.0) is None

    assert label_bars_since_swing_bucket(0.0) == "0-2"
    assert label_bars_since_swing_bucket(2.99) == "0-2"
    assert label_bars_since_swing_bucket(3.0) == "3-9"
    assert label_bars_since_swing_bucket(10.0) == "10-19"
    assert label_bars_since_swing_bucket(20.0) == "20+"
    assert label_bars_since_swing_bucket(-1.0) is None
    assert label_bars_since_swing_bucket(float("nan")) is None
