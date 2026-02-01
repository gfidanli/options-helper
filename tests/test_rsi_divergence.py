from __future__ import annotations

import pandas as pd

from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags


def test_bearish_divergence_detected_with_extension_gating() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="B")
    # Construct two recent swing highs: 103 -> 104 while RSI: 70 -> 68 (lower high).
    close = pd.Series([100.0] * 13 + [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0], index=idx)
    rsi = pd.Series([50.0] * 20, index=idx)
    rsi.iloc[16] = 70.0
    rsi.iloc[18] = 68.0
    ext_pct = pd.Series([99.0] * 20, index=idx)

    flags = compute_rsi_divergence_flags(
        close_series=close,
        rsi_series=rsi,
        extension_percentile_series=ext_pct,
        window_days=14,
        min_extension_days=5,
        min_extension_percentile=95.0,
        max_extension_percentile=5.0,
        min_price_delta_pct=0.0,
        min_rsi_delta=1.0,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
        require_rsi_extreme=False,
    )

    assert bool(flags.iloc[18]["bearish_divergence"]) is True
    assert flags.iloc[18]["divergence"] == "bearish"
    assert flags.iloc[18]["swing1_date"] == idx[16].date().isoformat()
    assert flags.iloc[18]["swing2_date"] == idx[18].date().isoformat()
    # rsi=68 is neutral for (70/30).
    assert flags.iloc[18]["rsi_regime"] == "neutral"


def test_bearish_divergence_can_be_gated_by_overbought() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="B")
    close = pd.Series([100.0] * 13 + [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0], index=idx)
    rsi = pd.Series([50.0] * 20, index=idx)
    rsi.iloc[16] = 70.0
    rsi.iloc[18] = 68.0  # not overbought
    ext_pct = pd.Series([99.0] * 20, index=idx)

    flags = compute_rsi_divergence_flags(
        close_series=close,
        rsi_series=rsi,
        extension_percentile_series=ext_pct,
        window_days=14,
        min_extension_days=5,
        min_extension_percentile=95.0,
        max_extension_percentile=5.0,
        min_rsi_delta=1.0,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
        require_rsi_extreme=True,
    )

    assert bool(flags.iloc[18]["bearish_divergence"]) is False


def test_bullish_divergence_detected_with_extension_gating_and_oversold_tag() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="B")
    # Two recent swing lows: 97 -> 96 (lower low) while RSI: 28 -> 32 (higher low).
    close = pd.Series([100.0] * 13 + [100.0, 98.0, 99.0, 97.0, 98.0, 96.0, 97.0], index=idx)
    rsi = pd.Series([50.0] * 20, index=idx)
    rsi.iloc[16] = 28.0
    rsi.iloc[18] = 32.0
    ext_pct = pd.Series([1.0] * 20, index=idx)

    flags = compute_rsi_divergence_flags(
        close_series=close,
        rsi_series=rsi,
        extension_percentile_series=ext_pct,
        window_days=14,
        min_extension_days=5,
        min_extension_percentile=95.0,
        max_extension_percentile=5.0,
        min_price_delta_pct=0.0,
        min_rsi_delta=1.0,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
        require_rsi_extreme=False,
    )

    assert bool(flags.iloc[18]["bullish_divergence"]) is True
    assert flags.iloc[18]["divergence"] == "bullish"
    assert flags.iloc[18]["swing1_date"] == idx[16].date().isoformat()
    assert flags.iloc[18]["swing2_date"] == idx[18].date().isoformat()
    assert flags.iloc[18]["rsi_regime"] == "neutral"  # rsi=32 is not <=30

    flags2 = compute_rsi_divergence_flags(
        close_series=close,
        rsi_series=rsi,
        extension_percentile_series=ext_pct,
        window_days=14,
        min_extension_days=5,
        max_extension_percentile=5.0,
        min_rsi_delta=1.0,
        rsi_overbought=70.0,
        rsi_oversold=35.0,
        require_rsi_extreme=True,
    )
    assert bool(flags2.iloc[18]["bullish_divergence"]) is True
    assert flags2.iloc[18]["rsi_regime"] == "oversold"

