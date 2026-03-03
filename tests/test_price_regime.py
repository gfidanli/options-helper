from __future__ import annotations

import numpy as np
import pandas as pd

from options_helper.analysis.price_regime import classify_price_regime, compute_choppiness_index


def _build_ohlc(close: pd.Series) -> pd.DataFrame:
    close_series = pd.to_numeric(close, errors="coerce").astype("float64")
    open_series = close_series.shift(1).fillna(close_series)
    high = pd.concat([open_series, close_series], axis=1).max(axis=1) + 0.8
    low = pd.concat([open_series, close_series], axis=1).min(axis=1) - 0.8
    return pd.DataFrame(
        {
            "Open": open_series,
            "High": high,
            "Low": low,
            "Close": close_series,
        },
        index=close_series.index,
    )


def test_classify_price_regime_uptrend() -> None:
    idx = pd.date_range("2025-01-02", periods=140, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx, dtype="float64")
    ohlc = _build_ohlc(close)

    tag, diagnostics = classify_price_regime(ohlc)

    assert tag == "trend_up"
    assert diagnostics["history_bars"] == len(idx)
    assert diagnostics["chop14"] is not None
    assert diagnostics["chop14"] <= 45.0
    assert diagnostics["ema21_cross_count_20"] <= 2


def test_classify_price_regime_choppy() -> None:
    idx = pd.date_range("2025-01-02", periods=140, freq="B")
    swing = np.where(np.arange(len(idx)) % 2 == 0, 2.0, -2.0)
    close = pd.Series(100.0 + swing, index=idx, dtype="float64")
    ohlc = _build_ohlc(close)

    tag, diagnostics = classify_price_regime(ohlc)

    assert tag == "choppy"
    assert diagnostics["chop14"] is not None
    assert diagnostics["chop14"] >= 60.0
    assert diagnostics["ema21_cross_count_20"] >= 4


def test_classify_price_regime_insufficient_history() -> None:
    idx = pd.date_range("2025-01-02", periods=40, freq="B")
    close = pd.Series(np.linspace(50.0, 70.0, len(idx)), index=idx, dtype="float64")
    ohlc = _build_ohlc(close)

    tag, diagnostics = classify_price_regime(ohlc)

    assert tag == "insufficient_data"
    assert diagnostics["reason"] == "insufficient_history"
    assert diagnostics["history_bars"] == 40


def test_compute_choppiness_index_is_nan_safe() -> None:
    idx = pd.date_range("2025-01-02", periods=50, freq="B")
    close = pd.Series(np.linspace(90.0, 120.0, len(idx)), index=idx, dtype="float64")
    high = close + 1.0
    low = close - 1.0
    high.iloc[20] = np.nan
    low.iloc[21] = np.nan

    chop = compute_choppiness_index(high=high, low=low, close=close, window=14)

    assert len(chop) == len(idx)
    assert chop.iloc[:13].isna().all()
    assert chop.iloc[20:35].isna().any()
    assert np.isfinite(chop.iloc[-1])


def test_classify_price_regime_handles_nan_segments() -> None:
    idx = pd.date_range("2025-01-02", periods=140, freq="B")
    close = pd.Series(np.linspace(100.0, 210.0, len(idx)), index=idx, dtype="float64")
    ohlc = _build_ohlc(close)
    ohlc.loc[idx[50:55], ["Open", "High", "Low", "Close"]] = np.nan

    first = classify_price_regime(ohlc)
    second = classify_price_regime(ohlc)

    assert first == second
    assert first[0] == "trend_up"
    assert first[1]["chop14"] is not None
