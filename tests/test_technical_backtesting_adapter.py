from __future__ import annotations

import pandas as pd

from options_helper.technicals_backtesting.adapter import standardize_ohlc


def test_standardize_ohlc_sorts_and_dedupes() -> None:
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-03", "2020-01-01"])
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.2, 2.2, 3.2, 4.2],
            "low": [0.9, 1.9, 2.9, 3.9],
            "close": [1.1, 2.1, 3.1, 4.1],
            "volume": [100, 200, 300, 400],
        },
        index=idx,
    )

    out = standardize_ohlc(
        df,
        required_cols=["Open", "High", "Low", "Close"],
        optional_cols=["Volume"],
        dropna_ohlc=True,
    )

    assert list(out.columns[:4]) == ["Open", "High", "Low", "Close"]
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates


def test_standardize_ohlc_dropna() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, None, 3.0],
            "High": [1.2, None, 3.2],
            "Low": [0.9, None, 2.9],
            "Close": [1.1, None, 3.1],
        },
        index=idx,
    )

    out = standardize_ohlc(
        df,
        required_cols=["Open", "High", "Low", "Close"],
        optional_cols=[],
        dropna_ohlc=True,
    )
    assert len(out) == 2

