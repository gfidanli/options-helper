from __future__ import annotations

import pytest
import pandas as pd

from options_helper.technicals_backtesting.adapter import apply_yfinance_price_adjustment


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [101.0, 111.0],
            "Low": [99.0, 109.0],
            "Close": [100.0, 110.0],
            "Adj Close": [50.0, 55.0],
            "Volume": [1, 2],
        },
        index=idx,
    )


def test_apply_yfinance_price_adjustment_auto_adjust() -> None:
    df = _sample_df()
    out = apply_yfinance_price_adjustment(df, auto_adjust=True, back_adjust=False)
    assert "Adj Close" not in out.columns
    assert out["Open"].tolist() == [50.0, 55.0]
    assert out["High"].tolist() == [50.5, 55.5]
    assert out["Low"].tolist() == [49.5, 54.5]
    assert out["Close"].tolist() == [50.0, 55.0]


def test_apply_yfinance_price_adjustment_back_adjust() -> None:
    df = _sample_df()
    out = apply_yfinance_price_adjustment(df, auto_adjust=False, back_adjust=True)
    assert "Adj Close" not in out.columns
    assert out["Open"].tolist() == [50.0, 55.0]
    assert out["High"].tolist() == [50.5, 55.5]
    assert out["Low"].tolist() == [49.5, 54.5]
    # yfinance-style back_adjust keeps Close unadjusted.
    assert out["Close"].tolist() == [100.0, 110.0]


def test_apply_yfinance_price_adjustment_rejects_conflicting_flags() -> None:
    df = _sample_df()
    with pytest.raises(ValueError, match="cannot both be true"):
        _ = apply_yfinance_price_adjustment(df, auto_adjust=True, back_adjust=True)

