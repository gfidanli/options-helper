from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.analysis.quote_quality import _parse_last_trade_dates, compute_quote_quality


def test_parse_last_trade_dates_handles_epoch_seconds() -> None:
    # 2026-02-24 21:00:02+00:00 (from a Yahoo underlying payload regularMarketTime example)
    parsed = _parse_last_trade_dates(pd.Series([1771966802]))
    assert parsed.iloc[0].year == 2026


def test_parse_last_trade_dates_handles_epoch_milliseconds() -> None:
    parsed = _parse_last_trade_dates(pd.Series([1771966802000]))
    assert parsed.iloc[0].year == 2026


def test_parse_last_trade_dates_handles_epoch_nanoseconds() -> None:
    parsed = _parse_last_trade_dates(pd.Series([1771966802000000000]))
    assert parsed.iloc[0].year == 2026


def test_quote_quality_does_not_mark_epoch_seconds_as_stale() -> None:
    df = pd.DataFrame(
        {
            "bid": [1.0],
            "ask": [1.1],
            "lastPrice": [1.05],
            "volume": [10],
            "openInterest": [100],
            "lastTradeDate": [1771966802],
        }
    )
    quality = compute_quote_quality(df, min_volume=0, min_open_interest=0, as_of=date(2026, 2, 24))
    warnings = quality.loc[0, "quality_warnings"]
    assert isinstance(warnings, list)
    assert "quote_stale" not in warnings

