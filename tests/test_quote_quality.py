from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_helper.analysis.quote_quality import compute_quote_quality


def test_quote_quality_good_quote() -> None:
    as_of = date(2026, 1, 10)
    df = pd.DataFrame(
        {
            "bid": [1.0],
            "ask": [1.1],
            "lastPrice": [1.05],
            "volume": [25],
            "openInterest": [200],
            "lastTradeDate": [as_of.isoformat()],
        }
    )
    out = compute_quote_quality(df, min_volume=10, min_open_interest=100, as_of=as_of)
    row = out.iloc[0]
    assert row["quality_label"] == "good"
    assert row["quality_score"] == 100.0
    assert row["last_trade_age_days"] == 0
    assert row["quality_warnings"] == []


def test_quote_quality_missing_bid_ask_flags() -> None:
    as_of = date(2026, 1, 10)
    df = pd.DataFrame(
        {
            "bid": [None],
            "ask": [None],
            "lastPrice": [1.0],
            "volume": [5],
            "openInterest": [50],
            "lastTradeDate": [as_of.isoformat()],
        }
    )
    out = compute_quote_quality(df, min_volume=10, min_open_interest=100, as_of=as_of)
    row = out.iloc[0]
    assert row["quality_label"] == "bad"
    assert "quote_missing_bid_ask" in row["quality_warnings"]


def test_quote_quality_stale_quote() -> None:
    as_of = date(2026, 1, 10)
    last_trade = as_of - timedelta(days=10)
    df = pd.DataFrame(
        {
            "bid": [1.0],
            "ask": [1.2],
            "volume": [100],
            "openInterest": [200],
            "lastTradeDate": [last_trade.isoformat()],
        }
    )
    out = compute_quote_quality(df, min_volume=10, min_open_interest=100, as_of=as_of)
    row = out.iloc[0]
    assert row["last_trade_age_days"] > 5
    assert "quote_stale" in row["quality_warnings"]


def test_quote_quality_invalid_spread() -> None:
    as_of = date(2026, 1, 10)
    df = pd.DataFrame(
        {
            "bid": [2.0],
            "ask": [1.0],
            "volume": [100],
            "openInterest": [200],
            "lastTradeDate": [as_of.isoformat()],
        }
    )
    out = compute_quote_quality(df, min_volume=10, min_open_interest=100, as_of=as_of)
    row = out.iloc[0]
    assert "quote_invalid" in row["quality_warnings"]


def test_quote_quality_unknown_when_no_fields() -> None:
    df = pd.DataFrame({"strike": [100.0]})
    out = compute_quote_quality(df, min_volume=10, min_open_interest=100, as_of=date(2026, 1, 10))
    row = out.iloc[0]
    assert row["quality_label"] == "unknown"
    assert pd.isna(row["quality_score"])
