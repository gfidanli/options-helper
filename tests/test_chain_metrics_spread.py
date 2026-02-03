from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.chain_metrics import compute_chain_report, compute_spread, compute_spread_pct, execution_quality


def test_compute_spread_and_pct_with_bid_ask() -> None:
    df = pd.DataFrame({"bid": [1.0], "ask": [1.5]})
    spread = compute_spread(df).iloc[0]
    spread_pct = compute_spread_pct(df).iloc[0]
    assert spread == 0.5
    assert spread_pct == pytest.approx(0.4)


def test_compute_spread_and_pct_missing_or_zero() -> None:
    df = pd.DataFrame({"bid": [0.0, 1.0], "ask": [1.0, None]})
    spread = compute_spread(df)
    spread_pct = compute_spread_pct(df)
    assert pd.isna(spread.iloc[0])
    assert pd.isna(spread_pct.iloc[0])
    assert pd.isna(spread.iloc[1])
    assert pd.isna(spread_pct.iloc[1])


def test_execution_quality_thresholds_and_inverted() -> None:
    assert execution_quality(None) == "unknown"
    assert execution_quality(float("nan")) == "unknown"
    assert execution_quality(0.10) == "good"
    assert execution_quality(0.20) == "ok"
    assert execution_quality(0.50) == "bad"
    assert execution_quality(-0.05) == "bad"


def test_inverted_market_spread_is_negative_and_bad() -> None:
    df = pd.DataFrame({"bid": [1.0], "ask": [0.9]})
    spread = compute_spread(df).iloc[0]
    spread_pct = compute_spread_pct(df).iloc[0]
    assert spread < 0
    assert spread_pct < 0
    assert execution_quality(spread_pct) == "bad"


def test_chain_report_includes_atm_iv_for_next_expiry() -> None:
    df = pd.DataFrame(
        [
            {
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "bid": 2.0,
                "ask": 2.2,
                "impliedVolatility": 0.20,
            },
            {
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "bid": 1.8,
                "ask": 2.0,
                "impliedVolatility": 0.20,
            },
            {
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 100.0,
                "bid": 3.0,
                "ask": 3.2,
                "impliedVolatility": 0.30,
            },
            {
                "optionType": "put",
                "expiry": "2026-03-20",
                "strike": 100.0,
                "bid": 2.8,
                "ask": 3.0,
                "impliedVolatility": 0.30,
            },
        ]
    )

    report = compute_chain_report(
        df,
        symbol="AAA",
        as_of=date(2026, 1, 2),
        spot=100.0,
        expiries_mode="near",
        top=5,
        best_effort=True,
    )

    assert len(report.expiries) >= 2
    assert report.expiries[0].atm_iv == 0.20
    assert report.expiries[1].atm_iv == 0.30
