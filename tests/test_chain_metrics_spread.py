from __future__ import annotations

import pandas as pd
import pytest

from options_helper.analysis.chain_metrics import compute_spread, compute_spread_pct, execution_quality


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
