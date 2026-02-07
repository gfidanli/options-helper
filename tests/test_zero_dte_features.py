from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.zero_dte_features import ZeroDTEFeatureConfig, compute_zero_dte_features
from options_helper.data.zero_dte_dataset import (
    QuoteQualityRules,
    resolve_strike_premium_snapshot,
)


def test_strike_snapshot_no_eligible_contracts_returns_skip_reason() -> None:
    snapshot = pd.DataFrame(
        {
            "contract_symbol": ["SPX260206P00100000"],
            "bid": [1.0],
            "ask": [1.1],
            "quote_timestamp": ["2026-02-06T15:30:00Z"],
        }
    )

    out = resolve_strike_premium_snapshot(
        option_snapshot=snapshot,
        option_bars=pd.DataFrame(),
        session_date=date(2026, 2, 6),
        previous_close=100.0,
        strike_returns=[-0.01],
        entry_anchor_ts="2026-02-06T15:30:00Z",
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["contract_symbol"] is None
    assert row["skip_reason"] == "no_eligible_contracts"
    assert row["quote_quality_status"] == "missing"


def test_strike_snapshot_prefers_higher_bid_when_distance_ties() -> None:
    snapshot = pd.DataFrame(
        {
            "contract_symbol": [
                "SPX260206P00100000",  # AM-settled (filtered)
                "SPXW260206P00098000",
                "SPXW260206P00100000",
            ],
            "bid": [9.0, 1.0, 1.4],
            "ask": [9.1, 1.2, 1.6],
            "quote_timestamp": [
                "2026-02-06T15:30:00Z",
                "2026-02-06T15:30:00Z",
                "2026-02-06T15:30:00Z",
            ],
        }
    )

    out = resolve_strike_premium_snapshot(
        option_snapshot=snapshot,
        option_bars=pd.DataFrame(),
        session_date=date(2026, 2, 6),
        previous_close=100.0,
        strike_returns=[-0.01],
        entry_anchor_ts="2026-02-06T15:30:00Z",
    )

    row = out.iloc[0]
    assert row["contract_symbol"] == "SPXW260206P00100000"
    assert row["settlement"] == "pm"
    assert row["strike_price"] == 100.0
    assert row["entry_premium_source"] == "quote_bid"


def test_strike_snapshot_falls_back_to_bar_when_quote_is_stale() -> None:
    snapshot = pd.DataFrame(
        {
            "contract_symbol": ["SPXW260206P00100000"],
            "bid": [1.0],
            "ask": [1.1],
            "quote_timestamp": ["2026-02-06T15:20:00Z"],
        }
    )
    option_bars = pd.DataFrame(
        {
            "contract_symbol": ["SPXW260206P00100000"],
            "timestamp": ["2026-02-06T15:30:30Z"],
            "close": [2.5],
        }
    )

    out = resolve_strike_premium_snapshot(
        option_snapshot=snapshot,
        option_bars=option_bars,
        session_date=date(2026, 2, 6),
        previous_close=100.0,
        strike_returns=[-0.01],
        entry_anchor_ts="2026-02-06T15:30:00Z",
        quote_quality_rules=QuoteQualityRules(max_quote_age_seconds=120.0, max_spread_pct=0.35),
    )

    row = out.iloc[0]
    assert row["quote_quality_status"] == "stale"
    assert row["entry_premium_source"] == "bar_close"
    assert row["skip_reason"] is None
    assert row["entry_premium"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("bid", "ask", "quote_ts", "expected_status"),
    [
        (1.2, 1.0, "2026-02-06T15:30:00Z", "crossed"),
        (1.0, 2.0, "2026-02-06T15:30:00Z", "wide"),
        (0.0, 1.0, "2026-02-06T15:30:00Z", "zero_bid"),
    ],
)
def test_strike_snapshot_bad_quote_quality_sets_skip_reason(
    bid: float,
    ask: float,
    quote_ts: str,
    expected_status: str,
) -> None:
    snapshot = pd.DataFrame(
        {
            "contract_symbol": ["SPXW260206P00100000"],
            "bid": [bid],
            "ask": [ask],
            "quote_timestamp": [quote_ts],
        }
    )

    out = resolve_strike_premium_snapshot(
        option_snapshot=snapshot,
        option_bars=pd.DataFrame(),
        session_date=date(2026, 2, 6),
        previous_close=100.0,
        strike_returns=[-0.01],
        entry_anchor_ts="2026-02-06T15:30:00Z",
        quote_quality_rules=QuoteQualityRules(max_quote_age_seconds=180.0, max_spread_pct=0.35),
    )

    row = out.iloc[0]
    assert row["quote_quality_status"] == expected_status
    assert row["skip_reason"] == "bad_quote_quality"
    assert row["entry_premium_source"] is None
    assert pd.isna(row["entry_premium"])


def test_compute_zero_dte_features_uses_strict_time_cutoff_and_formulas() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-02-06T14:30:00Z",
                "2026-02-06T14:31:00Z",
                "2026-02-06T14:32:00Z",
            ],
            "open": [100.0, 100.0, 101.0],
            "high": [101.0, 102.0, 120.0],
            "low": [99.0, 99.0, 80.0],
            "close": [100.0, 101.0, 110.0],
            "volume": [100.0, 100.0, 100.0],
            "vwap": [100.0, 100.5, 103.0],
        }
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06"],
            "decision_ts": ["2026-02-06T14:31:00Z", "2026-02-06T14:32:00Z"],
            "decision_ts_market": [
                "2026-02-06T09:31:00-05:00",
                "2026-02-06T09:32:00-05:00",
            ],
            "bar_ts": ["2026-02-06T14:31:00Z", "2026-02-06T14:32:00Z"],
            "bar_ts_market": [
                "2026-02-06T09:31:00-05:00",
                "2026-02-06T09:32:00-05:00",
            ],
            "status": ["ok", "ok"],
        }
    )
    iv_context = pd.DataFrame(
        {
            "timestamp": ["2026-02-06T14:30:00Z", "2026-02-06T14:32:00Z"],
            "iv_regime": ["low", "high"],
        }
    )

    out = compute_zero_dte_features(
        state_rows,
        bars,
        previous_close=99.0,
        iv_context=iv_context,
        config=ZeroDTEFeatureConfig(
            realized_vol_min_returns=1,
            bar_range_percentile_min_bars=1,
        ),
    )

    first = out.iloc[0]
    second = out.iloc[1]

    assert first["feature_status"] == "ok"
    assert first["bars_observed"] == 2
    assert first["intraday_return"] == pytest.approx((101.0 / 99.0) - 1.0)
    assert first["drawdown_from_open"] == pytest.approx((99.0 / 100.0) - 1.0)
    assert first["distance_from_vwap"] == pytest.approx((101.0 / 100.5) - 1.0)
    assert first["iv_regime"] == "low"

    # The 09:31 decision must not include the 09:32 outlier low.
    assert first["drawdown_from_open"] > second["drawdown_from_open"]
    assert second["iv_regime"] == "high"


def test_compute_zero_dte_features_handles_non_ok_state_and_invalid_previous_close() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-02-06T14:30:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [100.0],
            "vwap": [100.0],
        }
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06"],
            "decision_ts": ["2026-02-06T14:30:00Z", "2026-02-06T14:30:00Z"],
            "decision_ts_market": [
                "2026-02-06T09:30:00-05:00",
                "2026-02-06T09:30:00-05:00",
            ],
            "bar_ts": ["2026-02-06T14:30:00Z", "2026-02-06T14:30:00Z"],
            "bar_ts_market": [
                "2026-02-06T09:30:00-05:00",
                "2026-02-06T09:30:00-05:00",
            ],
            "status": ["outside_session", "ok"],
        }
    )

    out = compute_zero_dte_features(
        state_rows,
        bars,
        previous_close=0.0,
        config=ZeroDTEFeatureConfig(realized_vol_min_returns=1, bar_range_percentile_min_bars=1),
    )

    assert out.iloc[0]["feature_status"] == "state_outside_session"
    assert out.iloc[1]["feature_status"] == "invalid_previous_close"
    assert pd.isna(out.iloc[1]["intraday_return"])
