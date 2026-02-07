from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.exposure import (
    aggregate_net_exposure_by_strike,
    compute_exposure_slice,
    compute_exposure_slices,
    compute_flip_strike,
)


def _base_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "call", "openInterest": 10, "bs_gamma": 0.010},
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "put", "openInterest": 8, "bs_gamma": 0.020},
            {"expiry": "2026-02-20", "strike": 105.0, "optionType": "call", "openInterest": 5, "bs_gamma": 0.010},
            {"expiry": "2026-02-20", "strike": 95.0, "optionType": "put", "openInterest": 4, "bs_gamma": 0.020},
            {"expiry": "2026-03-20", "strike": 100.0, "optionType": "call", "openInterest": 6, "bs_gamma": 0.015},
            {"expiry": "2026-03-20", "strike": 100.0, "optionType": "put", "openInterest": 3, "bs_gamma": 0.025},
        ]
    )


def test_compute_exposure_slice_builds_deterministic_rows_and_summary() -> None:
    result = compute_exposure_slice(
        _base_snapshot(),
        symbol="spy",
        as_of=date(2026, 2, 7),
        spot=100.0,
        mode="all",
        top_n=3,
    )

    strikes = [(row["expiry"], row["strike"]) for row in result.strike_rows]
    assert strikes == [
        ("2026-02-20", 95.0),
        ("2026-02-20", 100.0),
        ("2026-02-20", 105.0),
        ("2026-03-20", 100.0),
    ]

    row_map = {(row["expiry"], row["strike"]): row for row in result.strike_rows}
    row_0220_100 = row_map[("2026-02-20", 100.0)]
    assert row_0220_100["call_oi"] == pytest.approx(10.0)
    assert row_0220_100["put_oi"] == pytest.approx(8.0)
    assert row_0220_100["call_gex"] == pytest.approx(1000.0)
    assert row_0220_100["put_gex"] == pytest.approx(1600.0)
    assert row_0220_100["net_gex"] == pytest.approx(-600.0)

    summary = result.summary
    assert summary["symbol"] == "SPY"
    assert summary["as_of"] == "2026-02-07"
    assert summary["total_call_gex"] == pytest.approx(2400.0)
    assert summary["total_put_gex"] == pytest.approx(3150.0)
    assert summary["total_net_gex"] == pytest.approx(-750.0)
    assert summary["flip_strike"] is None
    assert summary["warnings"] == []

    top = result.top_abs_net_levels
    assert [level.strike for level in top] == [95.0, 105.0, 100.0]
    assert [level.net_gex for level in top] == pytest.approx([-800.0, 500.0, -450.0])


def test_flip_strike_uses_cumulative_crossing() -> None:
    snapshot = pd.DataFrame(
        [
            {"expiry": "2026-02-20", "strike": 90.0, "optionType": "put", "openInterest": 20, "bs_gamma": 0.010},
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "call", "openInterest": 10, "bs_gamma": 0.020},
            {"expiry": "2026-02-20", "strike": 110.0, "optionType": "call", "openInterest": 10, "bs_gamma": 0.010},
        ]
    )
    result = compute_exposure_slice(
        snapshot,
        symbol="XYZ",
        as_of="2026-02-07",
        spot=100.0,
        mode="all",
    )
    assert result.summary["flip_strike"] == pytest.approx(100.0)


def test_top_abs_net_tie_breaks_by_lower_strike() -> None:
    snapshot = pd.DataFrame(
        [
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "call", "openInterest": 10, "bs_gamma": 0.010},
            {"expiry": "2026-02-20", "strike": 105.0, "optionType": "put", "openInterest": 10, "bs_gamma": 0.010},
        ]
    )
    result = compute_exposure_slice(
        snapshot,
        symbol="XYZ",
        as_of="2026-02-07",
        spot=100.0,
        mode="all",
        top_n=2,
    )
    assert [level.strike for level in result.top_abs_net_levels] == [100.0, 105.0]
    assert [level.abs_net_gex for level in result.top_abs_net_levels] == pytest.approx([1000.0, 1000.0])


def test_non_positive_spot_skips_gex_but_keeps_oi_rows() -> None:
    result = compute_exposure_slice(
        _base_snapshot(),
        symbol="SPY",
        as_of=date(2026, 2, 7),
        spot=0.0,
        mode="all",
    )
    assert result.summary["total_call_gex"] is None
    assert result.summary["total_put_gex"] is None
    assert result.summary["total_net_gex"] is None
    assert result.summary["flip_strike"] is None
    assert "non_positive_spot" in result.summary["warnings"]
    assert all(row["call_gex"] is None for row in result.strike_rows)
    assert all(row["put_gex"] is None for row in result.strike_rows)
    assert all(row["net_gex"] is None for row in result.strike_rows)


def test_missing_oi_and_gamma_columns_default_to_zero() -> None:
    snapshot = pd.DataFrame(
        [
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "call"},
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "put"},
        ]
    )
    result = compute_exposure_slice(
        snapshot,
        symbol="SPY",
        as_of="2026-02-07",
        spot=100.0,
        mode="all",
    )
    assert len(result.strike_rows) == 1
    row = result.strike_rows[0]
    assert row["call_oi"] == 0.0
    assert row["put_oi"] == 0.0
    assert row["call_gex"] == 0.0
    assert row["put_gex"] == 0.0
    assert row["net_gex"] == 0.0
    assert "missing_openInterest" in result.summary["warnings"]
    assert "missing_bs_gamma" in result.summary["warnings"]


def test_compute_exposure_slices_near_monthly_all_selection() -> None:
    snapshot = pd.DataFrame(
        [
            {"expiry": "2026-02-13", "strike": 100.0, "optionType": "call", "openInterest": 1, "bs_gamma": 0.010},
            {"expiry": "2026-02-20", "strike": 100.0, "optionType": "call", "openInterest": 1, "bs_gamma": 0.010},
            {"expiry": "2026-03-20", "strike": 100.0, "optionType": "call", "openInterest": 1, "bs_gamma": 0.010},
        ]
    )
    slices = compute_exposure_slices(
        snapshot,
        symbol="SPY",
        as_of="2026-02-07",
        spot=100.0,
        near_n=1,
    )

    assert slices["near"].included_expiries == ["2026-02-13"]
    assert slices["monthly"].included_expiries == ["2026-02-20", "2026-03-20"]
    assert slices["all"].included_expiries == ["2026-02-13", "2026-02-20", "2026-03-20"]

    near_expiries = [row["expiry"] for row in slices["near"].strike_rows]
    monthly_expiries = [row["expiry"] for row in slices["monthly"].strike_rows]
    all_expiries = [row["expiry"] for row in slices["all"].strike_rows]
    assert near_expiries == ["2026-02-13"]
    assert monthly_expiries == ["2026-02-20", "2026-03-20"]
    assert all_expiries == ["2026-02-13", "2026-02-20", "2026-03-20"]


def test_helpers_aggregate_net_and_flip_with_no_cross() -> None:
    strike_rows = [
        {"strike": 95.0, "net_gex": -200.0},
        {"strike": 100.0, "net_gex": -100.0},
        {"strike": 100.0, "net_gex": 40.0},
        {"strike": 105.0, "net_gex": -10.0},
    ]
    net_by_strike = aggregate_net_exposure_by_strike(strike_rows)
    assert net_by_strike == [(95.0, -200.0), (100.0, -60.0), (105.0, -10.0)]
    assert compute_flip_strike(net_by_strike) is None
