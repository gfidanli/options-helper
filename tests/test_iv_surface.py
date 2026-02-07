from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.iv_surface import (
    IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS,
    IV_SURFACE_TENOR_CHANGE_FIELDS,
    compute_iv_surface,
)
from options_helper.schemas.research_metrics_contracts import (
    DELTA_BUCKET_ORDER,
    IV_SURFACE_DELTA_BUCKET_FIELDS,
    IV_SURFACE_TENOR_FIELDS,
    IV_SURFACE_TENOR_TARGETS_DTE,
)


def _build_expiry_rows(expiry: str, *, iv_shift: float = 0.0, mark_shift: float = 0.0) -> list[dict[str, object]]:
    # strike, call_delta, put_delta, call_iv, put_iv, call_bid, call_ask, put_bid, put_ask
    specs = [
        (90.0, 0.90, -0.10, 0.30, 0.21, 12.0, 12.4, 0.4, 0.6),
        (95.0, 0.70, -0.30, 0.24, 0.24, 7.0, 7.4, 0.9, 1.1),
        (100.0, 0.50, -0.50, 0.20, 0.22, 2.0, 2.2, 1.8, 2.0),
        (105.0, 0.30, -0.70, 0.19, 0.28, 0.9, 1.1, 5.5, 5.9),
        (110.0, 0.10, -0.90, 0.17, 0.34, 0.4, 0.6, 11.5, 11.9),
    ]

    rows: list[dict[str, object]] = []
    for strike, call_delta, put_delta, call_iv, put_iv, call_bid, call_ask, put_bid, put_ask in specs:
        rows.append(
            {
                "expiry": expiry,
                "optionType": "call",
                "strike": strike,
                "impliedVolatility": call_iv + iv_shift,
                "bs_delta": call_delta,
                "bid": call_bid + mark_shift,
                "ask": call_ask + mark_shift,
                "lastPrice": (call_bid + call_ask) / 2.0 + mark_shift,
            }
        )
        rows.append(
            {
                "expiry": expiry,
                "optionType": "put",
                "strike": strike,
                "impliedVolatility": put_iv + iv_shift,
                "bs_delta": put_delta,
                "bid": put_bid + mark_shift,
                "ask": put_ask + mark_shift,
                "lastPrice": (put_bid + put_ask) / 2.0 + mark_shift,
            }
        )

    return rows


def _is_null(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def test_compute_iv_surface_extracts_tenors_and_delta_buckets_deterministically() -> None:
    snapshot = pd.DataFrame(
        _build_expiry_rows("2026-01-08", iv_shift=0.00)
        + _build_expiry_rows("2026-01-15", iv_shift=0.01)
        + _build_expiry_rows("2026-01-29", iv_shift=0.02)
        + _build_expiry_rows("2026-02-02", iv_shift=0.03)
        + _build_expiry_rows("2026-03-02", iv_shift=0.04)
    )

    result = compute_iv_surface(snapshot, symbol="aaa", as_of=date(2026, 1, 1), spot=100.0)

    assert list(result.tenor.columns) == list(IV_SURFACE_TENOR_FIELDS)
    assert list(result.delta_buckets.columns) == list(IV_SURFACE_DELTA_BUCKET_FIELDS)
    assert tuple(result.tenor["tenor_target_dte"].tolist()) == IV_SURFACE_TENOR_TARGETS_DTE

    row_7 = result.tenor[result.tenor["tenor_target_dte"] == 7].iloc[0]
    assert row_7["expiry"] == "2026-01-08"
    assert row_7["dte"] == 7
    assert row_7["atm_strike"] == 100.0
    assert row_7["atm_iv"] == pytest.approx(0.21)
    assert row_7["atm_mark"] == pytest.approx(2.0)
    assert row_7["straddle_mark"] == pytest.approx(4.0)
    assert row_7["expected_move_pct"] == pytest.approx(0.04)
    assert row_7["skew_25d_pp"] == pytest.approx(5.0)
    assert row_7["skew_10d_pp"] == pytest.approx(4.0)
    assert row_7["contracts_used"] == 10
    assert row_7["warnings"] == []

    row_30 = result.tenor[result.tenor["tenor_target_dte"] == 30].iloc[0]
    assert row_30["expiry"] == "2026-01-29"
    assert row_30["dte"] == 28
    assert row_30["tenor_gap_dte"] == -2

    row_90 = result.tenor[result.tenor["tenor_target_dte"] == 90].iloc[0]
    assert row_90["expiry"] == "2026-03-02"
    assert row_90["dte"] == 60

    expected_bucket_rows = len(IV_SURFACE_TENOR_TARGETS_DTE) * 2 * len(DELTA_BUCKET_ORDER)
    assert len(result.delta_buckets) == expected_bucket_rows

    call_d20_40 = result.delta_buckets[
        (result.delta_buckets["tenor_target_dte"] == 7)
        & (result.delta_buckets["option_type"] == "call")
        & (result.delta_buckets["delta_bucket"] == "d20_40")
    ].iloc[0]
    assert call_d20_40["n_contracts"] == 1
    assert call_d20_40["avg_iv"] == pytest.approx(0.19)
    assert call_d20_40["median_iv"] == pytest.approx(0.19)

    assert result.tenor_changes.empty
    assert result.delta_bucket_changes.empty
    assert result.warnings == ()


def test_compute_iv_surface_handles_missing_iv_delta_and_quotes_with_warnings() -> None:
    snapshot = pd.DataFrame(
        [
            {"expiry": "2026-01-08", "optionType": "call", "strike": 100.0},
            {"expiry": "2026-01-08", "optionType": "put", "strike": 100.0},
        ]
    )

    result = compute_iv_surface(snapshot, symbol="AAA", as_of=date(2026, 1, 1), spot=100.0)

    assert "missing_impliedVolatility" in result.warnings
    assert "missing_bs_delta" in result.warnings
    assert "missing_quotes" in result.warnings

    row_7 = result.tenor[result.tenor["tenor_target_dte"] == 7].iloc[0]
    assert row_7["expiry"] == "2026-01-08"
    assert row_7["atm_strike"] == 100.0
    assert _is_null(row_7["atm_iv"])
    assert _is_null(row_7["atm_mark"])
    assert _is_null(row_7["straddle_mark"])
    assert _is_null(row_7["expected_move_pct"])

    row_warnings = row_7["warnings"]
    assert "missing_impliedVolatility" in row_warnings
    assert "missing_bs_delta" in row_warnings
    assert "missing_quotes" in row_warnings

    bucket_row = result.delta_buckets[
        (result.delta_buckets["tenor_target_dte"] == 7)
        & (result.delta_buckets["option_type"] == "call")
        & (result.delta_buckets["delta_bucket"] == "d40_60")
    ].iloc[0]
    assert bucket_row["n_contracts"] == 0
    assert _is_null(bucket_row["avg_iv"])
    assert _is_null(bucket_row["median_iv"])
    assert "empty_bucket" in bucket_row["warnings"]


def test_compute_iv_surface_can_skip_optional_10d_skew() -> None:
    snapshot = pd.DataFrame(_build_expiry_rows("2026-01-08"))

    result = compute_iv_surface(
        snapshot,
        symbol="AAA",
        as_of=date(2026, 1, 1),
        spot=100.0,
        include_skew_10d=False,
    )

    row_7 = result.tenor[result.tenor["tenor_target_dte"] == 7].iloc[0]
    assert _is_null(row_7["skew_10d_pp"])
    assert "missing_skew_10d" not in row_7["warnings"]


def test_compute_iv_surface_computes_day_over_day_changes_from_previous_surface() -> None:
    prev_snapshot = pd.DataFrame(_build_expiry_rows("2026-01-08", iv_shift=0.00, mark_shift=0.00))
    cur_snapshot = pd.DataFrame(_build_expiry_rows("2026-01-08", iv_shift=0.02, mark_shift=0.20))

    prev = compute_iv_surface(prev_snapshot, symbol="AAA", as_of=date(2026, 1, 1), spot=100.0)
    cur = compute_iv_surface(
        cur_snapshot,
        symbol="AAA",
        as_of=date(2026, 1, 2),
        spot=100.0,
        previous_tenor=prev.tenor,
        previous_delta_buckets=prev.delta_buckets,
    )

    assert list(cur.tenor_changes.columns) == list(IV_SURFACE_TENOR_CHANGE_FIELDS)
    assert list(cur.delta_bucket_changes.columns) == list(IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS)

    tenor_change_7 = cur.tenor_changes[cur.tenor_changes["tenor_target_dte"] == 7].iloc[0]
    assert tenor_change_7["atm_iv_change_pp"] == pytest.approx(2.0)
    assert tenor_change_7["atm_mark_change"] == pytest.approx(0.2)
    assert tenor_change_7["straddle_mark_change"] == pytest.approx(0.4)
    assert tenor_change_7["expected_move_pct_change_pp"] == pytest.approx(0.4)
    assert tenor_change_7["skew_25d_pp_change"] == pytest.approx(0.0)
    assert tenor_change_7["skew_10d_pp_change"] == pytest.approx(0.0)
    assert tenor_change_7["contracts_used_change"] == 0
    assert "missing_previous_row" not in tenor_change_7["warnings"]

    delta_change_call_d20_40 = cur.delta_bucket_changes[
        (cur.delta_bucket_changes["tenor_target_dte"] == 7)
        & (cur.delta_bucket_changes["option_type"] == "call")
        & (cur.delta_bucket_changes["delta_bucket"] == "d20_40")
    ].iloc[0]
    assert delta_change_call_d20_40["avg_iv_change_pp"] == pytest.approx(2.0)
    assert delta_change_call_d20_40["median_iv_change_pp"] == pytest.approx(2.0)
    assert delta_change_call_d20_40["n_contracts_change"] == 0
