from __future__ import annotations

from datetime import date, timedelta

import pytest

import options_helper.analysis.scenarios as scenarios_module
from options_helper.analysis.greeks import black_scholes_price
from options_helper.analysis.scenarios import (
    DEFAULT_DAYS_FORWARD,
    DEFAULT_IV_MOVES_PP,
    DEFAULT_SPOT_MOVES_PCT,
    compute_position_scenarios,
)
from options_helper.schemas.research_metrics_contracts import SCENARIO_GRID_FIELDS, SCENARIO_SUMMARY_FIELDS


def _build_call_result(*, mark: float | None, basis: float | None) -> scenarios_module.PositionScenarioResult:
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=60)
    spot = 100.0
    strike = 100.0
    iv = 0.25
    if mark is None:
        mark = black_scholes_price(option_type="call", s=spot, k=strike, t_years=60 / 365.0, sigma=iv, r=0.0)
    assert mark is not None
    return compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260303C100",
        option_type="call",
        side="long",
        contracts=2,
        spot=spot,
        strike=strike,
        expiry=expiry,
        mark=float(mark),
        iv=iv,
        basis=basis,
    )


def test_compute_position_scenarios_default_schema_and_baseline() -> None:
    result = _build_call_result(mark=None, basis=5.0)

    assert list(result.summary.keys()) == list(SCENARIO_SUMMARY_FIELDS)
    assert result.summary["warnings"] == []
    assert list(result.grid[0].keys()) == list(SCENARIO_GRID_FIELDS)
    assert len(result.grid) == len(DEFAULT_SPOT_MOVES_PCT) * len(DEFAULT_IV_MOVES_PP) * len(DEFAULT_DAYS_FORWARD)

    baseline = [
        row
        for row in result.grid
        if row["spot_change_pct"] == 0.0 and row["iv_change_pp"] == 0.0 and row["days_forward"] == 0
    ]
    assert len(baseline) == 1
    assert baseline[0]["scenario_spot"] == pytest.approx(100.0, abs=1e-9)
    assert baseline[0]["scenario_iv"] == pytest.approx(0.25, abs=1e-9)
    assert baseline[0]["days_to_expiry"] == 60


def test_compute_position_scenarios_bs_monotonicity() -> None:
    result = _build_call_result(mark=None, basis=5.0)

    rows_spot = [row for row in result.grid if row["iv_change_pp"] == 0.0 and row["days_forward"] == 0]
    rows_spot = sorted(rows_spot, key=lambda row: float(row["spot_change_pct"]))
    spot_prices = [float(row["theoretical_price"]) for row in rows_spot]
    assert spot_prices == sorted(spot_prices)

    rows_iv = [row for row in result.grid if row["spot_change_pct"] == 0.0 and row["days_forward"] == 0]
    rows_iv = sorted(rows_iv, key=lambda row: float(row["iv_change_pp"]))
    iv_prices = [float(row["theoretical_price"]) for row in rows_iv]
    assert iv_prices == sorted(iv_prices)

    rows_time = [row for row in result.grid if row["spot_change_pct"] == 0.0 and row["iv_change_pp"] == 0.0]
    rows_time = sorted(rows_time, key=lambda row: int(row["days_forward"]))
    time_prices = [float(row["theoretical_price"]) for row in rows_time]
    assert time_prices[0] >= time_prices[1] >= time_prices[2] >= time_prices[3]


def test_compute_position_scenarios_clamps_days_to_expiry() -> None:
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=10)
    result = compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260112C100",
        option_type="call",
        side="long",
        contracts=1,
        spot=100.0,
        strike=100.0,
        expiry=expiry,
        mark=2.0,
        iv=0.25,
        basis=2.0,
        spot_moves_pct=(0.0,),
        iv_moves_pp=(0.0,),
        days_forward=(0, 30),
    )

    row_now = [row for row in result.grid if row["days_forward"] == 0][0]
    row_fwd = [row for row in result.grid if row["days_forward"] == 30][0]

    assert row_now["days_to_expiry"] == 10
    assert row_fwd["days_to_expiry"] == 0
    assert row_fwd["theoretical_price"] == pytest.approx(0.0, abs=1e-9)


def test_compute_position_scenarios_past_expiry_returns_empty_grid() -> None:
    as_of = date(2026, 1, 10)
    expiry = as_of - timedelta(days=1)
    result = compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260109C100",
        option_type="call",
        side="long",
        contracts=1,
        spot=100.0,
        strike=100.0,
        expiry=expiry,
        mark=1.5,
        iv=0.30,
        basis=1.2,
    )

    assert result.grid == []
    assert "past_expiry" in result.summary["warnings"]


def test_compute_position_scenarios_missing_inputs_returns_partial_output() -> None:
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=30)
    result = compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260201C100",
        option_type="call",
        side="long",
        contracts=1,
        spot=None,
        strike=100.0,
        expiry=expiry,
        mark=None,
        iv=None,
        basis=2.0,
    )

    assert result.grid == []
    assert result.summary["intrinsic"] is None
    assert result.summary["extrinsic"] is None
    assert result.summary["theta_burn_dollars_day"] is None
    warnings = set(result.summary["warnings"])
    assert {"missing_spot", "missing_iv", "missing_mark"}.issubset(warnings)


def test_compute_position_scenarios_iv_non_positive_never_calls_black_scholes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Black-Scholes helper should not be called")

    monkeypatch.setattr(scenarios_module, "black_scholes_price", _boom)
    monkeypatch.setattr(scenarios_module, "black_scholes_greeks", _boom)

    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=30)
    result = compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260201C100",
        option_type="call",
        side="long",
        contracts=1,
        spot=100.0,
        strike=100.0,
        expiry=expiry,
        mark=2.0,
        iv=0.0,
        basis=2.0,
    )

    assert result.grid == []
    assert "missing_iv" in result.summary["warnings"]


def test_compute_position_scenarios_missing_mark_keeps_grid_with_basis_pnl() -> None:
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=30)
    result = compute_position_scenarios(
        symbol="XYZ",
        as_of=as_of,
        contract_symbol="XYZ_20260201C100",
        option_type="call",
        side="long",
        contracts=1,
        spot=100.0,
        strike=100.0,
        expiry=expiry,
        mark=None,
        iv=0.25,
        basis=2.5,
        spot_moves_pct=(0.0,),
        iv_moves_pp=(0.0,),
        days_forward=(0,),
    )

    assert "missing_mark" in result.summary["warnings"]
    assert len(result.grid) == 1
    row = result.grid[0]
    assert row["theoretical_price"] is not None
    assert row["pnl_per_contract"] == pytest.approx(float(row["theoretical_price"]) - 2.5, abs=1e-9)
