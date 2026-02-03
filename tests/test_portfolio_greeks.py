from __future__ import annotations

from datetime import date, timedelta

import pytest

from options_helper.analysis.advice import PositionMetrics
from options_helper.analysis.greeks import black_scholes_greeks
from options_helper.analysis.portfolio_risk import StressScenario, compute_portfolio_exposure, run_stress
from options_helper.models import Position


def _make_metrics(
    *,
    position_id: str,
    symbol: str,
    option_type: str,
    expiry: date,
    strike: float,
    contracts: int,
    cost_basis: float = 1.0,
    spot: float | None,
    iv: float | None,
    dte: int | None,
    delta: float | None = None,
    theta_per_day: float | None = None,
    mark: float | None = None,
    as_of: date | None = None,
) -> PositionMetrics:
    position = Position(
        id=position_id,
        symbol=symbol,
        option_type=option_type,  # type: ignore[arg-type]
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        cost_basis=cost_basis,
        opened_at=None,
    )
    return PositionMetrics(
        position=position,
        underlying_price=spot,
        mark=mark,
        bid=None,
        ask=None,
        spread=None,
        spread_pct=None,
        execution_quality=None,
        last=None,
        implied_vol=iv,
        open_interest=None,
        volume=None,
        quality_label=None,
        last_trade_age_days=None,
        quality_warnings=[],
        dte=dte,
        moneyness=None,
        pnl_abs=None,
        pnl_pct=None,
        sma20=None,
        sma50=None,
        rsi14=None,
        ema20=None,
        ema50=None,
        close_3d=None,
        rsi14_3d=None,
        ema20_3d=None,
        ema50_3d=None,
        close_w=None,
        rsi14_w=None,
        ema20_w=None,
        ema50_w=None,
        near_support_w=None,
        breakout_w=None,
        delta=delta,
        theta_per_day=theta_per_day,
        as_of=as_of,
        next_earnings_date=None,
    )


def test_compute_portfolio_exposure_aggregates_greeks() -> None:
    spot = 100.0
    iv = 0.2
    dte = 30
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=dte)

    call_greeks = black_scholes_greeks(option_type="call", s=spot, k=100.0, t_years=dte / 365.0, sigma=iv)
    put_greeks = black_scholes_greeks(option_type="put", s=spot, k=95.0, t_years=dte / 365.0, sigma=iv)
    assert call_greeks is not None
    assert put_greeks is not None

    metrics_call = _make_metrics(
        position_id="c1",
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        strike=100.0,
        contracts=2,
        spot=spot,
        iv=iv,
        dte=dte,
        as_of=as_of,
    )
    metrics_put = _make_metrics(
        position_id="p1",
        symbol="TEST",
        option_type="put",
        expiry=expiry,
        strike=95.0,
        contracts=1,
        spot=spot,
        iv=iv,
        dte=dte,
        as_of=as_of,
    )

    exposure = compute_portfolio_exposure([metrics_call, metrics_put])

    expected_delta = (call_greeks.delta * 2.0 + put_greeks.delta * 1.0) * 100.0
    expected_theta = (call_greeks.theta_per_day * 2.0 + put_greeks.theta_per_day * 1.0) * 100.0
    expected_vega = (call_greeks.vega * 2.0 + put_greeks.vega * 1.0) * 100.0

    assert exposure.total_delta_shares == pytest.approx(expected_delta, abs=1e-6)
    assert exposure.total_theta_dollars_per_day == pytest.approx(expected_theta, abs=1e-6)
    assert exposure.total_vega_dollars_per_iv == pytest.approx(expected_vega, abs=1e-6)
    assert exposure.missing_greeks == 0
    assert exposure.as_of == as_of


def test_run_stress_directional_effects() -> None:
    spot = 100.0
    iv = 0.25
    dte = 30
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=dte)

    metrics = _make_metrics(
        position_id="c1",
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        strike=100.0,
        contracts=1,
        spot=spot,
        iv=iv,
        dte=dte,
        as_of=as_of,
    )

    exposure = compute_portfolio_exposure([metrics])
    scenarios = [
        StressScenario(name="spot_up", spot_pct=0.10),
        StressScenario(name="vol_up", vol_pp=0.05),
        StressScenario(name="time_pass", days=7),
    ]
    results = {r.name: r for r in run_stress(exposure, scenarios)}

    assert results["spot_up"].pnl is not None and results["spot_up"].pnl > 0
    assert results["vol_up"].pnl is not None and results["vol_up"].pnl > 0
    assert results["time_pass"].pnl is not None and results["time_pass"].pnl < 0


def test_missing_iv_skips_stress() -> None:
    spot = 100.0
    dte = 30
    as_of = date(2026, 1, 2)
    expiry = as_of + timedelta(days=dte)

    metrics = _make_metrics(
        position_id="c1",
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        strike=100.0,
        contracts=1,
        spot=spot,
        iv=None,
        dte=dte,
        as_of=as_of,
    )

    exposure = compute_portfolio_exposure([metrics])
    results = run_stress(exposure, [StressScenario(name="spot_up", spot_pct=0.10)])

    assert results[0].pnl is None
    assert any("stress_missing_inputs" in w for w in results[0].warnings)
