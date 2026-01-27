from __future__ import annotations

import pytest

from options_helper.analysis.greeks import black_scholes_greeks


def test_black_scholes_greeks_atm_call() -> None:
    # Standard sanity check: ATM call with r=0, T=1y, sigma=20%
    g = black_scholes_greeks(option_type="call", s=100.0, k=100.0, t_years=1.0, sigma=0.2)
    assert g is not None

    assert g.price == pytest.approx(7.9656, abs=1e-4)
    assert g.delta == pytest.approx(0.5398, abs=1e-4)
    assert g.gamma == pytest.approx(0.01985, abs=1e-5)
    assert g.vega == pytest.approx(39.695, abs=1e-3)
    assert g.theta_per_day == pytest.approx(-0.01088, abs=1e-5)
