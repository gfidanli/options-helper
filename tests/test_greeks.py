from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.greeks import (
    add_black_scholes_greeks_to_chain,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility_from_price,
)


def test_black_scholes_greeks_atm_call() -> None:
    # Standard sanity check: ATM call with r=0, T=1y, sigma=20%
    g = black_scholes_greeks(option_type="call", s=100.0, k=100.0, t_years=1.0, sigma=0.2)
    assert g is not None

    assert g.price == pytest.approx(7.9656, abs=1e-4)
    assert g.delta == pytest.approx(0.5398, abs=1e-4)
    assert g.gamma == pytest.approx(0.01985, abs=1e-5)
    assert g.vega == pytest.approx(39.695, abs=1e-3)
    assert g.theta_per_day == pytest.approx(-0.01088, abs=1e-5)


def test_implied_volatility_from_price_round_trip_call() -> None:
    price = black_scholes_price(option_type="call", s=100.0, k=100.0, t_years=1.0, sigma=0.30, r=0.0)
    assert price is not None
    iv = implied_volatility_from_price(option_type="call", s=100.0, k=100.0, t_years=1.0, price=price, r=0.0)
    assert iv == pytest.approx(0.30, abs=1e-2)


def test_add_black_scholes_greeks_to_chain_fills_placeholder_iv_from_last_price() -> None:
    as_of = date(2026, 1, 1)
    expiry = date(2027, 1, 1)  # 365 days -> t_years == 1.0
    spot = 100.0
    strike = 100.0
    sigma = 0.30

    last_price = black_scholes_price(option_type="call", s=spot, k=strike, t_years=1.0, sigma=sigma, r=0.0)
    assert last_price is not None

    df = pd.DataFrame(
        [
            {
                "contractSymbol": "TEST_CALL",
                "optionType": "call",
                "expiry": expiry.isoformat(),
                "strike": strike,
                "lastPrice": float(last_price),
                "bid": 0.0,
                "ask": 0.0,
                "volume": 1.0,
                "openInterest": 1,
                "impliedVolatility": 1e-5,  # placeholder from Yahoo in many illiquid strikes
            }
        ]
    )

    out = add_black_scholes_greeks_to_chain(df, spot=spot, expiry=expiry, as_of=as_of, r=0.0)
    assert out.loc[0, "impliedVolatility"] == pytest.approx(sigma, abs=1e-2)
    assert out.loc[0, "bs_price"] == pytest.approx(float(last_price), abs=1e-2)
    assert out.loc[0, "bs_gamma"] > 0
    assert out.loc[0, "bs_vega"] > 0


def test_add_black_scholes_greeks_to_chain_keeps_placeholder_iv_missing_when_no_volume() -> None:
    as_of = date(2026, 1, 1)
    expiry = date(2027, 1, 1)
    spot = 100.0
    strike = 100.0
    sigma = 0.30

    last_price = black_scholes_price(option_type="call", s=spot, k=strike, t_years=1.0, sigma=sigma, r=0.0)
    assert last_price is not None

    df = pd.DataFrame(
        [
            {
                "contractSymbol": "TEST_CALL",
                "optionType": "call",
                "expiry": expiry.isoformat(),
                "strike": strike,
                "lastPrice": float(last_price),
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0.0,  # no trade today -> don't infer IV from lastPrice
                "openInterest": 1,
                "impliedVolatility": 1e-5,
            }
        ]
    )

    out = add_black_scholes_greeks_to_chain(df, spot=spot, expiry=expiry, as_of=as_of, r=0.0)
    assert pd.isna(out.loc[0, "impliedVolatility"])
    assert pd.isna(out.loc[0, "bs_delta"])
