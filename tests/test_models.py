from __future__ import annotations

from datetime import date

import pytest

from options_helper.models import Leg, MultiLegPosition, Portfolio, Position


def test_position_premium_paid() -> None:
    p = Position(
        id="t",
        symbol="UROY",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=5.0,
        contracts=2,
        cost_basis=0.5,
    )
    assert p.premium_paid == pytest.approx(100.0)


def test_portfolio_rejects_duplicate_ids() -> None:
    p1 = Position(
        id="dup",
        symbol="UROY",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=5.0,
        contracts=1,
        cost_basis=0.5,
    )
    p2 = p1.model_copy()

    with pytest.raises(ValueError, match="duplicate"):
        Portfolio(positions=[p1, p2])


def test_multi_leg_requires_two_legs() -> None:
    leg = Leg(
        side="long",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=100.0,
        contracts=1,
    )
    with pytest.raises(ValueError, match="at least 2 legs"):
        MultiLegPosition(id="spread-1", symbol="AAPL", legs=[leg])


def test_multi_leg_premium_paid_net_debit() -> None:
    leg1 = Leg(
        side="long",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=100.0,
        contracts=1,
    )
    leg2 = Leg(
        side="short",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=105.0,
        contracts=1,
    )
    debit = MultiLegPosition(id="spread-2", symbol="AAPL", legs=[leg1, leg2], net_debit=125.0)
    assert debit.premium_paid == pytest.approx(125.0)

    credit = MultiLegPosition(id="spread-3", symbol="AAPL", legs=[leg1, leg2], net_debit=-50.0)
    assert credit.premium_paid == pytest.approx(0.0)

    missing = MultiLegPosition(id="spread-4", symbol="AAPL", legs=[leg1, leg2])
    assert missing.premium_paid == pytest.approx(0.0)


def test_portfolio_parses_multi_leg_position() -> None:
    payload = {
        "cash": 0.0,
        "positions": [
            {
                "id": "spread-5",
                "symbol": "AAPL",
                "net_debit": 120.0,
                "legs": [
                    {
                        "side": "long",
                        "option_type": "call",
                        "expiry": "2026-04-17",
                        "strike": 100.0,
                        "contracts": 1,
                    },
                    {
                        "side": "short",
                        "option_type": "call",
                        "expiry": "2026-04-17",
                        "strike": 105.0,
                        "contracts": 1,
                    },
                ],
            }
        ],
    }
    portfolio = Portfolio.model_validate(payload)
    assert isinstance(portfolio.positions[0], MultiLegPosition)
    assert portfolio.premium_at_risk() == pytest.approx(120.0)
