from __future__ import annotations

from datetime import date

import pytest

from options_helper.models import Portfolio, Position


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

