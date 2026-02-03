from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.cli import _position_metrics
from options_helper.models import Position, RiskProfile


def test_position_metrics_accepts_snapshot_row_without_client() -> None:
    position = Position(
        id="a",
        symbol="AAA",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=5.0,
        contracts=1,
        cost_basis=1.0,
    )
    snap = {
        "bid": 1.0,
        "ask": 1.2,
        "lastPrice": 1.1,
        "impliedVolatility": 0.50,
        "openInterest": 500,
        "volume": 20,
    }

    metrics = _position_metrics(
        None,
        position,
        risk_profile=RiskProfile(tolerance="high", max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        underlying_history=pd.DataFrame(),
        underlying_last_price=10.0,
        as_of=date(2026, 1, 30),
        next_earnings_date=None,
        snapshot_row=snap,
    )

    assert metrics.bid == 1.0
    assert metrics.ask == 1.2
    assert metrics.mark == pytest.approx(1.1)
    assert metrics.implied_vol == pytest.approx(0.50)
    assert metrics.open_interest == 500
    assert metrics.volume == 20


def test_position_metrics_requires_client_when_snapshot_row_missing() -> None:
    position = Position(
        id="a",
        symbol="AAA",
        option_type="call",
        expiry=date(2026, 4, 17),
        strike=5.0,
        contracts=1,
        cost_basis=1.0,
    )

    with pytest.raises(ValueError, match="client is required"):
        _position_metrics(
            None,
            position,
            risk_profile=RiskProfile(tolerance="high", max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
            underlying_history=pd.DataFrame(),
            underlying_last_price=10.0,
            as_of=date(2026, 1, 30),
        )
