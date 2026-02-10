from datetime import date

from options_helper.data.streaming.subscriptions import build_subscription_plan
from options_helper.models import Leg, MultiLegPosition, Portfolio, Position


def test_build_subscription_plan_enumerates_single_and_multileg_legs() -> None:
    portfolio = Portfolio(
        positions=[
            Position(
                id="single-1",
                symbol="brk.b",
                option_type="call",
                expiry=date(2026, 3, 20),
                strike=300.0,
                contracts=1,
                cost_basis=4.2,
            ),
            MultiLegPosition(
                id="spread-1",
                symbol="spy",
                net_debit=125.0,
                legs=[
                    Leg(
                        side="long",
                        option_type="call",
                        expiry=date(2026, 3, 20),
                        strike=500.0,
                        contracts=1,
                    ),
                    Leg(
                        side="short",
                        option_type="call",
                        expiry=date(2026, 3, 20),
                        strike=505.0,
                        contracts=1,
                    ),
                    Leg(
                        side="long",
                        option_type="put",
                        expiry=date(2026, 3, 20),
                        strike=470.0,
                        contracts=1,
                    ),
                ],
            ),
        ]
    )

    plan = build_subscription_plan(
        portfolio,
        stream_stocks=True,
        stream_options=True,
        max_option_contracts=10,
    )

    assert plan.stocks == ["BRK.B", "SPY"]
    assert plan.option_contracts == [
        "BRK.B260320C00300000",
        "SPY260320C00500000",
        "SPY260320C00505000",
        "SPY260320P00470000",
    ]
    assert plan.warnings == []
    assert plan.truncated is False
    assert plan.truncated_count == 0


def test_build_subscription_plan_truncates_option_contracts_with_warning() -> None:
    portfolio = Portfolio(
        positions=[
            Position(
                id="single-call",
                symbol="AAPL",
                option_type="call",
                expiry=date(2026, 3, 20),
                strike=180.0,
                contracts=1,
                cost_basis=5.0,
            ),
            Position(
                id="single-put",
                symbol="AAPL",
                option_type="put",
                expiry=date(2026, 3, 20),
                strike=160.0,
                contracts=1,
                cost_basis=4.0,
            ),
            MultiLegPosition(
                id="spread-1",
                symbol="SPY",
                net_debit=200.0,
                legs=[
                    Leg(
                        side="long",
                        option_type="call",
                        expiry=date(2026, 3, 20),
                        strike=500.0,
                        contracts=1,
                    ),
                    Leg(
                        side="short",
                        option_type="call",
                        expiry=date(2026, 3, 20),
                        strike=505.0,
                        contracts=1,
                    ),
                    Leg(
                        side="long",
                        option_type="put",
                        expiry=date(2026, 3, 20),
                        strike=470.0,
                        contracts=1,
                    ),
                ],
            ),
        ]
    )

    plan = build_subscription_plan(
        portfolio,
        stream_stocks=False,
        stream_options=True,
        max_option_contracts=3,
    )

    assert plan.stocks == []
    assert plan.option_contracts == [
        "AAPL260320C00180000",
        "AAPL260320P00160000",
        "SPY260320C00500000",
    ]
    assert plan.truncated is True
    assert plan.truncated_count == 2
    assert len(plan.warnings) == 1
    assert "truncated" in plan.warnings[0].lower()
    assert "max_option_contracts=3" in plan.warnings[0]


def test_build_subscription_plan_respects_stream_toggles() -> None:
    portfolio = Portfolio(
        positions=[
            Position(
                id="single-1",
                symbol="SPY",
                option_type="put",
                expiry=date(2026, 3, 20),
                strike=450.0,
                contracts=1,
                cost_basis=3.0,
            ),
        ]
    )

    plan = build_subscription_plan(
        portfolio,
        stream_stocks=False,
        stream_options=False,
        max_option_contracts=0,
    )

    assert plan.stocks == []
    assert plan.option_contracts == []
    assert plan.warnings == []
    assert plan.truncated is False
    assert plan.truncated_count == 0
