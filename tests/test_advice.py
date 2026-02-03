from __future__ import annotations

from datetime import date, timedelta

from options_helper.analysis.advice import Action, Confidence, PositionMetrics, advise
from options_helper.models import Portfolio, Position, RiskProfile


def _pos(
    *,
    contracts: int = 1,
    cost_basis: float = 1.0,
    option_type: str = "call",
    expiry: date | None = None,
) -> Position:
    return Position(
        id="p",
        symbol="UROY",
        option_type=option_type,  # type: ignore[arg-type]
        expiry=expiry or (date.today() + timedelta(days=365)),
        strike=5.0,
        contracts=contracts,
        cost_basis=cost_basis,
    )


def _metrics(
    position: Position,
    *,
    mark: float,
    dte: int = 365,
    daily: tuple[float, float, float] = (10.0, 9.5, 9.0),  # close, ema20, ema50
    three: tuple[float, float, float] = (10.0, 9.5, 9.0),
    weekly: tuple[float, float, float] = (10.0, 9.5, 9.0),
    rsi_d: float | None = 55.0,
    rsi_w: float | None = 55.0,
    breakout_w: bool | None = None,
    near_support_w: bool | None = None,
    spread: float | None = None,
    spread_pct: float | None = None,
    execution_quality: str | None = "unknown",
    as_of: date | None = None,
    next_earnings_date: date | None = None,
) -> PositionMetrics:
    pnl_abs = (mark - position.cost_basis) * 100.0 * position.contracts
    pnl_pct = (mark - position.cost_basis) / position.cost_basis if position.cost_basis else None

    close_d, ema20_d, ema50_d = daily
    close_3, ema20_3, ema50_3 = three
    close_w, ema20_w, ema50_w = weekly

    return PositionMetrics(
        position=position,
        underlying_price=close_d,
        mark=mark,
        bid=None,
        ask=None,
        spread=spread,
        spread_pct=spread_pct,
        execution_quality=execution_quality,
        last=None,
        implied_vol=0.5,
        open_interest=1000,
        volume=100,
        dte=dte,
        moneyness=close_d / position.strike,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        sma20=None,
        sma50=None,
        rsi14=rsi_d,
        ema20=ema20_d,
        ema50=ema50_d,
        close_3d=close_3,
        rsi14_3d=None,
        ema20_3d=ema20_3,
        ema50_3d=ema50_3,
        close_w=close_w,
        rsi14_w=rsi_w,
        ema20_w=ema20_w,
        ema50_w=ema50_w,
        near_support_w=near_support_w,
        breakout_w=breakout_w,
        delta=0.5,
        theta_per_day=-0.01,
        as_of=as_of,
        next_earnings_date=next_earnings_date,
    )


def test_weekly_breakout_recommends_add() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    m = _metrics(portfolio.positions[0], mark=1.0, breakout_w=True, rsi_w=62.0, rsi_d=60.0)
    a = advise(m, portfolio)
    assert a.action == Action.ADD


def test_weekly_breakdown_recommends_close_or_reduce() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    adverse = (8.0, 8.5, 9.0)  # close below ema50, ema20 below ema50
    m = _metrics(portfolio.positions[0], mark=0.5, daily=adverse, three=adverse, weekly=adverse, rsi_w=40.0)
    a = advise(m, portfolio)
    assert a.action in {Action.CLOSE, Action.REDUCE}


def test_stop_loss_does_not_trigger_without_trend_breakdown() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(stop_loss_pct=0.35, max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    # Big drawdown, but weekly still supportive.
    m = _metrics(
        portfolio.positions[0],
        mark=0.4,
        weekly=(10.0, 9.5, 9.0),
        three=(10.0, 9.5, 9.0),
        rsi_w=50.0,
        rsi_d=50.0,
    )
    a = advise(m, portfolio)
    assert a.action == Action.HOLD


def test_low_dte_roll_when_weekly_supportive() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(roll_dte_threshold=21, max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    m = _metrics(portfolio.positions[0], mark=1.0, dte=10, weekly=(10.0, 9.5, 9.0))
    a = advise(m, portfolio)
    assert a.action == Action.ROLL


def test_wide_spread_warns_and_lowers_confidence() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    m = _metrics(portfolio.positions[0], mark=1.0, spread=0.5, spread_pct=0.50, execution_quality="bad")
    a = advise(m, portfolio)
    assert a.confidence == Confidence.LOW
    assert any("fills may be poor" in w for w in a.warnings)


def test_missing_bid_ask_warns() -> None:
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(max_portfolio_risk_pct=None, max_single_position_risk_pct=None),
        positions=[_pos()],
    )
    m = _metrics(portfolio.positions[0], mark=1.0, spread=None, spread_pct=None, execution_quality="unknown")
    a = advise(m, portfolio)
    assert any("quote quality low" in w for w in a.warnings)


def test_earnings_event_warnings_are_added() -> None:
    as_of = date(2026, 1, 2)
    earnings = as_of + timedelta(days=5)
    position = _pos(expiry=as_of + timedelta(days=30))
    portfolio = Portfolio(
        cash=500.0,
        risk_profile=RiskProfile(
            max_portfolio_risk_pct=None,
            max_single_position_risk_pct=None,
            earnings_warn_days=21,
            earnings_avoid_days=0,
        ),
        positions=[position],
    )
    m = _metrics(position, mark=1.0, as_of=as_of, next_earnings_date=earnings)
    a = advise(m, portfolio)
    assert "earnings_within_21d" in a.warnings
    assert "expiry_crosses_earnings" in a.warnings
