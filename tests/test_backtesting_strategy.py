from __future__ import annotations

from datetime import date

from options_helper.backtesting.strategy import BaselineLongCallStrategy, DayContext, PositionContext


def test_baseline_strategy_entry_conditions() -> None:
    strat = BaselineLongCallStrategy(extension_low_pct=20.0)
    day_ok = DayContext(as_of=date(2026, 1, 2), spot=100.0, weekly_trend_up=True, extension_percentile=10.0)
    day_bad_trend = DayContext(
        as_of=date(2026, 1, 2), spot=100.0, weekly_trend_up=False, extension_percentile=10.0
    )
    day_bad_ext = DayContext(as_of=date(2026, 1, 2), spot=100.0, weekly_trend_up=True, extension_percentile=30.0)

    assert strat.should_enter(day_ok) is True
    assert strat.should_enter(day_bad_trend) is False
    assert strat.should_enter(day_bad_ext) is False


def test_baseline_strategy_exit_conditions() -> None:
    strat = BaselineLongCallStrategy(take_profit_pct=0.5, stop_loss_pct=0.3, max_holding_days=5)
    base = PositionContext(
        entry_date=date(2026, 1, 2),
        days_held=1,
        entry_price=1.0,
        mark=1.1,
        pnl_pct=0.1,
        max_favorable=None,
        max_adverse=None,
    )
    assert strat.should_exit(base) is False

    take_profit = PositionContext(
        entry_date=base.entry_date,
        days_held=base.days_held,
        entry_price=base.entry_price,
        mark=base.mark,
        pnl_pct=0.6,
        max_favorable=base.max_favorable,
        max_adverse=base.max_adverse,
    )
    assert strat.should_exit(take_profit) is True

    stop_loss = PositionContext(
        entry_date=base.entry_date,
        days_held=base.days_held,
        entry_price=base.entry_price,
        mark=base.mark,
        pnl_pct=-0.4,
        max_favorable=base.max_favorable,
        max_adverse=base.max_adverse,
    )
    assert strat.should_exit(stop_loss) is True

    time_stop = PositionContext(
        entry_date=base.entry_date,
        days_held=6,
        entry_price=base.entry_price,
        mark=base.mark,
        pnl_pct=base.pnl_pct,
        max_favorable=base.max_favorable,
        max_adverse=base.max_adverse,
    )
    assert strat.should_exit(time_stop) is True
