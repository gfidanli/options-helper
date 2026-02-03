from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class DayContext:
    as_of: date
    spot: float | None
    weekly_trend_up: bool | None
    extension_percentile: float | None


@dataclass(frozen=True)
class PositionContext:
    entry_date: date
    days_held: int
    entry_price: float
    mark: float | None
    pnl_pct: float | None
    max_favorable: float | None
    max_adverse: float | None


class Strategy(Protocol):
    name: str

    def should_enter(self, day_ctx: DayContext) -> bool:
        ...

    def should_exit(self, pos_ctx: PositionContext) -> bool:
        ...


@dataclass(frozen=True)
class BaselineLongCallStrategy:
    name: str = "baseline_long_call"
    extension_low_pct: float = 20.0
    take_profit_pct: float = 0.8
    stop_loss_pct: float = 0.5
    max_holding_days: int = 15

    def should_enter(self, day_ctx: DayContext) -> bool:
        if day_ctx.weekly_trend_up is not True:
            return False
        if day_ctx.extension_percentile is None:
            return False
        return day_ctx.extension_percentile <= float(self.extension_low_pct)

    def should_exit(self, pos_ctx: PositionContext) -> bool:
        if pos_ctx.days_held >= int(self.max_holding_days):
            return True
        if pos_ctx.pnl_pct is None:
            return False
        if pos_ctx.pnl_pct >= float(self.take_profit_pct):
            return True
        if pos_ctx.pnl_pct <= -float(self.stop_loss_pct):
            return True
        return False

