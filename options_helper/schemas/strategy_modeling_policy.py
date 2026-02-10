from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SizingRule = Literal["risk_pct_of_equity"]
GapFillPolicy = Literal["fill_at_open"]
EntryTsAnchorPolicy = Literal["first_tradable_bar_open_after_signal_confirmed_ts"]
PriceAdjustmentPolicy = Literal["adjusted_ohlc"]
MaxHoldUnit = Literal["entry", "min", "h", "d", "w"]
StopMoveTrigger = Literal["close_r"]

_MAX_HOLD_TIMEFRAME_PATTERN = re.compile(r"^(?P<count>\d+)\s*(?P<unit>m|min|h|d|w)$", re.IGNORECASE)


def normalize_max_hold_timeframe(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError("max_hold_timeframe must be non-empty")

    lowered = token.lower()
    if lowered in {"entry", "entry_timeframe"}:
        return "entry"

    match = _MAX_HOLD_TIMEFRAME_PATTERN.match(lowered)
    if match is None:
        raise ValueError("max_hold_timeframe must be 'entry' or use units like 10Min, 1H, 1D, 1W")

    count = int(match.group("count"))
    if count < 1:
        raise ValueError("max_hold_timeframe count must be >= 1")

    raw_unit = match.group("unit")
    if raw_unit in {"m", "min"}:
        return f"{count}Min"
    if raw_unit == "h":
        return f"{count}H"
    if raw_unit == "d":
        return f"{count}D"
    return f"{count}W"


def parse_max_hold_timeframe(value: object) -> tuple[MaxHoldUnit, int]:
    normalized = normalize_max_hold_timeframe(value)
    if normalized == "entry":
        return ("entry", 1)
    if normalized.endswith("Min"):
        return ("min", int(normalized[:-3]))
    count = int(normalized[:-1])
    if normalized.endswith("H"):
        return ("h", count)
    if normalized.endswith("D"):
        return ("d", count)
    return ("w", count)


class StopMoveRule(BaseModel):
    """Close-confirmed stop moves (applied starting on the next bar).

    `trigger_r` and `stop_r` are expressed in R multiples from entry, where:
    - `0.0R` means breakeven
    - `1.0R` means lock in +1R
    - `-1.0R` corresponds to the initial stop (by definition of initial risk)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trigger_r: float = Field(ge=0.0)
    stop_r: float = Field(ge=-1.0)
    trigger: StopMoveTrigger = "close_r"

    @model_validator(mode="after")
    def _validate_stop_rule(self) -> StopMoveRule:
        if self.stop_r > self.trigger_r:
            raise ValueError("stop_r must be <= trigger_r")
        return self


class StrategyModelingPolicyConfig(BaseModel):
    """Baseline policy contract for strategy-modeling simulations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    require_intraday_bars: bool = True
    # When None, hold the trade until the last available intraday bar of the entry session.
    max_hold_bars: int | None = Field(default=None, ge=1)
    max_hold_timeframe: str = "entry"
    sizing_rule: SizingRule = "risk_pct_of_equity"
    risk_per_trade_pct: float = Field(default=1.0, gt=0.0, le=100.0)
    one_open_per_symbol: bool = True
    gap_fill_policy: GapFillPolicy = "fill_at_open"
    entry_ts_anchor_policy: EntryTsAnchorPolicy = (
        "first_tradable_bar_open_after_signal_confirmed_ts"
    )
    price_adjustment_policy: PriceAdjustmentPolicy = "adjusted_ohlc"
    stop_move_rules: tuple[StopMoveRule, ...] = Field(default_factory=tuple)

    @field_validator("max_hold_timeframe", mode="before")
    @classmethod
    def _normalize_max_hold_timeframe(cls, value: object) -> str:
        return normalize_max_hold_timeframe(value)

    @field_validator("stop_move_rules", mode="before")
    @classmethod
    def _normalize_stop_move_rules(cls, value: object) -> tuple[StopMoveRule, ...]:
        if value is None:
            return ()
        if isinstance(value, StopMoveRule):
            raw_items: list[object] = [value]
        elif isinstance(value, tuple | list):
            raw_items = list(value)
        else:
            raise TypeError("stop_move_rules must be a sequence")

        rules = tuple(StopMoveRule.model_validate(item) for item in raw_items)
        if not rules:
            return ()

        sorted_rules = tuple(sorted(rules, key=lambda rule: (float(rule.trigger_r), float(rule.stop_r))))
        last_stop_r = -1.0
        seen_triggers: set[float] = set()
        for rule in sorted_rules:
            trigger_r = float(rule.trigger_r)
            if trigger_r in seen_triggers:
                raise ValueError("stop_move_rules must not contain duplicate trigger_r values")
            seen_triggers.add(trigger_r)

            stop_r = float(rule.stop_r)
            if stop_r + 1e-12 < last_stop_r:
                raise ValueError("stop_move_rules must tighten stops (non-decreasing stop_r)")
            last_stop_r = stop_r

        return sorted_rules
