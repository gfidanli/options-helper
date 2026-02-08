from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SizingRule = Literal["risk_pct_of_equity"]
GapFillPolicy = Literal["fill_at_open"]
EntryTsAnchorPolicy = Literal["first_tradable_bar_open_after_signal_confirmed_ts"]
PriceAdjustmentPolicy = Literal["adjusted_ohlc"]


class StrategyModelingPolicyConfig(BaseModel):
    """Baseline policy contract for strategy-modeling simulations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    require_intraday_bars: bool = True
    # When None, hold the trade until the last available intraday bar of the entry session.
    max_hold_bars: int | None = Field(default=None, ge=1)
    sizing_rule: SizingRule = "risk_pct_of_equity"
    risk_per_trade_pct: float = Field(default=1.0, gt=0.0, le=100.0)
    one_open_per_symbol: bool = True
    gap_fill_policy: GapFillPolicy = "fill_at_open"
    entry_ts_anchor_policy: EntryTsAnchorPolicy = (
        "first_tradable_bar_open_after_signal_confirmed_ts"
    )
    price_adjustment_policy: PriceAdjustmentPolicy = "adjusted_ohlc"
