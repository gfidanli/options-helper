from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

OptionType = Literal["call", "put"]
RiskTolerance = Literal["low", "medium", "high"]


class RiskProfile(BaseModel):
    tolerance: RiskTolerance = "medium"

    # Risk budgets (fractions of portfolio capital)
    # Set to null to disable risk gating.
    max_portfolio_risk_pct: float | None = Field(default=0.25, ge=0.0, le=1.0)
    max_single_position_risk_pct: float | None = Field(default=0.07, ge=0.0, le=1.0)

    # Trade management
    # Set to null to disable PnL-based exits.
    take_profit_pct: float | None = Field(default=0.60, ge=0.0)
    stop_loss_pct: float | None = Field(default=0.35, ge=0.0)

    # Rolling
    roll_dte_threshold: int = Field(default=21, ge=0)
    preferred_roll_dte: int = Field(default=60, ge=1)
    earnings_avoid_days: int = Field(default=0, ge=0)
    earnings_warn_days: int = Field(default=21, ge=0)

    # Technical analysis
    support_proximity_pct: float = Field(default=0.03, ge=0.0, le=0.5)
    breakout_lookback_weeks: int = Field(default=20, ge=2)
    breakout_volume_mult: float = Field(default=1.25, ge=0.0)

    # Liquidity
    min_open_interest: int = Field(default=100, ge=0)
    min_volume: int = Field(default=10, ge=0)


class Position(BaseModel):
    id: str
    symbol: str
    option_type: OptionType
    expiry: date
    strike: float = Field(gt=0.0)
    contracts: int = Field(gt=0)
    cost_basis: float = Field(ge=0.0)
    opened_at: date | None = None

    @property
    def premium_paid(self) -> float:
        return self.cost_basis * 100.0 * self.contracts


class Portfolio(BaseModel):
    base_currency: str = "USD"
    cash: float = Field(default=0.0)
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)
    positions: list[Position] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_position_ids(self) -> "Portfolio":
        ids = [p.id for p in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("positions contain duplicate 'id' values")
        return self

    def premium_at_risk(self) -> float:
        return sum(p.premium_paid for p in self.positions)

    def capital_cost_basis(self) -> float:
        return float(self.cash) + self.premium_at_risk()
