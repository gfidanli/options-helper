from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


OrbStopPolicy = Literal["base", "orb_range", "tighten"]
VolatilityRegime = Literal["low", "normal", "high"]


class StrategyEntryFilterConfig(BaseModel):
    """Filter/gate contract for strategy-modeling entry selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_shorts: bool = True
    enable_orb_confirmation: bool = False
    orb_range_minutes: int = Field(default=15, ge=1, le=120)
    orb_confirmation_cutoff_et: str = "10:30"
    orb_stop_policy: OrbStopPolicy = "base"

    enable_atr_stop_floor: bool = False
    atr_stop_floor_multiple: float = Field(default=0.5, gt=0.0)
    enable_rsi_extremes: bool = False
    enable_ema9_regime: bool = False
    ema9_slope_lookback_bars: int = Field(default=3, ge=1)
    enable_volatility_regime: bool = False
    allowed_volatility_regimes: tuple[VolatilityRegime, ...] = ("low", "normal", "high")

    @field_validator("orb_confirmation_cutoff_et")
    @classmethod
    def _validate_orb_confirmation_cutoff_et(cls, value: str) -> str:
        text = str(value).strip()
        try:
            datetime.strptime(text, "%H:%M")
        except ValueError as exc:
            raise ValueError("orb_confirmation_cutoff_et must be HH:MM in 24-hour time") from exc
        return text

    @field_validator("allowed_volatility_regimes")
    @classmethod
    def _validate_allowed_volatility_regimes(
        cls,
        value: tuple[VolatilityRegime, ...],
    ) -> tuple[VolatilityRegime, ...]:
        if not value:
            raise ValueError("allowed_volatility_regimes must contain at least one regime")
        if len(set(value)) != len(value):
            raise ValueError("allowed_volatility_regimes must not contain duplicates")
        return value


__all__ = [
    "OrbStopPolicy",
    "StrategyEntryFilterConfig",
    "VolatilityRegime",
]
