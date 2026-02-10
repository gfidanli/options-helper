from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from options_helper.schemas.common import ArtifactBase, utc_now
from options_helper.schemas.strategy_modeling_contracts import StrategyId
from options_helper.schemas.strategy_modeling_filters import OrbStopPolicy, VolatilityRegime
from options_helper.schemas.strategy_modeling_policy import (
    GapFillPolicy,
    StopMoveRule,
    normalize_max_hold_timeframe,
)


STRATEGY_MODELING_PROFILE_SCHEMA_VERSION = 1
MaType = Literal["sma", "ema"]


class StrategyModelingProfile(ArtifactBase):
    """Reusable run-focused strategy-modeling inputs for CLI + dashboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: StrategyId = "sfp"
    symbols: tuple[str, ...] = Field(default_factory=tuple)
    start_date: date | None = None
    end_date: date | None = None
    intraday_timeframe: str = "5Min"
    intraday_source: str = "alpaca"

    starting_capital: float = Field(default=100_000.0, gt=0.0)
    risk_per_trade_pct: float = Field(default=1.0, gt=0.0, le=100.0)
    gap_fill_policy: GapFillPolicy = "fill_at_open"
    max_hold_bars: int | None = Field(default=None, ge=1)
    max_hold_timeframe: str = "entry"
    one_open_per_symbol: bool = True
    stop_move_rules: tuple[StopMoveRule, ...] = Field(default_factory=tuple)

    r_ladder_min_tenths: int = Field(default=10, ge=1)
    r_ladder_max_tenths: int = Field(default=20, ge=1)
    r_ladder_step_tenths: int = Field(default=1, ge=1)

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

    ma_fast_window: int = Field(default=20, ge=1)
    ma_slow_window: int = Field(default=50, ge=1)
    ma_trend_window: int = Field(default=200, ge=1)
    ma_fast_type: MaType = "sma"
    ma_slow_type: MaType = "sma"
    ma_trend_type: MaType = "sma"
    trend_slope_lookback_bars: int = Field(default=3, ge=1)
    atr_window: int = Field(default=14, ge=1)
    atr_stop_multiple: float = Field(default=2.0, gt=0.0)

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalize_strategy(cls, value: object) -> str:
        return str(value or "").strip().lower()

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values: list[object] = [item for item in value.split(",")]
        elif isinstance(value, tuple | list):
            raw_values = list(value)
        else:
            raise TypeError("symbols must be a sequence or comma-separated string")

        out: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            symbol = str(raw or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return tuple(out)

    @field_validator("intraday_timeframe", "intraday_source", mode="before")
    @classmethod
    def _normalize_non_empty_string(cls, value: object) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("value must be non-empty")
        return token

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
        elif isinstance(value, str):
            token = value.strip()
            raw_items = [] if not token else [item for item in token.split(",")]
        elif isinstance(value, tuple | list):
            raw_items = list(value)
        else:
            raise TypeError("stop_move_rules must be a sequence or comma-separated string")

        parsed: list[StopMoveRule] = []
        for raw in raw_items:
            if raw is None:
                continue
            if isinstance(raw, StopMoveRule):
                parsed.append(raw)
                continue
            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    continue
                if ":" not in text:
                    raise ValueError("stop_move_rules items must be 'trigger_r:stop_r'")
                trigger_text, stop_text = text.split(":", 1)
                parsed.append(
                    StopMoveRule(
                        trigger_r=float(trigger_text.strip()),
                        stop_r=float(stop_text.strip()),
                    )
                )
                continue
            parsed.append(StopMoveRule.model_validate(raw))

        if not parsed:
            return ()

        sorted_rules = tuple(sorted(parsed, key=lambda rule: (float(rule.trigger_r), float(rule.stop_r))))
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

    @field_validator("orb_confirmation_cutoff_et", mode="before")
    @classmethod
    def _normalize_orb_cutoff(cls, value: object) -> str:
        token = str(value or "").strip()
        try:
            datetime.strptime(token, "%H:%M")
        except ValueError as exc:
            raise ValueError("orb_confirmation_cutoff_et must be HH:MM in 24-hour time") from exc
        return token

    @field_validator("allowed_volatility_regimes", mode="before")
    @classmethod
    def _normalize_allowed_volatility_regimes(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = [item for item in value.split(",")]
        elif isinstance(value, tuple | list):
            raw_values = list(value)
        else:
            raise TypeError("allowed_volatility_regimes must be a sequence or comma-separated string")

        out: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            token = str(raw or "").strip().lower()
            if not token:
                continue
            if token in seen:
                raise ValueError("allowed_volatility_regimes must not contain duplicates")
            seen.add(token)
            out.append(token)
        return tuple(out)

    @field_validator("ma_fast_type", "ma_slow_type", "ma_trend_type", mode="before")
    @classmethod
    def _normalize_ma_type(cls, value: object) -> str:
        return str(value or "").strip().lower()

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> StrategyModelingProfile:
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
        if self.r_ladder_max_tenths < self.r_ladder_min_tenths:
            raise ValueError("r_ladder_max_tenths must be >= r_ladder_min_tenths")
        if self.strategy == "ma_crossover" and self.ma_fast_window >= self.ma_slow_window:
            raise ValueError("ma_fast_window must be < ma_slow_window when strategy=ma_crossover")
        if not self.allowed_volatility_regimes:
            raise ValueError("allowed_volatility_regimes must contain at least one regime")
        return self


class StrategyModelingProfileStore(ArtifactBase):
    """Versioned on-disk store for named strategy-modeling profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = STRATEGY_MODELING_PROFILE_SCHEMA_VERSION
    updated_at: datetime = Field(default_factory=utc_now)
    profiles: dict[str, StrategyModelingProfile] = Field(default_factory=dict)

    @field_validator("updated_at", mode="before")
    @classmethod
    def _normalize_updated_at(cls, value: object) -> datetime:
        if value is None:
            return utc_now()
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @field_validator("profiles")
    @classmethod
    def _validate_profile_names(
        cls,
        value: dict[str, StrategyModelingProfile],
    ) -> dict[str, StrategyModelingProfile]:
        normalized: dict[str, StrategyModelingProfile] = {}
        for name, profile in value.items():
            token = str(name or "").strip()
            if not token:
                raise ValueError("profile names must be non-empty")
            normalized[token] = profile
        return normalized

    @classmethod
    def empty(cls) -> StrategyModelingProfileStore:
        return cls(schema_version=STRATEGY_MODELING_PROFILE_SCHEMA_VERSION, profiles={})

    def to_json_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        ordered_profiles: dict[str, Any] = {}
        raw_profiles = dict(payload.get("profiles") or {})
        for name in sorted(raw_profiles):
            ordered_profiles[name] = raw_profiles[name]
        payload["profiles"] = ordered_profiles
        return payload


__all__ = [
    "MaType",
    "STRATEGY_MODELING_PROFILE_SCHEMA_VERSION",
    "StrategyModelingProfile",
    "StrategyModelingProfileStore",
]
