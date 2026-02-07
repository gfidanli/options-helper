from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from options_helper.schemas.common import ArtifactBase, utc_now


class DecisionMode(str, Enum):
    FIXED_TIME = "fixed_time"
    ROLLING = "rolling"


class FillModel(str, Enum):
    BID = "bid"
    MID = "mid"
    ASK = "ask"


class ExitMode(str, Enum):
    HOLD_TO_CLOSE = "hold_to_close"
    ADAPTIVE_EXIT = "adaptive_exit"


class DecisionBarPolicy(str, Enum):
    BAR_CLOSE_SNAPSHOT = "bar_close_snapshot"


class EntryAnchorPolicy(str, Enum):
    NEXT_TRADABLE_BAR_OR_QUOTE = "next_tradable_bar_or_quote_after_decision"


class SettlementRule(str, Enum):
    SAME_DAY_CLOSE_INTRINSIC = "same_day_close_intrinsic_proxy"


class QuoteQualityStatus(str, Enum):
    GOOD = "good"
    STALE = "stale"
    WIDE = "wide"
    CROSSED = "crossed"
    ZERO_BID = "zero_bid"
    MISSING = "missing"
    UNKNOWN = "unknown"


class SkipReason(str, Enum):
    NO_ENTRY_ANCHOR = "no_entry_anchor"
    NO_ELIGIBLE_CONTRACTS = "no_eligible_contracts"
    BAD_QUOTE_QUALITY = "bad_quote_quality"
    OUTSIDE_DECISION_WINDOW = "outside_decision_window"
    INSUFFICIENT_DATA = "insufficient_data"


DEFAULT_RISK_TIERS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
DEFAULT_EXIT_MODES: tuple[ExitMode, ...] = (
    ExitMode.HOLD_TO_CLOSE,
    ExitMode.ADAPTIVE_EXIT,
)
DEFAULT_DECISION_MODES: tuple[DecisionMode, ...] = (
    DecisionMode.FIXED_TIME,
    DecisionMode.ROLLING,
)


class ZeroDteStudyAssumptions(ArtifactBase):
    """Locked assumptions for the SPXW 0DTE put study contract."""

    proxy_underlying_symbol: str = "SPY"
    target_underlying_symbol: str = "SPXW"
    benchmark_decision_mode: DecisionMode = DecisionMode.FIXED_TIME
    benchmark_fixed_time_et: str = Field(default="10:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    rolling_interval_minutes: int = Field(default=15, ge=1, le=120)
    supported_decision_modes: tuple[DecisionMode, ...] = DEFAULT_DECISION_MODES
    fill_model: FillModel = FillModel.BID
    fill_slippage_bps: float = Field(default=0.0, ge=0.0)
    risk_tier_breach_probabilities: tuple[float, ...] = DEFAULT_RISK_TIERS
    exit_modes: tuple[ExitMode, ...] = DEFAULT_EXIT_MODES
    max_open_positions_per_symbol: int = Field(default=1, ge=1)
    max_open_positions_total: int = Field(default=1, ge=1)
    decision_bar: DecisionBarPolicy = DecisionBarPolicy.BAR_CLOSE_SNAPSHOT
    entry_anchor: EntryAnchorPolicy = EntryAnchorPolicy.NEXT_TRADABLE_BAR_OR_QUOTE
    settlement_rule: SettlementRule = SettlementRule.SAME_DAY_CLOSE_INTRINSIC
    lookahead_guardrail: str = (
        "Signals known at decision-bar close must anchor entries at the next tradable bar or quote."
    )

    @field_validator("risk_tier_breach_probabilities")
    @classmethod
    def _validate_risk_tiers(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("risk_tier_breach_probabilities must not be empty")
        if sorted(value) != list(value):
            raise ValueError("risk_tier_breach_probabilities must be sorted ascending")
        for tier in value:
            if tier <= 0.0 or tier >= 1.0:
                raise ValueError("risk tier probabilities must be between 0 and 1")
        return value

    @field_validator("exit_modes")
    @classmethod
    def _validate_exit_modes(cls, value: tuple[ExitMode, ...]) -> tuple[ExitMode, ...]:
        required = {ExitMode.HOLD_TO_CLOSE, ExitMode.ADAPTIVE_EXIT}
        if not required.issubset(set(value)):
            raise ValueError("exit_modes must include hold_to_close and adaptive_exit")
        return value


class DecisionAnchorMetadata(ArtifactBase):
    session_date: date
    decision_ts: datetime
    decision_bar_completed_ts: datetime
    close_label_ts: datetime
    decision_mode: DecisionMode
    decision_bar: DecisionBarPolicy = DecisionBarPolicy.BAR_CLOSE_SNAPSHOT
    entry_anchor: EntryAnchorPolicy = EntryAnchorPolicy.NEXT_TRADABLE_BAR_OR_QUOTE
    entry_anchor_ts: datetime | None = None


class AnchoredStudyRow(ArtifactBase):
    anchor: DecisionAnchorMetadata
    quote_quality_status: QuoteQualityStatus
    skip_reason: SkipReason | None = None

    @model_validator(mode="after")
    def _validate_no_entry_anchor_skip_reason(self) -> AnchoredStudyRow:
        if self.anchor.entry_anchor_ts is None and self.skip_reason != SkipReason.NO_ENTRY_ANCHOR:
            raise ValueError(
                "skip_reason must be no_entry_anchor when anchor.entry_anchor_ts is missing"
            )
        if self.anchor.entry_anchor_ts is not None and self.skip_reason == SkipReason.NO_ENTRY_ANCHOR:
            raise ValueError(
                "skip_reason=no_entry_anchor is only valid when anchor.entry_anchor_ts is missing"
            )
        return self


class ZeroDteProbabilityRow(AnchoredStudyRow):
    symbol: str = "SPY"
    risk_tier: float = Field(gt=0.0, lt=1.0)
    strike_return: float
    breach_probability: float = Field(ge=0.0, le=1.0)
    breach_probability_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    breach_probability_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_size: int = Field(ge=0)
    model_version: str
    assumptions_hash: str

    @model_validator(mode="after")
    def _validate_confidence_bounds(self) -> ZeroDteProbabilityRow:
        lo = self.breach_probability_ci_low
        hi = self.breach_probability_ci_high
        if lo is None and hi is None:
            return self
        if lo is None or hi is None:
            raise ValueError("breach_probability_ci_low and breach_probability_ci_high must both exist")
        if lo > hi:
            raise ValueError("breach_probability_ci_low must be <= breach_probability_ci_high")
        return self


class ZeroDteStrikeLadderRow(AnchoredStudyRow):
    symbol: str = "SPY"
    risk_tier: float = Field(gt=0.0, lt=1.0)
    ladder_rank: int = Field(ge=1)
    strike_price: float
    strike_return: float
    breach_probability: float = Field(ge=0.0, le=1.0)
    premium_estimate: float | None = None
    fill_model: FillModel = FillModel.BID


class ZeroDteSimulationRow(AnchoredStudyRow):
    symbol: str = "SPY"
    contract_symbol: str | None = None
    risk_tier: float = Field(gt=0.0, lt=1.0)
    exit_mode: ExitMode
    fill_model: FillModel = FillModel.BID
    settlement_rule: SettlementRule = SettlementRule.SAME_DAY_CLOSE_INTRINSIC
    entry_premium: float | None = None
    exit_premium: float | None = None
    pnl_per_contract: float | None = None
    max_loss_proxy: float | None = None


class ZeroDteDisclaimerMetadata(ArtifactBase):
    not_financial_advice: str = "Not financial advice. For informational/educational use only."
    informational_use_only: bool = True
    spy_proxy_caveat: str = (
        "This study uses SPY as a proxy for SPX/SPXW intraday state and may diverge from true SPX/SPXW behavior."
    )
    lookahead_notice: str = (
        "If a decision is computed at bar close, entry anchors use the next tradable bar or quote timestamp."
    )


class ZeroDtePutStudyArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime = Field(default_factory=utc_now)
    as_of: date
    assumptions: ZeroDteStudyAssumptions = Field(default_factory=ZeroDteStudyAssumptions)
    disclaimer: ZeroDteDisclaimerMetadata = Field(default_factory=ZeroDteDisclaimerMetadata)
    probability_rows: list[ZeroDteProbabilityRow] = Field(default_factory=list)
    strike_ladder_rows: list[ZeroDteStrikeLadderRow] = Field(default_factory=list)
    simulation_rows: list[ZeroDteSimulationRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

