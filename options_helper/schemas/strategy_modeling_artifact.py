from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase, utc_now
from options_helper.schemas.strategy_modeling_contracts import (
    StrategyEquityPoint,
    StrategyId,
    StrategyPortfolioMetrics,
    StrategyRLadderStat,
    StrategySegmentRecord,
    StrategySignalEvent,
    StrategyTradeSimulation,
)
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig


STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION = 1


class StrategyModelingArtifact(ArtifactBase):
    """Versioned artifact contract for strategy-modeling outputs."""

    schema_version: Literal[1] = STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    run_id: str
    strategy: StrategyId
    symbols: list[str]
    from_date: date | None = None
    to_date: date | None = None
    policy: StrategyModelingPolicyConfig
    portfolio_metrics: StrategyPortfolioMetrics
    target_hit_rates: list[StrategyRLadderStat]
    segment_records: list[StrategySegmentRecord]
    equity_curve: list[StrategyEquityPoint]
    trade_simulations: list[StrategyTradeSimulation]
    signal_events: list[StrategySignalEvent]
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = "Not financial advice. For informational/educational use only."


__all__ = [
    "STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION",
    "StrategyModelingArtifact",
]
