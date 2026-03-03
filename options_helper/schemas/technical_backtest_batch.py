from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from options_helper.schemas.common import ArtifactBase, utc_now


TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION = 1


class _TechnicalBacktestBatchContractBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TechnicalBacktestBatchMetrics(_TechnicalBacktestBatchContractBase):
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualized_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    trade_count: int = Field(ge=0)
    win_rate: float | None = None
    profit_factor: float | None = None
    invested_pct: float | None = None


class TechnicalBacktestBatchFailure(_TechnicalBacktestBatchContractBase):
    symbol: str
    stage: str
    error: str


class TechnicalBacktestBatchSymbolMetrics(_TechnicalBacktestBatchContractBase):
    symbol: str
    period_start: date
    period_end: date
    metrics: TechnicalBacktestBatchMetrics


class TechnicalBacktestBatchEquityPoint(_TechnicalBacktestBatchContractBase):
    session_date: date
    aggregate_equity: float
    benchmark_equity: float | None = None
    aggregate_drawdown_pct: float | None = None
    benchmark_drawdown_pct: float | None = None


class TechnicalBacktestBatchMonthlyReturn(_TechnicalBacktestBatchContractBase):
    year: int
    month: int = Field(ge=1, le=12)
    aggregate_return_pct: float
    benchmark_return_pct: float | None = None


class TechnicalBacktestBatchYearlyReturn(_TechnicalBacktestBatchContractBase):
    year: int
    aggregate_return_pct: float
    benchmark_return_pct: float | None = None


class TechnicalBacktestBatchSummaryArtifact(ArtifactBase):
    """Versioned contract for technical backtest batch summary artifacts."""

    schema_version: Literal[1] = TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    as_of: date
    run_id: str
    strategy: str
    benchmark_symbol: str
    requested_symbols: list[str]
    modeled_symbols: list[str]
    failed_symbols: list[TechnicalBacktestBatchFailure]
    period_start: date
    period_end: date
    aggregate_metrics: TechnicalBacktestBatchMetrics
    benchmark_metrics: TechnicalBacktestBatchMetrics
    per_symbol_metrics: list[TechnicalBacktestBatchSymbolMetrics]
    equity_curve: list[TechnicalBacktestBatchEquityPoint]
    monthly_returns: list[TechnicalBacktestBatchMonthlyReturn]
    yearly_returns: list[TechnicalBacktestBatchYearlyReturn]
    warnings: list[str]
    disclaimer: str = "Not financial advice. For informational/educational use only."


def validate_technical_backtest_batch_summary_payload(
    payload: Mapping[str, Any],
) -> TechnicalBacktestBatchSummaryArtifact:
    """Validate writer payloads against the strict summary artifact contract."""
    return TechnicalBacktestBatchSummaryArtifact.model_validate(dict(payload))


__all__ = [
    "TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION",
    "TechnicalBacktestBatchEquityPoint",
    "TechnicalBacktestBatchFailure",
    "TechnicalBacktestBatchMetrics",
    "TechnicalBacktestBatchMonthlyReturn",
    "TechnicalBacktestBatchSummaryArtifact",
    "TechnicalBacktestBatchSymbolMetrics",
    "TechnicalBacktestBatchYearlyReturn",
    "validate_technical_backtest_batch_summary_payload",
]
