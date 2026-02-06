from __future__ import annotations

from datetime import datetime

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class TailRiskConfigArtifact(ArtifactBase):
    lookback_days: int
    horizon_days: int
    num_simulations: int
    seed: int
    var_confidence: float
    end_percentiles: list[float] = Field(default_factory=list)
    chart_percentiles: list[float] = Field(default_factory=list)
    sample_paths: int
    trading_days_per_year: int


class TailRiskPercentileRow(ArtifactBase):
    percentile: float
    price: float | None = None
    return_pct: float | None = None


class TailRiskIVContext(ArtifactBase):
    label: str
    reason: str
    iv_rv_20d: float | None = None
    low: float
    high: float
    atm_iv_near: float | None = None
    rv_20d: float | None = None
    atm_iv_near_percentile: float | None = None
    iv_term_slope: float | None = None


class TailRiskArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    disclaimer: str = "Not financial advice."
    config: TailRiskConfigArtifact
    spot: float
    realized_vol_annual: float
    expected_return_annual: float
    var_return: float
    cvar_return: float | None = None
    end_percentiles: list[TailRiskPercentileRow] = Field(default_factory=list)
    iv_context: TailRiskIVContext | None = None
    warnings: list[str] = Field(default_factory=list)

