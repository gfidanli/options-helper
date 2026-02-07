from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase
from options_helper.schemas.research_metrics_contracts import SIGNED_EXPOSURE_CONVENTION, SignedExposureConvention


class ExposureStrikeRow(ArtifactBase):
    symbol: str
    as_of: str
    expiry: str
    strike: float
    call_oi: float
    put_oi: float
    call_gex: float | None = None
    put_gex: float | None = None
    net_gex: float | None = None


class ExposureSummaryRow(ArtifactBase):
    symbol: str
    as_of: str
    spot: float
    flip_strike: float | None = None
    total_call_gex: float | None = None
    total_put_gex: float | None = None
    total_net_gex: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ExposureTopLevelRow(ArtifactBase):
    strike: float
    net_gex: float
    abs_net_gex: float


class ExposureSliceArtifact(ArtifactBase):
    mode: Literal["near", "monthly", "all"]
    available_expiries: list[str] = Field(default_factory=list)
    included_expiries: list[str] = Field(default_factory=list)
    strike_rows: list[ExposureStrikeRow] = Field(default_factory=list)
    summary: ExposureSummaryRow
    top_abs_net_levels: list[ExposureTopLevelRow] = Field(default_factory=list)


class ExposureArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    spot: float
    signed_exposure_convention: SignedExposureConvention = SIGNED_EXPOSURE_CONVENTION
    disclaimer: str = "Not financial advice."
    slices: list[ExposureSliceArtifact] = Field(default_factory=list)
