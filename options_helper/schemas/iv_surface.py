from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase
from options_helper.schemas.research_metrics_contracts import DeltaBucketName, IV_SURFACE_TENOR_TARGETS_DTE


class IvSurfaceTenorRow(ArtifactBase):
    symbol: str
    as_of: str
    tenor_target_dte: int
    expiry: str | None = None
    dte: int | None = None
    tenor_gap_dte: int | None = None
    atm_strike: float | None = None
    atm_iv: float | None = None
    atm_mark: float | None = None
    straddle_mark: float | None = None
    expected_move_pct: float | None = None
    skew_25d_pp: float | None = None
    skew_10d_pp: float | None = None
    contracts_used: int
    warnings: list[str] = Field(default_factory=list)


class IvSurfaceDeltaBucketRow(ArtifactBase):
    symbol: str
    as_of: str
    tenor_target_dte: int
    expiry: str | None = None
    option_type: Literal["call", "put"]
    delta_bucket: DeltaBucketName
    avg_iv: float | None = None
    median_iv: float | None = None
    n_contracts: int
    warnings: list[str] = Field(default_factory=list)


class IvSurfaceTenorChangeRow(ArtifactBase):
    symbol: str
    as_of: str
    tenor_target_dte: int
    expiry: str | None = None
    dte: int | None = None
    atm_iv_change_pp: float | None = None
    atm_mark_change: float | None = None
    straddle_mark_change: float | None = None
    expected_move_pct_change_pp: float | None = None
    skew_25d_pp_change: float | None = None
    skew_10d_pp_change: float | None = None
    contracts_used_change: int | None = None
    warnings: list[str] = Field(default_factory=list)


class IvSurfaceDeltaBucketChangeRow(ArtifactBase):
    symbol: str
    as_of: str
    tenor_target_dte: int
    expiry: str | None = None
    option_type: Literal["call", "put"]
    delta_bucket: DeltaBucketName
    avg_iv_change_pp: float | None = None
    median_iv_change_pp: float | None = None
    n_contracts_change: int | None = None
    warnings: list[str] = Field(default_factory=list)


class IvSurfaceArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    spot: float | None = None
    disclaimer: str = "Not financial advice."
    tenor_targets_dte: list[int] = Field(default_factory=lambda: list(IV_SURFACE_TENOR_TARGETS_DTE))
    tenor: list[IvSurfaceTenorRow] = Field(default_factory=list)
    delta_buckets: list[IvSurfaceDeltaBucketRow] = Field(default_factory=list)
    tenor_changes: list[IvSurfaceTenorChangeRow] = Field(default_factory=list)
    delta_bucket_changes: list[IvSurfaceDeltaBucketChangeRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
