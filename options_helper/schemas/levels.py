from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class LevelsSummaryRow(ArtifactBase):
    symbol: str
    as_of: str
    spot: float | None = None
    prev_close: float | None = None
    session_open: float | None = None
    gap_pct: float | None = None
    prior_high: float | None = None
    prior_low: float | None = None
    rolling_high: float | None = None
    rolling_low: float | None = None
    rs_ratio: float | None = None
    beta_20d: float | None = None
    corr_20d: float | None = None
    warnings: list[str] = Field(default_factory=list)


class LevelsAnchoredVwapRow(ArtifactBase):
    symbol: str
    as_of: str
    anchor_id: str
    anchor_type: Literal["session_open", "timestamp", "date", "breakout_day"]
    anchor_ts_utc: datetime | None = None
    anchor_price: float | None = None
    anchored_vwap: float | None = None
    distance_from_spot_pct: float | None = None


class LevelsVolumeProfileRow(ArtifactBase):
    symbol: str
    as_of: str
    price_bin_low: float
    price_bin_high: float
    volume: float
    volume_pct: float
    is_poc: bool
    is_hvn: bool
    is_lvn: bool


class LevelsArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    disclaimer: str = "Not financial advice."
    summary: LevelsSummaryRow
    anchored_vwap: list[LevelsAnchoredVwapRow] = Field(default_factory=list)
    volume_profile: list[LevelsVolumeProfileRow] = Field(default_factory=list)
    volume_profile_poc: float | None = None
    volume_profile_hvn_candidates: list[float] = Field(default_factory=list)
    volume_profile_lvn_candidates: list[float] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
