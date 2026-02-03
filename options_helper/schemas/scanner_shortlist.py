from __future__ import annotations

from datetime import datetime

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class ScannerShortlistRow(ArtifactBase):
    symbol: str
    score: float | None = None
    coverage: float | None = None
    top_reasons: str | None = None


class ScannerShortlistArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    run_id: str
    universe: str
    tail_low_pct: float
    tail_high_pct: float
    all_watchlist_name: str
    shortlist_watchlist_name: str
    rows: list[ScannerShortlistRow] = Field(default_factory=list)
