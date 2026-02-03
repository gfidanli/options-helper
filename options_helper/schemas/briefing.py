from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class BriefingTechnicals(ArtifactBase):
    source: str
    config_path: str | None = None


class BriefingPortfolio(ArtifactBase):
    exposure: dict[str, Any] | None = None
    stress: list[dict[str, Any]] = Field(default_factory=list)


class BriefingSection(ArtifactBase):
    symbol: str
    as_of: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    derived_updated: bool = False
    next_earnings_date: str | None = None
    technicals: dict[str, Any] | None = None
    confluence: dict[str, Any] | None = None
    chain: dict[str, Any] | None = None
    derived: dict[str, Any] | None = None
    compare: dict[str, Any] | None = None
    flow_net: list[dict[str, Any]] | None = None
    quote_quality: dict[str, Any] | None = None


class BriefingArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    disclaimer: str
    report_date: str
    portfolio_path: str
    symbols: list[str]
    top: int
    technicals: BriefingTechnicals
    portfolio: BriefingPortfolio
    sections: list[BriefingSection] = Field(default_factory=list)
