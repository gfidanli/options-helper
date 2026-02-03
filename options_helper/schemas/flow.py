from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class FlowNetRow(ArtifactBase):
    contract_symbol: str | None = None
    expiry: str | None = None
    option_type: str | None = None
    strike: float | None = None
    delta_oi: float | None = None
    delta_oi_notional: float | None = None
    volume_notional: float | None = None
    delta_notional: float | None = None
    n_pairs: int | None = None


class FlowArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    from_date: str
    to_date: str
    window: int
    group_by: Literal["contract", "strike", "expiry", "expiry-strike"]
    snapshot_dates: list[str] = Field(default_factory=list)
    net: list[FlowNetRow] = Field(default_factory=list)
