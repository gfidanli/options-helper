from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class ScenarioSummaryRow(ArtifactBase):
    symbol: str
    as_of: str
    contract_symbol: str
    option_type: Literal["call", "put"]
    side: Literal["long", "short"]
    contracts: int
    spot: float | None = None
    strike: float
    expiry: str
    mark: float | None = None
    iv: float | None = None
    intrinsic: float | None = None
    extrinsic: float | None = None
    theta_burn_dollars_day: float | None = None
    theta_burn_pct_premium_day: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ScenarioGridRow(ArtifactBase):
    symbol: str
    as_of: str
    contract_symbol: str
    spot_change_pct: float
    iv_change_pp: float
    days_forward: int
    scenario_spot: float
    scenario_iv: float
    days_to_expiry: int
    theoretical_price: float | None = None
    pnl_per_contract: float | None = None
    pnl_position: float | None = None


class ScenariosArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    contract_symbol: str
    disclaimer: str = "Not financial advice."
    summary: ScenarioSummaryRow
    grid: list[ScenarioGridRow] = Field(default_factory=list)
