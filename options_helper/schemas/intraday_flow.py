from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from options_helper.schemas.common import ArtifactBase
from options_helper.schemas.research_metrics_contracts import DeltaBucketName


class IntradayFlowContractRow(ArtifactBase):
    symbol: str
    market_date: str
    source: str
    contract_symbol: str
    expiry: str | None = None
    option_type: Literal["call", "put"] | None = None
    strike: float | None = None
    delta_bucket: DeltaBucketName | None = None
    buy_volume: float
    sell_volume: float
    unknown_volume: float
    buy_notional: float
    sell_notional: float
    net_notional: float
    trade_count: int
    unknown_trade_share: float
    quote_coverage_pct: float
    warnings: list[str] = Field(default_factory=list)


class IntradayFlowTimeBucketRow(ArtifactBase):
    symbol: str
    market_date: str
    bucket_start_utc: datetime
    bucket_minutes: int
    contract_symbol: str
    expiry: str | None = None
    option_type: Literal["call", "put"] | None = None
    strike: float | None = None
    delta_bucket: DeltaBucketName | None = None
    buy_notional: float
    sell_notional: float
    net_notional: float
    trade_count: int
    unknown_trade_share: float


class IntradayFlowArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    market_date: str
    source: str
    bucket_minutes: int
    disclaimer: str = "Not financial advice."
    contract_flow: list[IntradayFlowContractRow] = Field(default_factory=list)
    time_buckets: list[IntradayFlowTimeBucketRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
