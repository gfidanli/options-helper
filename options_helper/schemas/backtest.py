from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from options_helper.schemas.common import ArtifactBase, utc_now


class BacktestTradeRow(BaseModel):
    symbol: str
    contract_symbol: str | None = None
    expiry: date
    strike: float
    option_type: str
    quantity: int
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    holding_days: int
    pnl: float
    pnl_pct: float | None
    max_favorable: float | None
    max_adverse: float | None


class BacktestSkipRow(BaseModel):
    as_of: date
    action: str
    reason: str


class BacktestRollRow(BaseModel):
    as_of: date
    from_contract_symbol: str | None
    to_contract_symbol: str | None
    reason: str | None = None


class BacktestSummaryStats(BaseModel):
    total_pnl: float
    total_pnl_pct: float | None
    trade_count: int
    win_rate: float | None
    avg_pnl: float | None
    avg_pnl_pct: float | None


class OpenPositionSummary(BaseModel):
    contract_symbol: str | None
    expiry: date
    strike: float
    option_type: str
    quantity: int
    entry_date: date
    entry_price: float
    max_favorable: float | None
    max_adverse: float | None


class BacktestSummaryArtifact(ArtifactBase):
    schema_version: int = Field(default=1)
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    symbol: str
    contract_symbol: str | None
    start: date | None
    end: date | None
    strategy_name: str
    fill_mode: str
    slippage_factor: float
    quantity: int
    initial_cash: float
    final_cash: float
    stats: BacktestSummaryStats
    skips: list[BacktestSkipRow] = Field(default_factory=list)
    rolls: list[BacktestRollRow] = Field(default_factory=list)
    open_position: OpenPositionSummary | None = None
