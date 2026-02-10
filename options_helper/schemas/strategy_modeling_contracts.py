from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field

from options_helper.schemas.common import ArtifactBase


StrategyId = Literal["sfp", "msb", "orb", "ma_crossover", "trend_following"]
SignalDirection = Literal["long", "short"]
EntryPriceSource = Literal[
    "first_tradable_bar_open_after_signal_confirmed_ts",
    "next_bar_open",
    "session_open",
]

TradeStatus = Literal["closed", "open", "rejected"]
TradeExitReason = Literal["target_hit", "stop_hit", "time_stop", "signal_flip", "forced_close"]
TradeRejectCode = Literal[
    "invalid_signal",
    "missing_intraday_coverage",
    "missing_entry_bar",
    "entry_open_outside_signal_range",
    "non_positive_risk",
    "insufficient_future_bars",
]

SegmentDimension = Literal[
    "symbol",
    "direction",
    "extension_bucket",
    "rsi_regime",
    "rsi_divergence",
    "volatility_regime",
    "bars_since_swing_bucket",
]


STRATEGY_SIGNAL_EVENT_FIELDS: Final[tuple[str, ...]] = (
    "event_id",
    "strategy",
    "symbol",
    "timeframe",
    "direction",
    "signal_ts",
    "signal_confirmed_ts",
    "entry_ts",
    "entry_price_source",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "stop_price",
    "notes",
)

STRATEGY_TRADE_SIMULATION_FIELDS: Final[tuple[str, ...]] = (
    "trade_id",
    "event_id",
    "strategy",
    "symbol",
    "direction",
    "signal_ts",
    "signal_confirmed_ts",
    "entry_ts",
    "entry_price_source",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_ts",
    "exit_price",
    "status",
    "exit_reason",
    "reject_code",
    "initial_risk",
    "realized_r",
    "mae_r",
    "mfe_r",
    "holding_bars",
    "gap_fill_applied",
)

STRATEGY_EQUITY_POINT_FIELDS: Final[tuple[str, ...]] = (
    "ts",
    "equity",
    "cash",
    "drawdown_pct",
    "open_trade_count",
    "closed_trade_count",
)

STRATEGY_R_LADDER_STAT_FIELDS: Final[tuple[str, ...]] = (
    "target_label",
    "target_r",
    "trade_count",
    "hit_count",
    "hit_rate",
    "avg_bars_to_hit",
    "median_bars_to_hit",
    "expectancy_r",
)

STRATEGY_SEGMENT_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "segment_dimension",
    "segment_value",
    "trade_count",
    "win_rate",
    "avg_realized_r",
    "expectancy_r",
    "profit_factor",
    "sharpe_ratio",
    "max_drawdown_pct",
)

STRATEGY_PORTFOLIO_METRICS_FIELDS: Final[tuple[str, ...]] = (
    "starting_capital",
    "ending_capital",
    "total_return_pct",
    "cagr_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "expectancy_r",
    "avg_realized_r",
    "trade_count",
    "win_rate",
    "loss_rate",
    "avg_hold_bars",
    "exposure_pct",
)


class StrategyModelingContractBase(ArtifactBase):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


class StrategySignalEvent(StrategyModelingContractBase):
    event_id: str
    strategy: StrategyId
    symbol: str
    timeframe: str = "1d"
    direction: SignalDirection
    signal_ts: datetime
    signal_confirmed_ts: datetime
    entry_ts: datetime
    entry_price_source: EntryPriceSource
    signal_open: float | None = None
    signal_high: float | None = None
    signal_low: float | None = None
    signal_close: float | None = None
    stop_price: float | None = None
    notes: list[str] = Field(default_factory=list)


class StrategyTradeSimulation(StrategyModelingContractBase):
    trade_id: str
    event_id: str
    strategy: StrategyId
    symbol: str
    direction: SignalDirection
    signal_ts: datetime
    signal_confirmed_ts: datetime
    entry_ts: datetime
    entry_price_source: EntryPriceSource
    entry_price: float
    stop_price: float | None = None
    target_price: float | None = None
    exit_ts: datetime | None = None
    exit_price: float | None = None
    status: TradeStatus = "closed"
    exit_reason: TradeExitReason | None = None
    reject_code: TradeRejectCode | None = None
    initial_risk: float
    realized_r: float | None = None
    mae_r: float | None = None
    mfe_r: float | None = None
    holding_bars: int = Field(ge=0)
    gap_fill_applied: bool = False


class StrategyEquityPoint(StrategyModelingContractBase):
    ts: datetime
    equity: float
    cash: float | None = None
    drawdown_pct: float | None = None
    open_trade_count: int = Field(default=0, ge=0)
    closed_trade_count: int = Field(default=0, ge=0)


class StrategyRLadderStat(StrategyModelingContractBase):
    target_label: str
    target_r: float
    trade_count: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    hit_rate: float | None = None
    avg_bars_to_hit: float | None = None
    median_bars_to_hit: float | None = None
    expectancy_r: float | None = None


class StrategySegmentRecord(StrategyModelingContractBase):
    segment_dimension: SegmentDimension
    segment_value: str
    trade_count: int = Field(ge=0)
    win_rate: float | None = None
    avg_realized_r: float | None = None
    expectancy_r: float | None = None
    profit_factor: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None


class StrategyPortfolioMetrics(StrategyModelingContractBase):
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    cagr_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe_ratio: float | None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    profit_factor: float | None = None
    expectancy_r: float | None = None
    avg_realized_r: float | None = None
    trade_count: int = Field(ge=0)
    win_rate: float | None = None
    loss_rate: float | None = None
    avg_hold_bars: float | None = None
    exposure_pct: float | None = None


def contract_field_names() -> dict[str, tuple[str, ...]]:
    return {
        "signal_event": STRATEGY_SIGNAL_EVENT_FIELDS,
        "trade_simulation": STRATEGY_TRADE_SIMULATION_FIELDS,
        "equity_point": STRATEGY_EQUITY_POINT_FIELDS,
        "r_ladder_stat": STRATEGY_R_LADDER_STAT_FIELDS,
        "segment_record": STRATEGY_SEGMENT_RECORD_FIELDS,
        "portfolio_metrics": STRATEGY_PORTFOLIO_METRICS_FIELDS,
    }


def contract_defaults() -> dict[str, Any]:
    return {
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "timeframe": "1d",
        "trade_status": "closed",
    }


__all__ = [
    "EntryPriceSource",
    "SegmentDimension",
    "StrategyId",
    "SignalDirection",
    "StrategyEquityPoint",
    "StrategyModelingContractBase",
    "StrategyPortfolioMetrics",
    "StrategyRLadderStat",
    "StrategySegmentRecord",
    "StrategySignalEvent",
    "StrategyTradeSimulation",
    "STRATEGY_EQUITY_POINT_FIELDS",
    "STRATEGY_PORTFOLIO_METRICS_FIELDS",
    "STRATEGY_R_LADDER_STAT_FIELDS",
    "STRATEGY_SEGMENT_RECORD_FIELDS",
    "STRATEGY_SIGNAL_EVENT_FIELDS",
    "STRATEGY_TRADE_SIMULATION_FIELDS",
    "TradeExitReason",
    "TradeRejectCode",
    "TradeStatus",
    "contract_defaults",
    "contract_field_names",
]
