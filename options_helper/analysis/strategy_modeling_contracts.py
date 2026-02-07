from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, TypeVar

from options_helper.schemas.strategy_modeling_contracts import (
    StrategyEquityPoint,
    StrategyPortfolioMetrics,
    StrategyRLadderStat,
    StrategySegmentRecord,
    StrategySignalEvent,
    StrategyTradeSimulation,
)


ContractT = TypeVar(
    "ContractT",
    StrategySignalEvent,
    StrategyTradeSimulation,
    StrategyEquityPoint,
    StrategyRLadderStat,
    StrategySegmentRecord,
)


def parse_strategy_signal_events(
    payloads: Iterable[Mapping[str, Any] | StrategySignalEvent],
) -> list[StrategySignalEvent]:
    return _parse_contract_rows(payloads, StrategySignalEvent)


def parse_strategy_trade_simulations(
    payloads: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
) -> list[StrategyTradeSimulation]:
    return _parse_contract_rows(payloads, StrategyTradeSimulation)


def parse_strategy_equity_points(
    payloads: Iterable[Mapping[str, Any] | StrategyEquityPoint],
) -> list[StrategyEquityPoint]:
    return _parse_contract_rows(payloads, StrategyEquityPoint)


def parse_strategy_r_ladder_stats(
    payloads: Iterable[Mapping[str, Any] | StrategyRLadderStat],
) -> list[StrategyRLadderStat]:
    return _parse_contract_rows(payloads, StrategyRLadderStat)


def parse_strategy_segment_records(
    payloads: Iterable[Mapping[str, Any] | StrategySegmentRecord],
) -> list[StrategySegmentRecord]:
    return _parse_contract_rows(payloads, StrategySegmentRecord)


def parse_strategy_portfolio_metrics(
    payload: Mapping[str, Any] | StrategyPortfolioMetrics,
) -> StrategyPortfolioMetrics:
    if isinstance(payload, StrategyPortfolioMetrics):
        return payload
    return StrategyPortfolioMetrics.model_validate(dict(payload))


def serialize_strategy_signal_events(events: Sequence[StrategySignalEvent]) -> list[dict[str, Any]]:
    rows = sorted(
        events,
        key=lambda row: (
            row.signal_confirmed_ts,
            row.signal_ts,
            row.entry_ts,
            row.symbol,
            row.event_id,
        ),
    )
    return [row.to_dict() for row in rows]


def serialize_strategy_trade_simulations(
    trades: Sequence[StrategyTradeSimulation],
) -> list[dict[str, Any]]:
    rows = sorted(
        trades,
        key=lambda row: (
            row.entry_ts,
            row.signal_confirmed_ts,
            row.signal_ts,
            row.symbol,
            row.trade_id,
        ),
    )
    return [row.to_dict() for row in rows]


def serialize_strategy_equity_points(
    equity_curve: Sequence[StrategyEquityPoint],
) -> list[dict[str, Any]]:
    return [row.to_dict() for row in sorted(equity_curve, key=lambda row: row.ts)]


def serialize_strategy_r_ladder_stats(
    ladder_stats: Sequence[StrategyRLadderStat],
) -> list[dict[str, Any]]:
    rows = sorted(ladder_stats, key=lambda row: (row.target_r, row.target_label))
    return [row.to_dict() for row in rows]


def serialize_strategy_segment_records(
    segment_rows: Sequence[StrategySegmentRecord],
) -> list[dict[str, Any]]:
    rows = sorted(segment_rows, key=lambda row: (row.segment_dimension, row.segment_value))
    return [row.to_dict() for row in rows]


def serialize_strategy_portfolio_metrics(metrics: StrategyPortfolioMetrics) -> dict[str, Any]:
    return metrics.to_dict()


def _parse_contract_rows(
    payloads: Iterable[Mapping[str, Any] | ContractT],
    model_cls: type[ContractT],
) -> list[ContractT]:
    rows: list[ContractT] = []
    for payload in payloads:
        if isinstance(payload, model_cls):
            rows.append(payload)
            continue
        rows.append(model_cls.model_validate(dict(payload)))
    return rows


__all__ = [
    "parse_strategy_equity_points",
    "parse_strategy_portfolio_metrics",
    "parse_strategy_r_ladder_stats",
    "parse_strategy_segment_records",
    "parse_strategy_signal_events",
    "parse_strategy_trade_simulations",
    "serialize_strategy_equity_points",
    "serialize_strategy_portfolio_metrics",
    "serialize_strategy_r_ladder_stats",
    "serialize_strategy_segment_records",
    "serialize_strategy_signal_events",
    "serialize_strategy_trade_simulations",
]
