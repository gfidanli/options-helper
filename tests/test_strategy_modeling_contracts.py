from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from options_helper.analysis.strategy_modeling_contracts import (
    parse_strategy_equity_points,
    parse_strategy_portfolio_metrics,
    parse_strategy_r_ladder_stats,
    parse_strategy_segment_records,
    parse_strategy_signal_events,
    parse_strategy_trade_simulations,
    serialize_strategy_equity_points,
    serialize_strategy_portfolio_metrics,
    serialize_strategy_r_ladder_stats,
    serialize_strategy_segment_records,
    serialize_strategy_signal_events,
    serialize_strategy_trade_simulations,
)
from options_helper.schemas.strategy_modeling_contracts import (
    StrategyEquityPoint,
    StrategyPortfolioMetrics,
    StrategyRLadderStat,
    StrategySegmentRecord,
    StrategySignalEvent,
    StrategyTradeSimulation,
)


def _ts(day: int, hour: int) -> datetime:
    return datetime(2026, 1, day, hour, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("strategy_id", ("orb", "ma_crossover", "trend_following", "fib_retracement"))
def test_strategy_contracts_accept_extended_strategy_ids(strategy_id: str) -> None:
    event = StrategySignalEvent(
        event_id=f"{strategy_id}-evt-1",
        strategy=strategy_id,  # type: ignore[arg-type]
        symbol="SPY",
        direction="long",
        signal_ts=_ts(3, 16),
        signal_confirmed_ts=_ts(3, 16),
        entry_ts=_ts(4, 9),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
    )
    trade = StrategyTradeSimulation(
        trade_id=f"{strategy_id}-tr-1",
        event_id=event.event_id,
        strategy=strategy_id,  # type: ignore[arg-type]
        symbol="SPY",
        direction="long",
        signal_ts=event.signal_ts,
        signal_confirmed_ts=event.signal_confirmed_ts,
        entry_ts=event.entry_ts,
        entry_price_source=event.entry_price_source,
        entry_price=100.0,
        initial_risk=1.0,
        holding_bars=1,
    )
    assert event.strategy == strategy_id
    assert trade.strategy == strategy_id


@pytest.mark.parametrize(
    "missing_field",
    ("signal_ts", "signal_confirmed_ts", "entry_ts", "entry_price_source"),
)
def test_strategy_signal_event_requires_anti_lookahead_fields(missing_field: str) -> None:
    payload = {
        "event_id": "evt-1",
        "strategy": "sfp",
        "symbol": "SPY",
        "direction": "long",
        "signal_ts": _ts(3, 16),
        "signal_confirmed_ts": _ts(3, 16),
        "entry_ts": _ts(4, 9),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        StrategySignalEvent.model_validate(payload)


@pytest.mark.parametrize(
    "missing_field",
    ("signal_ts", "signal_confirmed_ts", "entry_ts", "entry_price_source"),
)
def test_strategy_trade_simulation_requires_anti_lookahead_fields(missing_field: str) -> None:
    payload = {
        "trade_id": "tr-1",
        "event_id": "evt-1",
        "strategy": "sfp",
        "symbol": "SPY",
        "direction": "long",
        "signal_ts": _ts(3, 16),
        "signal_confirmed_ts": _ts(3, 16),
        "entry_ts": _ts(4, 9),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": 100.0,
        "initial_risk": 2.0,
        "holding_bars": 3,
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        StrategyTradeSimulation.model_validate(payload)


def test_strategy_contract_round_trip_serialization() -> None:
    signal = StrategySignalEvent(
        event_id="evt-1",
        strategy="sfp",
        symbol="SPY",
        direction="long",
        signal_ts=_ts(3, 16),
        signal_confirmed_ts=_ts(3, 16),
        entry_ts=_ts(4, 9),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
        signal_open=100.0,
        signal_high=102.0,
        signal_low=99.0,
        signal_close=101.0,
        stop_price=99.0,
    )
    trade = StrategyTradeSimulation(
        trade_id="tr-1",
        event_id="evt-1",
        strategy="sfp",
        symbol="SPY",
        direction="long",
        signal_ts=signal.signal_ts,
        signal_confirmed_ts=signal.signal_confirmed_ts,
        entry_ts=signal.entry_ts,
        entry_price_source=signal.entry_price_source,
        entry_price=101.2,
        stop_price=99.0,
        target_price=103.2,
        exit_ts=_ts(5, 10),
        exit_price=103.2,
        status="closed",
        exit_reason="target_hit",
        initial_risk=2.2,
        realized_r=1.0,
        mae_r=-0.3,
        mfe_r=1.1,
        holding_bars=5,
    )
    equity = StrategyEquityPoint(ts=_ts(5, 16), equity=10_300.0, cash=8_000.0, drawdown_pct=-0.02)
    ladder = StrategyRLadderStat(
        target_label="1.0R",
        target_r=1.0,
        trade_count=10,
        hit_count=6,
        hit_rate=0.6,
        avg_bars_to_hit=2.3,
        median_bars_to_hit=2.0,
        expectancy_r=0.25,
    )
    segment = StrategySegmentRecord(
        segment_dimension="direction",
        segment_value="long",
        trade_count=10,
        win_rate=0.6,
        avg_realized_r=0.2,
        expectancy_r=0.25,
        sharpe_ratio=1.1,
    )
    metrics = StrategyPortfolioMetrics(
        starting_capital=10_000.0,
        ending_capital=10_900.0,
        total_return_pct=0.09,
        sharpe_ratio=1.2,
        trade_count=10,
        win_rate=0.6,
        loss_rate=0.4,
    )

    for model in (signal, trade, equity, ladder, segment, metrics):
        payload = model.to_dict()
        cloned = type(model).from_dict(payload)
        from_json = type(model).from_json(model.model_dump_json())
        assert cloned == model
        assert from_json == model


def test_contract_helpers_are_deterministic_and_include_anti_lookahead_fields() -> None:
    signal_early = StrategySignalEvent(
        event_id="evt-early",
        strategy="sfp",
        symbol="QQQ",
        direction="short",
        signal_ts=_ts(2, 16),
        signal_confirmed_ts=_ts(2, 16),
        entry_ts=_ts(3, 9),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
    )
    signal_late = StrategySignalEvent(
        event_id="evt-late",
        strategy="sfp",
        symbol="SPY",
        direction="long",
        signal_ts=_ts(4, 16),
        signal_confirmed_ts=_ts(4, 16),
        entry_ts=_ts(5, 9),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
    )
    trade_early = StrategyTradeSimulation(
        trade_id="tr-early",
        event_id="evt-early",
        strategy="sfp",
        symbol="QQQ",
        direction="short",
        signal_ts=signal_early.signal_ts,
        signal_confirmed_ts=signal_early.signal_confirmed_ts,
        entry_ts=signal_early.entry_ts,
        entry_price_source=signal_early.entry_price_source,
        entry_price=100.0,
        initial_risk=2.0,
        holding_bars=2,
    )
    trade_late = StrategyTradeSimulation(
        trade_id="tr-late",
        event_id="evt-late",
        strategy="sfp",
        symbol="SPY",
        direction="long",
        signal_ts=signal_late.signal_ts,
        signal_confirmed_ts=signal_late.signal_confirmed_ts,
        entry_ts=signal_late.entry_ts,
        entry_price_source=signal_late.entry_price_source,
        entry_price=110.0,
        initial_risk=2.5,
        holding_bars=3,
    )

    signals = serialize_strategy_signal_events([signal_late, signal_early])
    trades = serialize_strategy_trade_simulations([trade_late, trade_early])

    assert [row["event_id"] for row in signals] == ["evt-early", "evt-late"]
    assert [row["trade_id"] for row in trades] == ["tr-early", "tr-late"]

    for row in signals + trades:
        assert "signal_ts" in row
        assert "signal_confirmed_ts" in row
        assert "entry_ts" in row
        assert "entry_price_source" in row

    assert [row.event_id for row in parse_strategy_signal_events(signals)] == ["evt-early", "evt-late"]
    assert [row.trade_id for row in parse_strategy_trade_simulations(trades)] == ["tr-early", "tr-late"]

    equity_rows = serialize_strategy_equity_points(
        [
            StrategyEquityPoint(ts=_ts(6, 16), equity=102.0),
            StrategyEquityPoint(ts=_ts(5, 16), equity=101.0),
        ]
    )
    assert [row["equity"] for row in equity_rows] == [101.0, 102.0]
    assert len(parse_strategy_equity_points(equity_rows)) == 2

    ladder_rows = serialize_strategy_r_ladder_stats(
        [
            StrategyRLadderStat(target_label="1.5R", target_r=1.5, trade_count=2, hit_count=1),
            StrategyRLadderStat(target_label="1.0R", target_r=1.0, trade_count=2, hit_count=2),
        ]
    )
    assert [row["target_label"] for row in ladder_rows] == ["1.0R", "1.5R"]
    assert len(parse_strategy_r_ladder_stats(ladder_rows)) == 2

    segment_rows = serialize_strategy_segment_records(
        [
            StrategySegmentRecord(segment_dimension="symbol", segment_value="SPY", trade_count=1),
            StrategySegmentRecord(segment_dimension="direction", segment_value="long", trade_count=1),
        ]
    )
    assert [row["segment_dimension"] for row in segment_rows] == ["direction", "symbol"]
    assert len(parse_strategy_segment_records(segment_rows)) == 2


def test_portfolio_metrics_requires_sharpe_ratio_field() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_portfolio_metrics(
            {
                "starting_capital": 10_000.0,
                "ending_capital": 10_500.0,
                "total_return_pct": 0.05,
                "trade_count": 5,
            }
        )

    metrics = parse_strategy_portfolio_metrics(
        {
            "starting_capital": 10_000.0,
            "ending_capital": 10_500.0,
            "total_return_pct": 0.05,
            "sharpe_ratio": None,
            "trade_count": 5,
        }
    )
    payload = serialize_strategy_portfolio_metrics(metrics)

    assert isinstance(metrics, StrategyPortfolioMetrics)
    assert "sharpe_ratio" in payload
    assert payload["sharpe_ratio"] is None
