from __future__ import annotations

from datetime import datetime, timezone

import pytest

from options_helper.analysis.strategy_metrics import (
    compute_strategy_metrics,
    compute_strategy_portfolio_metrics,
)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _trade(
    *,
    trade_id: str,
    entry_ts: str,
    exit_ts: str,
    entry_price: float,
    exit_price: float,
    initial_risk: float,
    target_r: float,
    exit_reason: str,
    holding_bars: int,
    direction: str = "long",
    symbol: str = "SPY",
) -> dict[str, object]:
    if direction == "long":
        target_price = entry_price + (initial_risk * target_r)
        realized_r = (exit_price - entry_price) / initial_risk
    else:
        target_price = entry_price - (initial_risk * target_r)
        realized_r = (entry_price - exit_price) / initial_risk

    return {
        "trade_id": trade_id,
        "event_id": f"evt-{trade_id}",
        "strategy": "sfp",
        "symbol": symbol,
        "direction": direction,
        "signal_ts": _ts("2025-01-01T21:00:00Z"),
        "signal_confirmed_ts": _ts("2025-01-01T21:00:00Z"),
        "entry_ts": _ts(entry_ts),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": entry_price,
        "stop_price": entry_price - initial_risk if direction == "long" else entry_price + initial_risk,
        "target_price": target_price,
        "exit_ts": _ts(exit_ts),
        "exit_price": exit_price,
        "status": "closed",
        "exit_reason": exit_reason,
        "reject_code": None,
        "initial_risk": initial_risk,
        "realized_r": realized_r,
        "mae_r": min(0.0, realized_r),
        "mfe_r": max(0.0, realized_r),
        "holding_bars": holding_bars,
        "gap_fill_applied": False,
    }


def _equity(
    *,
    ts: str,
    equity: float,
    open_trade_count: int,
    closed_trade_count: int,
) -> dict[str, object]:
    return {
        "ts": _ts(ts),
        "equity": equity,
        "cash": equity,
        "drawdown_pct": None,
        "open_trade_count": open_trade_count,
        "closed_trade_count": closed_trade_count,
    }


def test_compute_strategy_metrics_core_formulas_and_target_hit_rates() -> None:
    trades = [
        _trade(
            trade_id="evt-a:1.0R",
            entry_ts="2025-01-06T14:30:00Z",
            exit_ts="2025-01-06T16:00:00Z",
            entry_price=100.0,
            exit_price=102.0,
            initial_risk=2.0,
            target_r=1.0,
            exit_reason="target_hit",
            holding_bars=2,
        ),
        _trade(
            trade_id="evt-b:1.0R",
            entry_ts="2025-03-10T14:30:00Z",
            exit_ts="2025-03-10T16:00:00Z",
            entry_price=100.0,
            exit_price=98.0,
            initial_risk=2.0,
            target_r=1.0,
            exit_reason="stop_hit",
            holding_bars=3,
        ),
        _trade(
            trade_id="evt-c:2.0R",
            entry_ts="2025-06-09T14:30:00Z",
            exit_ts="2025-06-09T16:00:00Z",
            entry_price=50.0,
            exit_price=54.0,
            initial_risk=2.0,
            target_r=2.0,
            exit_reason="target_hit",
            holding_bars=4,
        ),
        _trade(
            trade_id="evt-d:2.0R",
            entry_ts="2025-09-08T14:30:00Z",
            exit_ts="2025-09-08T16:00:00Z",
            entry_price=60.0,
            exit_price=59.0,
            initial_risk=2.0,
            target_r=2.0,
            exit_reason="time_stop",
            holding_bars=5,
        ),
    ]

    equity_curve = [
        _equity(ts="2025-01-01T00:00:00Z", equity=10_000.0, open_trade_count=1, closed_trade_count=0),
        _equity(ts="2025-04-01T00:00:00Z", equity=10_100.0, open_trade_count=1, closed_trade_count=1),
        _equity(ts="2025-07-01T00:00:00Z", equity=9_900.0, open_trade_count=0, closed_trade_count=2),
        _equity(ts="2025-10-01T00:00:00Z", equity=10_400.0, open_trade_count=1, closed_trade_count=3),
        _equity(ts="2026-01-01T00:00:00Z", equity=10_300.0, open_trade_count=0, closed_trade_count=4),
    ]

    result = compute_strategy_metrics(trades, equity_curve)
    metrics = result.portfolio_metrics

    assert metrics.starting_capital == pytest.approx(10_000.0)
    assert metrics.ending_capital == pytest.approx(10_300.0)
    assert metrics.total_return_pct == pytest.approx(0.03)
    assert metrics.cagr_pct == pytest.approx(0.030020853338703857)
    assert metrics.max_drawdown_pct == pytest.approx(-0.01980198019801982)
    assert metrics.sharpe_ratio == pytest.approx(0.578107475927207)
    assert metrics.sortino_ratio == pytest.approx(3.0528678678438874)
    assert metrics.calmar_ratio == pytest.approx(1.5160530936045433)

    assert metrics.trade_count == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.loss_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.expectancy_r == pytest.approx(0.375)
    assert metrics.avg_realized_r == pytest.approx(0.375)
    assert metrics.avg_hold_bars == pytest.approx(3.5)
    assert metrics.exposure_pct == pytest.approx(273.0 / 365.0)
    assert result.expectancy_dollars == pytest.approx(0.75)

    assert len(result.target_hit_rates) == 2

    one_r, two_r = result.target_hit_rates
    assert one_r.target_label == "1.0R"
    assert one_r.target_r == pytest.approx(1.0)
    assert one_r.trade_count == 2
    assert one_r.hit_count == 1
    assert one_r.hit_rate == pytest.approx(0.5)
    assert one_r.avg_bars_to_hit == pytest.approx(2.0)
    assert one_r.median_bars_to_hit == pytest.approx(2.0)
    assert one_r.expectancy_r == pytest.approx(0.0)

    assert two_r.target_label == "2.0R"
    assert two_r.target_r == pytest.approx(2.0)
    assert two_r.trade_count == 2
    assert two_r.hit_count == 1
    assert two_r.hit_rate == pytest.approx(0.5)
    assert two_r.avg_bars_to_hit == pytest.approx(4.0)
    assert two_r.median_bars_to_hit == pytest.approx(4.0)
    assert two_r.expectancy_r == pytest.approx(0.75)


def test_compute_strategy_portfolio_metrics_handles_zero_variance_and_no_losers() -> None:
    trades = [
        _trade(
            trade_id="evt-e:1.0R",
            entry_ts="2025-01-06T14:30:00Z",
            exit_ts="2025-01-06T16:00:00Z",
            entry_price=20.0,
            exit_price=21.0,
            initial_risk=1.0,
            target_r=1.0,
            exit_reason="target_hit",
            holding_bars=1,
        ),
        _trade(
            trade_id="evt-f:1.0R",
            entry_ts="2025-02-06T14:30:00Z",
            exit_ts="2025-02-06T16:00:00Z",
            entry_price=20.0,
            exit_price=21.0,
            initial_risk=1.0,
            target_r=1.0,
            exit_reason="target_hit",
            holding_bars=1,
        ),
    ]
    equity_curve = [
        _equity(ts="2025-01-01T00:00:00Z", equity=100.0, open_trade_count=1, closed_trade_count=0),
        _equity(ts="2025-02-01T00:00:00Z", equity=110.0, open_trade_count=1, closed_trade_count=1),
        _equity(ts="2025-03-01T00:00:00Z", equity=121.0, open_trade_count=0, closed_trade_count=2),
    ]

    metrics = compute_strategy_portfolio_metrics(trades, equity_curve)

    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.max_drawdown_pct == pytest.approx(0.0)
    assert metrics.calmar_ratio is None

    assert metrics.trade_count == 2
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.loss_rate == pytest.approx(0.0)
    assert metrics.profit_factor is None


def test_compute_strategy_portfolio_metrics_no_winners_returns_zero_profit_factor() -> None:
    trades = [
        _trade(
            trade_id="evt-g:1.0R",
            entry_ts="2025-01-06T14:30:00Z",
            exit_ts="2025-01-06T16:00:00Z",
            entry_price=20.0,
            exit_price=19.0,
            initial_risk=1.0,
            target_r=1.0,
            exit_reason="stop_hit",
            holding_bars=1,
        ),
    ]
    equity_curve = [
        _equity(ts="2025-01-01T00:00:00Z", equity=100.0, open_trade_count=1, closed_trade_count=0),
        _equity(ts="2025-02-01T00:00:00Z", equity=99.0, open_trade_count=0, closed_trade_count=1),
    ]

    metrics = compute_strategy_portfolio_metrics(trades, equity_curve)

    assert metrics.trade_count == 1
    assert metrics.win_rate == pytest.approx(0.0)
    assert metrics.loss_rate == pytest.approx(1.0)
    assert metrics.profit_factor == pytest.approx(0.0)


def test_compute_strategy_metrics_handles_sparse_inputs_without_crashing() -> None:
    result = compute_strategy_metrics([], [], starting_capital=10_000.0)
    metrics = result.portfolio_metrics

    assert metrics.starting_capital == pytest.approx(10_000.0)
    assert metrics.ending_capital == pytest.approx(10_000.0)
    assert metrics.total_return_pct == pytest.approx(0.0)
    assert metrics.cagr_pct is None
    assert metrics.max_drawdown_pct is None
    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.calmar_ratio is None

    assert metrics.trade_count == 0
    assert metrics.win_rate is None
    assert metrics.loss_rate is None
    assert metrics.profit_factor is None
    assert metrics.expectancy_r is None
    assert metrics.avg_realized_r is None
    assert metrics.avg_hold_bars is None
    assert metrics.exposure_pct is None

    assert result.expectancy_dollars is None
    assert result.target_hit_rates == ()
