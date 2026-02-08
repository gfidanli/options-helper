from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping

import pandas as pd

from options_helper.analysis.strategy_modeling_contracts import (
    parse_strategy_equity_points,
    parse_strategy_trade_simulations,
)
from options_helper.schemas.strategy_modeling_contracts import (
    StrategyEquityPoint,
    StrategyPortfolioMetrics,
    StrategyRLadderStat,
    StrategyTradeSimulation,
)

_EPSILON = 1e-12
_SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class StrategyMetricsResult:
    portfolio_metrics: StrategyPortfolioMetrics
    target_hit_rates: tuple[StrategyRLadderStat, ...]
    expectancy_dollars: float | None = None


def compute_strategy_metrics(
    trades: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
    equity_curve: Iterable[Mapping[str, Any] | StrategyEquityPoint],
    *,
    starting_capital: float | None = None,
    annualization_periods: float | None = None,
) -> StrategyMetricsResult:
    """Compute deterministic portfolio + target-level metrics from simulated trades and equity."""

    parsed_trades = parse_strategy_trade_simulations(trades)
    parsed_equity = parse_strategy_equity_points(equity_curve)

    portfolio_metrics = _compute_portfolio_metrics_from_parsed(
        parsed_trades,
        parsed_equity,
        starting_capital=starting_capital,
        annualization_periods=annualization_periods,
    )
    target_hit_rates = compute_strategy_target_hit_rates(parsed_trades)
    expectancy_dollars = compute_strategy_expectancy_dollars(parsed_trades)

    return StrategyMetricsResult(
        portfolio_metrics=portfolio_metrics,
        target_hit_rates=target_hit_rates,
        expectancy_dollars=expectancy_dollars,
    )


def compute_strategy_portfolio_metrics(
    trades: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
    equity_curve: Iterable[Mapping[str, Any] | StrategyEquityPoint],
    *,
    starting_capital: float | None = None,
    annualization_periods: float | None = None,
) -> StrategyPortfolioMetrics:
    parsed_trades = parse_strategy_trade_simulations(trades)
    parsed_equity = parse_strategy_equity_points(equity_curve)
    return _compute_portfolio_metrics_from_parsed(
        parsed_trades,
        parsed_equity,
        starting_capital=starting_capital,
        annualization_periods=annualization_periods,
    )


def compute_strategy_target_hit_rates(
    trades: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
) -> tuple[StrategyRLadderStat, ...]:
    parsed_trades = parse_strategy_trade_simulations(trades)

    buckets: dict[float, dict[str, Any]] = {}
    for trade in parsed_trades:
        if str(trade.status).strip().lower() != "closed":
            continue

        target_r = _target_r_for_trade(trade)
        if target_r is None:
            continue

        key = round(target_r, 6)
        bucket = buckets.setdefault(
            key,
            {
                "target_r": target_r,
                "trade_count": 0,
                "hit_count": 0,
                "hit_bars": [],
                "realized_r_values": [],
            },
        )

        bucket["trade_count"] += 1
        if str(trade.exit_reason).strip().lower() == "target_hit":
            bucket["hit_count"] += 1
            bucket["hit_bars"].append(int(max(0, trade.holding_bars)))

        realized_r = _finite_float(trade.realized_r)
        if realized_r is not None:
            bucket["realized_r_values"].append(realized_r)

    rows: list[StrategyRLadderStat] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        trade_count = int(bucket["trade_count"])
        hit_count = int(bucket["hit_count"])
        hit_bars = list(bucket["hit_bars"])
        realized_r_values = list(bucket["realized_r_values"])

        hit_rate = (hit_count / trade_count) if trade_count > 0 else None
        avg_bars_to_hit = mean(hit_bars) if hit_bars else None
        median_bars_to_hit = median(hit_bars) if hit_bars else None
        expectancy_r = mean(realized_r_values) if realized_r_values else None

        rows.append(
            StrategyRLadderStat(
                target_label=_format_target_label(float(bucket["target_r"])),
                target_r=float(bucket["target_r"]),
                trade_count=trade_count,
                hit_count=hit_count,
                hit_rate=hit_rate,
                avg_bars_to_hit=avg_bars_to_hit,
                median_bars_to_hit=median_bars_to_hit,
                expectancy_r=expectancy_r,
            )
        )

    return tuple(rows)


def compute_strategy_expectancy_dollars(
    trades: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
) -> float | None:
    parsed_trades = parse_strategy_trade_simulations(trades)
    closed_trades = _closed_trades(parsed_trades)
    pnl_values = [_trade_pnl_per_unit(trade) for trade in closed_trades]
    finite_pnl_values = [value for value in pnl_values if value is not None]
    if not finite_pnl_values:
        return None
    return mean(finite_pnl_values)


def _compute_portfolio_metrics_from_parsed(
    parsed_trades: list[StrategyTradeSimulation],
    parsed_equity: list[StrategyEquityPoint],
    *,
    starting_capital: float | None,
    annualization_periods: float | None,
) -> StrategyPortfolioMetrics:
    if starting_capital is not None:
        start_override = float(starting_capital)
        if not math.isfinite(start_override) or start_override <= 0.0:
            raise ValueError("starting_capital must be > 0 when provided")
    else:
        start_override = None

    annualization = None
    if annualization_periods is not None:
        annualization = float(annualization_periods)
        if not math.isfinite(annualization) or annualization <= 0.0:
            raise ValueError("annualization_periods must be > 0 when provided")

    closed_trades = _closed_trades(parsed_trades)
    sorted_equity = sorted(parsed_equity, key=lambda row: _timestamp_sort_key(row.ts))

    pnl_values = [_trade_pnl_per_unit(trade) for trade in closed_trades]
    finite_pnl_values = [value for value in pnl_values if value is not None]
    realized_r_values = [value for value in (_finite_float(trade.realized_r) for trade in closed_trades) if value is not None]
    hold_values = [float(trade.holding_bars) for trade in closed_trades]

    if sorted_equity:
        start_equity = float(sorted_equity[0].equity)
        end_equity = float(sorted_equity[-1].equity)
    else:
        start_equity = 0.0
        end_equity = 0.0

    if start_override is not None:
        start_equity = start_override
        if not sorted_equity:
            end_equity = start_equity + sum(finite_pnl_values)

    if not math.isfinite(start_equity):
        start_equity = 0.0
    if not math.isfinite(end_equity):
        end_equity = start_equity

    total_return_pct = 0.0
    if start_equity > _EPSILON:
        total_return_pct = (end_equity / start_equity) - 1.0

    years = _years_covered(sorted_equity, closed_trades)
    cagr_pct = _compute_cagr(start_equity, end_equity, years)

    max_drawdown_pct = _max_drawdown_pct(sorted_equity)
    step_returns = _step_returns(sorted_equity)

    annualization_factor = annualization
    if annualization_factor is None and years is not None and years > _EPSILON and step_returns:
        annualization_factor = float(len(step_returns)) / years

    sharpe_ratio = _compute_sharpe(step_returns, annualization_factor)
    sortino_ratio = _compute_sortino(step_returns, annualization_factor)

    calmar_ratio = None
    if cagr_pct is not None and max_drawdown_pct is not None and max_drawdown_pct < -_EPSILON:
        calmar_ratio = cagr_pct / abs(max_drawdown_pct)

    trade_count = len(closed_trades)
    winners = sum(1 for value in pnl_values if value is not None and value > 0.0)
    losers = sum(1 for value in pnl_values if value is not None and value < 0.0)

    win_rate = (winners / trade_count) if trade_count > 0 else None
    loss_rate = (losers / trade_count) if trade_count > 0 else None

    gross_profit = sum(value for value in finite_pnl_values if value > 0.0)
    gross_loss = abs(sum(value for value in finite_pnl_values if value < 0.0))
    profit_factor = None
    if finite_pnl_values and gross_loss > _EPSILON:
        profit_factor = gross_profit / gross_loss

    expectancy_r = mean(realized_r_values) if realized_r_values else None
    avg_realized_r = expectancy_r
    avg_hold_bars = mean(hold_values) if hold_values else None

    exposure_pct = _exposure_from_equity(sorted_equity)
    if exposure_pct is None:
        exposure_pct = _exposure_from_trades(closed_trades)

    return StrategyPortfolioMetrics(
        starting_capital=start_equity,
        ending_capital=end_equity,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        profit_factor=profit_factor,
        expectancy_r=expectancy_r,
        avg_realized_r=avg_realized_r,
        trade_count=trade_count,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_hold_bars=avg_hold_bars,
        exposure_pct=exposure_pct,
    )


def _closed_trades(trades: list[StrategyTradeSimulation]) -> list[StrategyTradeSimulation]:
    return [trade for trade in trades if str(trade.status).strip().lower() == "closed"]


def _target_r_for_trade(trade: StrategyTradeSimulation) -> float | None:
    target_price = _finite_float(trade.target_price)
    entry_price = _finite_float(trade.entry_price)
    initial_risk = _finite_float(trade.initial_risk)
    if target_price is None or entry_price is None or initial_risk is None or initial_risk <= _EPSILON:
        return None

    direction = str(trade.direction).strip().lower()
    if direction == "long":
        target_r = (target_price - entry_price) / initial_risk
    elif direction == "short":
        target_r = (entry_price - target_price) / initial_risk
    else:
        return None

    if not math.isfinite(target_r) or target_r <= _EPSILON:
        return None
    return target_r


def _trade_pnl_per_unit(trade: StrategyTradeSimulation) -> float | None:
    entry_price = _finite_float(trade.entry_price)
    exit_price = _finite_float(trade.exit_price)
    direction = str(trade.direction).strip().lower()

    if entry_price is not None and exit_price is not None and direction in {"long", "short"}:
        if direction == "long":
            return exit_price - entry_price
        return entry_price - exit_price

    realized_r = _finite_float(trade.realized_r)
    initial_risk = _finite_float(trade.initial_risk)
    if realized_r is None or initial_risk is None:
        return None
    return realized_r * initial_risk


def _years_covered(equity_curve: list[StrategyEquityPoint], trades: list[StrategyTradeSimulation]) -> float | None:
    if len(equity_curve) >= 2:
        start_ts = _to_utc_timestamp(equity_curve[0].ts)
        end_ts = _to_utc_timestamp(equity_curve[-1].ts)
        if start_ts is not None and end_ts is not None and end_ts > start_ts:
            return (end_ts - start_ts).total_seconds() / _SECONDS_PER_YEAR

    entry_times = [_to_utc_timestamp(trade.entry_ts) for trade in trades]
    exit_times = [_to_utc_timestamp(trade.exit_ts) for trade in trades]
    valid_entries = [ts for ts in entry_times if ts is not None]
    valid_exits = [ts for ts in exit_times if ts is not None]
    if valid_entries and valid_exits:
        start_ts = min(valid_entries)
        end_ts = max(valid_exits)
        if end_ts > start_ts:
            return (end_ts - start_ts).total_seconds() / _SECONDS_PER_YEAR
    return None


def _compute_cagr(start_equity: float, end_equity: float, years: float | None) -> float | None:
    if years is None or years <= _EPSILON:
        return None
    if start_equity <= _EPSILON or end_equity <= _EPSILON:
        return None
    return (end_equity / start_equity) ** (1.0 / years) - 1.0


def _max_drawdown_pct(equity_curve: list[StrategyEquityPoint]) -> float | None:
    if not equity_curve:
        return None

    peak: float | None = None
    max_drawdown = 0.0
    for row in equity_curve:
        equity_value = _finite_float(row.equity)
        if equity_value is None or equity_value <= 0.0:
            continue

        if peak is None or equity_value > peak:
            peak = equity_value
        if peak is None or peak <= _EPSILON:
            continue

        drawdown = (equity_value / peak) - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def _step_returns(equity_curve: list[StrategyEquityPoint]) -> list[float]:
    out: list[float] = []
    for prev, cur in zip(equity_curve, equity_curve[1:], strict=False):
        prev_equity = _finite_float(prev.equity)
        cur_equity = _finite_float(cur.equity)
        if prev_equity is None or cur_equity is None or prev_equity <= _EPSILON:
            continue
        step_return = (cur_equity / prev_equity) - 1.0
        if math.isfinite(step_return):
            out.append(step_return)
    return out


def _compute_sharpe(returns: list[float], annualization_periods: float | None) -> float | None:
    if not returns:
        return None
    if annualization_periods is None or annualization_periods <= _EPSILON:
        return None

    volatility = pstdev(returns)
    if volatility <= _EPSILON:
        return None
    return mean(returns) / volatility * math.sqrt(annualization_periods)


def _compute_sortino(returns: list[float], annualization_periods: float | None) -> float | None:
    if not returns:
        return None
    if annualization_periods is None or annualization_periods <= _EPSILON:
        return None

    downside_returns = [value for value in returns if value < 0.0]
    if not downside_returns:
        return None

    downside_deviation = pstdev(downside_returns)
    if downside_deviation <= _EPSILON:
        return None
    return mean(returns) / downside_deviation * math.sqrt(annualization_periods)


def _exposure_from_equity(equity_curve: list[StrategyEquityPoint]) -> float | None:
    if len(equity_curve) < 2:
        return None

    open_seconds = 0.0
    total_seconds = 0.0
    for prev, cur in zip(equity_curve, equity_curve[1:], strict=False):
        start_ts = _to_utc_timestamp(prev.ts)
        end_ts = _to_utc_timestamp(cur.ts)
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            continue

        interval_seconds = (end_ts - start_ts).total_seconds()
        total_seconds += interval_seconds
        if int(prev.open_trade_count) > 0:
            open_seconds += interval_seconds

    if total_seconds <= _EPSILON:
        return None
    return open_seconds / total_seconds


def _exposure_from_trades(trades: list[StrategyTradeSimulation]) -> float | None:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for trade in trades:
        entry_ts = _to_utc_timestamp(trade.entry_ts)
        exit_ts = _to_utc_timestamp(trade.exit_ts)
        if entry_ts is None or exit_ts is None or exit_ts <= entry_ts:
            continue
        intervals.append((entry_ts, exit_ts))

    if not intervals:
        return None

    intervals.sort(key=lambda interval: (interval[0].value, interval[1].value))
    min_start = intervals[0][0]
    max_end = intervals[-1][1]
    total_seconds = (max_end - min_start).total_seconds()
    if total_seconds <= _EPSILON:
        return None

    merged_open_seconds = 0.0
    cur_start, cur_end = intervals[0]
    for start_ts, end_ts in intervals[1:]:
        if start_ts <= cur_end:
            if end_ts > cur_end:
                cur_end = end_ts
            continue
        merged_open_seconds += (cur_end - cur_start).total_seconds()
        cur_start, cur_end = start_ts, end_ts
    merged_open_seconds += (cur_end - cur_start).total_seconds()

    return merged_open_seconds / total_seconds


def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _timestamp_sort_key(value: object) -> int:
    ts = _to_utc_timestamp(value)
    if ts is None:
        return -1
    return int(ts.value)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_target_label(target_r: float) -> str:
    tenths = round(target_r * 10.0)
    if abs(target_r - (tenths / 10.0)) <= 1e-6:
        return f"{tenths / 10.0:.1f}R"

    text = f"{target_r:.4f}".rstrip("0").rstrip(".")
    return f"{text}R"


__all__ = [
    "StrategyMetricsResult",
    "compute_strategy_expectancy_dollars",
    "compute_strategy_metrics",
    "compute_strategy_portfolio_metrics",
    "compute_strategy_target_hit_rates",
]
