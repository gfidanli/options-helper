from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from options_helper.backtesting.ledger import TradeLogRow
from options_helper.backtesting.runner import BacktestRun, RollEvent, SkipEvent
from options_helper.schemas.backtest import (
    BacktestRollRow,
    BacktestSkipRow,
    BacktestSummaryArtifact,
    BacktestSummaryStats,
    BacktestTradeRow,
    OpenPositionSummary,
)


@dataclass(frozen=True)
class BacktestArtifactPaths:
    run_dir: Path
    trade_log_csv: Path
    trade_log_json: Path
    summary_json: Path
    report_md: Path


def build_backtest_artifact_paths(base_dir: Path, *, run_id: str) -> BacktestArtifactPaths:
    run_dir = base_dir / run_id
    return BacktestArtifactPaths(
        run_dir=run_dir,
        trade_log_csv=run_dir / "trades.csv",
        trade_log_json=run_dir / "trades.json",
        summary_json=run_dir / "summary.json",
        report_md=run_dir / "report.md",
    )


def _trade_to_model(trade: TradeLogRow) -> BacktestTradeRow:
    return BacktestTradeRow(
        symbol=trade.symbol,
        contract_symbol=trade.contract_symbol,
        expiry=trade.expiry,
        strike=trade.strike,
        option_type=trade.option_type,
        quantity=trade.quantity,
        entry_date=trade.entry_date,
        entry_price=trade.entry_price,
        exit_date=trade.exit_date,
        exit_price=trade.exit_price,
        holding_days=trade.holding_days,
        pnl=trade.pnl,
        pnl_pct=trade.pnl_pct,
        max_favorable=trade.max_favorable,
        max_adverse=trade.max_adverse,
    )


def _skip_to_model(skip: SkipEvent) -> BacktestSkipRow:
    return BacktestSkipRow(as_of=skip.as_of, action=skip.action, reason=skip.reason)


def _roll_to_model(roll: RollEvent) -> BacktestRollRow:
    return BacktestRollRow(
        as_of=roll.as_of,
        from_contract_symbol=roll.from_contract_symbol,
        to_contract_symbol=roll.to_contract_symbol,
        reason=roll.reason,
    )


def _compute_stats(trades: list[TradeLogRow], *, initial_cash: float) -> BacktestSummaryStats:
    if not trades:
        return BacktestSummaryStats(
            total_pnl=0.0,
            total_pnl_pct=0.0 if initial_cash else None,
            trade_count=0,
            win_rate=None,
            avg_pnl=None,
            avg_pnl_pct=None,
        )

    total_pnl = sum(t.pnl for t in trades)
    total_pnl_pct = (total_pnl / initial_cash) if initial_cash else None
    trade_count = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / trade_count if trade_count else None
    avg_pnl = total_pnl / trade_count if trade_count else None
    pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else None
    return BacktestSummaryStats(
        total_pnl=float(total_pnl),
        total_pnl_pct=None if total_pnl_pct is None else float(total_pnl_pct),
        trade_count=trade_count,
        win_rate=None if win_rate is None else float(win_rate),
        avg_pnl=None if avg_pnl is None else float(avg_pnl),
        avg_pnl_pct=None if avg_pnl_pct is None else float(avg_pnl_pct),
    )


def write_backtest_artifacts(
    run: BacktestRun,
    *,
    run_id: str,
    reports_dir: Path,
    strategy_name: str,
    quantity: int,
) -> BacktestArtifactPaths:
    paths = build_backtest_artifact_paths(reports_dir, run_id=run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    trade_models = [_trade_to_model(t) for t in run.trades]
    skip_models = [_skip_to_model(s) for s in run.skips]
    roll_models = [_roll_to_model(r) for r in run.rolls]

    stats = _compute_stats(run.trades, initial_cash=run.initial_cash)
    final_cash = run.initial_cash + stats.total_pnl

    open_position = None
    if run.open_position is not None:
        pos = run.open_position
        open_position = OpenPositionSummary(
            contract_symbol=pos.contract_symbol,
            expiry=pos.expiry,
            strike=pos.strike,
            option_type=pos.option_type,
            quantity=pos.quantity,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
        )

    summary = BacktestSummaryArtifact(
        run_id=run_id,
        symbol=run.symbol,
        contract_symbol=run.contract_symbol,
        start=run.start,
        end=run.end,
        strategy_name=strategy_name,
        fill_mode=run.fill_mode,
        slippage_factor=run.slippage_factor,
        quantity=quantity,
        initial_cash=run.initial_cash,
        final_cash=final_cash,
        stats=stats,
        skips=skip_models,
        rolls=roll_models,
        open_position=open_position,
    )

    with paths.trade_log_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BacktestTradeRow.model_fields.keys())
        writer.writeheader()
        for trade in trade_models:
            writer.writerow(trade.model_dump(mode="json"))

    paths.trade_log_json.write_text(
        json.dumps([t.model_dump(mode="json") for t in trade_models], indent=2),
        encoding="utf-8",
    )
    paths.summary_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    report_lines = _render_backtest_report(
        run_id=run_id,
        run=run,
        stats=stats,
        final_cash=final_cash,
        quantity=quantity,
        trade_count=len(trade_models),
    )
    paths.report_md.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return paths


def _render_backtest_report(
    *,
    run_id: str,
    run: BacktestRun,
    stats: BacktestSummaryStats,
    final_cash: float,
    quantity: int,
    trade_count: int,
) -> list[str]:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# Backtest Report — {run.symbol}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated: {now}",
        f"- Fill mode: `{run.fill_mode}` (slippage={run.slippage_factor:g})",
        f"- Quantity: `{quantity}` contract(s)",
        f"- Trades: `{trade_count}`",
        f"- Total P&L: `{stats.total_pnl:+.2f}`",
        f"- Win rate: `{(stats.win_rate or 0.0)*100.0:.1f}%`" if stats.win_rate is not None else "- Win rate: `n/a`",
        f"- Ending cash (closed trades only): `{final_cash:.2f}`",
        "",
        "Not financial advice.",
    ]
    return lines
