from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from options_helper.analysis.advice import Advice, PositionMetrics
from options_helper.models import MultiLegPosition, Portfolio


@dataclass(frozen=True)
class MultiLegSummary:
    position: MultiLegPosition
    leg_metrics: list[PositionMetrics]
    net_mark: float | None
    net_pnl_abs: float | None
    net_pnl_pct: float | None
    dte_min: int | None
    dte_max: int | None
    warnings: list[str]


def _fmt_money(val: float | None) -> str:
    if val is None:
        return "-"
    return f"${val:,.2f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.1%}"


def render_summary(console: Console, portfolio: Portfolio) -> None:
    capital = portfolio.capital_cost_basis()
    risk = portfolio.premium_at_risk()

    table = Table(title="Portfolio Summary")
    table.add_column("Cash", justify="right")
    table.add_column("Premium at Risk", justify="right")
    table.add_column("Capital (Cost Basis)", justify="right")
    table.add_row(_fmt_money(portfolio.cash), _fmt_money(risk), _fmt_money(capital))
    console.print(table)


def render_positions(
    console: Console,
    portfolio: Portfolio,
    metrics_list: list[PositionMetrics],
    advice_by_id: dict[str, Advice],
) -> None:
    table = Table(title="Positions")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Type")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Mark", justify="right")
    table.add_column("PnL $", justify="right")
    table.add_column("PnL %", justify="right")
    table.add_column("DTE", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Action")

    for m in metrics_list:
        advice = advice_by_id.get(m.position.id)
        pnl_style = None
        if m.pnl_abs is not None:
            pnl_style = "green" if m.pnl_abs > 0 else "red" if m.pnl_abs < 0 else None
        action = advice.action.value if advice else "-"
        table.add_row(
            m.position.id,
            m.position.symbol,
            m.position.option_type,
            m.position.expiry.isoformat(),
            f"{m.position.strike:g}",
            str(m.position.contracts),
            _fmt_money(m.position.cost_basis),
            _fmt_money(m.mark),
            _fmt_money(m.pnl_abs),
            _fmt_pct(m.pnl_pct),
            "-" if m.dte is None else str(m.dte),
            "-" if m.implied_vol is None else f"{m.implied_vol:.1%}",
            "-" if m.delta is None else f"{m.delta:.2f}",
            action,
            style=pnl_style,
        )

    console.print(table)

    console.print("\nAdvice")
    for m in metrics_list:
        advice = advice_by_id.get(m.position.id)
        if not advice:
            continue
        console.print(f"\n[bold]{m.position.id}[/bold] — {advice.action.value} ({advice.confidence.value})")
        for r in advice.reasons:
            console.print(f"  - {r}")
        for w in advice.warnings:
            console.print(f"  - [yellow]Warning:[/yellow] {w}")


def render_multi_leg_positions(console: Console, summaries: list[MultiLegSummary]) -> None:
    if not summaries:
        return

    table = Table(title="Multi-leg Positions")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Legs", justify="right")
    table.add_column("DTE", justify="right")
    table.add_column("Net Debit", justify="right")
    table.add_column("Net Mark", justify="right")
    table.add_column("PnL $", justify="right")
    table.add_column("PnL %", justify="right")
    table.add_column("Warnings")

    for summary in summaries:
        position = summary.position
        dte_range = "-"
        if summary.dte_min is not None and summary.dte_max is not None:
            dte_range = str(summary.dte_min) if summary.dte_min == summary.dte_max else f"{summary.dte_min}-{summary.dte_max}"
        warn_text = "-" if not summary.warnings else ", ".join(summary.warnings)
        table.add_row(
            position.id,
            position.symbol,
            str(len(position.legs)),
            dte_range,
            _fmt_money(position.net_debit),
            _fmt_money(summary.net_mark),
            _fmt_money(summary.net_pnl_abs),
            _fmt_pct(summary.net_pnl_pct),
            warn_text,
        )

    console.print(table)

    for summary in summaries:
        position = summary.position
        leg_table = Table(title=f"{position.id} legs")
        leg_table.add_column("#", justify="right")
        leg_table.add_column("Side")
        leg_table.add_column("Type")
        leg_table.add_column("Expiry")
        leg_table.add_column("Strike", justify="right")
        leg_table.add_column("Ct", justify="right")
        leg_table.add_column("Mark", justify="right")
        leg_table.add_column("Δ", justify="right")
        leg_table.add_column("IV", justify="right")
        leg_table.add_column("OI", justify="right")
        leg_table.add_column("Vol", justify="right")
        leg_table.add_column("Spr%", justify="right")

        for idx, (leg, metrics) in enumerate(zip(position.legs, summary.leg_metrics), start=1):
            leg_table.add_row(
                str(idx),
                leg.side,
                leg.option_type,
                leg.expiry.isoformat(),
                f"{leg.strike:g}",
                str(leg.contracts),
                _fmt_money(metrics.mark),
                "-" if metrics.delta is None else f"{metrics.delta:+.2f}",
                "-" if metrics.implied_vol is None else f"{metrics.implied_vol:.1%}",
                "-" if metrics.open_interest is None else f"{metrics.open_interest:d}",
                "-" if metrics.volume is None else f"{metrics.volume:d}",
                _fmt_pct(metrics.spread_pct),
            )

        console.print(leg_table)
