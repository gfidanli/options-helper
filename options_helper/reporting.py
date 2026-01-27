from __future__ import annotations

from rich.console import Console
from rich.table import Table

from options_helper.analysis.advice import Advice, PositionMetrics
from options_helper.models import Portfolio


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
