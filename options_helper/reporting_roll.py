from __future__ import annotations

from rich.console import Console
from rich.table import Table

from options_helper.analysis.roll_plan import RollPlanReport


def _fmt_money(val: float | None) -> str:
    if val is None:
        return "-"
    return f"${val:,.2f}"


def _fmt_num(val: float | None, *, digits: int = 2) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _fmt_pct(val: float | None, *, digits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val * 100.0:.{digits}f}%"


def _fmt_exec(val: str | None) -> str:
    if not val:
        return "-"
    return val


def render_roll_plan_console(console: Console, report: RollPlanReport) -> None:
    console.print(
        f"\n[bold]{report.symbol}[/bold] roll plan as-of {report.as_of} | spot={report.spot:.2f} | "
        f"intent={report.intent} horizon={report.horizon_months}m (target {report.target_dte} DTE) | "
        f"shape={report.shape}"
    )

    cur = report.current
    cur_table = Table(title=f"Current ({report.position_id}, {report.contracts} contract(s))")
    cur_table.add_column("Expiry")
    cur_table.add_column("Strike", justify="right")
    cur_table.add_column("DTE", justify="right")
    cur_table.add_column("Mark", justify="right")
    cur_table.add_column("Δ", justify="right")
    cur_table.add_column("Θ/day", justify="right")
    cur_table.add_column("IV", justify="right")
    cur_table.add_column("OI", justify="right")
    cur_table.add_column("Vol", justify="right")
    cur_table.add_column("Spr%", justify="right")
    cur_table.add_column("Exec", justify="right")
    cur_table.add_row(
        cur.expiry,
        f"{cur.strike:g}",
        str(cur.dte),
        _fmt_money(cur.mark),
        "-" if cur.delta is None else f"{cur.delta:+.2f}",
        "-" if cur.theta_per_day is None else f"{cur.theta_per_day:+.4f}",
        _fmt_pct(cur.implied_vol),
        "-" if cur.open_interest is None else f"{cur.open_interest:d}",
        "-" if cur.volume is None else f"{cur.volume:d}",
        _fmt_pct(cur.spread_pct),
        _fmt_exec(cur.execution_quality),
    )
    console.print(cur_table)

    if not report.candidates:
        console.print("\n[yellow]No roll candidates found.[/yellow]")
    else:
        cand_table = Table(title="Roll candidates (ranked)")
        cand_table.add_column("#", justify="right")
        cand_table.add_column("Expiry")
        cand_table.add_column("Strike", justify="right")
        cand_table.add_column("DTE", justify="right")
        cand_table.add_column("Mark", justify="right")
        cand_table.add_column("Roll $", justify="right")
        cand_table.add_column("Δ", justify="right")
        cand_table.add_column("Θ/day", justify="right")
        cand_table.add_column("IV", justify="right")
        cand_table.add_column("OI", justify="right")
        cand_table.add_column("Vol", justify="right")
        cand_table.add_column("Spr%", justify="right")
        cand_table.add_column("Exec", justify="right")
        cand_table.add_column("Warn")

        for i, cand in enumerate(report.candidates, start=1):
            c = cand.contract
            style = None
            if not cand.liquidity_ok:
                style = "yellow"
            warn = "-" if not cand.warnings else ", ".join(cand.warnings)
            cand_table.add_row(
                str(i),
                c.expiry,
                f"{c.strike:g}",
                str(c.dte),
                _fmt_money(c.mark),
                "-" if cand.roll_debit is None else f"{cand.roll_debit:+.0f}",
                "-" if c.delta is None else f"{c.delta:+.2f}",
                "-" if c.theta_per_day is None else f"{c.theta_per_day:+.4f}",
                _fmt_pct(c.implied_vol),
                "-" if c.open_interest is None else f"{c.open_interest:d}",
                "-" if c.volume is None else f"{c.volume:d}",
                _fmt_pct(c.spread_pct),
                _fmt_exec(c.execution_quality),
                warn,
                style=style,
            )

        console.print(cand_table)

        top = report.candidates[0]
        console.print("\n[bold]Top pick rationale[/bold]")
        for r in top.rationale[:10]:
            console.print(f"- {r}")

        if top.issues:
            console.print("\n[yellow]Top pick flags[/yellow]")
            for w in sorted(set(top.issues)):
                console.print(f"- {w}")

        if top.warnings:
            console.print("\n[yellow]Top pick warnings[/yellow]")
            for w in sorted(set(top.warnings)):
                console.print(f"- {w}")

    if report.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for w in report.warnings:
            console.print(f"- {w}")
