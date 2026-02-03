from __future__ import annotations

from rich.console import Console
from rich.table import Table

from options_helper.analysis.roll_plan import RollPlanReport
from options_helper.analysis.roll_plan_multileg import MultiLegRollPlanReport


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


def _fmt_stale(age_days: int | None) -> str:
    if age_days is None:
        return "-"
    age = int(age_days)
    return f"{age}d" if age > 5 else "-"


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
    cur_table.add_column("Quality", justify="right")
    cur_table.add_column("Stale", justify="right")
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
        _fmt_exec(cur.quality_label),
        _fmt_stale(cur.last_trade_age_days),
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
        cand_table.add_column("Quality", justify="right")
        cand_table.add_column("Stale", justify="right")
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
                _fmt_exec(c.quality_label),
                _fmt_stale(c.last_trade_age_days),
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


def _strikes_summary(legs) -> str:  # noqa: ANN001
    strikes = [f"{leg.strike:g}" for leg in legs]
    return "/".join(strikes)


def render_roll_plan_multileg_console(console: Console, report: MultiLegRollPlanReport) -> None:
    console.print(
        f"\n[bold]{report.symbol}[/bold] multi-leg roll plan as-of {report.as_of} | spot={report.spot:.2f} | "
        f"horizon={report.horizon_months}m (target {report.target_dte} DTE) | structure={report.structure}"
    )

    cur_table = Table(title=f"Current ({report.position_id})")
    cur_table.add_column("Side")
    cur_table.add_column("Type")
    cur_table.add_column("Expiry")
    cur_table.add_column("Strike", justify="right")
    cur_table.add_column("Ct", justify="right")
    cur_table.add_column("DTE", justify="right")
    cur_table.add_column("Mark", justify="right")
    cur_table.add_column("OI", justify="right")
    cur_table.add_column("Vol", justify="right")
    cur_table.add_column("Spr%", justify="right")
    cur_table.add_column("Exec", justify="right")

    for leg in report.current_legs:
        cur_table.add_row(
            leg.side,
            leg.option_type,
            leg.expiry,
            f"{leg.strike:g}",
            str(leg.contracts),
            str(leg.dte),
            _fmt_money(leg.mark),
            "-" if leg.open_interest is None else f"{leg.open_interest:d}",
            "-" if leg.volume is None else f"{leg.volume:d}",
            _fmt_pct(leg.spread_pct),
            _fmt_exec(leg.execution_quality),
        )
    console.print(cur_table)

    if report.current_net_mark is not None or report.current_net_debit is not None:
        console.print(
            f"Net mark: {_fmt_money(report.current_net_mark)} | "
            f"Net debit: {_fmt_money(report.current_net_debit)}"
        )

    if not report.candidates:
        console.print("\n[yellow]No roll candidates found.[/yellow]")
    else:
        cand_table = Table(title="Roll candidates (ranked)")
        cand_table.add_column("#", justify="right")
        cand_table.add_column("Expiry")
        cand_table.add_column("Strikes", justify="right")
        cand_table.add_column("DTE", justify="right")
        cand_table.add_column("Net Mark", justify="right")
        cand_table.add_column("Roll $", justify="right")
        cand_table.add_column("Warn")

        for i, cand in enumerate(report.candidates, start=1):
            expiry = cand.legs[0].expiry if cand.legs else "-"
            dte = cand.legs[0].dte if cand.legs else "-"
            warn = "-" if not cand.warnings else ", ".join(cand.warnings)
            cand_table.add_row(
                str(i),
                expiry,
                _strikes_summary(cand.legs),
                str(dte),
                _fmt_money(cand.net_mark),
                "-" if cand.roll_debit is None else f"{cand.roll_debit:+.0f}",
                warn,
            )

        console.print(cand_table)

        top = report.candidates[0]
        if top.rationale:
            console.print("\n[bold]Top pick rationale[/bold]")
            for r in top.rationale[:10]:
                console.print(f"- {r}")

        if top.warnings:
            console.print("\n[yellow]Top pick warnings[/yellow]")
            for w in sorted(set(top.warnings)):
                console.print(f"- {w}")

    if report.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for w in report.warnings:
            console.print(f"- {w}")
