from __future__ import annotations

from rich.console import Console
from rich.table import Table

from options_helper.analysis.chain_metrics import ChainReport
from options_helper.analysis.compare_metrics import CompareReport


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


def _fmt_pp(val: float | None, *, digits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val:+.{digits}f}pp"


def _build_chain_totals_table(report: ChainReport) -> Table:
    totals = report.totals
    table = Table(title="Totals")
    table.add_column("")
    table.add_column("Calls", justify="right")
    table.add_column("Puts", justify="right")
    table.add_column("P/C", justify="right")
    table.add_row(
        "OI",
        _fmt_num(totals.calls_oi, digits=0),
        _fmt_num(totals.puts_oi, digits=0),
        _fmt_num(totals.pc_oi_ratio),
    )
    table.add_row(
        "Volume",
        _fmt_num(totals.calls_volume, digits=0),
        _fmt_num(totals.puts_volume, digits=0),
        _fmt_num(totals.pc_volume_ratio),
    )
    table.add_row(
        "Vol Notional",
        _fmt_money(totals.calls_volume_notional),
        _fmt_money(totals.puts_volume_notional),
        "-",
    )
    return table


def _build_chain_term_structure_table(report: ChainReport) -> Table:
    table = Table(title="Term Structure (selected expiries)")
    table.add_column("Expiry")
    table.add_column("ATM IV", justify="right")
    table.add_column("Skew 25Δ", justify="right")
    table.add_column("EM $", justify="right")
    table.add_column("EM %", justify="right")
    table.add_column("P/C OI", justify="right")
    table.add_column("P/C Vol", justify="right")
    for row in report.expiries:
        table.add_row(
            row.expiry,
            _fmt_pct(row.atm_iv),
            _fmt_pp(row.skew_25d_pp),
            _fmt_money(row.expected_move),
            _fmt_pct(row.expected_move_pct),
            _fmt_num(row.pc_oi_ratio),
            _fmt_num(row.pc_volume_ratio),
        )
    return table


def _build_chain_walls_table(report: ChainReport) -> tuple[Table, list[object], list[object]]:
    table = Table(title="Walls (overall)")
    table.add_column("Call Strike", justify="right")
    table.add_column("Call OI", justify="right")
    table.add_column("Put Strike", justify="right")
    table.add_column("Put OI", justify="right")
    calls = report.walls_overall.calls
    puts = report.walls_overall.puts
    for i in range(max(len(calls), len(puts))):
        c = calls[i] if i < len(calls) else None
        p = puts[i] if i < len(puts) else None
        table.add_row(
            "-" if c is None else f"{c.strike:g}",
            "-" if c is None else f"{c.open_interest:,.0f}",
            "-" if p is None else f"{p.strike:g}",
            "-" if p is None else f"{p.open_interest:,.0f}",
        )
    return table, calls, puts


def _build_chain_gamma_table(report: ChainReport) -> Table:
    table = Table(title="Gamma Concentration (gross, top strikes)")
    table.add_column("Strike", justify="right")
    table.add_column("Gamma $/1% move", justify="right")
    for level in report.gamma.top:
        table.add_row(f"{level.strike:g}", f"{level.gamma_1pct:,.0f}")
    return table


def _chain_takeaways(report: ChainReport, *, calls: list[object], puts: list[object]) -> list[str]:
    takeaways: list[str] = []
    if calls:
        takeaways.append(f"Top call wall: {calls[0].strike:g} (OI {calls[0].open_interest:,.0f})")
    if puts:
        takeaways.append(f"Top put wall: {puts[0].strike:g} (OI {puts[0].open_interest:,.0f})")
    if report.expiries:
        near = report.expiries[0]
        if near.expected_move_pct is not None:
            takeaways.append(f"Near expiry EM: {near.expiry} ≈ {near.expected_move_pct * 100.0:.1f}%")
        if near.atm_iv is not None:
            takeaways.append(f"Near expiry ATM IV: {near.expiry} ≈ {near.atm_iv * 100.0:.1f}%")
    if report.gamma.peak_strike is not None:
        takeaways.append(f"Gamma peak strike: {report.gamma.peak_strike:g}")
    return takeaways[:10]


def _render_warning_list(console: Console, warnings: list[str]) -> None:
    if not warnings:
        return
    console.print("\n[yellow]Warnings[/yellow]")
    for warning in warnings:
        console.print(f"- {warning}")


def render_chain_report_console(console: Console, report: ChainReport) -> None:
    console.print(f"\n[bold]{report.symbol}[/bold] chain report as-of {report.as_of} | spot={report.spot:.2f}")
    console.print(_build_chain_totals_table(report))
    if report.expiries:
        console.print(_build_chain_term_structure_table(report))
    walls_table, calls, puts = _build_chain_walls_table(report)
    console.print(walls_table)
    if report.gamma.top:
        console.print(_build_chain_gamma_table(report))
    takeaways = _chain_takeaways(report, calls=calls, puts=puts)
    if takeaways:
        console.print("\n[bold]Takeaways[/bold]")
        for takeaway in takeaways:
            console.print(f"- {takeaway}")
    _render_warning_list(console, report.warnings)


def render_chain_report_markdown(report: ChainReport) -> str:
    lines: list[str] = []
    lines.append(f"# {report.symbol} chain report ({report.as_of})")
    lines.append("")
    lines.append(f"- Spot: `{report.spot:.2f}`")
    lines.append(f"- Included expiries: `{', '.join(report.included_expiries) or '-'}`")
    lines.append("")

    t = report.totals
    lines.append("## Totals")
    lines.append("")
    lines.append("| Metric | Calls | Puts | P/C |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| OI | {t.calls_oi:,.0f} | {t.puts_oi:,.0f} | {'-' if t.pc_oi_ratio is None else f'{t.pc_oi_ratio:.2f}'} |")
    lines.append(
        f"| Volume | {t.calls_volume:,.0f} | {t.puts_volume:,.0f} | {'-' if t.pc_volume_ratio is None else f'{t.pc_volume_ratio:.2f}'} |"
    )
    lines.append(
        f"| Vol Notional | {_fmt_money(t.calls_volume_notional)} | {_fmt_money(t.puts_volume_notional)} | - |"
    )

    if report.expiries:
        lines.append("")
        lines.append("## Term structure (selected expiries)")
        lines.append("")
        lines.append("| Expiry | ATM IV | Skew 25Δ | EM $ | EM % | P/C OI | P/C Vol |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for e in report.expiries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        e.expiry,
                        "-" if e.atm_iv is None else f"{e.atm_iv * 100.0:.1f}%",
                        "-" if e.skew_25d_pp is None else f"{e.skew_25d_pp:+.1f}pp",
                        "-" if e.expected_move is None else f"${e.expected_move:.2f}",
                        "-" if e.expected_move_pct is None else f"{e.expected_move_pct * 100.0:.1f}%",
                        "-" if e.pc_oi_ratio is None else f"{e.pc_oi_ratio:.2f}",
                        "-" if e.pc_volume_ratio is None else f"{e.pc_volume_ratio:.2f}",
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Walls (overall)")
    lines.append("")
    lines.append("### Calls")
    for lvl in report.walls_overall.calls:
        lines.append(f"- `{lvl.strike:g}` OI `{lvl.open_interest:,.0f}`")
    if not report.walls_overall.calls:
        lines.append("- (none)")

    lines.append("")
    lines.append("### Puts")
    for lvl in report.walls_overall.puts:
        lines.append(f"- `{lvl.strike:g}` OI `{lvl.open_interest:,.0f}`")
    if not report.walls_overall.puts:
        lines.append("- (none)")

    if report.gamma.top:
        lines.append("")
        lines.append("## Gamma concentration (gross)")
        lines.append("")
        lines.append("| Strike | Gamma $/1% move |")
        lines.append("|---:|---:|")
        for lvl in report.gamma.top:
            lines.append(f"| {lvl.strike:g} | {lvl.gamma_1pct:,.0f} |")

    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines).rstrip() + "\n"


def render_compare_report_console(console: Console, report: CompareReport) -> None:
    console.print(
        f"\n[bold]{report.symbol}[/bold] compare {report.from_date} → {report.to_date} | "
        f"spot {report.spot_from:.2f} → {report.spot_to:.2f} ({report.spot_change:+.2f}, {_fmt_pct(report.spot_change_pct)})"
    )
    summary = Table(title="Summary")
    summary.add_column("Metric")
    summary.add_column("From", justify="right")
    summary.add_column("To", justify="right")
    summary.add_column("Δ", justify="right")
    summary.add_row(
        "P/C OI",
        "-" if report.pc_oi_ratio_from is None else f"{report.pc_oi_ratio_from:.2f}",
        "-" if report.pc_oi_ratio_to is None else f"{report.pc_oi_ratio_to:.2f}",
        "-" if report.pc_oi_ratio_change is None else f"{report.pc_oi_ratio_change:+.2f}",
    )
    summary.add_row(
        "P/C Vol",
        "-" if report.pc_volume_ratio_from is None else f"{report.pc_volume_ratio_from:.2f}",
        "-" if report.pc_volume_ratio_to is None else f"{report.pc_volume_ratio_to:.2f}",
        "-" if report.pc_volume_ratio_change is None else f"{report.pc_volume_ratio_change:+.2f}",
    )
    console.print(summary)
    if report.expiries:
        expiry_changes = Table(title="Expiry changes (selected)")
        expiry_changes.add_column("Expiry")
        expiry_changes.add_column("ATM IV Δ", justify="right")
        expiry_changes.add_column("EM $ Δ", justify="right")
        expiry_changes.add_column("EM % Δ", justify="right")
        for row in report.expiries:
            expiry_changes.add_row(
                row.expiry,
                _fmt_pp(row.atm_iv_change_pp),
                "-" if row.expected_move_change is None else f"{row.expected_move_change:+.2f}",
                _fmt_pp(row.expected_move_pct_change_pp),
            )
        console.print(expiry_changes)
    walls = Table(title="Walls ΔOI (overall, top)")
    walls.add_column("Type")
    walls.add_column("Strike", justify="right")
    walls.add_column("OI From", justify="right")
    walls.add_column("OI To", justify="right")
    walls.add_column("ΔOI", justify="right")
    for lvl in report.walls_overall.calls:
        walls.add_row("call", f"{lvl.strike:g}", f"{lvl.oi_from:,.0f}", f"{lvl.oi_to:,.0f}", f"{lvl.delta_oi:+.0f}")
    for lvl in report.walls_overall.puts:
        walls.add_row("put", f"{lvl.strike:g}", f"{lvl.oi_from:,.0f}", f"{lvl.oi_to:,.0f}", f"{lvl.delta_oi:+.0f}")
    console.print(walls)
    if report.flow_top_contracts:
        contracts = Table(title="Top contracts by |ΔOI$|")
        contracts.add_column("Expiry")
        contracts.add_column("Type")
        contracts.add_column("Strike", justify="right")
        contracts.add_column("ΔOI", justify="right")
        contracts.add_column("ΔOI$", justify="right")
        contracts.add_column("Class")
        for row in report.flow_top_contracts:
            contracts.add_row(
                "-" if row.expiry is None else row.expiry,
                "-" if row.option_type is None else row.option_type,
                "-" if row.strike is None else f"{row.strike:g}",
                "-" if row.delta_oi is None else f"{row.delta_oi:+.0f}",
                "-" if row.delta_oi_notional is None else f"{row.delta_oi_notional:+,.0f}",
                "-" if row.flow_class is None else row.flow_class,
            )
        console.print(contracts)
    if report.flow_top_expiries:
        expiries = Table(title="Top expiries by |net ΔOI$|")
        expiries.add_column("Expiry")
        expiries.add_column("Net ΔOI$", justify="right")
        for row in report.flow_top_expiries:
            expiries.add_row(row.expiry, f"{row.delta_oi_notional:+,.0f}")
        console.print(expiries)
    _render_warning_list(console, report.warnings)
