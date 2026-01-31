from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from options_helper.analysis.chain_metrics import ChainReport
from options_helper.analysis.compare_metrics import CompareReport


def _fmt_pct_ratio(val: float | None, *, digits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val * 100.0:.{digits}f}%"


def _fmt_pp(val: float | None, *, digits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val:+.{digits}f}pp"


def _fmt_money(val: float | None, *, digits: int = 0) -> str:
    if val is None:
        return "-"
    return f"${val:,.{digits}f}"


def chain_takeaways(report: ChainReport) -> list[str]:
    takeaways: list[str] = []

    takeaways.append(f"Spot: `{report.spot:.2f}`")
    if report.included_expiries:
        takeaways.append(f"Included expiries: `{', '.join(report.included_expiries)}`")

    if report.walls_overall.calls:
        lvl = report.walls_overall.calls[0]
        takeaways.append(f"Top call wall: `{lvl.strike:g}` (OI `{lvl.open_interest:,.0f}`)")
    if report.walls_overall.puts:
        lvl = report.walls_overall.puts[0]
        takeaways.append(f"Top put wall: `{lvl.strike:g}` (OI `{lvl.open_interest:,.0f}`)")

    if report.expiries:
        near = report.expiries[0]
        if near.expected_move is not None or near.expected_move_pct is not None:
            takeaways.append(
                f"Near expiry EM ({near.expiry}): {_fmt_money(near.expected_move, digits=2)} / `{_fmt_pct_ratio(near.expected_move_pct)}`"
            )
        if near.atm_iv is not None:
            takeaways.append(f"Near expiry ATM IV ({near.expiry}): `{_fmt_pct_ratio(near.atm_iv)}`")
        if near.skew_25d_pp is not None:
            takeaways.append(f"Near expiry skew 25Δ ({near.expiry}): `{_fmt_pp(near.skew_25d_pp)}`")

    if report.gamma.peak_strike is not None:
        takeaways.append(f"Gamma peak strike: `{report.gamma.peak_strike:g}`")

    return takeaways


def compare_takeaways(diff: CompareReport) -> list[str]:
    out: list[str] = []

    out.append(
        "Spot: "
        f"`{diff.spot_from:.2f} → {diff.spot_to:.2f}` "
        f"(`{diff.spot_change:+.2f}`, `{_fmt_pct_ratio(diff.spot_change_pct)}`)"
    )

    if diff.pc_oi_ratio_change is not None:
        out.append(
            "P/C OI: "
            f"`{diff.pc_oi_ratio_from:.2f} → {diff.pc_oi_ratio_to:.2f}` "
            f"(`{diff.pc_oi_ratio_change:+.2f}`)"
        )
    if diff.pc_volume_ratio_change is not None:
        out.append(
            "P/C Vol: "
            f"`{diff.pc_volume_ratio_from:.2f} → {diff.pc_volume_ratio_to:.2f}` "
            f"(`{diff.pc_volume_ratio_change:+.2f}`)"
        )

    if diff.expiries:
        near = diff.expiries[0]
        parts: list[str] = []
        if near.atm_iv_change_pp is not None:
            parts.append(_fmt_pp(near.atm_iv_change_pp))
        if near.expected_move_pct_change_pp is not None:
            parts.append(f"EM% {_fmt_pp(near.expected_move_pct_change_pp)}")
        if near.expected_move_change is not None:
            parts.append(f"EM$ {near.expected_move_change:+.2f}")
        if parts:
            out.append(f"Near expiry ({near.expiry}): " + ", ".join(f"`{p}`" for p in parts))

    wall_bits: list[str] = []
    if diff.walls_overall.calls:
        c = diff.walls_overall.calls[0]
        wall_bits.append(f"call `{c.strike:g}` ΔOI `{c.delta_oi:+.0f}`")
    if diff.walls_overall.puts:
        p = diff.walls_overall.puts[0]
        wall_bits.append(f"put `{p.strike:g}` ΔOI `{p.delta_oi:+.0f}`")
    if wall_bits:
        out.append("Top wall ΔOI: " + ", ".join(wall_bits))

    if diff.warnings:
        out.append("Warnings: " + ", ".join(f"`{w}`" for w in diff.warnings))

    return out


def flow_takeaways(net: pd.DataFrame, *, top: int) -> list[str]:
    if net is None or net.empty or "deltaOI_notional" not in net.columns:
        return []

    df = net.copy()
    df["deltaOI_notional"] = pd.to_numeric(df["deltaOI_notional"], errors="coerce")
    df["deltaOI"] = pd.to_numeric(df.get("deltaOI"), errors="coerce") if "deltaOI" in df.columns else float("nan")
    df["_abs"] = df["deltaOI_notional"].abs()
    df = df.sort_values(["_abs", "optionType", "strike"], ascending=[False, True, True], na_position="last").drop(columns=["_abs"])

    def _fmt_row(row) -> str:
        opt_type = str(row.get("optionType", "-"))
        strike = row.get("strike")
        strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
        delta_oi = row.get("deltaOI")
        delta_oi_val = float(delta_oi) if delta_oi is not None and not pd.isna(delta_oi) else None
        delta_notional = row.get("deltaOI_notional")
        delta_notional_val = (
            float(delta_notional) if delta_notional is not None and not pd.isna(delta_notional) else None
        )
        strike_str = "-" if strike_val is None else f"{strike_val:g}"
        s = f"{opt_type} `{strike_str}`"
        s += f": ΔOI$ `{_fmt_money(delta_notional_val, digits=0)}`"
        if delta_oi_val is not None:
            s += f", ΔOI `{delta_oi_val:+.0f}`"
        return s

    out: list[str] = []
    building = df[df["deltaOI_notional"] > 0].head(top)
    unwinding = df[df["deltaOI_notional"] < 0].head(top)

    if not building.empty:
        out.append("Building:")
        for _, row in building.iterrows():
            out.append(f"- {_fmt_row(row)}")
    if not unwinding.empty:
        out.append("Unwinding:")
        for _, row in unwinding.iterrows():
            out.append(f"- {_fmt_row(row)}")

    return out


@dataclass(frozen=True)
class BriefingSymbolSection:
    symbol: str
    as_of: str
    chain: ChainReport | None
    compare: CompareReport | None
    flow_net: pd.DataFrame | None
    errors: list[str]
    warnings: list[str]
    derived_updated: bool = False


def render_briefing_markdown(
    *,
    report_date: str,
    portfolio_path: str,
    symbol_sections: list[BriefingSymbolSection],
    portfolio_table_md: str | None = None,
) -> str:
    lines: list[str] = []

    lines.append(f"# Daily briefing ({report_date})")
    lines.append("")
    lines.append("> Not financial advice. For informational/educational use only.")
    lines.append("")
    lines.append(f"- Portfolio: `{portfolio_path}`")
    lines.append(f"- Symbols: `{', '.join(s.symbol for s in symbol_sections) or '-'}`")
    lines.append("")

    if portfolio_table_md:
        lines.append("## Portfolio")
        lines.append("")
        lines.append(portfolio_table_md.rstrip())
        lines.append("")

    for sec in symbol_sections:
        lines.append(f"## {sec.symbol} ({sec.as_of})")
        lines.append("")

        if sec.errors and sec.chain is None:
            lines.append("Errors:")
            for e in sec.errors:
                lines.append(f"- {e}")
            lines.append("")
            continue

        if sec.warnings:
            lines.append("Warnings:")
            for w in sec.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if sec.chain is not None:
            lines.append("### Chain")
            for b in chain_takeaways(sec.chain):
                lines.append(f"- {b}")
            if sec.chain.warnings:
                lines.append("- Warnings: " + ", ".join(f"`{w}`" for w in sec.chain.warnings))
            lines.append("")

        if sec.compare is not None:
            lines.append("### Compare")
            for b in compare_takeaways(sec.compare):
                lines.append(f"- {b}")
            lines.append("")

        if sec.flow_net is not None:
            ft = flow_takeaways(sec.flow_net, top=3)
            if ft:
                lines.append("### Flow (net, aggregated by strike)")
                # `flow_takeaways` returns a small mixed list (section headers + bullets).
                for line in ft:
                    if line.startswith("- "):
                        lines.append(line)
                    elif line.endswith(":"):
                        lines.append(f"- {line}")
                    else:
                        lines.append(f"- {line}")
                lines.append("")

        if sec.derived_updated:
            lines.append("- Derived: updated")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
