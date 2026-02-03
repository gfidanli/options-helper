from __future__ import annotations

import math
from dataclasses import dataclass, is_dataclass, asdict
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from options_helper.analysis.chain_metrics import ChainReport
from options_helper.analysis.compare_metrics import CompareReport
from options_helper.analysis.events import format_next_earnings_line
from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot


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


def vol_regime_takeaway(row: DerivedRow | None) -> str:
    rv20 = "-" if row is None else _fmt_pct_ratio(row.rv_20d)
    rv60 = "-" if row is None else _fmt_pct_ratio(row.rv_60d)
    iv_rv = "-" if row is None or row.iv_rv_20d is None else f"{row.iv_rv_20d:.2f}x"
    iv_pct = "-" if row is None or row.atm_iv_near_percentile is None else f"{row.atm_iv_near_percentile:.0f}"
    slope = "-" if row is None or row.iv_term_slope is None else _fmt_pp(row.iv_term_slope * 100.0)

    return f"Vol regime: RV20 `{rv20}`, RV60 `{rv60}`, IV/RV20 `{iv_rv}`, IV pct `{iv_pct}`, Term slope `{slope}`"


def _fmt_quote_quality(summary: dict[str, Any]) -> str:
    contracts = summary.get("contracts")
    missing_pct = summary.get("missing_bid_ask_pct")
    missing_count = summary.get("missing_bid_ask_count")
    spread_median = summary.get("spread_pct_median")
    spread_worst = summary.get("spread_pct_worst")
    stale_count = summary.get("stale_quotes")
    stale_pct = summary.get("stale_pct")

    def _pct(val: float | None) -> str:
        if val is None:
            return "-"
        return f"{val * 100.0:.1f}%"

    missing_ratio = "-" if contracts is None or missing_count is None else f"{missing_count}/{contracts}"
    stale_ratio = "-" if stale_count is None else f"{stale_count}"
    if stale_pct is not None:
        stale_ratio = f"{stale_ratio} ({_pct(stale_pct)})"

    return (
        "Quote quality: "
        f"missing bid/ask {_pct(missing_pct)} ({missing_ratio}); "
        f"median spread {_pct(spread_median)}; "
        f"worst spread {_pct(spread_worst)}; "
        f"stale {stale_ratio}"
    )


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


def technicals_takeaways(snapshot: TechnicalSnapshot) -> list[str]:
    out: list[str] = []
    out.append(f"Candles as-of: `{snapshot.asof}`")
    if snapshot.close is not None:
        out.append(f"Close: `{snapshot.close:.2f}`")
    if snapshot.weekly_trend_up is not None:
        out.append(f"Weekly trend up: `{snapshot.weekly_trend_up}`")
    if snapshot.atr is not None:
        atrp = "-" if snapshot.atrp is None else f"{snapshot.atrp * 100.0:.2f}%"
        out.append(f"ATR{snapshot.atr_window}: `{snapshot.atr:.2f}` (ATR% `{atrp}`)")
    if snapshot.zscore is not None:
        out.append(f"Z{snapshot.z_window}: `{snapshot.zscore:+.2f}`")
    if snapshot.extension_atr is not None:
        out.append(f"Extension (Close vs SMA{snapshot.sma_window}, in ATR): `{snapshot.extension_atr:+.2f}`")
    if snapshot.bb_pband is not None or snapshot.bb_wband is not None:
        pband = "-" if snapshot.bb_pband is None else f"{snapshot.bb_pband:.2f}"
        wband = "-" if snapshot.bb_wband is None else f"{snapshot.bb_wband:.2f}"
        out.append(f"BB{snapshot.bb_window} dev={snapshot.bb_dev:g}: pband `{pband}`, wband `{wband}`")
    if snapshot.rsi_window is not None and snapshot.rsi is not None:
        out.append(f"RSI{snapshot.rsi_window}: `{snapshot.rsi:.0f}`")
    if snapshot.extension_percentiles is not None:
        for label, ext in [("Daily", snapshot.extension_percentiles.daily), ("Weekly", snapshot.extension_percentiles.weekly)]:
            if ext.current_percentiles:
                parts = [f"{years}y `{pct:.1f}`" for years, pct in sorted(ext.current_percentiles.items())]
                out.append(f"Extension percentile ({label}): " + ", ".join(parts))
            if ext.quantiles_by_window:
                rows = []
                for years, q in sorted(ext.quantiles_by_window.items()):
                    if q.p5 is None or q.p50 is None or q.p95 is None:
                        continue
                    rows.append(f"{years}y p5/p50/p95: `{q.p5:+.2f}/{q.p50:+.2f}/{q.p95:+.2f}`")
                if rows:
                    out.append(f"Extension quantiles ({label}): " + " | ".join(rows))
    return out


def strike_map_takeaways(
    *,
    chain: ChainReport,
    flow_net: pd.DataFrame | None,
    technicals: TechnicalSnapshot | None,
    top: int,
) -> list[str]:
    spot = float(chain.spot)
    atr = None if technicals is None else technicals.atr

    def _dist_atr(strike: float) -> str:
        if atr is None or atr <= 0:
            return "-"
        return f"{(strike - spot) / atr:+.2f} ATR"

    def _fmt_strike(label: str, strike: float | None) -> str | None:
        if strike is None:
            return None
        return f"{label}: `{strike:g}` ({_dist_atr(float(strike))})"

    out: list[str] = []

    call_wall = chain.walls_overall.calls[0].strike if chain.walls_overall.calls else None
    put_wall = chain.walls_overall.puts[0].strike if chain.walls_overall.puts else None
    gamma_peak = chain.gamma.peak_strike

    for label, strike in [("Call wall", call_wall), ("Put wall", put_wall), ("Gamma peak", gamma_peak)]:
        line = _fmt_strike(label, strike)
        if line is not None:
            out.append(line)

    if flow_net is None or flow_net.empty or "deltaOI_notional" not in flow_net.columns:
        return out

    df = flow_net.copy()
    df["deltaOI_notional"] = pd.to_numeric(df["deltaOI_notional"], errors="coerce")
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    df = df.dropna(subset=["deltaOI_notional", "strike"])
    if df.empty:
        return out

    df["_abs"] = df["deltaOI_notional"].abs()
    df = df.sort_values(["_abs", "optionType", "strike"], ascending=[False, True, True], na_position="last")

    def _fmt_flow_row(row) -> str:
        opt_type = str(row.get("optionType", "-"))
        strike = float(row.get("strike"))
        dnoi = float(row.get("deltaOI_notional"))
        return f"{opt_type} `{strike:g}`: ΔOI$ `{_fmt_money(dnoi, digits=0)}` ({_dist_atr(strike)})"

    building = df[df["deltaOI_notional"] > 0].head(top)
    unwinding = df[df["deltaOI_notional"] < 0].head(top)
    if not building.empty:
        out.append("Flow building (top):")
        for _, row in building.iterrows():
            out.append(f"- {_fmt_flow_row(row)}")
    if not unwinding.empty:
        out.append("Flow unwinding (top):")
        for _, row in unwinding.iterrows():
            out.append(f"- {_fmt_flow_row(row)}")
    return out


def _jsonable(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        import numpy as np

        if isinstance(val, np.generic):
            return _jsonable(val.item())
        if isinstance(val, np.ndarray):
            return [_jsonable(v) for v in val.tolist()]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(val, (datetime, pd.Timestamp, date)):
        return val.isoformat()
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonable(v) for v in val]
    if isinstance(val, pd.Series):
        return {str(k): _jsonable(v) for k, v in val.to_dict().items()}
    if isinstance(val, pd.DataFrame):
        return [_jsonable(r) for r in val.to_dict(orient="records")]
    if is_dataclass(val):
        return _jsonable(asdict(val))
    if isinstance(val, ChainReport):
        return _jsonable(val.model_dump())
    if isinstance(val, CompareReport):
        return _jsonable(val.model_dump())
    if isinstance(val, DerivedRow):
        return _jsonable(val.model_dump())
    return str(val)


def build_briefing_payload(
    *,
    report_date: str,
    portfolio_path: str,
    symbol_sections: list["BriefingSymbolSection"],
    top: int = 3,
    technicals_config: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Not financial advice. For informational/educational use only.",
        "report_date": report_date,
        "portfolio_path": portfolio_path,
        "symbols": [s.symbol for s in symbol_sections],
        "top": int(top),
        "technicals": {
            "source": "technicals_backtesting",
            "config_path": technicals_config,
        },
        "sections": [
            {
                "symbol": sec.symbol,
                "as_of": sec.as_of,
                "errors": list(sec.errors),
                "warnings": list(sec.warnings),
                "derived_updated": bool(sec.derived_updated),
                "next_earnings_date": None
                if sec.next_earnings_date is None
                else sec.next_earnings_date.isoformat(),
                "technicals": _jsonable(sec.technicals),
                "chain": _jsonable(sec.chain),
                "derived": _jsonable(sec.derived),
                "compare": _jsonable(sec.compare),
                "flow_net": _jsonable(sec.flow_net),
                "quote_quality": _jsonable(sec.quote_quality),
            }
            for sec in symbol_sections
        ],
    }


@dataclass(frozen=True)
class BriefingSymbolSection:
    symbol: str
    as_of: str
    chain: ChainReport | None
    compare: CompareReport | None
    flow_net: pd.DataFrame | None
    technicals: TechnicalSnapshot | None
    errors: list[str]
    warnings: list[str]
    quote_quality: dict[str, Any] | None = None
    derived_updated: bool = False
    derived: DerivedRow | None = None
    next_earnings_date: date | None = None


def render_portfolio_table_markdown(
    rows: list[dict[str, str]],
    *,
    include_spread: bool = False,
) -> str:
    if not rows:
        return ""

    headers = ["ID", "Sym", "Type", "Exp", "Strike", "Ct", "Cost", "Mark"]
    if include_spread:
        headers.append("Spr%")
    headers += ["PnL $", "PnL %", "As-of"]

    sep = ["---", "---", "---", "---", "---:", "---:", "---:", "---:"]
    if include_spread:
        sep.append("---:")
    sep += ["---:", "---:", "---"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        base = [
            r["id"],
            r["symbol"],
            r["type"],
            r["expiry"],
            r["strike"],
            r["ct"],
            r["cost"],
            r["mark"],
        ]
        if include_spread:
            base.append(r.get("spr_%", "-"))
        base += [
            r["pnl_$"],
            r["pnl_%"],
            r["as_of"],
        ]
        lines.append("| " + " | ".join(base) + " |")

    return "\n".join(lines)


def render_briefing_markdown(
    *,
    report_date: str,
    portfolio_path: str,
    symbol_sections: list[BriefingSymbolSection],
    portfolio_table_md: str | None = None,
    top: int = 3,
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

        as_of_date = None
        try:
            as_of_date = date.fromisoformat(sec.as_of)
        except Exception:  # noqa: BLE001
            as_of_date = date.today()
        lines.append(f"- {format_next_earnings_line(as_of_date, sec.next_earnings_date)}")
        if sec.quote_quality:
            lines.append(f"- {_fmt_quote_quality(sec.quote_quality)}")
        vol_line = vol_regime_takeaway(sec.derived)
        if vol_line:
            lines.append(f"- {vol_line}")
        lines.append("")

        if sec.errors and sec.chain is None and sec.technicals is None:
            lines.append("Errors:")
            for e in sec.errors:
                lines.append(f"- {e}")
            lines.append("")
            continue
        if sec.errors:
            lines.append("Errors:")
            for e in sec.errors:
                lines.append(f"- {e}")
            lines.append("")

        if sec.warnings:
            lines.append("Warnings:")
            for w in sec.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if sec.technicals is not None:
            lines.append("### Technicals (canonical: technicals_backtesting)")
            for b in technicals_takeaways(sec.technicals):
                lines.append(f"- {b}")
            lines.append("")

        if sec.chain is not None:
            if sec.technicals is not None:
                sm = strike_map_takeaways(
                    chain=sec.chain,
                    flow_net=sec.flow_net,
                    technicals=sec.technicals,
                    top=top,
                )
                if sm:
                    lines.append("### Strike map (spot + ATR distance)")
                    lines.append(f"- Spot (snapshot): `{sec.chain.spot:.2f}`")
                    for line in sm:
                        if line.startswith("- "):
                            lines.append(line)
                        elif line.endswith(":"):
                            lines.append(f"- {line}")
                        else:
                            lines.append(f"- {line}")
                    lines.append("")

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
            ft = flow_takeaways(sec.flow_net, top=top)
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
