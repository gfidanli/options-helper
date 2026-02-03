from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.scanner_shortlist import ScannerShortlistArtifact

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class DailyBriefingPaths:
    date: str
    json_path: Path
    md_path: Path


def daily_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "daily"


def resolve_briefing_json(reports_dir: Path, date_spec: str) -> Path:
    daily_dir = daily_reports_dir(reports_dir)
    if date_spec == "latest":
        if not daily_dir.exists():
            raise FileNotFoundError(f"No daily reports directory at {daily_dir}")
        candidates = [p for p in daily_dir.glob("*.json") if p.is_file()]
        if not candidates:
            raise FileNotFoundError(f"No daily briefing JSON files under {daily_dir}")
        return sorted(candidates, key=lambda p: p.stem)[-1]

    try:
        date.fromisoformat(date_spec)
    except ValueError as exc:
        raise ValueError(f"Invalid date spec: {date_spec} (use YYYY-MM-DD or 'latest')") from exc

    path = daily_dir / f"{date_spec}.json"
    if not path.exists():
        raise FileNotFoundError(f"Briefing JSON not found: {path}")
    return path


def resolve_briefing_paths(reports_dir: Path, date_spec: str) -> DailyBriefingPaths:
    json_path = resolve_briefing_json(reports_dir, date_spec)
    report_date = json_path.stem
    return DailyBriefingPaths(
        date=report_date,
        json_path=json_path,
        md_path=json_path.with_suffix(".md"),
    )


def load_briefing_artifact(path: Path) -> BriefingArtifact:
    if not path.exists():
        raise FileNotFoundError(f"Briefing JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BriefingArtifact.model_validate(payload)


def load_scanner_shortlist(
    run_dir: Path,
    *,
    report_date: str | None = None,
    run_id: str | None = None,
) -> ScannerShortlistArtifact | None:
    if run_id:
        candidate = run_dir / run_id / "shortlist.json"
        if not candidate.exists():
            return None
        return ScannerShortlistArtifact.model_validate_json(candidate.read_text(encoding="utf-8"))

    if not run_dir.exists():
        return None

    runs = [p for p in run_dir.iterdir() if p.is_dir()]
    if report_date:
        runs = [p for p in runs if p.name.startswith(report_date)]
    if not runs:
        return None

    selected = sorted(runs, key=lambda p: p.name)[-1]
    shortlist_path = selected / "shortlist.json"
    if not shortlist_path.exists():
        return None
    return ScannerShortlistArtifact.model_validate_json(shortlist_path.read_text(encoding="utf-8"))


def render_dashboard(
    *,
    artifact: BriefingArtifact,
    console: Console,
    reports_dir: Path,
    scanner_run_dir: Path,
    scanner_run_id: str | None = None,
    max_shortlist_rows: int = 20,
) -> None:
    console.rule(f"Daily Briefing Dashboard — {artifact.report_date}")
    console.print(f"Portfolio: {artifact.portfolio_path}")
    console.print(f"Symbols: {', '.join(artifact.symbols) if artifact.symbols else '-'}")
    console.print(f"Generated: {artifact.generated_at}")

    if artifact.portfolio_rows:
        console.print()
        console.print(_render_portfolio_table(artifact))

    sections_by_symbol = {sec.symbol.upper(): sec for sec in artifact.sections}
    sources_by_symbol = {row.symbol.upper(): row.sources for row in artifact.symbol_sources}

    console.print()
    console.print(_render_symbol_summary(artifact, sources_by_symbol))

    if artifact.watchlists:
        console.print()
        console.rule("Watchlists")
        for wl in artifact.watchlists:
            table = Table(title=f"Watchlist: {wl.name} ({len(wl.symbols)})")
            table.add_column("Symbol")
            table.add_column("As-of")
            table.add_column("Sources")
            table.add_column("Chain")
            table.add_column("Compare")
            table.add_column("Flow")
            table.add_column("Derived")
            table.add_column("Technicals")

            for sym in wl.symbols:
                section = sections_by_symbol.get(sym.upper())
                paths = _artifact_path_hints(reports_dir, sym, section)
                sources = ", ".join(sources_by_symbol.get(sym.upper(), [])) or "-"
                table.add_row(
                    sym.upper(),
                    "-" if section is None else section.as_of,
                    sources,
                    paths["chain"],
                    paths["compare"],
                    paths["flow"],
                    paths["derived"],
                    paths["technicals"],
                )
            console.print(table)

    shortlist = load_scanner_shortlist(
        scanner_run_dir,
        report_date=artifact.report_date,
        run_id=scanner_run_id,
    )
    if shortlist is not None:
        console.print()
        console.rule("Scanner Shortlist")
        table = Table(title=f"Scanner Shortlist — {shortlist.run_id} (as-of {shortlist.as_of})")
        table.add_column("Symbol")
        table.add_column("Score", justify="right")
        table.add_column("Coverage", justify="right")
        table.add_column("Top reasons")
        for row in shortlist.rows[:max_shortlist_rows]:
            score = "-" if row.score is None else f"{row.score:.1f}"
            coverage = "-" if row.coverage is None else f"{row.coverage * 100.0:.0f}%"
            table.add_row(row.symbol, score, coverage, row.top_reasons or "-")
        console.print(table)


def _render_portfolio_table(artifact: BriefingArtifact) -> Table:
    table = Table(title="Portfolio Positions")
    table.add_column("ID")
    table.add_column("Sym")
    table.add_column("Type")
    table.add_column("Exp")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Mark", justify="right")
    table.add_column("PnL $", justify="right")
    table.add_column("PnL %", justify="right")
    table.add_column("Spr%", justify="right")
    table.add_column("As-of")

    for row in artifact.portfolio_rows:
        table.add_row(
            row.id,
            row.symbol,
            row.option_type,
            row.expiry,
            f"{row.strike:g}",
            str(row.contracts),
            _fmt_number(row.cost_basis, digits=2),
            _fmt_number(row.mark, digits=2),
            _fmt_number(row.pnl, digits=0, sign=True),
            _fmt_pct(row.pnl_pct, digits=1, sign=True),
            _fmt_pct(row.spr_pct, digits=1),
            row.as_of or "-",
        )
    return table


def _render_symbol_summary(
    artifact: BriefingArtifact,
    sources_by_symbol: dict[str, list[str]],
) -> Table:
    table = Table(title="Symbol Summary")
    table.add_column("Symbol")
    table.add_column("As-of")
    table.add_column("Sources")
    table.add_column("Earnings")
    table.add_column("Confluence", justify="right")
    table.add_column("Flags")

    for sec in artifact.sections:
        sources = ", ".join(sources_by_symbol.get(sec.symbol.upper(), [])) or "-"
        earnings = _format_earnings(sec.as_of, sec.next_earnings_date)
        confluence = _format_confluence(sec.confluence)
        flags = _format_flags(sec.errors, sec.warnings)
        table.add_row(sec.symbol, sec.as_of, sources, earnings, confluence, flags)
    return table


def _format_flags(errors: list[str], warnings: list[str]) -> str:
    parts: list[str] = []
    if errors:
        parts.append("errors: " + ", ".join(errors))
    if warnings:
        parts.append("warnings: " + ", ".join(warnings))
    return " | ".join(parts) if parts else "-"


def _format_confluence(confluence: dict[str, Any] | None) -> str:
    if not confluence:
        return "-"
    total = _safe_float(confluence.get("total"))
    coverage = _safe_float(confluence.get("coverage"))
    if total is None and coverage is None:
        return "-"
    if total is None:
        return f"{coverage * 100.0:.0f}%"
    if coverage is None:
        return f"{float(total):.0f}"
    return f"{float(total):.0f} ({coverage * 100.0:.0f}%)"


def _format_earnings(as_of: str | None, next_date: str | None) -> str:
    if next_date is None:
        return "-"
    as_of_date = _parse_iso_date(as_of)
    next_earnings = _parse_iso_date(next_date)
    if next_earnings is None:
        return "-"
    if as_of_date is None:
        return next_earnings.isoformat()
    delta_days = (next_earnings - as_of_date).days
    if delta_days == 0:
        suffix = "today"
    elif delta_days > 0:
        suffix = f"in {delta_days}d"
    else:
        suffix = f"{abs(delta_days)}d ago"
    return f"{next_earnings.isoformat()} ({suffix})"


def _artifact_path_hints(
    reports_dir: Path,
    symbol: str,
    section: Any | None,
) -> dict[str, str]:
    sym = symbol.upper()
    chain_dir = reports_dir / "chains" / sym
    compare_dir = reports_dir / "compare" / sym
    flow_dir = reports_dir / "flow" / sym
    derived_dir = reports_dir / "derived" / sym
    technicals_dir = reports_dir / "technicals" / "extension" / sym

    as_of = None
    if section is not None:
        as_of = getattr(section, "as_of", None)

    chain_path = chain_dir
    if _is_iso_date(as_of):
        candidate_md = chain_dir / f"{as_of}.md"
        candidate_json = chain_dir / f"{as_of}.json"
        if candidate_md.exists():
            chain_path = candidate_md
        elif candidate_json.exists():
            chain_path = candidate_json

    compare_path = compare_dir
    flow_path = flow_dir
    if section is not None and getattr(section, "compare", None):
        compare = getattr(section, "compare")
        if isinstance(compare, dict):
            from_date = compare.get("from_date")
            to_date = compare.get("to_date")
            if _is_iso_date(from_date) and _is_iso_date(to_date):
                compare_file = compare_dir / f"{from_date}_to_{to_date}.json"
                flow_file = flow_dir / f"{from_date}_to_{to_date}_w1_expiry-strike.json"
                if compare_file.exists():
                    compare_path = compare_file
                if flow_file.exists():
                    flow_path = flow_file

    return {
        "chain": str(chain_path),
        "compare": str(compare_path),
        "flow": str(flow_path),
        "derived": str(derived_dir),
        "technicals": str(technicals_dir),
    }


def _fmt_number(val: float | None, *, digits: int = 2, sign: bool = False) -> str:
    if val is None:
        return "-"
    if sign:
        return f"{val:+.{digits}f}"
    return f"{val:.{digits}f}"


def _fmt_pct(val: float | None, *, digits: int = 1, sign: bool = False) -> str:
    if val is None:
        return "-"
    scaled = val * 100.0
    return _fmt_number(scaled, digits=digits, sign=sign) + "%"


def _parse_iso_date(value: str | None) -> date | None:
    if not value or value == "-":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _is_iso_date(value: str | None) -> bool:
    return _parse_iso_date(value) is not None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
