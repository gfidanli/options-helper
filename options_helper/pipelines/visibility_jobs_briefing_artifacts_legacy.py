from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from rich.console import RenderableType

from options_helper.reporting_briefing import build_briefing_payload, render_briefing_markdown
from options_helper.schemas.briefing import BriefingArtifact

from .visibility_jobs_briefing_models_legacy import _PortfolioOutputs, _SymbolSelection


def _resolve_output_path(*, report_date: str, out: Path | None) -> Path:
    if out is None:
        return Path("data/reports/daily") / f"{report_date}.md"
    if out.suffix.lower() == ".md":
        return out
    return out / f"{report_date}.md"


def _write_briefing_artifacts(
    *,
    report_date: str,
    portfolio_path: Path,
    sections: list[Any],
    portfolio_outputs: _PortfolioOutputs,
    top: int,
    out: Path | None,
    write_json: bool,
    strict: bool,
    technicals_config: Path,
    selection: _SymbolSelection,
    portfolio_exposure: Any,
    portfolio_stress: Any,
    renderables: list[RenderableType],
) -> tuple[str, Path, Path | None]:
    markdown = render_briefing_markdown(
        report_date=report_date,
        portfolio_path=str(portfolio_path),
        symbol_sections=sections,
        portfolio_table_md=portfolio_outputs.table_markdown,
        top=top,
    )

    markdown_path = _resolve_output_path(report_date=report_date, out=out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    renderables.append(f"Saved: {markdown_path}")

    json_path: Path | None = None
    if write_json:
        payload = build_briefing_payload(
            report_date=report_date,
            as_of=report_date,
            portfolio_path=str(portfolio_path),
            symbol_sections=sections,
            top=top,
            technicals_config=str(technicals_config),
            portfolio_exposure=portfolio_exposure,
            portfolio_stress=portfolio_stress,
            portfolio_rows=portfolio_outputs.rows_payload,
            symbol_sources=selection.symbol_sources_payload,
            watchlists=selection.watchlists_payload,
        )
        if strict:
            BriefingArtifact.model_validate(payload)
        json_path = markdown_path.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        renderables.append(f"Saved: {json_path}")

    return markdown, markdown_path, json_path
