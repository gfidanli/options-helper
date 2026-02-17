from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import reports_legacy as legacy


def briefing(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). Adds to portfolio symbols.",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only include a single symbol (overrides portfolio/watchlists selection).",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    compare: str = typer.Option(
        "-1",
        "--compare",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technical context).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (canonical indicator definitions).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path (Markdown) or directory. Default: data/reports/daily/{ASOF}.md",
    ),
    print_to_console: bool = typer.Option(
        False,
        "--print/--no-print",
        help="Print the briefing to the console (in addition to writing files).",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write a JSON version of the briefing alongside the Markdown (LLM-friendly).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    update_derived: bool = typer.Option(
        True,
        "--update-derived/--no-update-derived",
        help="Update derived metrics for included symbols (per-symbol CSV).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used when --update-derived).",
    ),
    top: int = typer.Option(3, "--top", min=1, max=10, help="Top rows to include in compare/flow sections."),
) -> None:
    """Generate a daily Markdown briefing for portfolio + optional watchlists (offline-first)."""
    console = Console(width=200)

    import options_helper.commands.reports as reports_pkg

    with legacy._observed_run(
        console=console,
        job_name=legacy.JOB_BUILD_BRIEFING,
        args={
            "portfolio_path": str(portfolio_path),
            "watchlists_path": str(watchlists_path),
            "watchlist": watchlist,
            "symbol": symbol,
            "as_of": as_of,
            "compare": compare,
            "cache_dir": str(cache_dir),
            "candle_cache_dir": str(candle_cache_dir),
            "technicals_config": str(technicals_config),
            "out": None if out is None else str(out),
            "print_to_console": print_to_console,
            "write_json": write_json,
            "strict": strict,
            "update_derived": update_derived,
            "derived_dir": str(derived_dir),
            "top": top,
        },
    ) as run_logger:
        try:
            result = reports_pkg.run_briefing_job(
                portfolio_path=portfolio_path,
                watchlists_path=watchlists_path,
                watchlist=watchlist,
                symbol=symbol,
                as_of=as_of,
                compare=compare,
                cache_dir=cache_dir,
                candle_cache_dir=candle_cache_dir,
                technicals_config=technicals_config,
                out=out,
                print_to_console=print_to_console,
                write_json=write_json,
                strict=strict,
                update_derived=update_derived,
                derived_dir=derived_dir,
                top=top,
                snapshot_store_builder=cli_deps.build_snapshot_store,
                derived_store_builder=cli_deps.build_derived_store,
                candle_store_builder=cli_deps.build_candle_store,
                earnings_store_builder=cli_deps.build_earnings_store,
                safe_next_earnings_date_fn=reports_pkg.safe_next_earnings_date,
            )
        except legacy.VisibilityJobExecutionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

        report_day = legacy._coerce_iso_date(result.report_date)
        md_bytes = result.markdown_path.stat().st_size if result.markdown_path.exists() else None
        run_logger.log_asset_success(
            asset_key=legacy.ASSET_BRIEFING_MARKDOWN,
            asset_kind="file",
            partition_key=result.report_date,
            bytes_written=md_bytes,
            min_event_ts=report_day,
            max_event_ts=report_day,
        )
        if report_day is not None:
            run_logger.upsert_watermark(
                asset_key=legacy.ASSET_BRIEFING_MARKDOWN,
                scope_key="ALL",
                watermark_ts=report_day,
            )

        if result.json_path is not None:
            json_bytes = result.json_path.stat().st_size if result.json_path.exists() else None
            run_logger.log_asset_success(
                asset_key=legacy.ASSET_BRIEFING_JSON,
                asset_kind="file",
                partition_key=result.report_date,
                bytes_written=json_bytes,
                min_event_ts=report_day,
                max_event_ts=report_day,
            )
            if report_day is not None:
                run_logger.upsert_watermark(
                    asset_key=legacy.ASSET_BRIEFING_JSON,
                    scope_key="ALL",
                    watermark_ts=report_day,
                )
        else:
            run_logger.log_asset_skipped(
                asset_key=legacy.ASSET_BRIEFING_JSON,
                asset_kind="file",
                partition_key=result.report_date,
                extra={"reason": "write_json_disabled"},
            )

        for renderable in result.renderables:
            console.print(renderable)


def _resolve_dashboard_report_day(report_date: str, artifact: object) -> date | None:
    report_day = legacy._coerce_iso_date(report_date)
    if report_day is not None:
        return report_day
    if isinstance(artifact, dict):
        return legacy._coerce_iso_date(artifact.get("as_of") or artifact.get("report_date"))
    as_of = legacy._coerce_iso_date(getattr(artifact, "as_of", None))
    if as_of is not None:
        return as_of
    return legacy._coerce_iso_date(getattr(artifact, "report_date", None))


def dashboard(
    report_date: str = typer.Option(
        "latest",
        "--date",
        help="Briefing date (YYYY-MM-DD) or 'latest'.",
    ),
    reports_dir: Path = typer.Option(
        Path("data/reports"),
        "--reports-dir",
        help="Reports root (expects {reports_dir}/daily/{DATE}.json).",
    ),
    scanner_run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--scanner-run-dir",
        help="Scanner runs directory (for shortlist view).",
    ),
    scanner_run_id: str | None = typer.Option(
        None,
        "--scanner-run-id",
        help="Specific scanner run id to display (defaults to latest for the date).",
    ),
    max_shortlist_rows: int = typer.Option(
        20,
        "--max-shortlist-rows",
        min=1,
        max=200,
        help="Max rows to show in the scanner shortlist table.",
    ),
) -> None:
    """Render a read-only daily dashboard from briefing JSON + artifacts."""
    console = Console(width=200)

    import options_helper.commands.reports as reports_pkg

    with legacy._observed_run(
        console=console,
        job_name=legacy.JOB_BUILD_DASHBOARD,
        args={
            "report_date": report_date,
            "reports_dir": str(reports_dir),
            "scanner_run_dir": str(scanner_run_dir),
            "scanner_run_id": scanner_run_id,
            "max_shortlist_rows": max_shortlist_rows,
        },
    ) as run_logger:
        try:
            result = reports_pkg.run_dashboard_job(
                report_date=report_date,
                reports_dir=reports_dir,
            )
        except legacy.VisibilityJobExecutionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

        report_day = _resolve_dashboard_report_day(report_date, result.artifact)
        json_bytes = result.json_path.stat().st_size if result.json_path.exists() else None
        partition_key = report_day.isoformat() if report_day is not None else str(report_date)
        run_logger.log_asset_success(
            asset_key=legacy.ASSET_DASHBOARD,
            asset_kind="view",
            partition_key=partition_key,
            bytes_written=json_bytes,
            min_event_ts=report_day,
            max_event_ts=report_day,
        )
        if report_day is not None:
            run_logger.upsert_watermark(
                asset_key=legacy.ASSET_DASHBOARD,
                scope_key="ALL",
                watermark_ts=report_day,
            )

        reports_pkg.render_dashboard_report(
            result=result,
            reports_dir=reports_dir,
            scanner_run_dir=scanner_run_dir,
            scanner_run_id=scanner_run_id,
            max_shortlist_rows=max_shortlist_rows,
            render_console=console,
        )


__all__ = ["briefing", "dashboard"]
