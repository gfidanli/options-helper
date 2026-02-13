from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import reports_legacy as legacy
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


def flow_report(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Restrict flow report to a single symbol.",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Use symbols from watchlist (repeatable). Ignored when --all-watchlists is set.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Use all watchlists instead of portfolio symbols.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory containing options snapshot folders.",
    ),
    window: int = typer.Option(
        1,
        "--window",
        min=1,
        max=30,
        help="Number of snapshot-to-snapshot deltas to net (requires N+1 snapshots).",
    ),
    group_by: str = typer.Option(
        "contract",
        "--group-by",
        help="Aggregation mode: contract|strike|expiry|expiry-strike",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/flow/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    console = Console()

    import options_helper.commands.reports as reports_pkg

    with legacy._observed_run(
        console=console,
        job_name=legacy.JOB_COMPUTE_FLOW,
        args={
            "portfolio_path": str(portfolio_path),
            "symbol": symbol,
            "watchlists_path": str(watchlists_path),
            "watchlist": watchlist,
            "all_watchlists": all_watchlists,
            "cache_dir": str(cache_dir),
            "window": window,
            "group_by": group_by,
            "top": top,
            "out": None if out is None else str(out),
            "strict": strict,
        },
    ) as run_logger:
        try:
            result = reports_pkg.run_flow_report_job(
                portfolio_path=portfolio_path,
                symbol=symbol,
                watchlists_path=watchlists_path,
                watchlist=watchlist,
                all_watchlists=all_watchlists,
                cache_dir=cache_dir,
                window=window,
                group_by=group_by,
                top=top,
                out=out,
                strict=strict,
                snapshot_store_builder=cli_deps.build_snapshot_store,
                portfolio_loader=load_portfolio,
                watchlists_loader=load_watchlists,
            )
        except legacy.VisibilityJobParameterError as exc:
            if exc.param_hint:
                raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
            raise typer.BadParameter(str(exc)) from exc

        for renderable in result.renderables:
            console.print(renderable)

        if result.no_symbols:
            run_logger.log_asset_skipped(
                asset_key=legacy.ASSET_OPTIONS_FLOW,
                asset_kind="file",
                partition_key="ALL",
                extra={"reason": "no_symbols"},
            )
            raise typer.Exit(0)

        success_by_symbol, skipped_symbols = legacy._flow_renderable_statuses(result.renderables)
        for sym, flow_end_date in sorted(success_by_symbol.items()):
            run_logger.log_asset_success(
                asset_key=legacy.ASSET_OPTIONS_FLOW,
                asset_kind="file",
                partition_key=sym,
                min_event_ts=flow_end_date,
                max_event_ts=flow_end_date,
            )
            if flow_end_date is not None:
                run_logger.upsert_watermark(
                    asset_key=legacy.ASSET_OPTIONS_FLOW,
                    scope_key=sym,
                    watermark_ts=flow_end_date,
                )

        for sym in sorted(skipped_symbols):
            run_logger.log_asset_skipped(
                asset_key=legacy.ASSET_OPTIONS_FLOW,
                asset_kind="file",
                partition_key=sym,
                extra={"reason": "insufficient_snapshots"},
            )

        if success_by_symbol:
            latest = max((d for d in success_by_symbol.values() if d is not None), default=None)
            if latest is not None:
                run_logger.upsert_watermark(
                    asset_key=legacy.ASSET_OPTIONS_FLOW,
                    scope_key="ALL",
                    watermark_ts=latest,
                )
        elif not skipped_symbols:
            run_logger.log_asset_success(
                asset_key=legacy.ASSET_OPTIONS_FLOW,
                asset_kind="file",
                partition_key="ALL",
            )


__all__ = ["flow_report"]
