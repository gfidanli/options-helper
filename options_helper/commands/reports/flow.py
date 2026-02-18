from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import reports_legacy as legacy
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


_PORTFOLIO_PATH_ARG = typer.Argument(..., help="Path to portfolio JSON.")
_SYMBOL_OPT = typer.Option(None, "--symbol", help="Restrict flow report to a single symbol.")
_WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store.",
)
_WATCHLIST_OPT = typer.Option(
    [],
    "--watchlist",
    help="Use symbols from watchlist (repeatable). Ignored when --all-watchlists is set.",
)
_ALL_WATCHLISTS_OPT = typer.Option(
    False,
    "--all-watchlists",
    help="Use all watchlists instead of portfolio symbols.",
)
_CACHE_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--cache-dir",
    help="Directory containing options snapshot folders.",
)
_WINDOW_OPT = typer.Option(
    1,
    "--window",
    min=1,
    max=30,
    help="Number of snapshot-to-snapshot deltas to net (requires N+1 snapshots).",
)
_GROUP_BY_OPT = typer.Option(
    "contract",
    "--group-by",
    help="Aggregation mode: contract|strike|expiry|expiry-strike",
)
_TOP_OPT = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display.")
_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Output root for saved artifacts (writes under {out}/flow/{SYMBOL}/).",
)
_STRICT_OPT = typer.Option(False, "--strict", help="Validate JSON artifacts against schemas.")


def flow_report(
    portfolio_path: Path = _PORTFOLIO_PATH_ARG,
    symbol: str | None = _SYMBOL_OPT,
    watchlists_path: Path = _WATCHLISTS_PATH_OPT,
    watchlist: list[str] = _WATCHLIST_OPT,
    all_watchlists: bool = _ALL_WATCHLISTS_OPT,
    cache_dir: Path = _CACHE_DIR_OPT,
    window: int = _WINDOW_OPT,
    group_by: str = _GROUP_BY_OPT,
    top: int = _TOP_OPT,
    out: Path | None = _OUT_OPT,
    strict: bool = _STRICT_OPT,
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    console = Console()
    with legacy._observed_run(
        console=console,
        job_name=legacy.JOB_COMPUTE_FLOW,
        args=_flow_run_args(
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
        ),
    ) as run_logger:
        result = _run_flow_job(
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
        )
        _render_flow_output(console, result)
        _log_flow_assets(run_logger, result)


def _flow_run_args(
    *,
    portfolio_path: Path,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    cache_dir: Path,
    window: int,
    group_by: str,
    top: int,
    out: Path | None,
    strict: bool,
) -> dict[str, object]:
    return {
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
    }


def _run_flow_job(
    *,
    portfolio_path: Path,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    cache_dir: Path,
    window: int,
    group_by: str,
    top: int,
    out: Path | None,
    strict: bool,
) -> Any:
    import options_helper.commands.reports as reports_pkg

    try:
        return reports_pkg.run_flow_report_job(
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


def _render_flow_output(console: Console, result: Any) -> None:
    for renderable in result.renderables:
        console.print(renderable)


def _log_flow_assets(run_logger: Any, result: Any) -> None:
    if result.no_symbols:
        run_logger.log_asset_skipped(
            asset_key=legacy.ASSET_OPTIONS_FLOW,
            asset_kind="file",
            partition_key="ALL",
            extra={"reason": "no_symbols"},
        )
        raise typer.Exit(0)

    success_by_symbol, skipped_symbols = legacy._flow_renderable_statuses(result.renderables)
    _log_symbol_flow_statuses(run_logger, success_by_symbol=success_by_symbol, skipped_symbols=skipped_symbols)
    _log_global_flow_watermark(run_logger, success_by_symbol=success_by_symbol, skipped_symbols=skipped_symbols)


def _log_symbol_flow_statuses(run_logger: Any, *, success_by_symbol: dict[str, Any], skipped_symbols: set[str]) -> None:
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


def _log_global_flow_watermark(run_logger: Any, *, success_by_symbol: dict[str, Any], skipped_symbols: set[str]) -> None:
    if success_by_symbol:
        latest = max((d for d in success_by_symbol.values() if d is not None), default=None)
        if latest is not None:
            run_logger.upsert_watermark(
                asset_key=legacy.ASSET_OPTIONS_FLOW,
                scope_key="ALL",
                watermark_ts=latest,
            )
        return
    if not skipped_symbols:
        run_logger.log_asset_success(
            asset_key=legacy.ASSET_OPTIONS_FLOW,
            asset_kind="file",
            partition_key="ALL",
        )


__all__ = ["flow_report"]
