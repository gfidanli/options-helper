from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from .scanner_runtime import (
    _build_row_callback,
    _flush_tracking,
    _handle_empty_shortlist,
    _init_run_root,
    _load_confluence_cfg,
    _load_symbols,
    _resolve_tail_thresholds,
    _run_scan,
)
from .scanner_runtime_shortlist import (
    evaluate_shortlist,
    run_shortlist_reports,
    update_shortlist_watchlist,
    write_shortlist_outputs,
    write_shortlist_summary,
)


@dataclass(frozen=True)
class _ScannerSetup:
    cfg: dict[str, Any]
    confluence_cfg: Any
    tail_low_pct: float
    tail_high_pct: float
    symbols: list[str]
    exclude_symbols: set[str]
    scanned_symbols: set[str]
    run_stamp: str
    run_root: Path
    row_callback: Any
    new_error_symbols: set[str]
    new_scanned_symbols: set[str]


@dataclass(frozen=True)
class _ScanResult:
    provider: Any
    candle_store: Any
    scan_rows: list[Any]
    tail_symbols: list[str]
    scan_as_of: str


def _prepare_runtime(*, params: dict[str, Any], deps: SimpleNamespace, console: Console) -> _ScannerSetup:
    cfg = deps.load_technical_backtesting_config(params["config_path"])
    deps.setup_technicals_logging(cfg)
    confluence_cfg, _ = _load_confluence_cfg(deps=deps, console=console)
    tail_low_pct, tail_high_pct = _resolve_tail_thresholds(cfg=cfg, tail_pct=params["tail_pct"])
    symbols, exclude_symbols, scanned_symbols = _load_symbols(
        deps=deps,
        console=console,
        universe=params["universe"],
        universe_cache_dir=params["universe_cache_dir"],
        universe_refresh_days=params["universe_refresh_days"],
        prefilter_mode=params["prefilter_mode"],
        exclude_path=params["exclude_path"],
        scanned_path=params["scanned_path"],
        skip_scanned=bool(params["skip_scanned"]),
        write_scanned=bool(params["write_scanned"]),
        max_symbols=params["max_symbols"],
    )
    run_stamp, run_root = _init_run_root(run_dir=params["run_dir"], run_id=params["run_id"])
    row_callback, new_error_symbols, new_scanned_symbols = _build_row_callback(
        deps=deps,
        write_scanned=bool(params["write_scanned"]),
        scanned_path=params["scanned_path"],
        scanned_symbols=scanned_symbols,
        scanned_flush_every=int(params["scanned_flush_every"]),
        write_error_excludes=bool(params["write_error_excludes"]),
        exclude_path=params["exclude_path"],
        exclude_symbols=exclude_symbols,
        exclude_statuses=params["exclude_statuses"],
        error_flush_every=int(params["error_flush_every"]),
    )
    return _ScannerSetup(
        cfg=cfg,
        confluence_cfg=confluence_cfg,
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
        symbols=symbols,
        exclude_symbols=exclude_symbols,
        scanned_symbols=scanned_symbols,
        run_stamp=run_stamp,
        run_root=run_root,
        row_callback=row_callback,
        new_error_symbols=new_error_symbols,
        new_scanned_symbols=new_scanned_symbols,
    )


def _run_scan_stage(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    setup: _ScannerSetup,
) -> _ScanResult:
    provider, candle_store, scan_rows, tail_symbols, scan_as_of = _run_scan(
        deps=deps,
        params=params,
        console=console,
        cfg=setup.cfg,
        tail_low_pct=setup.tail_low_pct,
        tail_high_pct=setup.tail_high_pct,
        symbols=setup.symbols,
        row_callback=setup.row_callback,
    )
    return _ScanResult(
        provider=provider,
        candle_store=candle_store,
        scan_rows=scan_rows,
        tail_symbols=tail_symbols,
        scan_as_of=scan_as_of,
    )


def _write_scan_output(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    run_root: Path,
    scan_rows: list[Any],
) -> None:
    if not params["write_scan"]:
        return
    scan_path = run_root / "scan.csv"
    deps.write_scan_csv(scan_rows, scan_path)
    console.print(f"Wrote scan CSV: {scan_path}")


def _update_all_watchlist(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    tail_symbols: list[str],
) -> Any:
    watchlists = deps.load_watchlists(params["watchlists_path"])
    watchlists.set(params["all_watchlist_name"], tail_symbols)
    deps.save_watchlists(params["watchlists_path"], watchlists)
    console.print(
        f"Updated watchlist `{params['all_watchlist_name']}` ({len(tail_symbols)} symbol(s))"
    )
    return watchlists


def _maybe_backfill_candles(
    *,
    params: dict[str, Any],
    console: Console,
    candle_store: Any,
    tail_symbols: list[str],
) -> None:
    if not params["backfill"]:
        return
    console.print(f"Backfilling candles for {len(tail_symbols)} tail symbol(s)...")
    for sym in tail_symbols:
        try:
            candle_store.get_daily_history(sym, period="max")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {sym}: candle backfill failed: {exc}")


def _maybe_snapshot_options(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    provider: Any,
    tail_symbols: list[str],
) -> None:
    if not params["snapshot_options"]:
        return
    console.print(f"Snapshotting full options chains for {len(tail_symbols)} tail symbol(s)...")
    results = deps.snapshot_full_chain_for_symbols(
        tail_symbols,
        cache_dir=params["options_cache_dir"],
        candle_cache_dir=params["candle_cache_dir"],
        spot_period=params["spot_period"],
        max_expiries=None,
        risk_free_rate=params["risk_free_rate"],
        symbol_source="scanner",
        watchlists=[params["all_watchlist_name"]],
        provider=provider,
    )
    ok_count = sum(1 for result in results if result.status == "ok")
    console.print(f"Options snapshots complete: {ok_count}/{len(results)} ok")


def _post_scan_stage(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    setup: _ScannerSetup,
    scan: _ScanResult,
) -> tuple[Any, bool]:
    _flush_tracking(
        deps=deps,
        console=console,
        write_error_excludes=bool(params["write_error_excludes"]),
        exclude_path=params["exclude_path"],
        exclude_symbols=setup.exclude_symbols,
        new_error_symbols=setup.new_error_symbols,
        write_scanned=bool(params["write_scanned"]),
        scanned_path=params["scanned_path"],
        scanned_symbols=setup.scanned_symbols,
        new_scanned_symbols=setup.new_scanned_symbols,
    )
    _write_scan_output(
        params=params,
        deps=deps,
        console=console,
        run_root=setup.run_root,
        scan_rows=scan.scan_rows,
    )
    watchlists = _update_all_watchlist(
        params=params,
        deps=deps,
        console=console,
        tail_symbols=scan.tail_symbols,
    )
    is_empty = not scan.tail_symbols and _handle_empty_shortlist(
        deps=deps,
        params=params,
        console=console,
        wl=watchlists,
        run_root=setup.run_root,
        run_stamp=setup.run_stamp,
        scan_as_of=scan.scan_as_of,
        tail_low_pct=setup.tail_low_pct,
        tail_high_pct=setup.tail_high_pct,
    )
    return watchlists, is_empty


def _run_shortlist_stage(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    setup: _ScannerSetup,
    scan: _ScanResult,
    watchlists: Any,
) -> None:
    _maybe_backfill_candles(
        params=params,
        console=console,
        candle_store=scan.candle_store,
        tail_symbols=scan.tail_symbols,
    )
    _maybe_snapshot_options(
        params=params,
        deps=deps,
        console=console,
        provider=scan.provider,
        tail_symbols=scan.tail_symbols,
    )
    shortlist = evaluate_shortlist(
        params=params,
        deps=deps,
        candle_store=scan.candle_store,
        confluence_cfg=setup.confluence_cfg,
        scan_rows=scan.scan_rows,
        tail_symbols=scan.tail_symbols,
    )
    write_shortlist_outputs(
        params=params,
        deps=deps,
        console=console,
        run_root=setup.run_root,
        run_stamp=setup.run_stamp,
        scan_as_of=scan.scan_as_of,
        tail_low_pct=setup.tail_low_pct,
        tail_high_pct=setup.tail_high_pct,
        shortlist=shortlist,
    )
    update_shortlist_watchlist(
        params=params,
        deps=deps,
        console=console,
        watchlists=watchlists,
        shortlist_symbols=shortlist.shortlist_symbols,
    )
    run_shortlist_reports(
        params=params,
        deps=deps,
        console=console,
        shortlist_symbols=shortlist.shortlist_symbols,
    )
    write_shortlist_summary(
        params=params,
        console=console,
        run_root=setup.run_root,
        run_stamp=setup.run_stamp,
        tail_low_pct=setup.tail_low_pct,
        tail_high_pct=setup.tail_high_pct,
        shortlist=shortlist,
    )


def run_scanner_command(*, params: dict[str, Any], deps: SimpleNamespace) -> None:
    console = Console(width=200)
    setup = _prepare_runtime(params=params, deps=deps, console=console)
    scan = _run_scan_stage(params=params, deps=deps, console=console, setup=setup)
    watchlists, is_empty = _post_scan_stage(
        params=params,
        deps=deps,
        console=console,
        setup=setup,
        scan=scan,
    )
    if is_empty:
        return
    _run_shortlist_stage(
        params=params,
        deps=deps,
        console=console,
        setup=setup,
        scan=scan,
        watchlists=watchlists,
    )
    console.print("Not financial advice.")
