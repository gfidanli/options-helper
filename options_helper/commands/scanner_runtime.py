from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from rich.console import Console


def _resolve_tail_thresholds(*, cfg: dict[str, Any], tail_pct: float | None) -> tuple[float, float]:
    ext_cfg = cfg.get("extension_percentiles", {})
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_low_pct = tail_low_cfg
        tail_high_pct = tail_high_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp
    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")
    return tail_low_pct, tail_high_pct


def _load_confluence_cfg(*, deps: SimpleNamespace, console: Console) -> tuple[Any | None, str | None]:
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = deps.load_confluence_config()
    except deps.ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")
    return confluence_cfg, confluence_cfg_error


def _load_symbols(
    *,
    deps: SimpleNamespace,
    console: Console,
    universe: str,
    universe_cache_dir: Path,
    universe_refresh_days: int,
    prefilter_mode: str,
    exclude_path: Path,
    scanned_path: Path,
    skip_scanned: bool,
    write_scanned: bool,
    max_symbols: int | None,
) -> tuple[list[str], set[str], set[str]]:
    try:
        symbols = deps.load_universe_symbols(
            universe,
            cache_dir=universe_cache_dir,
            refresh_days=universe_refresh_days,
        )
    except deps.UniverseError as exc:
        console.print(f"[red]Universe error:[/red] {exc}")
        raise typer.Exit(1) from exc

    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    exclude_symbols = deps.read_exclude_symbols(exclude_path) if exclude_path else set()
    if exclude_symbols:
        console.print(f"Loaded {len(exclude_symbols)} excluded symbol(s) from {exclude_path}")

    scanned_symbols: set[str] = set()
    if scanned_path and (skip_scanned or write_scanned):
        scanned_symbols = deps.read_scanned_symbols(scanned_path)
        if scanned_symbols:
            console.print(f"Loaded {len(scanned_symbols)} scanned symbol(s) from {scanned_path}")

    filtered, dropped = deps.prefilter_symbols(
        symbols,
        mode=prefilter_mode,
        exclude=exclude_symbols,
        scanned=scanned_symbols if skip_scanned else None,
    )
    dropped_n = sum(dropped.values())
    if dropped_n:
        console.print(f"Prefiltered symbols: dropped {dropped_n} ({dropped})")
    symbols = filtered[: int(max_symbols)] if max_symbols is not None else filtered
    if not symbols:
        console.print("[yellow]No symbols found in universe.[/yellow]")
        raise typer.Exit(0)
    return symbols, exclude_symbols, scanned_symbols


def _init_run_root(*, run_dir: Path, run_id: str | None) -> tuple[str, Path]:
    run_stamp = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = run_dir / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)
    return run_stamp, run_root


def _build_row_callback(
    *,
    deps: SimpleNamespace,
    write_scanned: bool,
    scanned_path: Path | None,
    scanned_symbols: set[str],
    scanned_flush_every: int,
    write_error_excludes: bool,
    exclude_path: Path | None,
    exclude_symbols: set[str],
    exclude_statuses: str,
    error_flush_every: int,
) -> tuple[Any, set[str], set[str]]:
    new_error_symbols: set[str] = set()
    new_scanned_symbols: set[str] = set()
    status_set = {s.strip().lower() for s in exclude_statuses.split(",") if s.strip()}

    def _row_callback(row: Any) -> None:
        if write_scanned and scanned_path is not None:
            sym = row.symbol
            if sym not in scanned_symbols:
                scanned_symbols.add(sym)
                new_scanned_symbols.add(sym)
                if len(new_scanned_symbols) >= int(scanned_flush_every):
                    deps.write_scanned_symbols(scanned_path, scanned_symbols)
                    new_scanned_symbols.clear()
        if not write_error_excludes or exclude_path is None:
            return
        if str(row.status).strip().lower() not in status_set:
            return
        sym = row.symbol
        if sym not in exclude_symbols:
            exclude_symbols.add(sym)
            new_error_symbols.add(sym)
            if len(new_error_symbols) >= int(error_flush_every):
                deps.write_exclude_symbols(exclude_path, exclude_symbols)
                new_error_symbols.clear()

    return _row_callback, new_error_symbols, new_scanned_symbols


def _scan_as_of(scan_rows: list[Any]) -> str:
    scan_as_of_dates: list[date] = []
    for row in scan_rows:
        if not row.asof:
            continue
        try:
            scan_as_of_dates.append(date.fromisoformat(row.asof))
        except ValueError:
            continue
    return max(scan_as_of_dates).isoformat() if scan_as_of_dates else date.today().isoformat()


def _run_scan(
    *,
    deps: SimpleNamespace,
    params: dict[str, Any],
    console: Console,
    cfg: dict[str, Any],
    tail_low_pct: float,
    tail_high_pct: float,
    symbols: list[str],
    row_callback: Any,
) -> tuple[Any, Any, list[Any], list[str], str]:
    console.print(
        f"Scanning {len(symbols)} symbol(s) from `{params['universe']}` "
        f"(tail {tail_low_pct:.1f}/{tail_high_pct:.1f})..."
    )
    provider = deps.cli_deps.build_provider()
    candle_store = deps.cli_deps.build_candle_store(params["candle_cache_dir"], provider=provider)
    scan_rows, tail_symbols = deps.scan_symbols(
        symbols,
        candle_store=candle_store,
        cfg=cfg,
        scan_period=params["scan_period"],
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        percentile_window_years=params["percentile_window_years"],
        workers=params["workers"],
        batch_size=params["batch_size"],
        batch_sleep_seconds=params["batch_sleep_seconds"],
        row_callback=row_callback,
    )
    return provider, candle_store, scan_rows, tail_symbols, _scan_as_of(scan_rows)


def _flush_tracking(
    *,
    deps: SimpleNamespace,
    console: Console,
    write_error_excludes: bool,
    exclude_path: Path | None,
    exclude_symbols: set[str],
    new_error_symbols: set[str],
    write_scanned: bool,
    scanned_path: Path | None,
    scanned_symbols: set[str],
    new_scanned_symbols: set[str],
) -> None:
    if write_error_excludes and new_error_symbols and exclude_path is not None:
        deps.write_exclude_symbols(exclude_path, exclude_symbols)
        console.print(f"Wrote {len(new_error_symbols)} new excluded symbol(s) to {exclude_path}")
    if write_scanned and new_scanned_symbols and scanned_path is not None:
        deps.write_scanned_symbols(scanned_path, scanned_symbols)
        console.print(f"Wrote {len(new_scanned_symbols)} new scanned symbol(s) to {scanned_path}")


def _write_shortlist_json(
    *,
    deps: SimpleNamespace,
    path: Path,
    strict: bool,
    scan_as_of: str,
    run_stamp: str,
    universe: str,
    tail_low_pct: float,
    tail_high_pct: float,
    all_watchlist_name: str,
    shortlist_watchlist_name: str,
    rows: list[Any],
    console: Console,
) -> None:
    payload = deps.ScannerShortlistArtifact(
        schema_version=1,
        generated_at=deps.utc_now(),
        as_of=scan_as_of,
        run_id=run_stamp,
        universe=universe,
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        all_watchlist_name=all_watchlist_name,
        shortlist_watchlist_name=shortlist_watchlist_name,
        rows=rows,
    ).to_dict()
    if strict:
        deps.ScannerShortlistArtifact.model_validate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"Wrote shortlist JSON: {path}")


def _write_shortlist_md(
    *,
    path: Path,
    run_stamp: str,
    universe: str,
    tail_low_pct: float,
    tail_high_pct: float,
    all_watchlist_name: str,
    shortlist_watchlist_name: str,
    shortlist_symbols: list[str],
    rank_results: dict[str, Any],
    confluence_scores: dict[str, Any],
    reports_out: Path,
) -> None:
    lines = [
        f"# Scanner Shortlist — {run_stamp}",
        "",
        f"- Universe: `{universe}`",
        f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
        f"- Tail watchlist: `{all_watchlist_name}`",
        f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
        "- Ranking: `scanner score` (desc)",
        f"- Symbols: `{len(shortlist_symbols)}`",
        "",
        "Not financial advice.",
        "",
        "## Symbols",
    ]
    if shortlist_symbols:
        for sym in shortlist_symbols:
            parts: list[str] = []
            rank = rank_results.get(sym)
            if rank is not None:
                parts.append(f"scanner {rank.score:.0f}, cov {rank.coverage * 100.0:.0f}%")
            score = confluence_scores.get(sym)
            if score is not None:
                parts.append(f"confluence {score.total:.0f}, cov {score.coverage * 100.0:.0f}%")
            lines.append(f"- `{sym}` ({'; '.join(parts)}) → `{reports_out / sym}`" if parts else f"- `{sym}` → `{reports_out / sym}`")
    else:
        lines.append("- (empty)")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _handle_empty_shortlist(
    *,
    deps: SimpleNamespace,
    params: dict[str, Any],
    console: Console,
    wl: Any,
    run_root: Path,
    run_stamp: str,
    scan_as_of: str,
    tail_low_pct: float,
    tail_high_pct: float,
) -> bool:
    wl.set(params["shortlist_watchlist_name"], [])
    deps.save_watchlists(params["watchlists_path"], wl)
    console.print("[yellow]No tail symbols found; shortlist cleared.[/yellow]")
    if params["write_liquidity"]:
        liquidity_path = run_root / "liquidity.csv"
        deps.write_liquidity_csv([], liquidity_path)
        console.print(f"Wrote liquidity CSV: {liquidity_path}")
    if params["write_shortlist"]:
        shortlist_csv = run_root / "shortlist.csv"
        deps.write_shortlist_csv([], shortlist_csv)
        console.print(f"Wrote shortlist CSV: {shortlist_csv}")
        _write_shortlist_json(
            deps=deps,
            path=run_root / "shortlist.json",
            strict=bool(params["strict"]),
            scan_as_of=scan_as_of,
            run_stamp=run_stamp,
            universe=params["universe"],
            tail_low_pct=tail_low_pct,
            tail_high_pct=tail_high_pct,
            all_watchlist_name=params["all_watchlist_name"],
            shortlist_watchlist_name=params["shortlist_watchlist_name"],
            rows=[],
            console=console,
        )
    _write_shortlist_md(
        path=run_root / "shortlist.md",
        run_stamp=run_stamp,
        universe=params["universe"],
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
        all_watchlist_name=params["all_watchlist_name"],
        shortlist_watchlist_name=params["shortlist_watchlist_name"],
        shortlist_symbols=[],
        rank_results={},
        confluence_scores={},
        reports_out=params["reports_out"],
    )
    console.print(f"Wrote shortlist summary: {run_root / 'shortlist.md'}")
    console.print("Not financial advice.")
    return True


def _rank_shortlist(
    *,
    deps: SimpleNamespace,
    params: dict[str, Any],
    candle_store: Any,
    confluence_cfg: Any,
    scan_rows: list[Any],
    liquidity_rows: list[Any],
    shortlist_symbols: list[str],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    rank_results: dict[str, Any] = {}
    if shortlist_symbols:
        rank_cfg = confluence_cfg.get("scanner_rank") if isinstance(confluence_cfg, dict) else None
        derived_store = deps.cli_deps.build_derived_store(params["derived_dir"])
        rank_results = deps.rank_shortlist_candidates(
            shortlist_symbols,
            candle_store=candle_store,
            rank_cfg=rank_cfg,
            scan_rows=scan_rows,
            liquidity_rows=liquidity_rows,
            derived_store=derived_store,
            period=params["scan_period"],
        )
    scan_percentiles = {row.symbol.upper(): row.percentile for row in scan_rows}
    confluence_scores = deps.score_shortlist_confluence(
        shortlist_symbols,
        candle_store=candle_store,
        confluence_cfg=confluence_cfg,
        extension_percentiles=scan_percentiles,
        period=params["scan_period"],
    )
    if rank_results:
        shortlist_symbols = sorted(shortlist_symbols, key=lambda sym: (-(rank_results.get(sym).score if rank_results.get(sym) else -1.0), -(rank_results.get(sym).coverage if rank_results.get(sym) else -1.0), sym))
    elif confluence_scores:
        shortlist_symbols = sorted(shortlist_symbols, key=lambda sym: (-(confluence_scores.get(sym).coverage if confluence_scores.get(sym) else -1.0), -(confluence_scores.get(sym).total if confluence_scores.get(sym) else -1.0), sym))
    return shortlist_symbols, rank_results, confluence_scores
