from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.options_snapshotter import snapshot_full_chain_for_symbols
from options_helper.data.scanner import (
    ScannerShortlistRow,
    evaluate_liquidity_for_symbols,
    prefilter_symbols,
    rank_shortlist_candidates,
    read_exclude_symbols,
    read_scanned_symbols,
    scan_symbols,
    score_shortlist_confluence,
    write_exclude_symbols,
    write_liquidity_csv,
    write_scan_csv,
    write_scanned_symbols,
    write_shortlist_csv,
)
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.data.universe import UniverseError, load_universe_symbols
from options_helper.schemas.common import utc_now
from options_helper.schemas.scanner_shortlist import (
    ScannerShortlistArtifact,
    ScannerShortlistRow as ScannerShortlistRowSchema,
)
from options_helper.pipelines.technicals_extension_stats import run_extension_stats_for_symbol
from options_helper.watchlists import load_watchlists, save_watchlists

app = typer.Typer(help="Market opportunity scanner (not financial advice).")


@app.command("run")
def scanner_run(
    universe: str = typer.Option(
        "file:data/universe/sec_company_tickers.json",
        "--universe",
        help="Universe source: us-all/us-equities/us-etfs or file:/path/to/list.txt.",
    ),
    universe_cache_dir: Path = typer.Option(
        Path("data/universe"),
        "--universe-cache-dir",
        help="Directory for cached universe lists.",
    ),
    universe_refresh_days: int = typer.Option(
        1,
        "--universe-refresh-days",
        help="Refresh universe cache if older than this many days.",
    ),
    max_symbols: int | None = typer.Option(
        None,
        "--max-symbols",
        min=1,
        help="Optional cap on number of symbols scanned (for dev/testing).",
    ),
    prefilter_mode: str = typer.Option(
        "default",
        "--prefilter-mode",
        help="Prefilter mode: default, aggressive, or none.",
    ),
    exclude_path: Path = typer.Option(
        Path("data/universe/exclude_symbols.txt"),
        "--exclude-path",
        help="Path to exclude symbols file (one ticker per line).",
    ),
    scanned_path: Path = typer.Option(
        Path("data/scanner/scanned_symbols.txt"),
        "--scanned-path",
        help="Path to scanned symbols file (one ticker per line).",
    ),
    skip_scanned: bool = typer.Option(
        True,
        "--skip-scanned/--no-skip-scanned",
        help="Skip symbols already recorded in the scanned file.",
    ),
    write_scanned: bool = typer.Option(
        True,
        "--write-scanned/--no-write-scanned",
        help="Persist scanned symbols so future runs skip them.",
    ),
    write_error_excludes: bool = typer.Option(
        True,
        "--write-error-excludes/--no-write-error-excludes",
        help="Persist symbols that error to the exclude file.",
    ),
    exclude_statuses: str = typer.Option(
        "error,no_candles",
        "--exclude-statuses",
        help="Comma-separated scan statuses to add to the exclude file.",
    ),
    error_flush_every: int = typer.Option(
        50,
        "--error-flush-every",
        min=1,
        help="Flush exclude file after this many new error symbols.",
    ),
    scanned_flush_every: int = typer.Option(
        250,
        "--scanned-flush-every",
        min=1,
        help="Flush scanned file after this many new symbols.",
    ),
    scan_period: str = typer.Option(
        "max",
        "--scan-period",
        help="Candle period to pull for the scan (yfinance period format).",
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Symmetric tail threshold percentile (e.g. 2.5 => low<=2.5, high>=97.5).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Rolling window (years) for extension percentiles (default: auto 1y/3y).",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    all_watchlist_name: str = typer.Option(
        "Scanner - All",
        "--all-watchlist-name",
        help="Watchlist name for all tail symbols (replaced each run).",
    ),
    shortlist_watchlist_name: str = typer.Option(
        "Scanner - Shortlist",
        "--shortlist-watchlist-name",
        help="Watchlist name for liquid short list (replaced each run).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    options_cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--options-cache-dir",
        help="Directory for options chain snapshots.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot price when snapshotting options.",
    ),
    risk_free_rate: float = typer.Option(
        0.0,
        "--risk-free-rate",
        help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
    ),
    backfill: bool = typer.Option(
        False,
        "--backfill/--no-backfill",
        help="Backfill max candles for tail symbols (slow).",
    ),
    snapshot_options: bool = typer.Option(
        False,
        "--snapshot-options/--no-snapshot-options",
        help="Snapshot options for tail symbols (slow; uses --options-cache-dir).",
    ),
    liquidity_min_dte: int = typer.Option(
        60,
        "--liquidity-min-dte",
        help="Minimum DTE for liquidity screening.",
    ),
    liquidity_min_volume: int = typer.Option(
        10,
        "--liquidity-min-volume",
        help="Minimum volume for liquidity screening.",
    ),
    liquidity_min_oi: int = typer.Option(
        500,
        "--liquidity-min-oi",
        help="Minimum open interest for liquidity screening.",
    ),
    run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--run-dir",
        help="Output root for scanner runs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run id (default: timestamp).",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Max concurrent workers for scan (default: auto).",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        min=1,
        help="Batch size for scan requests.",
    ),
    batch_sleep_seconds: float = typer.Option(
        0.25,
        "--batch-sleep-seconds",
        min=0.0,
        help="Sleep between batches (seconds) to be polite to data sources.",
    ),
    reports_out: Path = typer.Option(
        Path("data/reports/technicals/extension"),
        "--reports-out",
        help="Output root for Extension Percentile Stats reports.",
    ),
    run_reports: bool = typer.Option(
        True,
        "--run-reports/--no-run-reports",
        help="Generate Extension Percentile Stats reports for shortlist symbols.",
    ),
    write_scan: bool = typer.Option(
        True,
        "--write-scan/--no-write-scan",
        help="Write scan CSV under the run directory.",
    ),
    write_liquidity: bool = typer.Option(
        True,
        "--write-liquidity/--no-write-liquidity",
        help="Write liquidity CSV under the run directory.",
    ),
    write_shortlist: bool = typer.Option(
        True,
        "--write-shortlist/--no-write-shortlist",
        help="Write shortlist CSV under the run directory.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--config",
        help="Config path.",
    ),
) -> None:
    """Scan the market for extension tails and build watchlists (not financial advice)."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")

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

    try:
        symbols = load_universe_symbols(
            universe,
            cache_dir=universe_cache_dir,
            refresh_days=universe_refresh_days,
        )
    except UniverseError as exc:
        console.print(f"[red]Universe error:[/red] {exc}")
        raise typer.Exit(1)

    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})

    exclude_symbols = read_exclude_symbols(exclude_path) if exclude_path else set()
    if exclude_symbols:
        console.print(f"Loaded {len(exclude_symbols)} excluded symbol(s) from {exclude_path}")

    scanned_symbols: set[str] = set()
    if scanned_path and (skip_scanned or write_scanned):
        scanned_symbols = read_scanned_symbols(scanned_path)
        if scanned_symbols:
            console.print(f"Loaded {len(scanned_symbols)} scanned symbol(s) from {scanned_path}")

    filtered, dropped = prefilter_symbols(
        symbols,
        mode=prefilter_mode,
        exclude=exclude_symbols,
        scanned=scanned_symbols if skip_scanned else None,
    )
    dropped_n = sum(dropped.values())
    if dropped_n:
        console.print(f"Prefiltered symbols: dropped {dropped_n} ({dropped})")
    symbols = filtered

    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]

    if not symbols:
        console.print("[yellow]No symbols found in universe.[/yellow]")
        raise typer.Exit(0)

    run_stamp = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = run_dir / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    new_error_symbols: set[str] = set()
    new_scanned_symbols: set[str] = set()

    status_set = {s.strip().lower() for s in exclude_statuses.split(",") if s.strip()}

    def _row_callback(row) -> None:  # noqa: ANN001
        if write_scanned and scanned_path is not None:
            sym = row.symbol
            if sym not in scanned_symbols:
                scanned_symbols.add(sym)
                new_scanned_symbols.add(sym)
                if len(new_scanned_symbols) >= int(scanned_flush_every):
                    write_scanned_symbols(scanned_path, scanned_symbols)
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
                write_exclude_symbols(exclude_path, exclude_symbols)
                new_error_symbols.clear()

    console.print(
        f"Scanning {len(symbols)} symbol(s) from `{universe}` (tail {tail_low_pct:.1f}/{tail_high_pct:.1f})..."
    )
    provider = cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    scan_rows, tail_symbols = scan_symbols(
        symbols,
        candle_store=candle_store,
        cfg=cfg,
        scan_period=scan_period,
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        percentile_window_years=percentile_window_years,
        workers=workers,
        batch_size=batch_size,
        batch_sleep_seconds=batch_sleep_seconds,
        row_callback=_row_callback,
    )

    scan_as_of_dates: list[date] = []
    for row in scan_rows:
        if not row.asof:
            continue
        try:
            scan_as_of_dates.append(date.fromisoformat(row.asof))
        except ValueError:
            continue
    scan_as_of = max(scan_as_of_dates).isoformat() if scan_as_of_dates else date.today().isoformat()

    if write_error_excludes and new_error_symbols and exclude_path is not None:
        write_exclude_symbols(exclude_path, exclude_symbols)
        console.print(f"Wrote {len(new_error_symbols)} new excluded symbol(s) to {exclude_path}")

    if write_scanned and new_scanned_symbols and scanned_path is not None:
        write_scanned_symbols(scanned_path, scanned_symbols)
        console.print(f"Wrote {len(new_scanned_symbols)} new scanned symbol(s) to {scanned_path}")

    if write_scan:
        scan_path = run_root / "scan.csv"
        write_scan_csv(scan_rows, scan_path)
        console.print(f"Wrote scan CSV: {scan_path}")

    wl = load_watchlists(watchlists_path)
    wl.set(all_watchlist_name, tail_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{all_watchlist_name}` ({len(tail_symbols)} symbol(s))")

    if not tail_symbols:
        wl.set(shortlist_watchlist_name, [])
        save_watchlists(watchlists_path, wl)
        console.print("[yellow]No tail symbols found; shortlist cleared.[/yellow]")
        if write_liquidity:
            liquidity_path = run_root / "liquidity.csv"
            write_liquidity_csv([], liquidity_path)
            console.print(f"Wrote liquidity CSV: {liquidity_path}")
        if write_shortlist:
            shortlist_csv = run_root / "shortlist.csv"
            write_shortlist_csv([], shortlist_csv)
            console.print(f"Wrote shortlist CSV: {shortlist_csv}")
            shortlist_json = run_root / "shortlist.json"
            payload = ScannerShortlistArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=scan_as_of,
                run_id=run_stamp,
                universe=universe,
                tail_low_pct=float(tail_low_pct),
                tail_high_pct=float(tail_high_pct),
                all_watchlist_name=all_watchlist_name,
                shortlist_watchlist_name=shortlist_watchlist_name,
                rows=[],
            ).to_dict()
            if strict:
                ScannerShortlistArtifact.model_validate(payload)
            shortlist_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"Wrote shortlist JSON: {shortlist_json}")
        shortlist_md = run_root / "shortlist.md"
        lines = [
            f"# Scanner Shortlist — {run_stamp}",
            "",
            f"- Universe: `{universe}`",
            f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
            f"- Tail watchlist: `{all_watchlist_name}`",
            f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
            "- Ranking: `scanner score` (desc)",
            "- Symbols: `0`",
            "",
            "Not financial advice.",
            "",
            "## Symbols",
            "- (empty)",
        ]
        shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        console.print(f"Wrote shortlist summary: {shortlist_md}")
        console.print("Not financial advice.")
        return

    if backfill:
        console.print(f"Backfilling candles for {len(tail_symbols)} tail symbol(s)...")
        for sym in tail_symbols:
            try:
                candle_store.get_daily_history(sym, period="max")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: candle backfill failed: {exc}")

    if snapshot_options:
        console.print(f"Snapshotting full options chains for {len(tail_symbols)} tail symbol(s)...")
        snapshot_results = snapshot_full_chain_for_symbols(
            tail_symbols,
            cache_dir=options_cache_dir,
            candle_cache_dir=candle_cache_dir,
            spot_period=spot_period,
            max_expiries=None,
            risk_free_rate=risk_free_rate,
            symbol_source="scanner",
            watchlists=[all_watchlist_name],
            provider=provider,
        )
        ok = sum(1 for r in snapshot_results if r.status == "ok")
        console.print(f"Options snapshots complete: {ok}/{len(snapshot_results)} ok")

    options_store = cli_deps.build_snapshot_store(options_cache_dir)
    liquidity_rows, shortlist_symbols = evaluate_liquidity_for_symbols(
        tail_symbols,
        store=options_store,
        min_dte=liquidity_min_dte,
        min_volume=liquidity_min_volume,
        min_open_interest=liquidity_min_oi,
    )

    rank_results = {}
    if shortlist_symbols:
        rank_cfg = None
        if confluence_cfg is not None and isinstance(confluence_cfg, dict):
            rank_cfg = confluence_cfg.get("scanner_rank")
        derived_store = cli_deps.build_derived_store(derived_dir)
        rank_results = rank_shortlist_candidates(
            shortlist_symbols,
            candle_store=candle_store,
            rank_cfg=rank_cfg,
            scan_rows=scan_rows,
            liquidity_rows=liquidity_rows,
            derived_store=derived_store,
            period=scan_period,
        )

    scan_percentiles = {row.symbol.upper(): row.percentile for row in scan_rows}
    confluence_scores = score_shortlist_confluence(
        shortlist_symbols,
        candle_store=candle_store,
        confluence_cfg=confluence_cfg,
        extension_percentiles=scan_percentiles,
        period=scan_period,
    )
    if rank_results:
        def _shortlist_rank_key(sym: str) -> tuple[float, float, str]:
            score = rank_results.get(sym)
            total = score.score if score is not None else -1.0
            coverage = score.coverage if score is not None else -1.0
            return (-total, -coverage, sym)

        shortlist_symbols = sorted(shortlist_symbols, key=_shortlist_rank_key)
    elif confluence_scores:
        def _shortlist_sort_key(sym: str) -> tuple[float, float, str]:
            score = confluence_scores.get(sym)
            coverage = score.coverage if score is not None else -1.0
            total = score.total if score is not None else -1.0
            return (-coverage, -total, sym)

        shortlist_symbols = sorted(shortlist_symbols, key=_shortlist_sort_key)

    if write_liquidity:
        liquidity_path = run_root / "liquidity.csv"
        write_liquidity_csv(liquidity_rows, liquidity_path)
        console.print(f"Wrote liquidity CSV: {liquidity_path}")

    if write_shortlist:
        shortlist_csv = run_root / "shortlist.csv"
        rows: list[ScannerShortlistRow] = []
        schema_rows: list[ScannerShortlistRowSchema] = []
        for sym in shortlist_symbols:
            rank = rank_results.get(sym)
            reasons = "; ".join(rank.top_reasons) if rank is not None else ""
            rows.append(
                ScannerShortlistRow(
                    symbol=sym,
                    score=rank.score if rank is not None else None,
                    coverage=rank.coverage if rank is not None else None,
                    top_reasons=reasons,
                )
            )
            schema_rows.append(
                ScannerShortlistRowSchema(
                    symbol=sym,
                    score=rank.score if rank is not None else None,
                    coverage=rank.coverage if rank is not None else None,
                    top_reasons=reasons or None,
                )
            )
        write_shortlist_csv(rows, shortlist_csv)
        console.print(f"Wrote shortlist CSV: {shortlist_csv}")
        shortlist_json = run_root / "shortlist.json"
        payload = ScannerShortlistArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=scan_as_of,
            run_id=run_stamp,
            universe=universe,
            tail_low_pct=float(tail_low_pct),
            tail_high_pct=float(tail_high_pct),
            all_watchlist_name=all_watchlist_name,
            shortlist_watchlist_name=shortlist_watchlist_name,
            rows=schema_rows,
        ).to_dict()
        if strict:
            ScannerShortlistArtifact.model_validate(payload)
        shortlist_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote shortlist JSON: {shortlist_json}")

    wl.set(shortlist_watchlist_name, shortlist_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{shortlist_watchlist_name}` ({len(shortlist_symbols)} symbol(s))")

    if run_reports and shortlist_symbols:
        console.print(f"Running Extension Percentile Stats for {len(shortlist_symbols)} symbol(s)...")
        for sym in shortlist_symbols:
            try:
                run_extension_stats_for_symbol(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=config_path,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=reports_out,
                    write_json=True,
                    write_md=True,
                    print_to_console=False,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: extension-stats failed: {exc}")

    shortlist_md = run_root / "shortlist.md"
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
            if parts:
                lines.append(f"- `{sym}` ({'; '.join(parts)}) → `{reports_out / sym}`")
            else:
                lines.append(f"- `{sym}` → `{reports_out / sym}`")
    else:
        lines.append("- (empty)")
    shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    console.print(f"Wrote shortlist summary: {shortlist_md}")
    console.print("Not financial advice.")
