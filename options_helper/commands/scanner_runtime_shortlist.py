from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from .scanner_runtime import (
    _rank_shortlist,
    _write_shortlist_json,
    _write_shortlist_md,
)


@dataclass(frozen=True)
class ShortlistResult:
    liquidity_rows: list[Any]
    shortlist_symbols: list[str]
    rank_results: dict[str, Any]
    confluence_scores: dict[str, Any]


def evaluate_shortlist(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    candle_store: Any,
    confluence_cfg: Any,
    scan_rows: list[Any],
    tail_symbols: list[str],
) -> ShortlistResult:
    options_store = deps.cli_deps.build_snapshot_store(params["options_cache_dir"])
    liquidity_rows, shortlist_symbols = deps.evaluate_liquidity_for_symbols(
        tail_symbols,
        store=options_store,
        min_dte=params["liquidity_min_dte"],
        min_volume=params["liquidity_min_volume"],
        min_open_interest=params["liquidity_min_oi"],
    )
    shortlist_symbols, rank_results, confluence_scores = _rank_shortlist(
        deps=deps,
        params=params,
        candle_store=candle_store,
        confluence_cfg=confluence_cfg,
        scan_rows=scan_rows,
        liquidity_rows=liquidity_rows,
        shortlist_symbols=shortlist_symbols,
    )
    return ShortlistResult(
        liquidity_rows=liquidity_rows,
        shortlist_symbols=shortlist_symbols,
        rank_results=rank_results,
        confluence_scores=confluence_scores,
    )


def _build_shortlist_rows(
    *, deps: SimpleNamespace, shortlist_symbols: list[str], rank_results: dict[str, Any]
) -> tuple[list[Any], list[Any]]:
    rows: list[Any] = []
    schema_rows: list[Any] = []
    for symbol in shortlist_symbols:
        rank = rank_results.get(symbol)
        reasons = "; ".join(rank.top_reasons) if rank is not None else ""
        score = rank.score if rank is not None else None
        coverage = rank.coverage if rank is not None else None
        rows.append(
            deps.ScannerShortlistRow(
                symbol=symbol,
                score=score,
                coverage=coverage,
                top_reasons=reasons,
            )
        )
        schema_rows.append(
            deps.ScannerShortlistRowSchema(
                symbol=symbol,
                score=score,
                coverage=coverage,
                top_reasons=reasons or None,
            )
        )
    return rows, schema_rows


def write_shortlist_outputs(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    run_root: Path,
    run_stamp: str,
    scan_as_of: str,
    tail_low_pct: float,
    tail_high_pct: float,
    shortlist: ShortlistResult,
) -> None:
    if params["write_liquidity"]:
        liquidity_path = run_root / "liquidity.csv"
        deps.write_liquidity_csv(shortlist.liquidity_rows, liquidity_path)
        console.print(f"Wrote liquidity CSV: {liquidity_path}")
    if not params["write_shortlist"]:
        return
    shortlist_csv = run_root / "shortlist.csv"
    rows, schema_rows = _build_shortlist_rows(
        deps=deps,
        shortlist_symbols=shortlist.shortlist_symbols,
        rank_results=shortlist.rank_results,
    )
    deps.write_shortlist_csv(rows, shortlist_csv)
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
        rows=schema_rows,
        console=console,
    )


def update_shortlist_watchlist(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    watchlists: Any,
    shortlist_symbols: list[str],
) -> None:
    watchlists.set(params["shortlist_watchlist_name"], shortlist_symbols)
    deps.save_watchlists(params["watchlists_path"], watchlists)
    console.print(
        f"Updated watchlist `{params['shortlist_watchlist_name']}` "
        f"({len(shortlist_symbols)} symbol(s))"
    )


def run_shortlist_reports(
    *,
    params: dict[str, Any],
    deps: SimpleNamespace,
    console: Console,
    shortlist_symbols: list[str],
) -> None:
    if not params["run_reports"] or not shortlist_symbols:
        return
    console.print(f"Running Extension Percentile Stats for {len(shortlist_symbols)} symbol(s)...")
    for sym in shortlist_symbols:
        try:
            deps.run_extension_stats_for_symbol(
                symbol=sym,
                ohlc_path=None,
                cache_dir=params["candle_cache_dir"],
                config_path=params["config_path"],
                tail_pct=params["tail_pct"],
                percentile_window_years=params["percentile_window_years"],
                out=params["reports_out"],
                write_json=True,
                write_md=True,
                print_to_console=False,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {sym}: extension-stats failed: {exc}")


def write_shortlist_summary(
    *,
    params: dict[str, Any],
    console: Console,
    run_root: Path,
    run_stamp: str,
    tail_low_pct: float,
    tail_high_pct: float,
    shortlist: ShortlistResult,
) -> None:
    shortlist_md = run_root / "shortlist.md"
    _write_shortlist_md(
        path=shortlist_md,
        run_stamp=run_stamp,
        universe=params["universe"],
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
        all_watchlist_name=params["all_watchlist_name"],
        shortlist_watchlist_name=params["shortlist_watchlist_name"],
        shortlist_symbols=shortlist.shortlist_symbols,
        rank_results=shortlist.rank_results,
        confluence_scores=shortlist.confluence_scores,
        reports_out=params["reports_out"],
    )
    console.print(f"Wrote shortlist summary: {shortlist_md}")
