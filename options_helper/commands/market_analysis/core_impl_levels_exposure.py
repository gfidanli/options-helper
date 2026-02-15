from __future__ import annotations

from datetime import date
import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.exposure import compute_exposure_slices
from options_helper.analysis.levels import compute_anchored_vwap, compute_levels_summary, compute_volume_profile
from options_helper.commands.market_analysis.core_artifacts import _build_exposure_artifact, _build_levels_artifact
from options_helper.commands.market_analysis.core_helpers import _normalize_output_format, _normalize_symbol
from options_helper.commands.market_analysis.core_io import (
    _duckdb_backend_enabled,
    _normalize_daily_history,
    _persist_exposure_strikes,
    _resolve_snapshot_spot,
    _save_artifact_json,
    _slice_history_to_as_of,
)
from options_helper.commands.market_analysis.core_renderers import _render_exposure_console, _render_levels_console
from options_helper.data.options_snapshots import OptionsSnapshotError
from options_helper.data.providers.runtime import get_default_provider_name


def _build_intraday_store(root_dir: Path) -> Any:
    market_analysis_pkg = importlib.import_module("options_helper.commands.market_analysis")
    intraday_store_cls = market_analysis_pkg.IntradayStore
    return intraday_store_cls(root_dir)


def exposure(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (spot fallback).",
    ),
    near_n: int = typer.Option(4, "--near-n", min=1, help="Number of nearest expiries in near slice."),
    top_n: int = typer.Option(10, "--top-n", min=1, help="Top absolute net GEX levels to report."),
    research_dir: Path = typer.Option(
        Path("data/research_metrics"),
        "--research-dir",
        help="Research metrics root used for DuckDB persistence wiring.",
    ),
    persist: bool = typer.Option(
        True,
        "--persist/--no-persist",
        help="Persist rows to DuckDB research tables when --storage duckdb.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for artifacts (writes under {out}/exposure/{SYMBOL}/).",
    ),
) -> None:
    """Build dealer exposure artifact from local snapshot chain (offline-first)."""
    _run_exposure(
        symbol=symbol,
        as_of=as_of,
        cache_dir=cache_dir,
        candle_cache_dir=candle_cache_dir,
        near_n=near_n,
        top_n=top_n,
        research_dir=research_dir,
        persist=persist,
        output_format=format,
        out=out,
    )


def _run_exposure(
    *,
    symbol: str,
    as_of: str,
    cache_dir: Path,
    candle_cache_dir: Path,
    near_n: int,
    top_n: int,
    research_dir: Path,
    persist: bool,
    output_format: str,
    out: Path | None,
) -> None:
    console = Console(width=200)
    fmt = _normalize_output_format(output_format)
    sym = _normalize_symbol(symbol)
    snapshot_store = cli_deps.build_snapshot_store(cache_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)
    as_of_date, snapshot = _resolve_snapshot_day(console, snapshot_store, sym=sym, as_of=as_of)
    spot, warnings = _resolve_snapshot_spot(
        meta=snapshot_store.load_meta(sym, as_of_date),
        candles=candle_store.load(sym),
        as_of=as_of_date,
    )
    slices = compute_exposure_slices(snapshot, symbol=sym, as_of=as_of_date, spot=spot, near_n=near_n, top_n=top_n)
    artifact = _build_exposure_artifact(symbol=sym, as_of=as_of_date, spot=spot, slices=slices, warnings=warnings)
    _emit_exposure_output(
        console,
        artifact=artifact,
        slices=slices,
        output_format=fmt,
        out=out,
        persist=persist,
        research_dir=research_dir,
    )


def _resolve_snapshot_day(console: Console, snapshot_store: object, *, sym: str, as_of: str) -> tuple[date, pd.DataFrame]:
    try:
        as_of_date = snapshot_store.resolve_date(sym, as_of)
    except OptionsSnapshotError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    snapshot = snapshot_store.load_day(sym, as_of_date)
    if snapshot.empty:
        console.print(
            f"No snapshot rows found for {sym} on {as_of_date.isoformat()}. "
            "Run snapshot ingestion first."
        )
        raise typer.Exit(1)
    return as_of_date, snapshot


def _emit_exposure_output(
    console: Console,
    *,
    artifact: object,
    slices: dict[str, Any],
    output_format: str,
    out: Path | None,
    persist: bool,
    research_dir: Path,
) -> None:
    if output_format == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_exposure_console(console, artifact)

    if out is not None:
        saved = _save_artifact_json(out, feature="exposure", symbol=artifact.symbol, as_of=artifact.as_of, artifact=artifact)
        if output_format != "json":
            console.print(f"Saved: {saved}")

    if persist and _duckdb_backend_enabled():
        all_slice = slices.get("all")
        strike_rows = pd.DataFrame(all_slice.strike_rows if all_slice is not None else [])
        persisted = _persist_exposure_strikes(strike_rows, research_dir)
        if output_format != "json":
            console.print(
                "Persisted dealer exposure rows "
                f"(provider={get_default_provider_name()}): strikes={persisted}"
            )


def levels(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol."),
    as_of: str = typer.Option("latest", "--as-of", help="Daily candle date (YYYY-MM-DD) or 'latest'."),
    benchmark: str = typer.Option("SPY", "--benchmark", help="Benchmark symbol for RS/Beta context."),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    intraday_dir: Path = typer.Option(
        Path("data/intraday"),
        "--intraday-dir",
        help="Directory containing captured intraday partitions.",
    ),
    intraday_timeframe: str = typer.Option(
        "1Min",
        "--intraday-timeframe",
        help="Intraday bars timeframe partition to read for anchored VWAP/profile.",
    ),
    rolling_window: int = typer.Option(20, "--rolling-window", min=1, help="Lookback for rolling high/low levels."),
    rs_window: int = typer.Option(20, "--rs-window", min=2, help="Lookback for beta/correlation."),
    volume_bins: int = typer.Option(20, "--volume-bins", min=1, help="Number of volume-profile bins."),
    hvn_quantile: float = typer.Option(0.8, "--hvn-quantile", min=0.0, max=1.0, help="HVN quantile threshold."),
    lvn_quantile: float = typer.Option(0.2, "--lvn-quantile", min=0.0, max=1.0, help="LVN quantile threshold."),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for artifacts (writes under {out}/levels/{SYMBOL}/).",
    ),
) -> None:
    """Compute actionable levels from cached daily candles + optional intraday bars (offline-only)."""
    _run_levels(
        symbol=symbol,
        as_of=as_of,
        benchmark=benchmark,
        candle_cache_dir=candle_cache_dir,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        rolling_window=rolling_window,
        rs_window=rs_window,
        volume_bins=volume_bins,
        hvn_quantile=hvn_quantile,
        lvn_quantile=lvn_quantile,
        output_format=format,
        out=out,
    )


def _run_levels(
    *,
    symbol: str,
    as_of: str,
    benchmark: str,
    candle_cache_dir: Path,
    intraday_dir: Path,
    intraday_timeframe: str,
    rolling_window: int,
    rs_window: int,
    volume_bins: int,
    hvn_quantile: float,
    lvn_quantile: float,
    output_format: str,
    out: Path | None,
) -> None:
    console = Console(width=200)
    fmt = _normalize_output_format(output_format)
    sym = _normalize_symbol(symbol)
    benchmark_sym = _normalize_symbol(benchmark)
    symbol_history, as_of_date, benchmark_history = _load_levels_history(
        console, sym=sym, benchmark_sym=benchmark_sym, as_of=as_of, candle_cache_dir=candle_cache_dir
    )
    summary = compute_levels_summary(
        symbol_history,
        benchmark_daily=benchmark_history,
        rolling_window=rolling_window,
        rs_window=rs_window,
    )
    intraday_store = _build_intraday_store(intraday_dir)
    intraday_bars = intraday_store.load_partition("stocks", "bars", intraday_timeframe, sym, as_of_date)
    anchored = compute_anchored_vwap(intraday_bars, anchor_type="session_open", spot=summary.spot)
    profile = compute_volume_profile(intraday_bars, num_bins=volume_bins, hvn_quantile=hvn_quantile, lvn_quantile=lvn_quantile)
    artifact = _build_levels_artifact(symbol=sym, as_of=as_of_date, summary=summary, anchored=anchored, profile=profile)
    if fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_levels_console(console, artifact)
    if out is not None:
        saved = _save_artifact_json(out, feature="levels", symbol=sym, as_of=artifact.as_of, artifact=artifact)
        if fmt != "json":
            console.print(f"Saved: {saved}")


def _load_levels_history(
    console: Console,
    *,
    sym: str,
    benchmark_sym: str,
    as_of: str,
    candle_cache_dir: Path,
) -> tuple[pd.DataFrame, date, pd.DataFrame | None]:
    candle_store = cli_deps.build_candle_store(candle_cache_dir)
    symbol_history = _normalize_daily_history(candle_store.load(sym))
    if symbol_history.empty:
        console.print(f"No candle history available for {sym}.")
        raise typer.Exit(1)

    try:
        symbol_history, as_of_date = _slice_history_to_as_of(symbol_history, as_of)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    benchmark_history_raw = _normalize_daily_history(candle_store.load(benchmark_sym))
    benchmark_history: pd.DataFrame | None = None
    if not benchmark_history_raw.empty:
        benchmark_slice = benchmark_history_raw[benchmark_history_raw["date"] <= pd.Timestamp(as_of_date)].copy()
        if not benchmark_slice.empty:
            benchmark_history = benchmark_slice
    return symbol_history, as_of_date, benchmark_history


__all__ = ["exposure", "levels"]
