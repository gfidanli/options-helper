from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.iv_surface import compute_iv_surface
from options_helper.analysis.tail_risk import TailRiskConfig, compute_tail_risk
from options_helper.commands.market_analysis.core_artifacts import (
    _build_iv_context,
    _build_iv_surface_artifact,
    _build_tail_risk_artifact,
)
from options_helper.commands.market_analysis.core_helpers import _normalize_output_format, _normalize_symbol
from options_helper.commands.market_analysis.core_io import (
    _duckdb_backend_enabled,
    _latest_derived_row,
    _persist_iv_surface,
    _resolve_snapshot_spot,
    _save_artifact_json,
    _select_close,
)
from options_helper.commands.market_analysis.core_impl_levels_exposure import exposure, levels
from options_helper.commands.market_analysis.core_renderers import _render_iv_surface_console, _render_tail_risk_console
from options_helper.data.options_snapshots import OptionsSnapshotError
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.tail_risk_artifacts import write_tail_risk_artifacts


def tail_risk(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol."),
    lookback_days: int = typer.Option(252 * 6, "--lookback-days", min=30, help="Lookback window in trading days."),
    horizon_days: int = typer.Option(60, "--horizon-days", min=1, help="Forecast horizon in trading days."),
    num_simulations: int = typer.Option(
        25_000,
        "--num-simulations",
        min=1_000,
        help="Number of Monte Carlo simulated paths.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible simulations."),
    var_confidence: float = typer.Option(
        0.95,
        "--var-confidence",
        min=0.001,
        max=0.999,
        help="VaR/CVaR confidence level (e.g., 0.95).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh/--no-refresh",
        help="Refresh candle cache before running analysis.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for artifacts (writes under {out}/tail_risk/{SYMBOL}/).",
    ),
) -> None:
    """Run Monte Carlo tail risk from local candle history with IV context (offline-first)."""
    _run_tail_risk(
        symbol=symbol,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        seed=seed,
        var_confidence=var_confidence,
        candle_cache_dir=candle_cache_dir,
        derived_dir=derived_dir,
        refresh=refresh,
        output_format=format,
        out=out,
    )


def _run_tail_risk(
    *,
    symbol: str,
    lookback_days: int,
    horizon_days: int,
    num_simulations: int,
    seed: int,
    var_confidence: float,
    candle_cache_dir: Path,
    derived_dir: Path,
    refresh: bool,
    output_format: str,
    out: Path | None,
) -> None:
    console = Console(width=200)
    sym = _normalize_symbol(symbol)
    fmt = _normalize_output_format(output_format)

    provider = cli_deps.build_provider() if refresh else None
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    derived_store = cli_deps.build_derived_store(derived_dir)

    close = _load_tail_risk_close(console, candle_store, sym=sym, refresh=refresh)
    cfg = TailRiskConfig(
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        seed=seed,
        var_confidence=var_confidence,
    )
    result = compute_tail_risk(close, config=cfg)
    derived_latest = _latest_derived_row(derived_store.load(sym))
    iv_context = _build_iv_context(derived_latest)
    artifact = _build_tail_risk_artifact(symbol=sym, config=cfg, result=result, iv_context=iv_context)
    _emit_tail_risk_output(console, artifact=artifact, output_format=fmt, out=out)


def _load_tail_risk_close(console: Console, candle_store: object, *, sym: str, refresh: bool) -> pd.Series:
    history = candle_store.get_daily_history(sym, period="max") if refresh else candle_store.load(sym)
    close = _select_close(history)
    if close.empty:
        console.print(
            f"No candle history available for {sym}. "
            f"Use `options-helper ingest candles --symbol {sym}` (or run with `--refresh`)."
        )
        raise typer.Exit(1)
    return close


def _emit_tail_risk_output(
    console: Console,
    *,
    artifact: object,
    output_format: str,
    out: Path | None,
) -> None:
    if output_format == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_tail_risk_console(console, artifact)

    if out is not None:
        paths = write_tail_risk_artifacts(artifact, out_dir=out)
        console.print(f"Saved: {paths.json_path}")
        console.print(f"Saved: {paths.report_path}")


def iv_surface(
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
        help="Output root for artifacts (writes under {out}/iv_surface/{SYMBOL}/).",
    ),
) -> None:
    """Build IV surface artifact from local snapshot chain (offline-first)."""
    _run_iv_surface(
        symbol=symbol,
        as_of=as_of,
        cache_dir=cache_dir,
        candle_cache_dir=candle_cache_dir,
        research_dir=research_dir,
        persist=persist,
        output_format=format,
        out=out,
    )


def _run_iv_surface(
    *,
    symbol: str,
    as_of: str,
    cache_dir: Path,
    candle_cache_dir: Path,
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
    candles = candle_store.load(sym)
    spot, spot_warnings = _resolve_snapshot_spot(
        meta=snapshot_store.load_meta(sym, as_of_date),
        candles=candles,
        as_of=as_of_date,
    )
    previous_tenor, previous_delta = _resolve_previous_surface_context(snapshot_store, sym=sym, as_of_date=as_of_date, candles=candles)
    surface = compute_iv_surface(
        snapshot,
        symbol=sym,
        as_of=as_of_date,
        spot=spot,
        previous_tenor=previous_tenor,
        previous_delta_buckets=previous_delta,
    )
    artifact = _build_iv_surface_artifact(symbol=sym, as_of=as_of_date, spot=spot, result=surface, warnings=spot_warnings)
    _emit_iv_surface_output(
        console,
        artifact=artifact,
        surface=surface,
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


def _resolve_previous_surface_context(
    snapshot_store: object,
    *,
    sym: str,
    as_of_date: date,
    candles: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        previous_date = snapshot_store.resolve_relative_date(sym, to_date=as_of_date, offset=-1)
    except OptionsSnapshotError:
        return None, None
    if previous_date is None:
        return None, None

    previous_snapshot = snapshot_store.load_day(sym, previous_date)
    if previous_snapshot.empty:
        return None, None
    previous_spot, _ = _resolve_snapshot_spot(
        meta=snapshot_store.load_meta(sym, previous_date),
        candles=candles,
        as_of=previous_date,
    )
    previous_surface = compute_iv_surface(previous_snapshot, symbol=sym, as_of=previous_date, spot=previous_spot)
    return previous_surface.tenor, previous_surface.delta_buckets


def _emit_iv_surface_output(
    console: Console,
    *,
    artifact: object,
    surface: object,
    output_format: str,
    out: Path | None,
    persist: bool,
    research_dir: Path,
) -> None:
    if output_format == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_iv_surface_console(console, artifact)

    if out is not None:
        saved = _save_artifact_json(out, feature="iv_surface", symbol=artifact.symbol, as_of=artifact.as_of, artifact=artifact)
        if output_format != "json":
            console.print(f"Saved: {saved}")

    if persist and _duckdb_backend_enabled():
        persisted_tenor, persisted_delta = _persist_iv_surface(surface.tenor, surface.delta_buckets, research_dir)
        if output_format != "json":
            console.print(
                "Persisted iv-surface rows "
                f"(provider={get_default_provider_name()}): "
                f"tenor={persisted_tenor}, delta_buckets={persisted_delta}"
            )


__all__ = ["tail_risk", "iv_surface", "exposure", "levels"]
