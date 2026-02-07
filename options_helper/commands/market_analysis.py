from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

import options_helper.cli_deps as cli_deps
from options_helper.analysis.exposure import ExposureSlice, compute_exposure_slices
from options_helper.analysis.iv_context import classify_iv_rv
from options_helper.analysis.iv_surface import compute_iv_surface
from options_helper.analysis.levels import compute_anchored_vwap, compute_levels_summary, compute_volume_profile
from options_helper.analysis.tail_risk import TailRiskConfig, TailRiskResult, compute_tail_risk
from options_helper.commands.common import _parse_date, _spot_from_meta
from options_helper.data.confluence_config import ConfigError, load_confluence_config
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.options_snapshots import OptionsSnapshotError
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.storage_runtime import get_storage_runtime_config
from options_helper.data.tail_risk_artifacts import write_tail_risk_artifacts
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.exposure import (
    ExposureArtifact,
    ExposureSliceArtifact,
    ExposureStrikeRow,
    ExposureSummaryRow,
    ExposureTopLevelRow,
)
from options_helper.schemas.iv_surface import (
    IvSurfaceArtifact,
    IvSurfaceDeltaBucketChangeRow,
    IvSurfaceDeltaBucketRow,
    IvSurfaceTenorChangeRow,
    IvSurfaceTenorRow,
)
from options_helper.schemas.levels import (
    LevelsAnchoredVwapRow,
    LevelsArtifact,
    LevelsSummaryRow,
    LevelsVolumeProfileRow,
)
from options_helper.schemas.research_metrics_contracts import DELTA_BUCKET_ORDER
from options_helper.schemas.tail_risk import (
    TailRiskArtifact,
    TailRiskConfigArtifact,
    TailRiskIVContext,
    TailRiskPercentileRow,
)


app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")


@app.command("tail-risk")
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
    console = Console(width=200)
    sym = _normalize_symbol(symbol)

    output_fmt = _normalize_output_format(format)

    provider = cli_deps.build_provider() if refresh else None
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    derived_store = cli_deps.build_derived_store(derived_dir)

    history = candle_store.get_daily_history(sym, period="max") if refresh else candle_store.load(sym)
    close = _select_close(history)
    if close.empty:
        console.print(
            f"No candle history available for {sym}. "
            f"Use `options-helper ingest candles --symbol {sym}` (or run with `--refresh`)."
        )
        raise typer.Exit(1)

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
    artifact = _build_tail_risk_artifact(
        symbol=sym,
        config=cfg,
        result=result,
        iv_context=iv_context,
    )

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_tail_risk_console(console, artifact)

    if out is not None:
        paths = write_tail_risk_artifacts(artifact, out_dir=out)
        console.print(f"Saved: {paths.json_path}")
        console.print(f"Saved: {paths.report_path}")


@app.command("iv-surface")
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
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)
    sym = _normalize_symbol(symbol)

    snapshot_store = cli_deps.build_snapshot_store(cache_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)

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

    candles = candle_store.load(sym)
    spot, spot_warnings = _resolve_snapshot_spot(
        meta=snapshot_store.load_meta(sym, as_of_date),
        candles=candles,
        as_of=as_of_date,
    )

    previous_tenor: pd.DataFrame | None = None
    previous_delta: pd.DataFrame | None = None
    try:
        previous_date = snapshot_store.resolve_relative_date(sym, to_date=as_of_date, offset=-1)
    except OptionsSnapshotError:
        previous_date = None
    if previous_date is not None:
        previous_snapshot = snapshot_store.load_day(sym, previous_date)
        if not previous_snapshot.empty:
            previous_spot, _ = _resolve_snapshot_spot(
                meta=snapshot_store.load_meta(sym, previous_date),
                candles=candles,
                as_of=previous_date,
            )
            previous_surface = compute_iv_surface(
                previous_snapshot,
                symbol=sym,
                as_of=previous_date,
                spot=previous_spot,
            )
            previous_tenor = previous_surface.tenor
            previous_delta = previous_surface.delta_buckets

    surface = compute_iv_surface(
        snapshot,
        symbol=sym,
        as_of=as_of_date,
        spot=spot,
        previous_tenor=previous_tenor,
        previous_delta_buckets=previous_delta,
    )

    artifact = _build_iv_surface_artifact(
        symbol=sym,
        as_of=as_of_date,
        spot=spot,
        result=surface,
        warnings=spot_warnings,
    )

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_iv_surface_console(console, artifact)

    if out is not None:
        saved = _save_artifact_json(out, feature="iv_surface", symbol=sym, as_of=artifact.as_of, artifact=artifact)
        if output_fmt != "json":
            console.print(f"Saved: {saved}")

    if persist and _duckdb_backend_enabled():
        persisted_tenor, persisted_delta = _persist_iv_surface(surface.tenor, surface.delta_buckets, research_dir)
        if output_fmt != "json":
            console.print(
                "Persisted iv-surface rows "
                f"(provider={get_default_provider_name()}): "
                f"tenor={persisted_tenor}, delta_buckets={persisted_delta}"
            )


@app.command("exposure")
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
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)
    sym = _normalize_symbol(symbol)

    snapshot_store = cli_deps.build_snapshot_store(cache_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)

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

    spot, spot_warnings = _resolve_snapshot_spot(
        meta=snapshot_store.load_meta(sym, as_of_date),
        candles=candle_store.load(sym),
        as_of=as_of_date,
    )

    slices = compute_exposure_slices(
        snapshot,
        symbol=sym,
        as_of=as_of_date,
        spot=spot,
        near_n=near_n,
        top_n=top_n,
    )
    artifact = _build_exposure_artifact(symbol=sym, as_of=as_of_date, spot=spot, slices=slices, warnings=spot_warnings)

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_exposure_console(console, artifact)

    if out is not None:
        saved = _save_artifact_json(out, feature="exposure", symbol=sym, as_of=artifact.as_of, artifact=artifact)
        if output_fmt != "json":
            console.print(f"Saved: {saved}")

    if persist and _duckdb_backend_enabled():
        all_slice = slices.get("all")
        strike_rows = pd.DataFrame(all_slice.strike_rows if all_slice is not None else [])
        persisted = _persist_exposure_strikes(strike_rows, research_dir)
        if output_fmt != "json":
            console.print(
                "Persisted dealer exposure rows "
                f"(provider={get_default_provider_name()}): strikes={persisted}"
            )


@app.command("levels")
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
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)
    sym = _normalize_symbol(symbol)
    benchmark_sym = _normalize_symbol(benchmark)

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

    summary = compute_levels_summary(
        symbol_history,
        benchmark_daily=benchmark_history,
        rolling_window=rolling_window,
        rs_window=rs_window,
    )

    intraday_store = IntradayStore(intraday_dir)
    intraday_bars = intraday_store.load_partition("stocks", "bars", intraday_timeframe, sym, as_of_date)

    anchored = compute_anchored_vwap(intraday_bars, anchor_type="session_open", spot=summary.spot)
    profile = compute_volume_profile(
        intraday_bars,
        num_bins=volume_bins,
        hvn_quantile=hvn_quantile,
        lvn_quantile=lvn_quantile,
    )

    artifact = _build_levels_artifact(
        symbol=sym,
        as_of=as_of_date,
        summary=summary,
        anchored=anchored,
        profile=profile,
    )

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_levels_console(console, artifact)

    if out is not None:
        saved = _save_artifact_json(out, feature="levels", symbol=sym, as_of=artifact.as_of, artifact=artifact)
        if output_fmt != "json":
            console.print(f"Saved: {saved}")


def _normalize_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def _normalize_output_format(value: str) -> str:
    output_fmt = str(value or "").strip().lower()
    if output_fmt not in {"console", "json"}:
        raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")
    return output_fmt


def _select_close(history: pd.DataFrame | None) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    if "Close" in history.columns:
        return pd.to_numeric(history["Close"], errors="coerce")
    if "close" in history.columns:
        return pd.to_numeric(history["close"], errors="coerce")
    return pd.Series(dtype="float64")


def _latest_derived_row(df: pd.DataFrame) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    temp = df.copy()
    if "date" in temp.columns:
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.sort_values(by="date", kind="stable")
    latest = temp.iloc[-1]
    return dict(latest.to_dict())


def _build_iv_context(derived_row: dict[str, Any] | None) -> TailRiskIVContext | None:
    if not derived_row:
        return None
    iv_rv = _as_float(derived_row.get("iv_rv_20d"))
    if iv_rv is None:
        return None

    low, high = _load_iv_thresholds()
    regime = classify_iv_rv(iv_rv, low=low, high=high)
    if regime is None:
        return None

    return TailRiskIVContext(
        label=regime.label,
        reason=regime.reason,
        iv_rv_20d=regime.iv_rv,
        low=regime.low,
        high=regime.high,
        atm_iv_near=_as_float(derived_row.get("atm_iv_near")),
        rv_20d=_as_float(derived_row.get("rv_20d")),
        atm_iv_near_percentile=_as_float(derived_row.get("atm_iv_near_percentile")),
        iv_term_slope=_as_float(derived_row.get("iv_term_slope")),
    )


def _load_iv_thresholds() -> tuple[float, float]:
    default_low = 0.8
    default_high = 1.2
    try:
        cfg = load_confluence_config()
    except (ConfigError, OSError, ValueError):
        return default_low, default_high

    iv_cfg = (cfg.get("iv_regime") or {}) if isinstance(cfg, dict) else {}
    try:
        low = float(iv_cfg.get("low", default_low))
        high = float(iv_cfg.get("high", default_high))
    except Exception:  # noqa: BLE001
        return default_low, default_high
    if low >= high:
        return default_low, default_high
    return low, high


def _build_tail_risk_artifact(
    *,
    symbol: str,
    config: TailRiskConfig,
    result: TailRiskResult,
    iv_context: TailRiskIVContext | None,
) -> TailRiskArtifact:
    end_rows: list[TailRiskPercentileRow] = []
    for percentile in sorted(result.end_price_percentiles):
        end_rows.append(
            TailRiskPercentileRow(
                percentile=float(percentile),
                price=_as_float(result.end_price_percentiles.get(percentile)),
                return_pct=_as_float(result.end_return_percentiles.get(percentile)),
            )
        )
    return TailRiskArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=result.as_of,
        symbol=symbol,
        disclaimer="Not financial advice.",
        config=TailRiskConfigArtifact(
            lookback_days=config.lookback_days,
            horizon_days=config.horizon_days,
            num_simulations=config.num_simulations,
            seed=config.seed,
            var_confidence=config.var_confidence,
            end_percentiles=[float(value) for value in config.end_percentiles],
            chart_percentiles=[float(value) for value in config.chart_percentiles],
            sample_paths=min(config.sample_paths, config.num_simulations),
            trading_days_per_year=config.trading_days_per_year,
        ),
        spot=result.spot,
        realized_vol_annual=result.realized_vol_annual,
        expected_return_annual=result.expected_return_annual,
        var_return=result.var_return,
        cvar_return=result.cvar_return,
        end_percentiles=end_rows,
        iv_context=iv_context,
        warnings=list(result.warnings),
    )


def _build_iv_surface_artifact(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    result: Any,
    warnings: list[str],
) -> IvSurfaceArtifact:
    return IvSurfaceArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=spot if spot > 0 else None,
        disclaimer="Not financial advice.",
        tenor=[IvSurfaceTenorRow.model_validate(row) for row in _frame_records(result.tenor)],
        delta_buckets=[IvSurfaceDeltaBucketRow.model_validate(row) for row in _frame_records(result.delta_buckets)],
        tenor_changes=[
            IvSurfaceTenorChangeRow.model_validate(row) for row in _frame_records(result.tenor_changes)
        ],
        delta_bucket_changes=[
            IvSurfaceDeltaBucketChangeRow.model_validate(row)
            for row in _frame_records(result.delta_bucket_changes)
        ],
        warnings=_dedupe([*result.warnings, *warnings]),
    )


def _build_exposure_artifact(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    slices: dict[str, ExposureSlice],
    warnings: list[str],
) -> ExposureArtifact:
    slice_artifacts: list[ExposureSliceArtifact] = []
    for mode in ("near", "monthly", "all"):
        slice_data = slices[mode]
        summary = ExposureSummaryRow.model_validate(clean_nan(slice_data.summary))
        summary.warnings = _dedupe([*summary.warnings, *warnings])
        slice_artifacts.append(
            ExposureSliceArtifact(
                mode=mode,
                available_expiries=list(slice_data.available_expiries),
                included_expiries=list(slice_data.included_expiries),
                strike_rows=[
                    ExposureStrikeRow.model_validate(row) for row in _frame_records(pd.DataFrame(slice_data.strike_rows))
                ],
                summary=summary,
                top_abs_net_levels=[
                    ExposureTopLevelRow(
                        strike=float(level.strike),
                        net_gex=float(level.net_gex),
                        abs_net_gex=float(level.abs_net_gex),
                    )
                    for level in slice_data.top_abs_net_levels
                ],
            )
        )

    return ExposureArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=float(spot),
        disclaimer="Not financial advice.",
        slices=slice_artifacts,
    )


def _build_levels_artifact(
    *,
    symbol: str,
    as_of: date,
    summary: Any,
    anchored: Any,
    profile: Any,
) -> LevelsArtifact:
    summary_row = LevelsSummaryRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        spot=summary.spot,
        prev_close=summary.prev_close,
        session_open=summary.session_open,
        gap_pct=summary.gap_pct,
        prior_high=summary.prior_high,
        prior_low=summary.prior_low,
        rolling_high=summary.rolling_high,
        rolling_low=summary.rolling_low,
        rs_ratio=summary.rs_ratio,
        beta_20d=summary.beta_20d,
        corr_20d=summary.corr_20d,
        warnings=list(summary.warnings),
    )

    anchor_row = LevelsAnchoredVwapRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        anchor_id="session_open",
        anchor_type="session_open",
        anchor_ts_utc=_to_python_datetime(anchored.anchor_ts_utc),
        anchor_price=anchored.anchor_price,
        anchored_vwap=anchored.anchored_vwap,
        distance_from_spot_pct=anchored.distance_from_spot_pct,
    )

    profile_rows = [
        LevelsVolumeProfileRow(
            symbol=symbol,
            as_of=as_of.isoformat(),
            price_bin_low=float(row.price_bin_low),
            price_bin_high=float(row.price_bin_high),
            volume=float(row.volume),
            volume_pct=float(row.volume_pct),
            is_poc=bool(row.is_poc),
            is_hvn=bool(row.is_hvn),
            is_lvn=bool(row.is_lvn),
        )
        for row in profile.bins
    ]

    return LevelsArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        disclaimer="Not financial advice.",
        summary=summary_row,
        anchored_vwap=[anchor_row],
        volume_profile=profile_rows,
        volume_profile_poc=profile.poc_price,
        volume_profile_hvn_candidates=[float(value) for value in profile.hvn_candidates],
        volume_profile_lvn_candidates=[float(value) for value in profile.lvn_candidates],
        warnings=_dedupe([*summary.warnings, *anchored.warnings, *profile.warnings]),
    )


def _render_tail_risk_console(console: Console, artifact: TailRiskArtifact) -> None:
    console.print(f"[bold]{artifact.symbol}[/bold] tail risk as-of {artifact.as_of}")
    console.print(
        f"Spot={artifact.spot:.2f} | RV={artifact.realized_vol_annual:.2%} | "
        f"ExpRet={artifact.expected_return_annual:.2%} | "
        f"VaR({artifact.config.var_confidence:.0%})={artifact.var_return:.2%} | "
        f"CVaR({artifact.config.var_confidence:.0%})="
        + ("-" if artifact.cvar_return is None else f"{artifact.cvar_return:.2%}")
    )

    table = Table(title="Horizon End Percentiles")
    table.add_column("Pct", justify="right")
    table.add_column("End Price", justify="right")
    table.add_column("End Return", justify="right")
    for row in artifact.end_percentiles:
        table.add_row(
            f"{row.percentile:.0f}",
            "-" if row.price is None else f"{row.price:.2f}",
            "-" if row.return_pct is None else f"{row.return_pct:.2%}",
        )
    console.print(table)

    if artifact.iv_context is None:
        console.print(
            "IV context unavailable. Run `options-helper derived update --symbol "
            f"{artifact.symbol}` after snapshot ingestion."
        )
    else:
        iv = artifact.iv_context
        console.print(
            "IV context: "
            f"{iv.label} (IV/RV20={iv.iv_rv_20d:.2f}x; low={iv.low:.2f}, high={iv.high:.2f})"
        )
        console.print(iv.reason)
        if iv.atm_iv_near is not None or iv.rv_20d is not None:
            console.print(
                "ATM IV near="
                + ("-" if iv.atm_iv_near is None else f"{iv.atm_iv_near:.2%}")
                + ", RV20="
                + ("-" if iv.rv_20d is None else f"{iv.rv_20d:.2%}")
                + ", IV percentile="
                + ("-" if iv.atm_iv_near_percentile is None else f"{iv.atm_iv_near_percentile:.0f}%")
                + ", IV term slope="
                + ("-" if iv.iv_term_slope is None else f"{iv.iv_term_slope:+.3f}")
            )

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _render_iv_surface_console(console: Console, artifact: IvSurfaceArtifact) -> None:
    spot_text = "-" if artifact.spot is None else f"{artifact.spot:.2f}"
    console.print(f"[bold]{artifact.symbol}[/bold] IV surface as-of {artifact.as_of} (spot={spot_text})")

    tenor_table = Table(title="Tenor Surface")
    tenor_table.add_column("Target DTE", justify="right")
    tenor_table.add_column("Expiry")
    tenor_table.add_column("DTE", justify="right")
    tenor_table.add_column("ATM IV", justify="right")
    tenor_table.add_column("Straddle", justify="right")
    tenor_table.add_column("ExpMove%", justify="right")
    tenor_table.add_column("Skew25 (pp)", justify="right")
    tenor_table.add_column("Contracts", justify="right")
    tenor_table.add_column("Warnings")

    for row in artifact.tenor:
        tenor_table.add_row(
            str(row.tenor_target_dte),
            row.expiry or "-",
            _fmt_int(row.dte),
            _fmt_pct(row.atm_iv),
            _fmt_num(row.straddle_mark),
            _fmt_pct(row.expected_move_pct),
            _fmt_num(row.skew_25d_pp),
            str(row.contracts_used),
            ", ".join(row.warnings) if row.warnings else "-",
        )
    console.print(tenor_table)

    bucket_rank = {name: idx for idx, name in enumerate(DELTA_BUCKET_ORDER)}
    ordered_buckets = sorted(
        artifact.delta_buckets,
        key=lambda row: (
            row.tenor_target_dte,
            row.option_type,
            bucket_rank.get(row.delta_bucket, 999),
        ),
    )

    delta_table = Table(title="Delta Buckets")
    delta_table.add_column("Target DTE", justify="right")
    delta_table.add_column("Type")
    delta_table.add_column("Bucket")
    delta_table.add_column("Avg IV", justify="right")
    delta_table.add_column("Median IV", justify="right")
    delta_table.add_column("N", justify="right")
    delta_table.add_column("Warnings")

    for row in ordered_buckets:
        delta_table.add_row(
            str(row.tenor_target_dte),
            row.option_type,
            row.delta_bucket,
            _fmt_pct(row.avg_iv),
            _fmt_pct(row.median_iv),
            str(row.n_contracts),
            ", ".join(row.warnings) if row.warnings else "-",
        )
    console.print(delta_table)

    if artifact.tenor_changes:
        change_table = Table(title="Tenor Changes")
        change_table.add_column("Target DTE", justify="right")
        change_table.add_column("ATM IV Δpp", justify="right")
        change_table.add_column("Straddle Δ", justify="right")
        change_table.add_column("ExpMove Δpp", justify="right")
        for row in artifact.tenor_changes:
            change_table.add_row(
                str(row.tenor_target_dte),
                _fmt_num(row.atm_iv_change_pp),
                _fmt_num(row.straddle_mark_change),
                _fmt_num(row.expected_move_pct_change_pp),
            )
        console.print(change_table)

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _render_exposure_console(console: Console, artifact: ExposureArtifact) -> None:
    console.print(f"[bold]{artifact.symbol}[/bold] dealer exposure as-of {artifact.as_of} (spot={artifact.spot:.2f})")

    for slice_artifact in artifact.slices:
        summary = slice_artifact.summary
        console.print(
            f"\n[bold]{slice_artifact.mode}[/bold] expiries="
            + (", ".join(slice_artifact.included_expiries) if slice_artifact.included_expiries else "-")
        )
        console.print(
            "flip="
            + _fmt_num(summary.flip_strike)
            + " | total_call_gex="
            + _fmt_num(summary.total_call_gex)
            + " | total_put_gex="
            + _fmt_num(summary.total_put_gex)
            + " | total_net_gex="
            + _fmt_num(summary.total_net_gex)
        )

        top_table = Table(title=f"Top |Net GEX| Levels ({slice_artifact.mode})")
        top_table.add_column("Strike", justify="right")
        top_table.add_column("Net GEX", justify="right")
        top_table.add_column("Abs Net GEX", justify="right")
        for row in slice_artifact.top_abs_net_levels:
            top_table.add_row(_fmt_num(row.strike), _fmt_num(row.net_gex), _fmt_num(row.abs_net_gex))
        console.print(top_table)

        if summary.warnings:
            console.print("Warnings: " + ", ".join(summary.warnings))

    console.print(artifact.disclaimer)


def _render_levels_console(console: Console, artifact: LevelsArtifact) -> None:
    summary = artifact.summary
    console.print(f"[bold]{artifact.symbol}[/bold] levels as-of {artifact.as_of}")

    summary_table = Table(title="Summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Spot", _fmt_num(summary.spot))
    summary_table.add_row("Prev Close", _fmt_num(summary.prev_close))
    summary_table.add_row("Session Open", _fmt_num(summary.session_open))
    summary_table.add_row("Gap %", _fmt_pct(summary.gap_pct))
    summary_table.add_row("Prior High", _fmt_num(summary.prior_high))
    summary_table.add_row("Prior Low", _fmt_num(summary.prior_low))
    summary_table.add_row("Rolling High", _fmt_num(summary.rolling_high))
    summary_table.add_row("Rolling Low", _fmt_num(summary.rolling_low))
    summary_table.add_row("RS Ratio", _fmt_num(summary.rs_ratio))
    summary_table.add_row("Beta (20d)", _fmt_num(summary.beta_20d))
    summary_table.add_row("Corr (20d)", _fmt_num(summary.corr_20d))
    console.print(summary_table)

    anchor_table = Table(title="Anchored VWAP")
    anchor_table.add_column("Anchor")
    anchor_table.add_column("Anchor TS (UTC)")
    anchor_table.add_column("Anchor Px", justify="right")
    anchor_table.add_column("Anchored VWAP", justify="right")
    anchor_table.add_column("Distance %", justify="right")

    for row in artifact.anchored_vwap:
        anchor_ts = row.anchor_ts_utc.isoformat() if row.anchor_ts_utc is not None else "-"
        anchor_table.add_row(
            row.anchor_id,
            anchor_ts,
            _fmt_num(row.anchor_price),
            _fmt_num(row.anchored_vwap),
            _fmt_pct(row.distance_from_spot_pct),
        )
    console.print(anchor_table)

    profile_table = Table(title="Volume Profile")
    profile_table.add_column("Bin Low", justify="right")
    profile_table.add_column("Bin High", justify="right")
    profile_table.add_column("Volume", justify="right")
    profile_table.add_column("Vol %", justify="right")
    profile_table.add_column("POC")
    profile_table.add_column("HVN")
    profile_table.add_column("LVN")

    for row in artifact.volume_profile:
        profile_table.add_row(
            _fmt_num(row.price_bin_low),
            _fmt_num(row.price_bin_high),
            _fmt_num(row.volume),
            _fmt_pct(row.volume_pct),
            "Y" if row.is_poc else "",
            "Y" if row.is_hvn else "",
            "Y" if row.is_lvn else "",
        )
    console.print(profile_table)

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _save_artifact_json(
    out_root: Path,
    *,
    feature: str,
    symbol: str,
    as_of: str,
    artifact: TailRiskArtifact | IvSurfaceArtifact | ExposureArtifact | LevelsArtifact,
) -> Path:
    base = out_root / feature / symbol
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{as_of}.json"
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [clean_nan(row) for row in frame.to_dict(orient="records")]


def _resolve_snapshot_spot(*, meta: dict[str, Any], candles: pd.DataFrame, as_of: date) -> tuple[float, list[str]]:
    warnings: list[str] = []

    spot = _spot_from_meta(meta)
    if spot is not None and spot > 0:
        return float(spot), warnings

    fallback = _spot_from_candles(candles, as_of)
    if fallback is not None and fallback > 0:
        return float(fallback), warnings

    warnings.append("missing_spot")
    return 0.0, warnings


def _spot_from_candles(candles: pd.DataFrame, as_of: date) -> float | None:
    frame = _normalize_daily_history(candles)
    if frame.empty:
        return None
    filtered = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
    if filtered.empty:
        return None
    close = pd.to_numeric(filtered["close"], errors="coerce").dropna()
    if close.empty:
        return None
    value = float(close.iloc[-1])
    return value if value > 0 else None


def _normalize_daily_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = history.copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        date_col = pd.to_datetime(frame["date"], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        date_col = pd.to_datetime(frame.index, errors="coerce")
    else:
        date_col = pd.Series([pd.NaT] * len(frame), index=frame.index)

    out = pd.DataFrame(index=frame.index)
    out["date"] = date_col
    out["open"] = pd.to_numeric(frame.get("open"), errors="coerce")
    out["high"] = pd.to_numeric(frame.get("high"), errors="coerce")
    out["low"] = pd.to_numeric(frame.get("low"), errors="coerce")
    out["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    out["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _slice_history_to_as_of(history: pd.DataFrame, as_of_spec: str) -> tuple[pd.DataFrame, date]:
    if history.empty:
        raise ValueError("empty daily candle history")

    spec = str(as_of_spec or "").strip().lower()
    if spec == "latest":
        target = history["date"].iloc[-1].date()
    else:
        target = _parse_date(as_of_spec)

    sliced = history[history["date"] <= pd.Timestamp(target)].copy()
    if sliced.empty:
        raise ValueError(f"No candles available on or before {target.isoformat()}.")
    resolved = sliced["date"].iloc[-1].date()
    return sliced.reset_index(drop=True), resolved


def _duckdb_backend_enabled() -> bool:
    return get_storage_runtime_config().backend == "duckdb"


def _persist_iv_surface(tenor: pd.DataFrame, delta_buckets: pd.DataFrame, research_dir: Path) -> tuple[int, int]:
    store = cli_deps.build_research_metrics_store(research_dir)
    provider = get_default_provider_name()
    tenor_rows = int(store.upsert_iv_surface_tenor(tenor, provider=provider))
    delta_rows = int(store.upsert_iv_surface_delta_buckets(delta_buckets, provider=provider))
    return tenor_rows, delta_rows


def _persist_exposure_strikes(strike_rows: pd.DataFrame, research_dir: Path) -> int:
    store = cli_deps.build_research_metrics_store(research_dir)
    provider = get_default_provider_name()
    return int(store.upsert_dealer_exposure_strikes(strike_rows, provider=provider))


def _to_python_datetime(value: pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "-"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out
