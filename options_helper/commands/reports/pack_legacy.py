from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

from options_helper.commands import reports_legacy as legacy

cli_deps = legacy.cli_deps
_spot_from_meta = legacy._spot_from_meta
_mark_price = legacy._mark_price
IntradayStore = legacy.IntradayStore
DERIVED_COLUMNS = legacy.DERIVED_COLUMNS

compute_chain_report = legacy.compute_chain_report
compute_compare_report = legacy.compute_compare_report
compute_derived_stats = legacy.compute_derived_stats
compute_exposure_slices = legacy.compute_exposure_slices
compute_flow = legacy.compute_flow
compute_iv_surface = legacy.compute_iv_surface
compute_levels_summary = legacy.compute_levels_summary
compute_position_scenarios = legacy.compute_position_scenarios
compute_anchored_vwap = legacy.compute_anchored_vwap
compute_volume_profile = legacy.compute_volume_profile
aggregate_flow_window = legacy.aggregate_flow_window

FlowGroupBy = legacy.FlowGroupBy
DerivedRow = legacy.DerivedRow
ChainReportArtifact = legacy.ChainReportArtifact
CompareArtifact = legacy.CompareArtifact
FlowArtifact = legacy.FlowArtifact
ExposureArtifact = legacy.ExposureArtifact
IvSurfaceArtifact = legacy.IvSurfaceArtifact
LevelsArtifact = legacy.LevelsArtifact
ScenariosArtifact = legacy.ScenariosArtifact
ScenarioGridRow = legacy.ScenarioGridRow
ScenarioSummaryRow = legacy.ScenarioSummaryRow

load_portfolio = legacy.load_portfolio
load_watchlists = legacy.load_watchlists
find_snapshot_row = legacy.find_snapshot_row
render_chain_report_markdown = legacy.render_chain_report_markdown
utc_now = legacy.utc_now

_ensure_pandas = legacy._ensure_pandas
_dedupe_strings = legacy._dedupe_strings
_safe_file_token = legacy._safe_file_token
_resolve_snapshot_spot_for_report_pack = legacy._resolve_snapshot_spot_for_report_pack
_build_iv_surface_artifact_for_report_pack = legacy._build_iv_surface_artifact_for_report_pack
_build_exposure_artifact_for_report_pack = legacy._build_exposure_artifact_for_report_pack
_build_levels_artifact_for_report_pack = legacy._build_levels_artifact_for_report_pack
_normalize_daily_history = legacy._normalize_daily_history
_slice_history_to_as_of = legacy._slice_history_to_as_of
_fallback_contract_symbol = legacy._fallback_contract_symbol
_row_float = legacy._row_float
_row_string = legacy._row_string
_build_report_pack_scenario_targets = legacy._build_report_pack_scenario_targets


_PORTFOLIO_PATH_ARG = typer.Argument(..., help="Path to portfolio JSON (required for watchlist defaults).")
_WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store.",
)
_WATCHLIST_OPT = typer.Option(
    [],
    "--watchlist",
    help="Watchlist name(s) to include (repeatable). Default: positions, monitor, Scanner - Shortlist.",
)
_AS_OF_OPT = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'.")
_COMPARE_FROM_OPT = typer.Option(
    "-1",
    "--compare-from",
    help="Compare-from date (relative negative offsets or YYYY-MM-DD). Use none to disable.",
)
_CACHE_DIR_OPT = typer.Option(Path("data/options_snapshots"), "--cache-dir", help="Directory for options chain snapshots.")
_DERIVED_DIR_OPT = typer.Option(Path("data/derived"), "--derived-dir", help="Directory for derived metric files.")
_CANDLE_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--candle-cache-dir",
    help="Directory for cached daily candles.",
)
_OUT_OPT = typer.Option(Path("data/reports"), "--out", help="Output root for report pack artifacts.")
_STRICT_OPT = typer.Option(False, "--strict", help="Validate JSON artifacts against schemas.")
_REQUIRE_SNAPSHOT_DATE_OPT = typer.Option(
    None,
    "--require-snapshot-date",
    help="Skip symbols unless the snapshot date matches this date (YYYY-MM-DD) or 'today'.",
)
_REQUIRE_SNAPSHOT_TZ_OPT = typer.Option(
    "America/New_York",
    "--require-snapshot-tz",
    help="Timezone used when --require-snapshot-date is 'today'.",
)
_CHAIN_OPT = typer.Option(True, "--chain/--no-chain", help="Generate chain report artifacts.")
_COMPARE_OPT = typer.Option(
    True,
    "--compare/--no-compare",
    help="Generate compare report artifacts (requires previous snapshot).",
)
_FLOW_OPT = typer.Option(True, "--flow/--no-flow", help="Generate flow artifacts (requires previous snapshot).")
_DERIVED_OPT = typer.Option(True, "--derived/--no-derived", help="Update derived metrics + emit derived stats artifacts.")
_TECHNICALS_OPT = typer.Option(
    True,
    "--technicals/--no-technicals",
    help="Generate technicals extension-stats artifacts (offline, from candle cache).",
)
_IV_SURFACE_OPT = typer.Option(
    True,
    "--iv-surface/--no-iv-surface",
    help="Generate IV surface artifacts from local snapshots.",
)
_EXPOSURE_OPT = typer.Option(
    True,
    "--exposure/--no-exposure",
    help="Generate dealer exposure artifacts from local snapshots.",
)
_LEVELS_OPT = typer.Option(
    True,
    "--levels/--no-levels",
    help="Generate levels artifacts from local candles and optional intraday partitions.",
)
_LEVELS_BENCHMARK_OPT = typer.Option(
    "SPY",
    "--levels-benchmark",
    help="Benchmark symbol used for RS/Beta in levels artifacts.",
)
_LEVELS_INTRADAY_DIR_OPT = typer.Option(
    Path("data/intraday"),
    "--levels-intraday-dir",
    help="Intraday partition root used for anchored VWAP and volume profile.",
)
_LEVELS_INTRADAY_TIMEFRAME_OPT = typer.Option(
    "1Min",
    "--levels-intraday-timeframe",
    help="Intraday partition timeframe for levels artifacts.",
)
_LEVELS_VOLUME_BINS_OPT = typer.Option(
    20,
    "--levels-volume-bins",
    min=1,
    max=200,
    help="Volume-profile bins for levels artifacts.",
)
_SCENARIOS_OPT = typer.Option(
    False,
    "--scenarios/--no-scenarios",
    help="Generate per-position scenarios artifacts for portfolio positions.",
)
_TECHNICALS_CONFIG_OPT = typer.Option(
    Path("config/technical_backtesting.yaml"),
    "--technicals-config",
    help="Technical backtesting config (used for extension-stats artifacts).",
)
_TOP_OPT = typer.Option(10, "--top", min=1, max=100, help="Top rows/strikes to include in reports.")
_DERIVED_WINDOW_OPT = typer.Option(60, "--derived-window", min=1, max=3650, help="Derived stats lookback window.")
_DERIVED_TREND_WINDOW_OPT = typer.Option(
    5, "--derived-trend-window", min=1, max=3650, help="Derived stats trend lookback window."
)
_TAIL_PCT_OPT = typer.Option(
    None,
    "--tail-pct",
    help="Optional symmetric tail threshold for technicals extension-stats (e.g. 5 => low<=5, high>=95).",
)
_PERCENTILE_WINDOW_YEARS_OPT = typer.Option(
    None,
    "--percentile-window-years",
    help="Optional rolling window (years) for extension percentiles in technicals extension-stats.",
)


@dataclass(slots=True)
class _ReportPackConfig:
    portfolio_path: Path
    watchlists_path: Path
    watchlist: list[str]
    as_of: str
    compare_from: str
    cache_dir: Path
    derived_dir: Path
    candle_cache_dir: Path
    out: Path
    strict: bool
    require_snapshot_date: str | None
    require_snapshot_tz: str
    chain: bool
    compare: bool
    flow: bool
    derived: bool
    technicals: bool
    iv_surface: bool
    exposure: bool
    levels: bool
    levels_benchmark: str
    levels_intraday_dir: Path
    levels_intraday_timeframe: str
    levels_volume_bins: int
    scenarios: bool
    technicals_config: Path
    top: int
    derived_window: int
    derived_trend_window: int
    tail_pct: float | None
    percentile_window_years: int | None


@dataclass(slots=True)
class _ReportPackRuntime:
    pd: Any
    console: Console
    store: Any
    derived_store: Any
    candle_store: Any
    intraday_store: IntradayStore
    out: Path
    required_date: date | None
    compare_norm: str
    compare_enabled: bool
    benchmark_symbol: str
    counts: dict[str, int]


@dataclass(slots=True)
class _SymbolSnapshotContext:
    symbol: str
    to_date: date
    df_to: Any
    candles: Any
    spot_to: float
    spot_warnings: list[str]


@dataclass(slots=True)
class _ScenarioContext:
    as_of: date
    day_df: Any
    spot: float | None
    warnings: list[str]


def _resolve_watchlist_symbols(*, console: Console, watchlists_path: Path, watchlist: list[str]) -> tuple[set[str], list[str]]:
    wl = load_watchlists(watchlists_path)
    watchlists_used = watchlist[:] if watchlist else ["positions", "monitor", "Scanner - Shortlist"]
    symbols: set[str] = set()
    for name in watchlists_used:
        syms = wl.get(name)
        if not syms:
            console.print(f"[yellow]Warning:[/yellow] watchlist '{name}' missing/empty in {watchlists_path}")
            continue
        symbols.update(syms)
    return {s.strip().upper() for s in symbols if s and s.strip()}, watchlists_used


def _parse_required_snapshot_date(*, require_snapshot_date: str | None, require_snapshot_tz: str) -> date | None:
    if require_snapshot_date is None:
        return None
    spec = require_snapshot_date.strip().lower()
    try:
        if spec in {"today", "now"}:
            return datetime.now(ZoneInfo(require_snapshot_tz)).date()
        return date.fromisoformat(spec)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(
            f"Invalid --require-snapshot-date/--require-snapshot-tz: {exc}",
            param_hint="--require-snapshot-date",
        ) from exc


def _normalize_compare_mode(compare_from: str) -> tuple[str, bool]:
    compare_norm = compare_from.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}
    return compare_norm, compare_enabled


def _prepare_output_dirs(out: Path) -> Path:
    out = out.expanduser()
    for subdir in (
        "chains",
        "compare",
        "flow",
        "derived",
        "iv_surface",
        "exposure",
        "levels",
        "scenarios",
        "technicals/extension",
    ):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    return out


def _initialize_counts(symbols_total: int) -> dict[str, int]:
    return {
        "symbols_total": symbols_total,
        "symbols_ok": 0,
        "chain_ok": 0,
        "compare_ok": 0,
        "flow_ok": 0,
        "derived_ok": 0,
        "iv_surface_ok": 0,
        "exposure_ok": 0,
        "levels_ok": 0,
        "technicals_ok": 0,
        "scenarios_ok": 0,
        "skipped_required_date": 0,
    }


def _build_runtime(
    *,
    config: _ReportPackConfig,
    console: Console,
    pd_module: Any,
    symbols_total: int,
) -> _ReportPackRuntime:
    benchmark_symbol = str(config.levels_benchmark or "").strip().upper() or "SPY"
    required_date = _parse_required_snapshot_date(
        require_snapshot_date=config.require_snapshot_date,
        require_snapshot_tz=config.require_snapshot_tz,
    )
    compare_norm, compare_enabled = _normalize_compare_mode(config.compare_from)
    return _ReportPackRuntime(
        pd=pd_module,
        console=console,
        store=cli_deps.build_snapshot_store(config.cache_dir),
        derived_store=cli_deps.build_derived_store(config.derived_dir),
        candle_store=cli_deps.build_candle_store(config.candle_cache_dir),
        intraday_store=IntradayStore(config.levels_intraday_dir),
        out=_prepare_output_dirs(config.out),
        required_date=required_date,
        compare_norm=compare_norm,
        compare_enabled=compare_enabled,
        benchmark_symbol=benchmark_symbol,
        counts=_initialize_counts(symbols_total),
    )


def _load_symbol_snapshot(
    *,
    sym: str,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
) -> _SymbolSnapshotContext | None:
    try:
        to_date = runtime.store.resolve_date(sym, config.as_of)
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {sym}: no snapshots ({exc})")
        return None
    if runtime.required_date is not None and to_date != runtime.required_date:
        runtime.counts["skipped_required_date"] += 1
        return None
    df_to = runtime.store.load_day(sym, to_date)
    meta_to = runtime.store.load_meta(sym, to_date)
    candles = runtime.candle_store.load(sym)
    spot_to, spot_warnings = _resolve_snapshot_spot_for_report_pack(meta=meta_to, candles=candles, as_of=to_date)
    if spot_to <= 0:
        runtime.console.print(
            f"[yellow]Warning:[/yellow] {sym}: missing spot for {to_date.isoformat()} "
            "(meta and candle fallback unavailable)"
        )
        return None
    if spot_warnings:
        runtime.console.print(f"[yellow]Warning:[/yellow] {sym}: " + ", ".join(spot_warnings))
    return _SymbolSnapshotContext(sym, to_date, df_to, candles, spot_to, spot_warnings)


def _compute_chain_report_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
) -> Any | None:
    if not (config.chain or config.derived):
        return None
    try:
        return compute_chain_report(
            snapshot.df_to,
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            spot=snapshot.spot_to,
            expiries_mode="near",
            top=config.top,
            best_effort=True,
        )
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: chain-report failed: {exc}")
        return None


def _write_chain_artifacts_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
    chain_report_model: Any | None,
) -> None:
    if not config.chain or chain_report_model is None:
        return
    try:
        base = runtime.out / "chains" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        json_path = base / f"{snapshot.to_date.isoformat()}.json"
        md_path = base / f"{snapshot.to_date.isoformat()}.md"
        chain_artifact = ChainReportArtifact(generated_at=utc_now(), **chain_report_model.model_dump())
        if config.strict:
            ChainReportArtifact.model_validate(chain_artifact.to_dict())
        json_path.write_text(chain_artifact.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(render_chain_report_markdown(chain_report_model), encoding="utf-8")
        runtime.counts["chain_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: failed writing chain artifacts: {exc}")


def _write_derived_artifacts_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
    chain_report_model: Any | None,
) -> None:
    if not config.derived or chain_report_model is None:
        return
    try:
        history = runtime.derived_store.load(snapshot.symbol)
        row = DerivedRow.from_chain_report(chain_report_model, candles=snapshot.candles, derived_history=history)
        runtime.derived_store.upsert(snapshot.symbol, row)
        df_derived = runtime.derived_store.load(snapshot.symbol)
        if df_derived.empty:
            return
        stats = compute_derived_stats(
            df_derived,
            symbol=snapshot.symbol,
            as_of="latest",
            window=config.derived_window,
            trend_window=config.derived_trend_window,
            metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
        )
        base = runtime.out / "derived" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        stats_path = base / f"{stats.as_of}_w{config.derived_window}_tw{config.derived_trend_window}.json"
        stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
        runtime.counts["derived_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: derived update/stats failed: {exc}")


def _write_iv_surface_artifact_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
) -> None:
    if not config.iv_surface:
        return
    try:
        previous_tenor = None
        previous_delta_buckets = None
        try:
            previous_date = runtime.store.resolve_relative_date(snapshot.symbol, to_date=snapshot.to_date, offset=-1)
        except Exception:  # noqa: BLE001
            previous_date = None
        if previous_date is not None:
            previous_snapshot = runtime.store.load_day(snapshot.symbol, previous_date)
            if not previous_snapshot.empty:
                previous_meta = runtime.store.load_meta(snapshot.symbol, previous_date)
                previous_spot, _previous_spot_warnings = _resolve_snapshot_spot_for_report_pack(
                    meta=previous_meta,
                    candles=snapshot.candles,
                    as_of=previous_date,
                )
                previous_surface = compute_iv_surface(
                    previous_snapshot,
                    symbol=snapshot.symbol,
                    as_of=previous_date,
                    spot=previous_spot,
                )
                previous_tenor = previous_surface.tenor
                previous_delta_buckets = previous_surface.delta_buckets
        current_surface = compute_iv_surface(
            snapshot.df_to,
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            spot=snapshot.spot_to,
            previous_tenor=previous_tenor,
            previous_delta_buckets=previous_delta_buckets,
        )
        iv_artifact = _build_iv_surface_artifact_for_report_pack(
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            spot=snapshot.spot_to,
            surface=current_surface,
            warnings=snapshot.spot_warnings,
        )
        if config.strict:
            IvSurfaceArtifact.model_validate(iv_artifact.to_dict())
        base = runtime.out / "iv_surface" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        iv_path = base / f"{snapshot.to_date.isoformat()}.json"
        iv_path.write_text(iv_artifact.model_dump_json(indent=2), encoding="utf-8")
        runtime.counts["iv_surface_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: iv-surface artifact skipped: {exc}")


def _write_exposure_artifact_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
) -> None:
    if not config.exposure:
        return
    try:
        exposure_slices = compute_exposure_slices(
            snapshot.df_to,
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            spot=snapshot.spot_to,
            near_n=4,
            top_n=config.top,
        )
        exposure_artifact = _build_exposure_artifact_for_report_pack(
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            spot=snapshot.spot_to,
            slices=exposure_slices,
            warnings=snapshot.spot_warnings,
        )
        if config.strict:
            ExposureArtifact.model_validate(exposure_artifact.to_dict())
        base = runtime.out / "exposure" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        exposure_path = base / f"{snapshot.to_date.isoformat()}.json"
        exposure_path.write_text(exposure_artifact.model_dump_json(indent=2), encoding="utf-8")
        runtime.counts["exposure_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: exposure artifact skipped: {exc}")


def _write_levels_artifact_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
) -> None:
    if not config.levels:
        return
    try:
        symbol_history = _normalize_daily_history(snapshot.candles)
        symbol_slice = _slice_history_to_as_of(symbol_history, snapshot.to_date)
        benchmark_history = _normalize_daily_history(runtime.candle_store.load(runtime.benchmark_symbol))
        benchmark_slice = _slice_history_to_as_of(benchmark_history, snapshot.to_date) if not benchmark_history.empty else None
        levels_summary = compute_levels_summary(
            symbol_slice,
            benchmark_daily=benchmark_slice,
            rolling_window=20,
            rs_window=20,
        )
        intraday_bars = runtime.intraday_store.load_partition(
            "stocks",
            "bars",
            config.levels_intraday_timeframe,
            snapshot.symbol,
            snapshot.to_date,
        )
        anchored = compute_anchored_vwap(intraday_bars, anchor_type="session_open", spot=levels_summary.spot)
        profile = compute_volume_profile(
            intraday_bars,
            num_bins=config.levels_volume_bins,
            hvn_quantile=0.8,
            lvn_quantile=0.2,
        )
        levels_artifact = _build_levels_artifact_for_report_pack(
            symbol=snapshot.symbol,
            as_of=snapshot.to_date,
            summary=levels_summary,
            anchored=anchored,
            profile=profile,
        )
        if config.strict:
            LevelsArtifact.model_validate(levels_artifact.to_dict())
        base = runtime.out / "levels" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        levels_path = base / f"{snapshot.to_date.isoformat()}.json"
        levels_path.write_text(levels_artifact.model_dump_json(indent=2), encoding="utf-8")
        runtime.counts["levels_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: levels artifact skipped: {exc}")


def _resolve_compare_from_date(*, runtime: _ReportPackRuntime, snapshot: _SymbolSnapshotContext) -> date:
    if runtime.compare_norm.startswith("-") and runtime.compare_norm[1:].isdigit():
        return runtime.store.resolve_relative_date(snapshot.symbol, to_date=snapshot.to_date, offset=int(runtime.compare_norm))
    return runtime.store.resolve_date(snapshot.symbol, runtime.compare_norm)


def _write_compare_artifact(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
    from_date: date,
    df_from: Any,
    spot_from: float,
) -> None:
    diff, report_from, report_to = compute_compare_report(
        symbol=snapshot.symbol,
        from_date=from_date,
        to_date=snapshot.to_date,
        from_df=df_from,
        to_df=snapshot.df_to,
        spot_from=spot_from,
        spot_to=snapshot.spot_to,
        top=config.top,
    )
    base = runtime.out / "compare" / snapshot.symbol.upper()
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / f"{from_date.isoformat()}_to_{snapshot.to_date.isoformat()}.json"
    payload = CompareArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=snapshot.to_date.isoformat(),
        symbol=snapshot.symbol.upper(),
        from_report=report_from.model_dump(),
        to_report=report_to.model_dump(),
        diff=diff.model_dump(),
    ).to_dict()
    if config.strict:
        CompareArtifact.model_validate(payload)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    runtime.counts["compare_ok"] += 1


def _write_flow_artifacts(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
    from_date: date,
    df_from: Any,
) -> None:
    pair_flow = compute_flow(snapshot.df_to, df_from, spot=snapshot.spot_to)
    if pair_flow.empty:
        return
    for group_by in ("contract", "expiry-strike"):
        net = aggregate_flow_window([pair_flow], group_by=cast(FlowGroupBy, group_by))
        base = runtime.out / "flow" / snapshot.symbol.upper()
        base.mkdir(parents=True, exist_ok=True)
        out_path = base / f"{from_date.isoformat()}_to_{snapshot.to_date.isoformat()}_w1_{group_by}.json"
        artifact_net = net.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "optionType": "option_type",
                "deltaOI": "delta_oi",
                "deltaOI_notional": "delta_oi_notional",
                "size": "n_pairs",
            }
        )
        payload = FlowArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=snapshot.to_date.isoformat(),
            symbol=snapshot.symbol.upper(),
            from_date=from_date.isoformat(),
            to_date=snapshot.to_date.isoformat(),
            window=1,
            group_by=group_by,
            snapshot_dates=[from_date.isoformat(), snapshot.to_date.isoformat()],
            net=artifact_net.where(runtime.pd.notna(artifact_net), None).to_dict(orient="records"),
        ).to_dict()
        if config.strict:
            FlowArtifact.model_validate(payload)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    runtime.counts["flow_ok"] += 1


def _write_compare_and_flow_artifacts_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    snapshot: _SymbolSnapshotContext,
) -> None:
    if not runtime.compare_enabled or not (config.compare or config.flow):
        return
    try:
        from_date = _resolve_compare_from_date(runtime=runtime, snapshot=snapshot)
        df_from = runtime.store.load_day(snapshot.symbol, from_date)
        meta_from = runtime.store.load_meta(snapshot.symbol, from_date)
        spot_from = _spot_from_meta(meta_from)
        if spot_from is None:
            raise ValueError("missing spot in from-date meta.json")
        if config.compare:
            _write_compare_artifact(
                config=config,
                runtime=runtime,
                snapshot=snapshot,
                from_date=from_date,
                df_from=df_from,
                spot_from=spot_from,
            )
        if config.flow:
            _write_flow_artifacts(config=config, runtime=runtime, snapshot=snapshot, from_date=from_date, df_from=df_from)
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {snapshot.symbol}: compare/flow skipped: {exc}")


def _write_technicals_artifacts_for_symbol(
    *,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
    symbol: str,
) -> None:
    if not config.technicals:
        return
    try:
        legacy.run_extension_stats_for_symbol(
            symbol=symbol,
            ohlc_path=None,
            cache_dir=config.candle_cache_dir,
            config_path=config.technicals_config,
            tail_pct=config.tail_pct,
            percentile_window_years=config.percentile_window_years,
            out=runtime.out / "technicals" / "extension",
            write_json=True,
            write_md=True,
            print_to_console=False,
            divergence_window_days=14,
            divergence_min_extension_days=5,
            divergence_min_extension_percentile=None,
            divergence_max_extension_percentile=None,
            divergence_min_price_delta_pct=0.0,
            divergence_min_rsi_delta=0.0,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            require_rsi_extreme=False,
        )
        runtime.counts["technicals_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {symbol}: technicals extension-stats failed: {exc}")


def _process_symbol(*, sym: str, config: _ReportPackConfig, runtime: _ReportPackRuntime) -> None:
    snapshot = _load_symbol_snapshot(sym=sym, config=config, runtime=runtime)
    if snapshot is None:
        return
    chain_report_model = _compute_chain_report_for_symbol(config=config, runtime=runtime, snapshot=snapshot)
    _write_chain_artifacts_for_symbol(config=config, runtime=runtime, snapshot=snapshot, chain_report_model=chain_report_model)
    _write_derived_artifacts_for_symbol(config=config, runtime=runtime, snapshot=snapshot, chain_report_model=chain_report_model)
    _write_iv_surface_artifact_for_symbol(config=config, runtime=runtime, snapshot=snapshot)
    _write_exposure_artifact_for_symbol(config=config, runtime=runtime, snapshot=snapshot)
    _write_levels_artifact_for_symbol(config=config, runtime=runtime, snapshot=snapshot)
    _write_compare_and_flow_artifacts_for_symbol(config=config, runtime=runtime, snapshot=snapshot)
    _write_technicals_artifacts_for_symbol(config=config, runtime=runtime, symbol=sym)
    runtime.counts["symbols_ok"] += 1


def _load_scenario_context(
    *,
    target_symbol: str,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
) -> _ScenarioContext:
    scenario_as_of = runtime.store.resolve_date(target_symbol, config.as_of)
    scenario_day = runtime.store.load_day(target_symbol, scenario_as_of)
    scenario_meta = runtime.store.load_meta(target_symbol, scenario_as_of)
    scenario_candles = runtime.candle_store.load(target_symbol)
    scenario_spot, scenario_spot_warnings = _resolve_snapshot_spot_for_report_pack(
        meta=scenario_meta,
        candles=scenario_candles,
        as_of=scenario_as_of,
    )
    return _ScenarioContext(
        as_of=scenario_as_of,
        day_df=scenario_day,
        spot=scenario_spot if scenario_spot > 0 else None,
        warnings=scenario_spot_warnings,
    )


def _write_scenario_artifact_for_target(
    *,
    target: Any,
    context: _ScenarioContext,
    config: _ReportPackConfig,
    runtime: _ReportPackRuntime,
) -> None:
    try:
        base_warnings = list(context.warnings)
        row = find_snapshot_row(context.day_df, expiry=target.expiry, strike=target.strike, option_type=target.option_type)
        if row is None:
            base_warnings.append("missing_snapshot_row")
        spot_value = context.spot
        if spot_value is None:
            spot_value = _row_float(row, "underlyingPrice", "underlying_price", "spot")
        contract_symbol = _row_string(row, "contractSymbol", "contract_symbol")
        if not contract_symbol:
            contract_symbol = _fallback_contract_symbol(
                symbol=target.symbol,
                expiry=target.expiry,
                option_type=target.option_type,
                strike=target.strike,
            )
        bid = _row_float(row, "bid")
        ask = _row_float(row, "ask")
        last = _row_float(row, "lastPrice", "last_price")
        mark = _mark_price(bid=bid, ask=ask, last=last)
        iv = _row_float(row, "impliedVolatility", "implied_volatility")
        computed = compute_position_scenarios(
            symbol=target.symbol,
            as_of=context.as_of,
            contract_symbol=contract_symbol,
            option_type=target.option_type,
            side=target.side,
            contracts=target.contracts,
            spot=spot_value,
            strike=target.strike,
            expiry=target.expiry,
            mark=mark,
            iv=iv,
            basis=target.basis,
        )
        summary_payload = dict(computed.summary)
        summary_warnings = summary_payload.get("warnings")
        summary_payload["warnings"] = _dedupe_strings(
            [*cast(list[str], summary_warnings if isinstance(summary_warnings, list) else []), *base_warnings]
        )
        scenario_artifact = ScenariosArtifact(
            generated_at=utc_now(),
            as_of=context.as_of.isoformat(),
            symbol=target.symbol,
            contract_symbol=contract_symbol,
            summary=ScenarioSummaryRow.model_validate(summary_payload),
            grid=[ScenarioGridRow.model_validate(item) for item in computed.grid],
        )
        if config.strict:
            ScenariosArtifact.model_validate(scenario_artifact.to_dict())
        scenario_base = runtime.out / "scenarios" / target.symbol / context.as_of.isoformat()
        scenario_base.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_base / f"{_safe_file_token(target.key)}.json"
        scenario_path.write_text(scenario_artifact.model_dump_json(indent=2), encoding="utf-8")
        runtime.counts["scenarios_ok"] += 1
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[yellow]Warning:[/yellow] {target.key}: scenarios artifact skipped: {exc}")


def _process_scenarios(*, config: _ReportPackConfig, runtime: _ReportPackRuntime, portfolio: Any) -> None:
    targets = _build_report_pack_scenario_targets(portfolio.positions)
    if not targets:
        runtime.console.print("[yellow]Warning:[/yellow] no portfolio option positions available for scenarios artifacts")
    scenario_contexts: dict[str, _ScenarioContext] = {}
    sort_key = lambda item: (item.symbol, item.key, item.expiry, item.strike, item.side)  # noqa: E731
    for target in sorted(targets, key=sort_key):
        context = scenario_contexts.get(target.symbol)
        if context is None:
            try:
                context = _load_scenario_context(target_symbol=target.symbol, config=config, runtime=runtime)
            except Exception as exc:  # noqa: BLE001
                runtime.console.print(
                    f"[yellow]Warning:[/yellow] {target.symbol}: scenarios skipped (snapshot unavailable: {exc})"
                )
                continue
            scenario_contexts[target.symbol] = context
        _write_scenario_artifact_for_target(target=target, context=context, config=config, runtime=runtime)


def _print_report_pack_summary(*, console: Console, counts: dict[str, int]) -> None:
    console.print(
        "Report pack complete: "
        f"symbols ok={counts['symbols_ok']}/{counts['symbols_total']} | "
        f"chain={counts['chain_ok']} compare={counts['compare_ok']} flow={counts['flow_ok']} "
        f"derived={counts['derived_ok']} iv_surface={counts['iv_surface_ok']} "
        f"exposure={counts['exposure_ok']} levels={counts['levels_ok']} "
        f"technicals={counts['technicals_ok']} scenarios={counts['scenarios_ok']} | "
        f"skipped(required_date)={counts['skipped_required_date']}"
    )


def _run_report_pack(config: _ReportPackConfig) -> None:
    _ensure_pandas()
    pd_module = legacy.pd
    assert pd_module is not None
    console = Console(width=200)
    portfolio = load_portfolio(config.portfolio_path)
    symbols, watchlists_used = _resolve_watchlist_symbols(
        console=console,
        watchlists_path=config.watchlists_path,
        watchlist=config.watchlist,
    )
    if not symbols:
        console.print("[yellow]No symbols selected (empty watchlists).[/yellow]")
        raise typer.Exit(0)
    runtime = _build_runtime(config=config, console=console, pd_module=pd_module, symbols_total=len(symbols))
    console.print(
        "Running offline report pack for "
        f"{len(symbols)} symbol(s) from watchlists: {', '.join([repr(x) for x in watchlists_used])}"
    )
    for sym in sorted(symbols):
        _process_symbol(sym=sym, config=config, runtime=runtime)
    if config.scenarios:
        _process_scenarios(config=config, runtime=runtime, portfolio=portfolio)
    _print_report_pack_summary(console=runtime.console, counts=runtime.counts)


def report_pack(
    portfolio_path: Path = _PORTFOLIO_PATH_ARG,
    watchlists_path: Path = _WATCHLISTS_PATH_OPT,
    watchlist: list[str] = _WATCHLIST_OPT,
    as_of: str = _AS_OF_OPT,
    compare_from: str = _COMPARE_FROM_OPT,
    cache_dir: Path = _CACHE_DIR_OPT,
    derived_dir: Path = _DERIVED_DIR_OPT,
    candle_cache_dir: Path = _CANDLE_CACHE_DIR_OPT,
    out: Path = _OUT_OPT,
    strict: bool = _STRICT_OPT,
    require_snapshot_date: str | None = _REQUIRE_SNAPSHOT_DATE_OPT,
    require_snapshot_tz: str = _REQUIRE_SNAPSHOT_TZ_OPT,
    chain: bool = _CHAIN_OPT,
    compare: bool = _COMPARE_OPT,
    flow: bool = _FLOW_OPT,
    derived: bool = _DERIVED_OPT,
    technicals: bool = _TECHNICALS_OPT,
    iv_surface: bool = _IV_SURFACE_OPT,
    exposure: bool = _EXPOSURE_OPT,
    levels: bool = _LEVELS_OPT,
    levels_benchmark: str = _LEVELS_BENCHMARK_OPT,
    levels_intraday_dir: Path = _LEVELS_INTRADAY_DIR_OPT,
    levels_intraday_timeframe: str = _LEVELS_INTRADAY_TIMEFRAME_OPT,
    levels_volume_bins: int = _LEVELS_VOLUME_BINS_OPT,
    scenarios: bool = _SCENARIOS_OPT,
    technicals_config: Path = _TECHNICALS_CONFIG_OPT,
    top: int = _TOP_OPT,
    derived_window: int = _DERIVED_WINDOW_OPT,
    derived_trend_window: int = _DERIVED_TREND_WINDOW_OPT,
    tail_pct: float | None = _TAIL_PCT_OPT,
    percentile_window_years: int | None = _PERCENTILE_WINDOW_YEARS_OPT,
) -> None:
    """Offline report pack from local snapshots/candles."""
    _run_report_pack(
        _ReportPackConfig(
            portfolio_path=portfolio_path,
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            as_of=as_of,
            compare_from=compare_from,
            cache_dir=cache_dir,
            derived_dir=derived_dir,
            candle_cache_dir=candle_cache_dir,
            out=out,
            strict=strict,
            require_snapshot_date=require_snapshot_date,
            require_snapshot_tz=require_snapshot_tz,
            chain=chain,
            compare=compare,
            flow=flow,
            derived=derived,
            technicals=technicals,
            iv_surface=iv_surface,
            exposure=exposure,
            levels=levels,
            levels_benchmark=levels_benchmark,
            levels_intraday_dir=levels_intraday_dir,
            levels_intraday_timeframe=levels_intraday_timeframe,
            levels_volume_bins=levels_volume_bins,
            scenarios=scenarios,
            technicals_config=technicals_config,
            top=top,
            derived_window=derived_window,
            derived_trend_window=derived_trend_window,
            tail_pct=tail_pct,
            percentile_window_years=percentile_window_years,
        )
    )


__all__ = ["report_pack"]
