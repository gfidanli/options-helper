from __future__ import annotations

import json
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


def report_pack(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (required for watchlist defaults)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to include (repeatable). Default: positions, monitor, Scanner - Shortlist.",
    ),
    as_of: str = typer.Option(
        "latest",
        "--as-of",
        help="Snapshot date (YYYY-MM-DD) or 'latest'.",
    ),
    compare_from: str = typer.Option(
        "-1",
        "--compare-from",
        help="Compare-from date (relative negative offsets or YYYY-MM-DD). Use none to disable.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    out: Path = typer.Option(
        Path("data/reports"),
        "--out",
        help="Output root for report pack artifacts.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    require_snapshot_date: str | None = typer.Option(
        None,
        "--require-snapshot-date",
        help="Skip symbols unless the snapshot date matches this date (YYYY-MM-DD) or 'today'.",
    ),
    require_snapshot_tz: str = typer.Option(
        "America/New_York",
        "--require-snapshot-tz",
        help="Timezone used when --require-snapshot-date is 'today'.",
    ),
    chain: bool = typer.Option(
        True,
        "--chain/--no-chain",
        help="Generate chain report artifacts.",
    ),
    compare: bool = typer.Option(
        True,
        "--compare/--no-compare",
        help="Generate compare report artifacts (requires previous snapshot).",
    ),
    flow: bool = typer.Option(
        True,
        "--flow/--no-flow",
        help="Generate flow artifacts (requires previous snapshot).",
    ),
    derived: bool = typer.Option(
        True,
        "--derived/--no-derived",
        help="Update derived metrics + emit derived stats artifacts.",
    ),
    technicals: bool = typer.Option(
        True,
        "--technicals/--no-technicals",
        help="Generate technicals extension-stats artifacts (offline, from candle cache).",
    ),
    iv_surface: bool = typer.Option(
        True,
        "--iv-surface/--no-iv-surface",
        help="Generate IV surface artifacts from local snapshots.",
    ),
    exposure: bool = typer.Option(
        True,
        "--exposure/--no-exposure",
        help="Generate dealer exposure artifacts from local snapshots.",
    ),
    levels: bool = typer.Option(
        True,
        "--levels/--no-levels",
        help="Generate levels artifacts from local candles and optional intraday partitions.",
    ),
    levels_benchmark: str = typer.Option(
        "SPY",
        "--levels-benchmark",
        help="Benchmark symbol used for RS/Beta in levels artifacts.",
    ),
    levels_intraday_dir: Path = typer.Option(
        Path("data/intraday"),
        "--levels-intraday-dir",
        help="Intraday partition root used for anchored VWAP and volume profile.",
    ),
    levels_intraday_timeframe: str = typer.Option(
        "1Min",
        "--levels-intraday-timeframe",
        help="Intraday partition timeframe for levels artifacts.",
    ),
    levels_volume_bins: int = typer.Option(
        20,
        "--levels-volume-bins",
        min=1,
        max=200,
        help="Volume-profile bins for levels artifacts.",
    ),
    scenarios: bool = typer.Option(
        False,
        "--scenarios/--no-scenarios",
        help="Generate per-position scenarios artifacts for portfolio positions.",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (used for extension-stats artifacts).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top rows/strikes to include in reports."),
    derived_window: int = typer.Option(60, "--derived-window", min=1, max=3650, help="Derived stats lookback window."),
    derived_trend_window: int = typer.Option(
        5, "--derived-trend-window", min=1, max=3650, help="Derived stats trend lookback window."
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Optional symmetric tail threshold for technicals extension-stats (e.g. 5 => low<=5, high>=95).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Optional rolling window (years) for extension percentiles in technicals extension-stats.",
    ),
) -> None:
    """
    Offline report pack from local snapshots/candles.

    Generates per-symbol artifacts under `--out`:
    - chains/{SYMBOL}/{YYYY-MM-DD}.json + .md
    - compare/{SYMBOL}/{FROM}_to_{TO}.json
    - flow/{SYMBOL}/{FROM}_to_{TO}_w1_{group_by}.json
    - derived/{SYMBOL}/{ASOF}_w{N}_tw{M}.json
    - iv_surface/{SYMBOL}/{ASOF}.json
    - exposure/{SYMBOL}/{ASOF}.json
    - levels/{SYMBOL}/{ASOF}.json
    - technicals/extension/{SYMBOL}/{ASOF}.json + .md
    - scenarios/{SYMBOL}/{ASOF}/{POSITION_KEY}.json (optional)
    """
    _ensure_pandas()
    pd = legacy.pd
    assert pd is not None
    console = Console(width=200)
    portfolio = load_portfolio(portfolio_path)

    wl = load_watchlists(watchlists_path)
    watchlists_used = watchlist[:] if watchlist else ["positions", "monitor", "Scanner - Shortlist"]
    symbols: set[str] = set()
    for name in watchlists_used:
        syms = wl.get(name)
        if not syms:
            console.print(f"[yellow]Warning:[/yellow] watchlist '{name}' missing/empty in {watchlists_path}")
            continue
        symbols.update(syms)

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        console.print("[yellow]No symbols selected (empty watchlists).[/yellow]")
        raise typer.Exit(0)

    store = cli_deps.build_snapshot_store(cache_dir)
    derived_store = cli_deps.build_derived_store(derived_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)
    intraday_store = IntradayStore(levels_intraday_dir)
    benchmark_symbol = str(levels_benchmark or "").strip().upper()
    if not benchmark_symbol:
        benchmark_symbol = "SPY"

    required_date: date | None = None
    if require_snapshot_date is not None:
        spec = require_snapshot_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_snapshot_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(
                f"Invalid --require-snapshot-date/--require-snapshot-tz: {exc}",
                param_hint="--require-snapshot-date",
            ) from exc

    compare_norm = compare_from.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    out = out.expanduser()
    (out / "chains").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "flow").mkdir(parents=True, exist_ok=True)
    (out / "derived").mkdir(parents=True, exist_ok=True)
    (out / "iv_surface").mkdir(parents=True, exist_ok=True)
    (out / "exposure").mkdir(parents=True, exist_ok=True)
    (out / "levels").mkdir(parents=True, exist_ok=True)
    (out / "scenarios").mkdir(parents=True, exist_ok=True)
    (out / "technicals" / "extension").mkdir(parents=True, exist_ok=True)

    console.print(
        "Running offline report pack for "
        f"{len(symbols)} symbol(s) from watchlists: {', '.join([repr(x) for x in watchlists_used])}"
    )

    counts = {
        "symbols_total": len(symbols),
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

    for sym in sorted(symbols):
        try:
            to_date = store.resolve_date(sym, as_of)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {sym}: no snapshots ({exc})")
            continue

        if required_date is not None and to_date != required_date:
            counts["skipped_required_date"] += 1
            continue

        df_to = store.load_day(sym, to_date)
        meta_to = store.load_meta(sym, to_date)
        candles = candle_store.load(sym)
        spot_to, spot_warnings = _resolve_snapshot_spot_for_report_pack(
            meta=meta_to,
            candles=candles,
            as_of=to_date,
        )
        if spot_to <= 0:
            console.print(
                f"[yellow]Warning:[/yellow] {sym}: missing spot for {to_date.isoformat()} "
                "(meta and candle fallback unavailable)"
            )
            continue
        if spot_warnings:
            console.print(f"[yellow]Warning:[/yellow] {sym}: " + ", ".join(spot_warnings))

        chain_report_model = None
        if chain or derived:
            try:
                chain_report_model = compute_chain_report(
                    df_to,
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    expiries_mode="near",
                    top=top,
                    best_effort=True,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: chain-report failed: {exc}")
                chain_report_model = None

        if chain and chain_report_model is not None:
            try:
                base = out / "chains" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                json_path = base / f"{to_date.isoformat()}.json"
                md_path = base / f"{to_date.isoformat()}.md"
                chain_artifact = ChainReportArtifact(
                    generated_at=utc_now(),
                    **chain_report_model.model_dump(),
                )
                if strict:
                    ChainReportArtifact.model_validate(chain_artifact.to_dict())
                json_path.write_text(chain_artifact.model_dump_json(indent=2), encoding="utf-8")
                md_path.write_text(render_chain_report_markdown(chain_report_model), encoding="utf-8")
                counts["chain_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: failed writing chain artifacts: {exc}")

        if derived and chain_report_model is not None:
            try:
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain_report_model, candles=candles, derived_history=history)
                derived_store.upsert(sym, row)
                df_derived = derived_store.load(sym)
                if not df_derived.empty:
                    stats = compute_derived_stats(
                        df_derived,
                        symbol=sym,
                        as_of="latest",
                        window=derived_window,
                        trend_window=derived_trend_window,
                        metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
                    )
                    base = out / "derived" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    stats_path = base / f"{stats.as_of}_w{derived_window}_tw{derived_trend_window}.json"
                    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
                    counts["derived_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: derived update/stats failed: {exc}")

        if iv_surface:
            try:
                previous_tenor = None
                previous_delta_buckets = None
                try:
                    previous_date = store.resolve_relative_date(sym, to_date=to_date, offset=-1)
                except Exception:  # noqa: BLE001
                    previous_date = None

                if previous_date is not None:
                    previous_snapshot = store.load_day(sym, previous_date)
                    if not previous_snapshot.empty:
                        previous_meta = store.load_meta(sym, previous_date)
                        previous_spot, _previous_spot_warnings = _resolve_snapshot_spot_for_report_pack(
                            meta=previous_meta,
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
                        previous_delta_buckets = previous_surface.delta_buckets

                current_surface = compute_iv_surface(
                    df_to,
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    previous_tenor=previous_tenor,
                    previous_delta_buckets=previous_delta_buckets,
                )
                iv_artifact = _build_iv_surface_artifact_for_report_pack(
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    surface=current_surface,
                    warnings=spot_warnings,
                )
                if strict:
                    IvSurfaceArtifact.model_validate(iv_artifact.to_dict())
                base = out / "iv_surface" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                iv_path = base / f"{to_date.isoformat()}.json"
                iv_path.write_text(iv_artifact.model_dump_json(indent=2), encoding="utf-8")
                counts["iv_surface_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: iv-surface artifact skipped: {exc}")

        if exposure:
            try:
                exposure_slices = compute_exposure_slices(
                    df_to,
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    near_n=4,
                    top_n=top,
                )
                exposure_artifact = _build_exposure_artifact_for_report_pack(
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    slices=exposure_slices,
                    warnings=spot_warnings,
                )
                if strict:
                    ExposureArtifact.model_validate(exposure_artifact.to_dict())
                base = out / "exposure" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                exposure_path = base / f"{to_date.isoformat()}.json"
                exposure_path.write_text(exposure_artifact.model_dump_json(indent=2), encoding="utf-8")
                counts["exposure_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: exposure artifact skipped: {exc}")

        if levels:
            try:
                symbol_history = _normalize_daily_history(candles)
                symbol_slice = _slice_history_to_as_of(symbol_history, to_date)

                benchmark_history = _normalize_daily_history(candle_store.load(benchmark_symbol))
                benchmark_slice = _slice_history_to_as_of(benchmark_history, to_date) if not benchmark_history.empty else None

                levels_summary = compute_levels_summary(
                    symbol_slice,
                    benchmark_daily=benchmark_slice,
                    rolling_window=20,
                    rs_window=20,
                )
                intraday_bars = intraday_store.load_partition(
                    "stocks",
                    "bars",
                    levels_intraday_timeframe,
                    sym,
                    to_date,
                )
                anchored = compute_anchored_vwap(intraday_bars, anchor_type="session_open", spot=levels_summary.spot)
                profile = compute_volume_profile(
                    intraday_bars,
                    num_bins=levels_volume_bins,
                    hvn_quantile=0.8,
                    lvn_quantile=0.2,
                )
                levels_artifact = _build_levels_artifact_for_report_pack(
                    symbol=sym,
                    as_of=to_date,
                    summary=levels_summary,
                    anchored=anchored,
                    profile=profile,
                )
                if strict:
                    LevelsArtifact.model_validate(levels_artifact.to_dict())
                base = out / "levels" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                levels_path = base / f"{to_date.isoformat()}.json"
                levels_path.write_text(levels_artifact.model_dump_json(indent=2), encoding="utf-8")
                counts["levels_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: levels artifact skipped: {exc}")

        if compare_enabled and (compare or flow):
            try:
                from_date: date
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                df_from = store.load_day(sym, from_date)
                meta_from = store.load_meta(sym, from_date)
                spot_from = _spot_from_meta(meta_from)
                if spot_from is None:
                    raise ValueError("missing spot in from-date meta.json")

                if compare:
                    diff, report_from, report_to = compute_compare_report(
                        symbol=sym,
                        from_date=from_date,
                        to_date=to_date,
                        from_df=df_from,
                        to_df=df_to,
                        spot_from=spot_from,
                        spot_to=spot_to,
                        top=top,
                    )
                    base = out / "compare" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}.json"
                    payload = CompareArtifact(
                        schema_version=1,
                        generated_at=utc_now(),
                        as_of=to_date.isoformat(),
                        symbol=sym.upper(),
                        from_report=report_from.model_dump(),
                        to_report=report_to.model_dump(),
                        diff=diff.model_dump(),
                    ).to_dict()
                    if strict:
                        CompareArtifact.model_validate(payload)
                    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    counts["compare_ok"] += 1

                if flow:
                    pair_flow = compute_flow(df_to, df_from, spot=spot_to)
                    if not pair_flow.empty:
                        for group_by in ("contract", "expiry-strike"):
                            net = aggregate_flow_window([pair_flow], group_by=cast(FlowGroupBy, group_by))
                            base = out / "flow" / sym.upper()
                            base.mkdir(parents=True, exist_ok=True)
                            out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}_w1_{group_by}.json"
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
                                as_of=to_date.isoformat(),
                                symbol=sym.upper(),
                                from_date=from_date.isoformat(),
                                to_date=to_date.isoformat(),
                                window=1,
                                group_by=group_by,
                                snapshot_dates=[from_date.isoformat(), to_date.isoformat()],
                                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                            ).to_dict()
                            if strict:
                                FlowArtifact.model_validate(payload)
                            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                        counts["flow_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: compare/flow skipped: {exc}")

        if technicals:
            try:
                legacy.run_extension_stats_for_symbol(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=technicals_config,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=out / "technicals" / "extension",
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
                counts["technicals_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: technicals extension-stats failed: {exc}")

        counts["symbols_ok"] += 1

    if scenarios:
        targets = _build_report_pack_scenario_targets(portfolio.positions)
        if not targets:
            console.print("[yellow]Warning:[/yellow] no portfolio option positions available for scenarios artifacts")
        scenario_contexts: dict[str, dict[str, Any]] = {}
        for target in sorted(targets, key=lambda item: (item.symbol, item.key, item.expiry, item.strike, item.side)):
            context = scenario_contexts.get(target.symbol)
            if context is None:
                try:
                    scenario_as_of = store.resolve_date(target.symbol, as_of)
                    scenario_day = store.load_day(target.symbol, scenario_as_of)
                    scenario_meta = store.load_meta(target.symbol, scenario_as_of)
                    scenario_candles = candle_store.load(target.symbol)
                    scenario_spot, scenario_spot_warnings = _resolve_snapshot_spot_for_report_pack(
                        meta=scenario_meta,
                        candles=scenario_candles,
                        as_of=scenario_as_of,
                    )
                    context = {
                        "as_of": scenario_as_of,
                        "day_df": scenario_day,
                        "spot": scenario_spot if scenario_spot > 0 else None,
                        "warnings": scenario_spot_warnings,
                    }
                    scenario_contexts[target.symbol] = context
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]Warning:[/yellow] {target.symbol}: scenarios skipped (snapshot unavailable: {exc})"
                    )
                    continue

            try:
                as_of_date = cast(date, context["as_of"])
                day_df = cast("pd.DataFrame", context["day_df"])
                base_warnings = list(cast(list[str], context["warnings"]))
                row = find_snapshot_row(
                    day_df,
                    expiry=target.expiry,
                    strike=target.strike,
                    option_type=target.option_type,
                )
                if row is None:
                    base_warnings.append("missing_snapshot_row")

                spot_value = cast(float | None, context["spot"])
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
                    as_of=as_of_date,
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
                    as_of=as_of_date.isoformat(),
                    symbol=target.symbol,
                    contract_symbol=contract_symbol,
                    summary=ScenarioSummaryRow.model_validate(summary_payload),
                    grid=[ScenarioGridRow.model_validate(item) for item in computed.grid],
                )
                if strict:
                    ScenariosArtifact.model_validate(scenario_artifact.to_dict())
                scenario_base = out / "scenarios" / target.symbol / as_of_date.isoformat()
                scenario_base.mkdir(parents=True, exist_ok=True)
                scenario_path = scenario_base / f"{_safe_file_token(target.key)}.json"
                scenario_path.write_text(scenario_artifact.model_dump_json(indent=2), encoding="utf-8")
                counts["scenarios_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {target.key}: scenarios artifact skipped: {exc}")

    console.print(
        "Report pack complete: "
        f"symbols ok={counts['symbols_ok']}/{counts['symbols_total']} | "
        f"chain={counts['chain_ok']} compare={counts['compare_ok']} flow={counts['flow_ok']} "
        f"derived={counts['derived_ok']} iv_surface={counts['iv_surface_ok']} "
        f"exposure={counts['exposure_ok']} levels={counts['levels_ok']} "
        f"technicals={counts['technicals_ok']} scenarios={counts['scenarios_ok']} | "
        f"skipped(required_date)={counts['skipped_required_date']}"
    )


__all__ = ["report_pack"]
