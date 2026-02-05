from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.advice import PositionMetrics
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, score_confluence
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.portfolio_risk import compute_portfolio_exposure, run_stress
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.roll_plan_multileg import compute_roll_plan_multileg
from options_helper.commands.common import _build_stress_scenarios, _parse_date, _spot_from_meta
from options_helper.commands.position_metrics import _extract_float, _mark_price, _position_metrics
from options_helper.commands.technicals import technicals_extension_stats
from options_helper.data.candles import close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.derived import DERIVED_COLUMNS
from options_helper.data.earnings import safe_next_earnings_date
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.models import MultiLegPosition, Position
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobExecutionError,
    VisibilityJobParameterError,
    render_dashboard_report,
    run_briefing_job,
    run_dashboard_job,
    run_flow_report_job,
)
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
    render_portfolio_table_markdown,
)
from options_helper.reporting_chain import (
    render_chain_report_console,
    render_chain_report_markdown,
    render_compare_report_console,
)
from options_helper.reporting_roll import render_roll_plan_console, render_roll_plan_multileg_console
from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.common import utc_now
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.storage import load_portfolio
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot
from options_helper.ui.dashboard import load_briefing_artifact, render_dashboard, resolve_briefing_paths
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


pd: object | None = None


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def register(app: typer.Typer) -> None:
    app.command("flow")(flow_report)
    app.command("chain-report")(chain_report)
    app.command("compare")(compare_report)
    app.command("report-pack")(report_pack)
    app.command("briefing")(briefing)
    app.command("dashboard")(dashboard)
    app.command("roll-plan")(roll_plan)


def flow_report(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Restrict flow report to a single symbol.",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Use symbols from watchlist (repeatable). Ignored when --all-watchlists is set.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Use all watchlists instead of portfolio symbols.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory containing options snapshot folders.",
    ),
    window: int = typer.Option(
        1,
        "--window",
        min=1,
        max=30,
        help="Number of snapshot-to-snapshot deltas to net (requires N+1 snapshots).",
    ),
    group_by: str = typer.Option(
        "contract",
        "--group-by",
        help="Aggregation mode: contract|strike|expiry|expiry-strike",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/flow/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    console = Console()
    try:
        result = run_flow_report_job(
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
    except VisibilityJobParameterError as exc:
        if exc.param_hint:
            raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
        raise typer.BadParameter(str(exc)) from exc

    for renderable in result.renderables:
        console.print(renderable)
    if result.no_symbols:
        raise typer.Exit(0)


def chain_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to report on."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        help="Output format: console|md|json",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/chains/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    include_expiry: list[str] = typer.Option(
        [],
        "--include-expiry",
        help="Include a specific expiry date (repeatable). When provided, overrides --expiries selection.",
    ),
    expiries: str = typer.Option(
        "near",
        "--expiries",
        help="Expiry selection mode: near|monthly|all (ignored when --include-expiry is used).",
    ),
    best_effort: bool = typer.Option(
        False,
        "--best-effort",
        help="Don't fail hard on missing fields; emit warnings and partial outputs.",
    ),
) -> None:
    """Offline options chain dashboard from local snapshot files."""
    console = Console()
    store = cli_deps.build_snapshot_store(cache_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        fmt = format.strip().lower()
        if fmt not in {"console", "md", "json"}:
            raise typer.BadParameter("Invalid --format (use console|md|json)", param_hint="--format")

        expiries_mode = expiries.strip().lower()
        if expiries_mode not in {"near", "monthly", "all"}:
            raise typer.BadParameter("Invalid --expiries (use near|monthly|all)", param_hint="--expiries")

        include_dates = [_parse_date(x) for x in include_expiry] if include_expiry else None

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode=expiries_mode,  # type: ignore[arg-type]
            include_expiries=include_dates,
            top=top,
            best_effort=best_effort,
        )
        report_artifact = ChainReportArtifact(
            generated_at=utc_now(),
            **report.model_dump(),
        )
        if strict:
            ChainReportArtifact.model_validate(report_artifact.to_dict())

        if fmt == "console":
            render_chain_report_console(console, report)
        elif fmt == "md":
            console.print(render_chain_report_markdown(report))
        else:
            console.print(report_artifact.model_dump_json(indent=2))

        if out is not None:
            base = out / "chains" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            json_path = base / f"{as_of_date.isoformat()}.json"
            json_path.write_text(report_artifact.model_dump_json(indent=2), encoding="utf-8")
            md_path = base / f"{as_of_date.isoformat()}.md"
            md_path.write_text(render_chain_report_markdown(report), encoding="utf-8")
            console.print(f"\nSaved: {json_path}")
            console.print(f"Saved: {md_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def compare_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to compare."),
    from_spec: str = typer.Option(
        "-1",
        "--from",
        help="From snapshot date (YYYY-MM-DD) or a negative offset relative to --to (e.g. -1).",
    ),
    to_spec: str = typer.Option("latest", "--to", help="To snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/compare/{SYMBOL}/).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Compare two snapshot days for a symbol (delta in OI/IV/Greeks)."""
    console = Console()
    store = cli_deps.build_snapshot_store(cache_dir)

    try:
        to_dt = store.resolve_date(symbol, to_spec)
        from_spec_norm = from_spec.strip().lower()
        if from_spec_norm.startswith("-") and from_spec_norm[1:].isdigit():
            from_dt = store.resolve_relative_date(symbol, to_date=to_dt, offset=int(from_spec_norm))
        else:
            from_dt = store.resolve_date(symbol, from_spec_norm)

        if from_dt == to_dt:
            raise typer.BadParameter("--from and --to must be different dates.")

        df_from = store.load_day(symbol, from_dt)
        df_to = store.load_day(symbol, to_dt)
        meta_from = store.load_meta(symbol, from_dt)
        meta_to = store.load_meta(symbol, to_dt)
        spot_from = _spot_from_meta(meta_from)
        spot_to = _spot_from_meta(meta_to)
        if spot_from is None or spot_to is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        diff, report_from, report_to = compute_compare_report(
            symbol=symbol,
            from_date=from_dt,
            to_date=to_dt,
            from_df=df_from,
            to_df=df_to,
            spot_from=spot_from,
            spot_to=spot_to,
            top=top,
        )

        artifact = CompareArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=to_dt.isoformat(),
            symbol=symbol.upper(),
            from_report=report_from.model_dump(),
            to_report=report_to.model_dump(),
            diff=diff.model_dump(),
        )
        if strict:
            CompareArtifact.model_validate(artifact.to_dict())

        render_compare_report_console(console, diff)

        if out is not None:
            base = out / "compare" / symbol.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{from_dt.isoformat()}_to_{to_dt.isoformat()}.json"
            out_path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


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
    - technicals/extension/{SYMBOL}/{ASOF}.json + .md
    """
    _ensure_pandas()
    console = Console(width=200)
    _ = load_portfolio(portfolio_path)

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
        "technicals_ok": 0,
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
        spot_to = _spot_from_meta(meta_to)
        if spot_to is None:
            console.print(f"[yellow]Warning:[/yellow] {sym}: missing spot in meta.json for {to_date.isoformat()}")
            continue

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
                candles = candle_store.load(sym)
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
                technicals_extension_stats(
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

    console.print(
        "Report pack complete: "
        f"symbols ok={counts['symbols_ok']}/{counts['symbols_total']} | "
        f"chain={counts['chain_ok']} compare={counts['compare_ok']} flow={counts['flow_ok']} "
        f"derived={counts['derived_ok']} technicals={counts['technicals_ok']} | "
        f"skipped(required_date)={counts['skipped_required_date']}"
    )


def briefing(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). Adds to portfolio symbols.",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only include a single symbol (overrides portfolio/watchlists selection).",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    compare: str = typer.Option(
        "-1",
        "--compare",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technical context).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (canonical indicator definitions).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path (Markdown) or directory. Default: data/reports/daily/{ASOF}.md",
    ),
    print_to_console: bool = typer.Option(
        False,
        "--print/--no-print",
        help="Print the briefing to the console (in addition to writing files).",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write a JSON version of the briefing alongside the Markdown (LLM-friendly).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    update_derived: bool = typer.Option(
        True,
        "--update-derived/--no-update-derived",
        help="Update derived metrics for included symbols (per-symbol CSV).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used when --update-derived).",
    ),
    top: int = typer.Option(3, "--top", min=1, max=10, help="Top rows to include in compare/flow sections."),
) -> None:
    """Generate a daily Markdown briefing for portfolio + optional watchlists (offline-first)."""
    console = Console(width=200)
    try:
        result = run_briefing_job(
            portfolio_path=portfolio_path,
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            as_of=as_of,
            compare=compare,
            cache_dir=cache_dir,
            candle_cache_dir=candle_cache_dir,
            technicals_config=technicals_config,
            out=out,
            print_to_console=print_to_console,
            write_json=write_json,
            strict=strict,
            update_derived=update_derived,
            derived_dir=derived_dir,
            top=top,
            snapshot_store_builder=cli_deps.build_snapshot_store,
            derived_store_builder=cli_deps.build_derived_store,
            candle_store_builder=cli_deps.build_candle_store,
            earnings_store_builder=cli_deps.build_earnings_store,
            safe_next_earnings_date_fn=safe_next_earnings_date,
        )
    except VisibilityJobExecutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    for renderable in result.renderables:
        console.print(renderable)


def dashboard(
    report_date: str = typer.Option(
        "latest",
        "--date",
        help="Briefing date (YYYY-MM-DD) or 'latest'.",
    ),
    reports_dir: Path = typer.Option(
        Path("data/reports"),
        "--reports-dir",
        help="Reports root (expects {reports_dir}/daily/{DATE}.json).",
    ),
    scanner_run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--scanner-run-dir",
        help="Scanner runs directory (for shortlist view).",
    ),
    scanner_run_id: str | None = typer.Option(
        None,
        "--scanner-run-id",
        help="Specific scanner run id to display (defaults to latest for the date).",
    ),
    max_shortlist_rows: int = typer.Option(
        20,
        "--max-shortlist-rows",
        min=1,
        max=200,
        help="Max rows to show in the scanner shortlist table.",
    ),
) -> None:
    """Render a read-only daily dashboard from briefing JSON + artifacts."""
    console = Console(width=200)
    try:
        result = run_dashboard_job(
            report_date=report_date,
            reports_dir=reports_dir,
        )
    except VisibilityJobExecutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    render_dashboard_report(
        result=result,
        reports_dir=reports_dir,
        scanner_run_dir=scanner_run_dir,
        scanner_run_id=scanner_run_id,
        max_shortlist_rows=max_shortlist_rows,
        render_console=console,
    )


def roll_plan(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (positions + risk profile)."),
    position_id: str = typer.Option(..., "--id", help="Position id to plan a roll for."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    intent: str = typer.Option(
        "max-upside",
        "--intent",
        help="Intent: max-upside|reduce-theta|increase-delta|de-risk",
    ),
    horizon_months: int = typer.Option(..., "--horizon-months", min=1, max=60),
    shape: str = typer.Option(
        "out-same-strike",
        "--shape",
        help="Roll shape: out-same-strike|out-up|out-down",
    ),
    top: int = typer.Option(10, "--top", min=1, max=50, help="Number of candidates to display."),
    max_debit: float | None = typer.Option(
        None,
        "--max-debit",
        help="Max roll debit in dollars (total for position size).",
    ),
    min_credit: float | None = typer.Option(
        None,
        "--min-credit",
        help="Min roll credit in dollars (total for position size).",
    ),
    min_open_interest: int | None = typer.Option(
        None,
        "--min-open-interest",
        help="Override minimum open interest liquidity gate (default from risk profile).",
    ),
    min_volume: int | None = typer.Option(
        None,
        "--min-volume",
        help="Override minimum volume liquidity gate (default from risk profile).",
    ),
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
    ),
) -> None:
    """Propose and rank roll candidates for a single position using offline snapshots."""
    console = Console(width=200)

    portfolio = load_portfolio(portfolio_path)
    position = next((p for p in portfolio.positions if p.id == position_id), None)
    if position is None:
        raise typer.BadParameter(f"No position found with id: {position_id}", param_hint="--id")

    intent_norm = intent.strip().lower()
    if intent_norm not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
        raise typer.BadParameter(
            "Invalid --intent (use max-upside|reduce-theta|increase-delta|de-risk)",
            param_hint="--intent",
        )

    shape_norm = shape.strip().lower()
    if shape_norm not in {"out-same-strike", "out-up", "out-down"}:
        raise typer.BadParameter("Invalid --shape (use out-same-strike|out-up|out-down)", param_hint="--shape")

    rp = portfolio.risk_profile
    min_oi = rp.min_open_interest if min_open_interest is None else int(min_open_interest)
    min_vol = rp.min_volume if min_volume is None else int(min_volume)

    store = cli_deps.build_snapshot_store(cache_dir)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    next_earnings_date = safe_next_earnings_date(earnings_store, position.symbol)

    try:
        as_of_date = store.resolve_date(position.symbol, as_of)
        df = store.load_day(position.symbol, as_of_date)
        meta = store.load_meta(position.symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        if isinstance(position, MultiLegPosition):
            report = compute_roll_plan_multileg(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                horizon_months=horizon_months,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
            )
            render_roll_plan_multileg_console(console, report)
        else:
            report = compute_roll_plan(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                intent=intent_norm,
                horizon_months=horizon_months,
                shape=shape_norm,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
            )
            render_roll_plan_console(console, report)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
