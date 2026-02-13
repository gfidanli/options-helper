from __future__ import annotations

from pathlib import Path

import typer
from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams


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
    sync_legacy_seams()
    return legacy.report_pack(
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


__all__ = ["report_pack"]
