from __future__ import annotations

from pathlib import Path

import typer
from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams

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
    sync_legacy_seams()
    kwargs = locals().copy()
    return legacy.report_pack(**kwargs)


__all__ = ["report_pack"]
