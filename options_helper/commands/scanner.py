from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import typer

import options_helper.cli_deps as cli_deps
from options_helper.commands.technicals_common import setup_technicals_logging
from .scanner_runtime_runner import run_scanner_command
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
from options_helper.pipelines.technicals_extension_stats import run_extension_stats_for_symbol
from options_helper.schemas.common import utc_now
from options_helper.schemas.scanner_shortlist import (
    ScannerShortlistArtifact,
    ScannerShortlistRow as ScannerShortlistRowSchema,
)
from options_helper.watchlists import load_watchlists, save_watchlists

app = typer.Typer(help="Market opportunity scanner (not financial advice).")

UNIVERSE_OPT = typer.Option(
    "file:data/universe/sec_company_tickers.json",
    "--universe",
    help="Universe source: us-all/us-equities/us-etfs or file:/path/to/list.txt.",
)
UNIVERSE_CACHE_DIR_OPT = typer.Option(
    Path("data/universe"),
    "--universe-cache-dir",
    help="Directory for cached universe lists.",
)
UNIVERSE_REFRESH_DAYS_OPT = typer.Option(
    1,
    "--universe-refresh-days",
    help="Refresh universe cache if older than this many days.",
)
MAX_SYMBOLS_OPT = typer.Option(
    None,
    "--max-symbols",
    min=1,
    help="Optional cap on number of symbols scanned (for dev/testing).",
)
PREFILTER_MODE_OPT = typer.Option(
    "default",
    "--prefilter-mode",
    help="Prefilter mode: default, aggressive, or none.",
)
EXCLUDE_PATH_OPT = typer.Option(
    Path("data/universe/exclude_symbols.txt"),
    "--exclude-path",
    help="Path to exclude symbols file (one ticker per line).",
)
SCANNED_PATH_OPT = typer.Option(
    Path("data/scanner/scanned_symbols.txt"),
    "--scanned-path",
    help="Path to scanned symbols file (one ticker per line).",
)
SKIP_SCANNED_OPT = typer.Option(
    True,
    "--skip-scanned/--no-skip-scanned",
    help="Skip symbols already recorded in the scanned file.",
)
WRITE_SCANNED_OPT = typer.Option(
    True,
    "--write-scanned/--no-write-scanned",
    help="Persist scanned symbols so future runs skip them.",
)
WRITE_ERROR_EXCLUDES_OPT = typer.Option(
    True,
    "--write-error-excludes/--no-write-error-excludes",
    help="Persist symbols that error to the exclude file.",
)
EXCLUDE_STATUSES_OPT = typer.Option(
    "error,no_candles",
    "--exclude-statuses",
    help="Comma-separated scan statuses to add to the exclude file.",
)
ERROR_FLUSH_EVERY_OPT = typer.Option(
    50,
    "--error-flush-every",
    min=1,
    help="Flush exclude file after this many new error symbols.",
)
SCANNED_FLUSH_EVERY_OPT = typer.Option(
    250,
    "--scanned-flush-every",
    min=1,
    help="Flush scanned file after this many new symbols.",
)
SCAN_PERIOD_OPT = typer.Option(
    "max",
    "--scan-period",
    help="Candle period to pull for the scan (yfinance period format).",
)
TAIL_PCT_OPT = typer.Option(
    None,
    "--tail-pct",
    help="Symmetric tail threshold percentile (e.g. 2.5 => low<=2.5, high>=97.5).",
)
PERCENTILE_WINDOW_YEARS_OPT = typer.Option(
    None,
    "--percentile-window-years",
    help="Rolling window (years) for extension percentiles (default: auto 1y/3y).",
)
WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store.",
)
ALL_WATCHLIST_NAME_OPT = typer.Option(
    "Scanner - All",
    "--all-watchlist-name",
    help="Watchlist name for all tail symbols (replaced each run).",
)
SHORTLIST_WATCHLIST_NAME_OPT = typer.Option(
    "Scanner - Shortlist",
    "--shortlist-watchlist-name",
    help="Watchlist name for liquid short list (replaced each run).",
)
CANDLE_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--candle-cache-dir",
    help="Directory for cached daily candles.",
)
OPTIONS_CACHE_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--options-cache-dir",
    help="Directory for options chain snapshots.",
)
DERIVED_DIR_OPT = typer.Option(
    Path("data/derived"),
    "--derived-dir",
    help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
)
SPOT_PERIOD_OPT = typer.Option(
    "10d",
    "--spot-period",
    help="Candle period used to estimate spot price when snapshotting options.",
)
RISK_FREE_RATE_OPT = typer.Option(
    0.0,
    "--risk-free-rate",
    help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
)
BACKFILL_OPT = typer.Option(
    False,
    "--backfill/--no-backfill",
    help="Backfill max candles for tail symbols (slow).",
)
SNAPSHOT_OPTIONS_OPT = typer.Option(
    False,
    "--snapshot-options/--no-snapshot-options",
    help="Snapshot options for tail symbols (slow; uses --options-cache-dir).",
)
LIQUIDITY_MIN_DTE_OPT = typer.Option(
    60,
    "--liquidity-min-dte",
    help="Minimum DTE for liquidity screening.",
)
LIQUIDITY_MIN_VOLUME_OPT = typer.Option(
    10,
    "--liquidity-min-volume",
    help="Minimum volume for liquidity screening.",
)
LIQUIDITY_MIN_OI_OPT = typer.Option(
    500,
    "--liquidity-min-oi",
    help="Minimum open interest for liquidity screening.",
)
RUN_DIR_OPT = typer.Option(
    Path("data/scanner/runs"),
    "--run-dir",
    help="Output root for scanner runs.",
)
RUN_ID_OPT = typer.Option(
    None,
    "--run-id",
    help="Optional run id (default: timestamp).",
)
WORKERS_OPT = typer.Option(
    None,
    "--workers",
    min=1,
    help="Max concurrent workers for scan (default: auto).",
)
BATCH_SIZE_OPT = typer.Option(
    50,
    "--batch-size",
    min=1,
    help="Batch size for scan requests.",
)
BATCH_SLEEP_SECONDS_OPT = typer.Option(
    0.25,
    "--batch-sleep-seconds",
    min=0.0,
    help="Sleep between batches (seconds) to be polite to data sources.",
)
REPORTS_OUT_OPT = typer.Option(
    Path("data/reports/technicals/extension"),
    "--reports-out",
    help="Output root for Extension Percentile Stats reports.",
)
RUN_REPORTS_OPT = typer.Option(
    True,
    "--run-reports/--no-run-reports",
    help="Generate Extension Percentile Stats reports for shortlist symbols.",
)
WRITE_SCAN_OPT = typer.Option(
    True,
    "--write-scan/--no-write-scan",
    help="Write scan CSV under the run directory.",
)
WRITE_LIQUIDITY_OPT = typer.Option(
    True,
    "--write-liquidity/--no-write-liquidity",
    help="Write liquidity CSV under the run directory.",
)
WRITE_SHORTLIST_OPT = typer.Option(
    True,
    "--write-shortlist/--no-write-shortlist",
    help="Write shortlist CSV under the run directory.",
)
STRICT_OPT = typer.Option(
    False,
    "--strict",
    help="Validate JSON artifacts against schemas.",
)
CONFIG_PATH_OPT = typer.Option(
    Path("config/technical_backtesting.yaml"),
    "--config",
    help="Config path.",
)


def _build_runtime_deps() -> SimpleNamespace:
    return SimpleNamespace(
        cli_deps=cli_deps,
        setup_technicals_logging=setup_technicals_logging,
        load_technical_backtesting_config=load_technical_backtesting_config,
        load_confluence_config=load_confluence_config,
        ConfluenceConfigError=ConfluenceConfigError,
        load_universe_symbols=load_universe_symbols,
        UniverseError=UniverseError,
        read_exclude_symbols=read_exclude_symbols,
        read_scanned_symbols=read_scanned_symbols,
        prefilter_symbols=prefilter_symbols,
        scan_symbols=scan_symbols,
        write_exclude_symbols=write_exclude_symbols,
        write_scanned_symbols=write_scanned_symbols,
        write_scan_csv=write_scan_csv,
        load_watchlists=load_watchlists,
        save_watchlists=save_watchlists,
        snapshot_full_chain_for_symbols=snapshot_full_chain_for_symbols,
        evaluate_liquidity_for_symbols=evaluate_liquidity_for_symbols,
        rank_shortlist_candidates=rank_shortlist_candidates,
        score_shortlist_confluence=score_shortlist_confluence,
        write_liquidity_csv=write_liquidity_csv,
        write_shortlist_csv=write_shortlist_csv,
        ScannerShortlistRow=ScannerShortlistRow,
        ScannerShortlistRowSchema=ScannerShortlistRowSchema,
        ScannerShortlistArtifact=ScannerShortlistArtifact,
        utc_now=utc_now,
        run_extension_stats_for_symbol=run_extension_stats_for_symbol,
    )


@app.command("run")
def scanner_run(
    universe: str = UNIVERSE_OPT,
    universe_cache_dir: Path = UNIVERSE_CACHE_DIR_OPT,
    universe_refresh_days: int = UNIVERSE_REFRESH_DAYS_OPT,
    max_symbols: int | None = MAX_SYMBOLS_OPT,
    prefilter_mode: str = PREFILTER_MODE_OPT,
    exclude_path: Path = EXCLUDE_PATH_OPT,
    scanned_path: Path = SCANNED_PATH_OPT,
    skip_scanned: bool = SKIP_SCANNED_OPT,
    write_scanned: bool = WRITE_SCANNED_OPT,
    write_error_excludes: bool = WRITE_ERROR_EXCLUDES_OPT,
    exclude_statuses: str = EXCLUDE_STATUSES_OPT,
    error_flush_every: int = ERROR_FLUSH_EVERY_OPT,
    scanned_flush_every: int = SCANNED_FLUSH_EVERY_OPT,
    scan_period: str = SCAN_PERIOD_OPT,
    tail_pct: float | None = TAIL_PCT_OPT,
    percentile_window_years: int | None = PERCENTILE_WINDOW_YEARS_OPT,
    watchlists_path: Path = WATCHLISTS_PATH_OPT,
    all_watchlist_name: str = ALL_WATCHLIST_NAME_OPT,
    shortlist_watchlist_name: str = SHORTLIST_WATCHLIST_NAME_OPT,
    candle_cache_dir: Path = CANDLE_CACHE_DIR_OPT,
    options_cache_dir: Path = OPTIONS_CACHE_DIR_OPT,
    derived_dir: Path = DERIVED_DIR_OPT,
    spot_period: str = SPOT_PERIOD_OPT,
    risk_free_rate: float = RISK_FREE_RATE_OPT,
    backfill: bool = BACKFILL_OPT,
    snapshot_options: bool = SNAPSHOT_OPTIONS_OPT,
    liquidity_min_dte: int = LIQUIDITY_MIN_DTE_OPT,
    liquidity_min_volume: int = LIQUIDITY_MIN_VOLUME_OPT,
    liquidity_min_oi: int = LIQUIDITY_MIN_OI_OPT,
    run_dir: Path = RUN_DIR_OPT,
    run_id: str | None = RUN_ID_OPT,
    workers: int | None = WORKERS_OPT,
    batch_size: int = BATCH_SIZE_OPT,
    batch_sleep_seconds: float = BATCH_SLEEP_SECONDS_OPT,
    reports_out: Path = REPORTS_OUT_OPT,
    run_reports: bool = RUN_REPORTS_OPT,
    write_scan: bool = WRITE_SCAN_OPT,
    write_liquidity: bool = WRITE_LIQUIDITY_OPT,
    write_shortlist: bool = WRITE_SHORTLIST_OPT,
    strict: bool = STRICT_OPT,
    config_path: Path = CONFIG_PATH_OPT,
) -> None:
    """Scan the market for extension tails and build watchlists (not financial advice)."""
    params = dict(locals())
    run_scanner_command(params=params, deps=_build_runtime_deps())
