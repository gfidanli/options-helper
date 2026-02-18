from __future__ import annotations

from pathlib import Path

import typer

from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS


DEFAULT_TUNE_CONFIG = Path("config/ingest_tuning.json")

_INGEST_CANDLES_WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store.",
)
_INGEST_CANDLES_WATCHLIST_OPT = typer.Option(
    list(DEFAULT_WATCHLISTS),
    "--watchlist",
    help="Watchlist name(s) to ingest (default: positions + monitor).",
)
_INGEST_CANDLES_SYMBOL_OPT = typer.Option(
    [],
    "--symbol",
    "-s",
    help="Optional symbol override (repeatable or comma-separated).",
)
_INGEST_CANDLES_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--candle-cache-dir",
    help="Directory for cached daily candles.",
)
_INGEST_CANDLES_CONCURRENCY_OPT = typer.Option(
    None,
    "--candles-concurrency",
    min=1,
    help="Concurrent stock-bars fetch workers for candle ingestion.",
)
_INGEST_CANDLES_MAX_RPS_OPT = typer.Option(
    None,
    "--candles-max-rps",
    min=0.1,
    help="Soft throttle for stock-bars requests per second.",
)
_INGEST_CANDLES_ALPACA_POOL_MAXSIZE_OPT = typer.Option(
    None,
    "--alpaca-http-pool-maxsize",
    min=1,
    help="Requests connection pool max size for Alpaca clients used by this run.",
)
_INGEST_CANDLES_ALPACA_POOL_CONNECTIONS_OPT = typer.Option(
    None,
    "--alpaca-http-pool-connections",
    min=1,
    help="Requests connection pool count for Alpaca clients used by this run.",
)
_INGEST_CANDLES_LOG_RATE_LIMITS_OPT = typer.Option(
    False,
    "--log-rate-limits",
    help="Enable per-request Alpaca rate-limit logging for this run.",
)
_INGEST_CANDLES_NO_LOG_RATE_LIMITS_OPT = typer.Option(
    False,
    "--no-log-rate-limits",
    help="Disable per-request Alpaca rate-limit logging for this run.",
)
_INGEST_CANDLES_AUTO_TUNE_OPT = typer.Option(
    False,
    "--auto-tune/--no-auto-tune",
    help="Auto-adjust ingestion tuning profile from observed endpoint stats.",
)
_INGEST_CANDLES_TUNE_CONFIG_OPT = typer.Option(
    DEFAULT_TUNE_CONFIG,
    "--tune-config",
    help="Path to local ingestion tuning profile JSON (local state).",
)

_INGEST_OPTIONS_BARS_WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store.",
)
_INGEST_OPTIONS_BARS_WATCHLIST_OPT = typer.Option(
    list(DEFAULT_WATCHLISTS),
    "--watchlist",
    help="Watchlist name(s) to ingest (default: positions + monitor).",
)
_INGEST_OPTIONS_BARS_SYMBOL_OPT = typer.Option(
    [],
    "--symbol",
    "-s",
    help="Optional symbol override (repeatable or comma-separated).",
)
_INGEST_OPTIONS_BARS_CONTRACTS_ROOT_SYMBOL_OPT = typer.Option(
    [],
    "--contracts-root-symbol",
    help=(
        "Optional option-contract root symbol filter (repeatable). "
        "Useful for index/weekly roots like SPXW."
    ),
)
_INGEST_OPTIONS_BARS_CONTRACTS_SYMBOL_PREFIX_OPT = typer.Option(
    None,
    "--contracts-symbol-prefix",
    help="Optional contractSymbol prefix filter applied after discovery (e.g. SPXW).",
)
_INGEST_OPTIONS_BARS_CONTRACTS_EXP_START_OPT = typer.Option(
    "2000-01-01",
    "--contracts-exp-start",
    help="Contracts expiration start date (YYYY-MM-DD).",
)
_INGEST_OPTIONS_BARS_CONTRACTS_EXP_END_OPT = typer.Option(
    None,
    "--contracts-exp-end",
    help="Contracts expiration end date (YYYY-MM-DD). Defaults to today + 5y.",
)
_INGEST_OPTIONS_BARS_CONTRACTS_STATUS_OPT = typer.Option(
    "all",
    "--contracts-status",
    help="Contracts discovery status filter: active, inactive, or all (default: all).",
)
_INGEST_OPTIONS_BARS_LOOKBACK_YEARS_OPT = typer.Option(
    10,
    "--lookback-years",
    min=1,
    help="Years of daily bars to backfill per expiry.",
)
_INGEST_OPTIONS_BARS_PAGE_LIMIT_OPT = typer.Option(
    200,
    "--page-limit",
    min=1,
    help="Max contract pages to request from Alpaca during contracts discovery.",
)
_INGEST_OPTIONS_BARS_CONTRACTS_PAGE_SIZE_OPT = typer.Option(
    None,
    "--contracts-page-size",
    min=1,
    help="Contracts page size (default 10000).",
)
_INGEST_OPTIONS_BARS_MAX_UNDERLYINGS_OPT = typer.Option(
    None,
    "--max-underlyings",
    min=1,
    help="Safety cap on number of underlyings to ingest.",
)
_INGEST_OPTIONS_BARS_MAX_CONTRACTS_OPT = typer.Option(
    None,
    "--max-contracts",
    min=1,
    help="Safety cap on total contracts to ingest.",
)
_INGEST_OPTIONS_BARS_MAX_EXPIRIES_OPT = typer.Option(
    None,
    "--max-expiries",
    min=1,
    help="Safety cap on expiries (most-recent first).",
)
_INGEST_OPTIONS_BARS_CONTRACTS_MAX_RPS_OPT = typer.Option(
    None,
    "--contracts-max-rps",
    min=0.1,
    help="Soft throttle for options-contracts requests per second.",
)
_INGEST_OPTIONS_BARS_BARS_CONCURRENCY_OPT = typer.Option(
    None,
    "--bars-concurrency",
    min=1,
    help="Concurrent option-contract bars fetches (reduced to 1 when --fail-fast).",
)
_INGEST_OPTIONS_BARS_BARS_MAX_RPS_OPT = typer.Option(
    None,
    "--bars-max-rps",
    min=0.1,
    help="Soft throttle for options-bars requests per second.",
)
_INGEST_OPTIONS_BARS_BARS_BATCH_MODE_OPT = typer.Option(
    None,
    "--bars-batch-mode",
    help="Bars fetch mode: adaptive or per-contract.",
)
_INGEST_OPTIONS_BARS_BARS_BATCH_SIZE_OPT = typer.Option(
    None,
    "--bars-batch-size",
    min=1,
    help="Initial batch size used when --bars-batch-mode adaptive.",
)
_INGEST_OPTIONS_BARS_BARS_WRITE_BATCH_SIZE_OPT = typer.Option(
    200,
    "--bars-write-batch-size",
    min=1,
    help="Contracts per DB/meta write batch for options-bars ingestion.",
)
_INGEST_OPTIONS_BARS_ALPACA_POOL_MAXSIZE_OPT = typer.Option(
    None,
    "--alpaca-http-pool-maxsize",
    min=1,
    help="Requests connection pool max size for Alpaca clients used by this run.",
)
_INGEST_OPTIONS_BARS_ALPACA_POOL_CONNECTIONS_OPT = typer.Option(
    None,
    "--alpaca-http-pool-connections",
    min=1,
    help="Requests connection pool count for Alpaca clients used by this run.",
)
_INGEST_OPTIONS_BARS_LOG_RATE_LIMITS_OPT = typer.Option(
    False,
    "--log-rate-limits",
    help="Enable per-request Alpaca rate-limit logging for this run.",
)
_INGEST_OPTIONS_BARS_NO_LOG_RATE_LIMITS_OPT = typer.Option(
    False,
    "--no-log-rate-limits",
    help="Disable per-request Alpaca rate-limit logging for this run.",
)
_INGEST_OPTIONS_BARS_RESUME_OPT = typer.Option(
    True,
    "--resume/--no-resume",
    help="Skip contracts already covered in option_bars_meta (ignored with --fetch-only).",
)
_INGEST_OPTIONS_BARS_CONTRACTS_ONLY_OPT = typer.Option(
    False,
    "--contracts-only",
    help="Discover + persist option contracts/OI snapshots, skip option-bars backfill.",
)
_INGEST_OPTIONS_BARS_DRY_RUN_OPT = typer.Option(
    False,
    "--dry-run",
    help="Do not write data; only print planned fetch ranges.",
)
_INGEST_OPTIONS_BARS_FETCH_ONLY_OPT = typer.Option(
    False,
    "--fetch-only",
    help="Fetch contracts/bars but skip all warehouse writes (benchmark mode).",
)
_INGEST_OPTIONS_BARS_FAIL_FAST_OPT = typer.Option(
    False,
    "--fail-fast/--best-effort",
    help="Stop on first error (default: best-effort).",
)
_INGEST_OPTIONS_BARS_AUTO_TUNE_OPT = typer.Option(
    False,
    "--auto-tune/--no-auto-tune",
    help="Auto-adjust ingestion tuning profile from observed endpoint stats.",
)
_INGEST_OPTIONS_BARS_TUNE_CONFIG_OPT = typer.Option(
    DEFAULT_TUNE_CONFIG,
    "--tune-config",
    help="Path to local ingestion tuning profile JSON (local state).",
)
