from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import ingest_common
from options_helper.commands import ingest_runtime_common
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.tuning import (
    default_provider_profile,
    load_tuning_profile,
    recommend_profile,
    save_tuning_profile,
)
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobParameterError,
    run_ingest_candles_job,
    run_ingest_options_bars_job,
)


app = typer.Typer(help="Ingestion utilities (not financial advice).")

JOB_INGEST_CANDLES = "ingest_candles"
JOB_INGEST_OPTIONS_BARS = "ingest_options_bars"

ASSET_CANDLES_DAILY = "candles_daily"
ASSET_OPTION_CONTRACTS = "option_contracts"
ASSET_OPTION_BARS = "option_bars"

NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)

DEFAULT_TUNE_CONFIG = Path("config/ingest_tuning.json")


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _coerce_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


def _latest_contract_expiry(discovery: object | None) -> date | None:
    if discovery is None:
        return None
    contracts = getattr(discovery, "contracts", None)
    if contracts is None or getattr(contracts, "empty", True):
        return None
    columns = getattr(contracts, "columns", [])
    if "expiry" not in columns:
        return None
    latest: date | None = None
    for raw in contracts["expiry"].tolist():
        parsed = _coerce_date(raw)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest


def _resolve_tuning_profile(*, auto_tune: bool, tune_config: Path, provider: str) -> dict[str, Any]:
    if auto_tune:
        return load_tuning_profile(tune_config, provider=provider)
    return default_provider_profile()


def _float_or_default(
    value: float | None,
    default: float,
) -> float:
    if value is None:
        return float(default)
    return float(value)


def _int_or_default(
    value: int | None,
    default: int,
) -> int:
    if value is None:
        return int(default)
    return int(value)


def _build_candles_provider(
    *,
    log_rate_limits_override: bool | None,
    alpaca_http_pool_maxsize: int | None,
    alpaca_http_pool_connections: int | None,
):
    provider_name = get_default_provider_name()
    if provider_name != "alpaca":
        return cli_deps.build_provider()

    kwargs: dict[str, object] = {}
    if log_rate_limits_override is not None:
        kwargs["log_rate_limits"] = log_rate_limits_override
    if alpaca_http_pool_maxsize is not None:
        kwargs["http_pool_maxsize"] = alpaca_http_pool_maxsize
    if alpaca_http_pool_connections is not None:
        kwargs["http_pool_connections"] = alpaca_http_pool_connections

    if not kwargs:
        return cli_deps.build_provider()

    from options_helper.data.providers.alpaca import AlpacaProvider
    try:
        client = AlpacaClient(**kwargs)
    except TypeError:
        client = AlpacaClient()
    return AlpacaProvider(client=client)


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


def _resolve_log_rate_limits_override(*, log_rate_limits: bool, no_log_rate_limits: bool) -> bool | None:
    if log_rate_limits and no_log_rate_limits:
        raise typer.BadParameter("Choose either --log-rate-limits or --no-log-rate-limits, not both.")
    if log_rate_limits:
        return True
    if no_log_rate_limits:
        return False
    return None


def _resolve_candles_run_config(
    *,
    auto_tune: bool,
    tune_config: Path,
    candles_concurrency: int | None,
    candles_max_requests_per_second: float | None,
) -> tuple[str, dict[str, Any], int, float]:
    provider_name = get_default_provider_name()
    profile = _resolve_tuning_profile(auto_tune=auto_tune, tune_config=tune_config, provider=provider_name)
    candles_profile = dict(profile.get("candles") or {})
    effective_concurrency = _int_or_default(candles_concurrency, int(candles_profile.get("concurrency") or 1))
    effective_max_rps = _float_or_default(
        candles_max_requests_per_second,
        float(candles_profile.get("max_rps") or 8.0),
    )
    return provider_name, profile, effective_concurrency, effective_max_rps


def _build_candles_run_args(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    candle_cache_dir: Path,
    candles_concurrency: int,
    candles_max_requests_per_second: float,
    alpaca_http_pool_maxsize: int | None,
    alpaca_http_pool_connections: int | None,
    log_rate_limits_override: bool | None,
    auto_tune: bool,
    tune_config: Path,
) -> dict[str, Any]:
    return {
        "watchlists_path": str(watchlists_path),
        "watchlist": watchlist,
        "symbol": symbol,
        "candle_cache_dir": str(candle_cache_dir),
        "candles_concurrency": candles_concurrency,
        "candles_max_requests_per_second": candles_max_requests_per_second,
        "alpaca_http_pool_maxsize": alpaca_http_pool_maxsize,
        "alpaca_http_pool_connections": alpaca_http_pool_connections,
        "log_rate_limits": log_rate_limits_override,
        "auto_tune": auto_tune,
        "tune_config": str(tune_config),
    }


def _print_ingest_warnings(console: Console, warnings: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _handle_candles_no_symbols(*, console: Console, run_logger: Any) -> None:
    run_logger.log_asset_skipped(
        asset_key=ASSET_CANDLES_DAILY,
        asset_kind="table",
        partition_key="ALL",
        extra={"reason": "no_symbols"},
    )
    console.print("No symbols found (empty watchlists and no --symbol override).")
    raise typer.Exit(0)


def _record_candles_results(*, console: Console, run_logger: Any, results: list[Any]) -> tuple[int, int, int]:
    ok = 0
    empty = 0
    error = 0
    for item in results:
        if item.status == "ok":
            ok += 1
            run_logger.log_asset_success(
                asset_key=ASSET_CANDLES_DAILY,
                asset_kind="table",
                partition_key=item.symbol,
                min_event_ts=item.last_date,
                max_event_ts=item.last_date,
            )
            if item.last_date is not None:
                run_logger.upsert_watermark(
                    asset_key=ASSET_CANDLES_DAILY,
                    scope_key=item.symbol,
                    watermark_ts=item.last_date,
                )
            console.print(f"{item.symbol}: cached through {item.last_date.isoformat()}")
        elif item.status == "empty":
            empty += 1
            run_logger.log_asset_skipped(
                asset_key=ASSET_CANDLES_DAILY,
                asset_kind="table",
                partition_key=item.symbol,
                extra={"reason": "empty"},
            )
            console.print(f"[yellow]Warning:[/yellow] {item.symbol}: no candles returned.")
        else:
            error += 1
            run_logger.log_asset_failure(
                asset_key=ASSET_CANDLES_DAILY,
                asset_kind="table",
                partition_key=item.symbol,
                extra={"error": item.error},
            )
            console.print(f"[red]Error:[/red] {item.symbol}: {item.error}")
    return ok, empty, error


def _print_candles_endpoint_stats(console: Console, endpoint_stats: Any) -> None:
    if endpoint_stats is None:
        return
    console.print(
        "Endpoint stats (/v2/stocks/bars): "
        f"calls={endpoint_stats.calls}, "
        f"429={endpoint_stats.rate_limit_429}, "
        f"timeouts={endpoint_stats.timeout_count}, "
        f"p50_ms={endpoint_stats.latency_p50_ms}, "
        f"p95_ms={endpoint_stats.latency_p95_ms}"
    )


def _apply_candles_auto_tune(
    *,
    console: Console,
    auto_tune: bool,
    profile: dict[str, Any],
    endpoint_stats: Any,
    tune_config: Path,
    provider_name: str,
) -> None:
    if not auto_tune:
        return
    updated = recommend_profile(profile, candles_stats=endpoint_stats)
    save_tuning_profile(tune_config, provider=provider_name, profile=updated)
    tuned_candles = updated.get("candles") or {}
    console.print(
        "Auto-tune saved: "
        f"candles.max_rps={tuned_candles.get('max_rps')} "
        f"candles.concurrency={tuned_candles.get('concurrency')} "
        f"({tune_config})"
    )


def _run_ingest_candles(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    candle_cache_dir: Path,
    candles_concurrency: int,
    candles_max_requests_per_second: float,
    log_rate_limits_override: bool | None,
    alpaca_http_pool_maxsize: int | None,
    alpaca_http_pool_connections: int | None,
):
    def _provider_builder():
        return _build_candles_provider(
            log_rate_limits_override=log_rate_limits_override,
            alpaca_http_pool_maxsize=alpaca_http_pool_maxsize,
            alpaca_http_pool_connections=alpaca_http_pool_connections,
        )

    return run_ingest_candles_job(
        watchlists_path=watchlists_path,
        watchlist=watchlist,
        symbol=symbol,
        candle_cache_dir=candle_cache_dir,
        candles_concurrency=candles_concurrency,
        candles_max_requests_per_second=candles_max_requests_per_second,
        provider_builder=_provider_builder,
        candle_store_builder=cli_deps.build_candle_store,
    )


@app.command("candles")
def ingest_candles_command(
    watchlists_path: Path = ingest_common._INGEST_CANDLES_WATCHLISTS_PATH_OPT,
    watchlist: list[str] = ingest_common._INGEST_CANDLES_WATCHLIST_OPT,
    symbol: list[str] = ingest_common._INGEST_CANDLES_SYMBOL_OPT,
    candle_cache_dir: Path = ingest_common._INGEST_CANDLES_CACHE_DIR_OPT,
    candles_concurrency: int | None = ingest_common._INGEST_CANDLES_CONCURRENCY_OPT,
    candles_max_requests_per_second: float | None = ingest_common._INGEST_CANDLES_MAX_RPS_OPT,
    alpaca_http_pool_maxsize: int | None = ingest_common._INGEST_CANDLES_ALPACA_POOL_MAXSIZE_OPT,
    alpaca_http_pool_connections: int | None = ingest_common._INGEST_CANDLES_ALPACA_POOL_CONNECTIONS_OPT,
    log_rate_limits: bool = ingest_common._INGEST_CANDLES_LOG_RATE_LIMITS_OPT,
    no_log_rate_limits: bool = ingest_common._INGEST_CANDLES_NO_LOG_RATE_LIMITS_OPT,
    auto_tune: bool = ingest_common._INGEST_CANDLES_AUTO_TUNE_OPT,
    tune_config: Path = ingest_common._INGEST_CANDLES_TUNE_CONFIG_OPT,
) -> None:
    """Backfill daily candles for watchlist symbols (period=max)."""
    console = Console(width=200)
    log_rate_limits_override = _resolve_log_rate_limits_override(
        log_rate_limits=log_rate_limits,
        no_log_rate_limits=no_log_rate_limits,
    )
    provider_name, profile, effective_concurrency, effective_max_rps = _resolve_candles_run_config(
        auto_tune=auto_tune,
        tune_config=tune_config,
        candles_concurrency=candles_concurrency,
        candles_max_requests_per_second=candles_max_requests_per_second,
    )
    with _observed_run(
        console=console,
        job_name=JOB_INGEST_CANDLES,
        args=_build_candles_run_args(
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            candle_cache_dir=candle_cache_dir,
            candles_concurrency=effective_concurrency,
            candles_max_requests_per_second=effective_max_rps,
            alpaca_http_pool_maxsize=alpaca_http_pool_maxsize,
            alpaca_http_pool_connections=alpaca_http_pool_connections,
            log_rate_limits_override=log_rate_limits_override,
            auto_tune=auto_tune,
            tune_config=tune_config,
        ),
    ) as run_logger:
        result = _run_ingest_candles(
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            candle_cache_dir=candle_cache_dir,
            candles_concurrency=effective_concurrency,
            candles_max_requests_per_second=effective_max_rps,
            log_rate_limits_override=log_rate_limits_override,
            alpaca_http_pool_maxsize=alpaca_http_pool_maxsize,
            alpaca_http_pool_connections=alpaca_http_pool_connections,
        )
        _print_ingest_warnings(console, result.warnings)
        if result.no_symbols:
            _handle_candles_no_symbols(console=console, run_logger=run_logger)
        ok, empty, error = _record_candles_results(console=console, run_logger=run_logger, results=result.results)
        console.print(f"Summary: {ok} ok, {empty} empty, {error} error(s) for {len(result.results)} symbol(s).")
        _print_candles_endpoint_stats(console, result.endpoint_stats)
        _apply_candles_auto_tune(
            console=console,
            auto_tune=auto_tune,
            profile=profile,
            endpoint_stats=result.endpoint_stats,
            tune_config=tune_config,
            provider_name=provider_name,
        )


def _validate_options_bars_flags(*, dry_run: bool, fetch_only: bool) -> None:
    if dry_run and fetch_only:
        raise typer.BadParameter("Choose either --dry-run or --fetch-only, not both.")


def _resolve_options_bars_run_config(
    *,
    auto_tune: bool,
    tune_config: Path,
    contracts_page_size: int | None,
    contracts_max_requests_per_second: float | None,
    bars_concurrency: int | None,
    bars_max_requests_per_second: float | None,
    bars_batch_mode: str | None,
    bars_batch_size: int | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    provider_name = get_default_provider_name()
    profile = _resolve_tuning_profile(auto_tune=auto_tune, tune_config=tune_config, provider=provider_name)
    contracts_profile = dict(profile.get("contracts") or {})
    bars_profile = dict(profile.get("bars") or {})
    effective_config = {
        "contracts_page_size": _int_or_default(contracts_page_size, int(contracts_profile.get("page_size") or 10000)),
        "contracts_max_rps": _float_or_default(
            contracts_max_requests_per_second,
            float(contracts_profile.get("max_rps") or 2.5),
        ),
        "bars_concurrency": _int_or_default(bars_concurrency, int(bars_profile.get("concurrency") or 8)),
        "bars_max_rps": _float_or_default(
            bars_max_requests_per_second,
            float(bars_profile.get("max_rps") or 30.0),
        ),
        "bars_batch_size": _int_or_default(bars_batch_size, int(bars_profile.get("batch_size") or 8)),
        "bars_batch_mode": str(bars_batch_mode or bars_profile.get("batch_mode") or "adaptive").strip().lower(),
    }
    if effective_config["bars_batch_mode"] not in {"adaptive", "per-contract"}:
        raise typer.BadParameter("bars-batch-mode must be one of: adaptive, per-contract")
    return provider_name, profile, effective_config


def _build_options_bars_run_args(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    contracts_root_symbol: list[str],
    contracts_symbol_prefix: str | None,
    contracts_exp_start: str,
    contracts_exp_end: str | None,
    contracts_status: str,
    lookback_years: int,
    page_limit: int,
    max_underlyings: int | None,
    max_contracts: int | None,
    max_expiries: int | None,
    bars_write_batch_size: int,
    log_rate_limits_override: bool | None,
    resume: bool,
    contracts_only: bool,
    dry_run: bool,
    fetch_only: bool,
    fail_fast: bool,
    auto_tune: bool,
    tune_config: Path,
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "watchlists_path": str(watchlists_path),
        "watchlist": watchlist,
        "symbol": symbol,
        "contracts_root_symbol": contracts_root_symbol,
        "contracts_symbol_prefix": contracts_symbol_prefix,
        "contracts_exp_start": contracts_exp_start,
        "contracts_exp_end": contracts_exp_end,
        "contracts_status": contracts_status,
        "lookback_years": lookback_years,
        "page_limit": page_limit,
        "contracts_page_size": effective_config["contracts_page_size"],
        "max_underlyings": max_underlyings,
        "max_contracts": max_contracts,
        "max_expiries": max_expiries,
        "contracts_max_requests_per_second": effective_config["contracts_max_rps"],
        "bars_concurrency": effective_config["bars_concurrency"],
        "bars_max_requests_per_second": effective_config["bars_max_rps"],
        "bars_batch_mode": effective_config["bars_batch_mode"],
        "bars_batch_size": effective_config["bars_batch_size"],
        "bars_write_batch_size": bars_write_batch_size,
        "alpaca_http_pool_maxsize": None,
        "alpaca_http_pool_connections": None,
        "log_rate_limits": log_rate_limits_override,
        "resume": resume,
        "contracts_only": contracts_only,
        "dry_run": dry_run,
        "fetch_only": fetch_only,
        "fail_fast": fail_fast,
        "auto_tune": auto_tune,
        "tune_config": str(tune_config),
    }


def _build_options_bars_job_kwargs(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    contracts_root_symbol: list[str],
    contracts_symbol_prefix: str | None,
    contracts_exp_start: str,
    contracts_exp_end: str | None,
    contracts_status: str,
    lookback_years: int,
    page_limit: int,
    max_underlyings: int | None,
    max_contracts: int | None,
    max_expiries: int | None,
    bars_write_batch_size: int,
    resume: bool,
    dry_run: bool,
    fail_fast: bool,
    contracts_only: bool,
    fetch_only: bool,
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "watchlists_path": watchlists_path,
        "watchlist": watchlist,
        "symbol": symbol,
        "contracts_root_symbols": contracts_root_symbol,
        "contract_symbol_prefix": contracts_symbol_prefix,
        "contracts_exp_start": contracts_exp_start,
        "contracts_exp_end": contracts_exp_end,
        "contracts_status": contracts_status,
        "lookback_years": lookback_years,
        "page_limit": page_limit,
        "contracts_page_size": effective_config["contracts_page_size"],
        "max_underlyings": max_underlyings,
        "max_contracts": max_contracts,
        "max_expiries": max_expiries,
        "contracts_max_requests_per_second": effective_config["contracts_max_rps"],
        "bars_concurrency": effective_config["bars_concurrency"],
        "bars_max_requests_per_second": effective_config["bars_max_rps"],
        "bars_batch_mode": effective_config["bars_batch_mode"],
        "bars_batch_size": effective_config["bars_batch_size"],
        "bars_write_batch_size": bars_write_batch_size,
        "resume": resume,
        "dry_run": dry_run,
        "fail_fast": fail_fast,
        "contracts_only": contracts_only,
        "fetch_only": fetch_only,
    }


def _build_options_client(
    *,
    log_rate_limits_override: bool | None,
    alpaca_http_pool_maxsize: int | None,
    alpaca_http_pool_connections: int | None,
) -> AlpacaClient:
    kwargs: dict[str, object] = {}
    if log_rate_limits_override is not None:
        kwargs["log_rate_limits"] = log_rate_limits_override
    if alpaca_http_pool_maxsize is not None:
        kwargs["http_pool_maxsize"] = alpaca_http_pool_maxsize
    if alpaca_http_pool_connections is not None:
        kwargs["http_pool_connections"] = alpaca_http_pool_connections
    if not kwargs:
        return AlpacaClient()
    try:
        return AlpacaClient(**kwargs)
    except TypeError:
        # Backward-compatible fallback for tests that monkeypatch AlpacaClient
        # with callables that do not accept keyword args.
        return AlpacaClient()


def _execute_ingest_options_bars(
    *,
    console: Console,
    run_day: date,
    log_rate_limits_override: bool | None,
    alpaca_http_pool_maxsize: int | None,
    alpaca_http_pool_connections: int | None,
    job_kwargs: dict[str, Any],
):
    def _build_client() -> AlpacaClient:
        return _build_options_client(
            log_rate_limits_override=log_rate_limits_override,
            alpaca_http_pool_maxsize=alpaca_http_pool_maxsize,
            alpaca_http_pool_connections=alpaca_http_pool_connections,
        )

    try:
        return run_ingest_options_bars_job(
            **job_kwargs,
            provider_builder=cli_deps.build_provider,
            contracts_store_builder=cli_deps.build_option_contracts_store,
            bars_store_builder=cli_deps.build_option_bars_store,
            client_factory=_build_client,
            contracts_store_dir=Path("data/option_contracts"),
            bars_store_dir=Path("data/option_bars"),
            today=run_day,
        )
    except VisibilityJobParameterError as exc:
        if exc.param_hint:
            raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
        raise typer.BadParameter(str(exc)) from exc
    except (OptionContractsStoreError, OptionBarsStoreError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _upsert_contracts_watermark(
    *,
    run_logger: Any,
    discovery: Any,
    dry_run: bool,
    fetch_only: bool,
) -> None:
    latest_expiry = _latest_contract_expiry(discovery)
    if latest_expiry is None or dry_run or fetch_only:
        return
    run_logger.upsert_watermark(
        asset_key=ASSET_OPTION_CONTRACTS,
        scope_key="ALL",
        watermark_ts=latest_expiry,
    )


def _apply_options_auto_tune(
    *,
    console: Console,
    auto_tune: bool,
    profile: dict[str, Any],
    result: Any,
    tune_config: Path,
    provider_name: str,
) -> None:
    if not auto_tune:
        return
    updated = recommend_profile(
        profile,
        contracts_stats=(
            result.contracts_endpoint_stats.endpoint_stats
            if result.contracts_endpoint_stats is not None
            else None
        ),
        bars_stats=(
            result.bars_endpoint_stats.endpoint_stats
            if result.bars_endpoint_stats is not None
            else None
        ),
    )
    save_tuning_profile(tune_config, provider=provider_name, profile=updated)
    tuned_contracts = updated.get("contracts") or {}
    tuned_bars = updated.get("bars") or {}
    console.print(
        "Auto-tune saved: "
        f"contracts.max_rps={tuned_contracts.get('max_rps')} "
        f"contracts.page_size={tuned_contracts.get('page_size')} "
        f"bars.max_rps={tuned_bars.get('max_rps')} "
        f"bars.concurrency={tuned_bars.get('concurrency')} "
        f"bars.batch_size={tuned_bars.get('batch_size')} "
        f"({tune_config})"
    )


def _build_options_bars_payloads(
    *,
    params: dict[str, Any],
    effective_config: dict[str, Any],
    log_rate_limits_override: bool | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_args = _build_options_bars_run_args(
        watchlists_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        symbol=params["symbol"],
        contracts_root_symbol=params["contracts_root_symbol"],
        contracts_symbol_prefix=params["contracts_symbol_prefix"],
        contracts_exp_start=params["contracts_exp_start"],
        contracts_exp_end=params["contracts_exp_end"],
        contracts_status=params["contracts_status"],
        lookback_years=params["lookback_years"],
        page_limit=params["page_limit"],
        max_underlyings=params["max_underlyings"],
        max_contracts=params["max_contracts"],
        max_expiries=params["max_expiries"],
        bars_write_batch_size=params["bars_write_batch_size"],
        log_rate_limits_override=log_rate_limits_override,
        resume=params["resume"],
        contracts_only=params["contracts_only"],
        dry_run=params["dry_run"],
        fetch_only=params["fetch_only"],
        fail_fast=params["fail_fast"],
        auto_tune=params["auto_tune"],
        tune_config=params["tune_config"],
        effective_config=effective_config,
    )
    run_args["alpaca_http_pool_maxsize"] = params["alpaca_http_pool_maxsize"]
    run_args["alpaca_http_pool_connections"] = params["alpaca_http_pool_connections"]
    job_kwargs = _build_options_bars_job_kwargs(
        watchlists_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        symbol=params["symbol"],
        contracts_root_symbol=params["contracts_root_symbol"],
        contracts_symbol_prefix=params["contracts_symbol_prefix"],
        contracts_exp_start=params["contracts_exp_start"],
        contracts_exp_end=params["contracts_exp_end"],
        contracts_status=params["contracts_status"],
        lookback_years=params["lookback_years"],
        page_limit=params["page_limit"],
        max_underlyings=params["max_underlyings"],
        max_contracts=params["max_contracts"],
        max_expiries=params["max_expiries"],
        bars_write_batch_size=params["bars_write_batch_size"],
        resume=params["resume"],
        dry_run=params["dry_run"],
        fail_fast=params["fail_fast"],
        contracts_only=params["contracts_only"],
        fetch_only=params["fetch_only"],
        effective_config=effective_config,
    )
    return run_args, job_kwargs


def _prepare_options_bars_context(params: dict[str, Any]) -> dict[str, Any]:
    console = Console(width=200)
    run_day = date.today()
    log_rate_limits_override = _resolve_log_rate_limits_override(
        log_rate_limits=params["log_rate_limits"],
        no_log_rate_limits=params["no_log_rate_limits"],
    )
    _validate_options_bars_flags(dry_run=params["dry_run"], fetch_only=params["fetch_only"])
    provider_name, profile, effective_config = _resolve_options_bars_run_config(
        auto_tune=params["auto_tune"],
        tune_config=params["tune_config"],
        contracts_page_size=params["contracts_page_size"],
        contracts_max_requests_per_second=params["contracts_max_requests_per_second"],
        bars_concurrency=params["bars_concurrency"],
        bars_max_requests_per_second=params["bars_max_requests_per_second"],
        bars_batch_mode=params["bars_batch_mode"],
        bars_batch_size=params["bars_batch_size"],
    )
    run_args, job_kwargs = _build_options_bars_payloads(
        params=params,
        effective_config=effective_config,
        log_rate_limits_override=log_rate_limits_override,
    )
    return {
        "console": console,
        "run_day": run_day,
        "log_rate_limits_override": log_rate_limits_override,
        "provider_name": provider_name,
        "profile": profile,
        "dry_run": params["dry_run"],
        "fetch_only": params["fetch_only"],
        "auto_tune": params["auto_tune"],
        "tune_config": params["tune_config"],
        "alpaca_http_pool_maxsize": params["alpaca_http_pool_maxsize"],
        "alpaca_http_pool_connections": params["alpaca_http_pool_connections"],
        "run_args": run_args,
        "job_kwargs": job_kwargs,
    }


def _run_options_bars_context(context: dict[str, Any]) -> None:
    with _observed_run(
        console=context["console"],
        job_name=JOB_INGEST_OPTIONS_BARS,
        args=context["run_args"],
    ) as run_logger:
        result = _execute_ingest_options_bars(
            console=context["console"],
            run_day=context["run_day"],
            log_rate_limits_override=context["log_rate_limits_override"],
            alpaca_http_pool_maxsize=context["alpaca_http_pool_maxsize"],
            alpaca_http_pool_connections=context["alpaca_http_pool_connections"],
            job_kwargs=context["job_kwargs"],
        )
        _print_ingest_warnings(context["console"], result.warnings)
        if result.no_symbols:
            ingest_runtime_common.handle_options_no_symbols(console=context["console"], run_logger=run_logger)
        ingest_runtime_common.print_underlying_limit_warning(
            console=context["console"],
            limited_underlyings=result.limited_underlyings,
            underlyings=result.underlyings,
        )
        discovery = result.discovery
        assert discovery is not None
        ingest_runtime_common.record_contract_discovery(
            console=context["console"],
            run_logger=run_logger,
            discovery=discovery,
            fetch_only=context["fetch_only"],
        )
        _upsert_contracts_watermark(
            run_logger=run_logger,
            discovery=discovery,
            dry_run=context["dry_run"],
            fetch_only=context["fetch_only"],
        )
        if result.no_contracts:
            ingest_runtime_common.handle_options_no_contracts(console=context["console"], run_logger=run_logger)
        if not ingest_runtime_common.handle_options_contracts_only(
            console=context["console"],
            run_logger=run_logger,
            contracts_only=result.contracts_only,
            discovery=discovery,
        ):
            ingest_runtime_common.print_options_fetch_mode_notice(
                console=context["console"],
                dry_run=context["dry_run"],
                fetch_only=context["fetch_only"],
                discovered_contracts=len(discovery.contracts),
            )
            if result.no_eligible_contracts:
                ingest_runtime_common.handle_options_no_eligible_contracts(
                    console=context["console"],
                    run_logger=run_logger,
                )
            summary = result.summary
            assert summary is not None
            ingest_runtime_common.record_options_bars_summary(
                console=context["console"],
                run_logger=run_logger,
                summary=summary,
                dry_run=context["dry_run"],
                fetch_only=context["fetch_only"],
                run_day=context["run_day"],
            )
        ingest_runtime_common.print_options_endpoint_stats(console=context["console"], result=result)
        _apply_options_auto_tune(
            console=context["console"],
            auto_tune=context["auto_tune"],
            profile=context["profile"],
            result=result,
            tune_config=context["tune_config"],
            provider_name=context["provider_name"],
        )


@app.command("options-bars")
def ingest_options_bars_command(
    watchlists_path: Path = ingest_common._INGEST_OPTIONS_BARS_WATCHLISTS_PATH_OPT,
    watchlist: list[str] = ingest_common._INGEST_OPTIONS_BARS_WATCHLIST_OPT,
    symbol: list[str] = ingest_common._INGEST_OPTIONS_BARS_SYMBOL_OPT,
    contracts_root_symbol: list[str] = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_ROOT_SYMBOL_OPT,
    contracts_symbol_prefix: str | None = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_SYMBOL_PREFIX_OPT,
    contracts_exp_start: str = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_EXP_START_OPT,
    contracts_exp_end: str | None = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_EXP_END_OPT,
    contracts_status: str = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_STATUS_OPT,
    lookback_years: int = ingest_common._INGEST_OPTIONS_BARS_LOOKBACK_YEARS_OPT,
    page_limit: int = ingest_common._INGEST_OPTIONS_BARS_PAGE_LIMIT_OPT,
    contracts_page_size: int | None = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_PAGE_SIZE_OPT,
    max_underlyings: int | None = ingest_common._INGEST_OPTIONS_BARS_MAX_UNDERLYINGS_OPT,
    max_contracts: int | None = ingest_common._INGEST_OPTIONS_BARS_MAX_CONTRACTS_OPT,
    max_expiries: int | None = ingest_common._INGEST_OPTIONS_BARS_MAX_EXPIRIES_OPT,
    contracts_max_requests_per_second: float | None = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_MAX_RPS_OPT,
    bars_concurrency: int | None = ingest_common._INGEST_OPTIONS_BARS_BARS_CONCURRENCY_OPT,
    bars_max_requests_per_second: float | None = ingest_common._INGEST_OPTIONS_BARS_BARS_MAX_RPS_OPT,
    bars_batch_mode: str | None = ingest_common._INGEST_OPTIONS_BARS_BARS_BATCH_MODE_OPT,
    bars_batch_size: int | None = ingest_common._INGEST_OPTIONS_BARS_BARS_BATCH_SIZE_OPT,
    bars_write_batch_size: int = ingest_common._INGEST_OPTIONS_BARS_BARS_WRITE_BATCH_SIZE_OPT,
    alpaca_http_pool_maxsize: int | None = ingest_common._INGEST_OPTIONS_BARS_ALPACA_POOL_MAXSIZE_OPT,
    alpaca_http_pool_connections: int | None = ingest_common._INGEST_OPTIONS_BARS_ALPACA_POOL_CONNECTIONS_OPT,
    log_rate_limits: bool = ingest_common._INGEST_OPTIONS_BARS_LOG_RATE_LIMITS_OPT,
    no_log_rate_limits: bool = ingest_common._INGEST_OPTIONS_BARS_NO_LOG_RATE_LIMITS_OPT,
    resume: bool = ingest_common._INGEST_OPTIONS_BARS_RESUME_OPT,
    contracts_only: bool = ingest_common._INGEST_OPTIONS_BARS_CONTRACTS_ONLY_OPT,
    dry_run: bool = ingest_common._INGEST_OPTIONS_BARS_DRY_RUN_OPT,
    fetch_only: bool = ingest_common._INGEST_OPTIONS_BARS_FETCH_ONLY_OPT,
    fail_fast: bool = ingest_common._INGEST_OPTIONS_BARS_FAIL_FAST_OPT,
    auto_tune: bool = ingest_common._INGEST_OPTIONS_BARS_AUTO_TUNE_OPT,
    tune_config: Path = ingest_common._INGEST_OPTIONS_BARS_TUNE_CONFIG_OPT,
) -> None:
    """Discover Alpaca option contracts and backfill daily bars."""
    context = _prepare_options_bars_context(locals())
    _run_options_bars_context(context)
