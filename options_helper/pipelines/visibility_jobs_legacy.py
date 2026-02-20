from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from rich.console import RenderableType

import options_helper.cli_deps as cli_deps
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.candles import CandleStore
from options_helper.data.derived import DerivedStore
from options_helper.data.earnings import safe_next_earnings_date
from options_helper.data.ingestion.candles import CandleIngestResult
from options_helper.data.ingestion.options_bars import (
    BarsEndpointStats,
    BarsBackfillSummary,
    ContractDiscoveryOutput,
    ContractDiscoveryStats,
    PreparedContracts,
    backfill_option_bars,
    discover_option_contracts,
    prepare_contracts_for_bars,
)
from options_helper.data.ingestion.tuning import EndpointStats
from options_helper.data.quality_checks import (
    persist_quality_checks,
    run_candle_quality_checks,
    run_derived_quality_checks,
    run_flow_quality_checks,
    run_options_bars_quality_checks,
    run_snapshot_quality_checks,
)
from options_helper.pipelines.visibility_jobs_derived_dashboard_legacy import (
    render_dashboard_report_impl,
    run_dashboard_job_impl,
    run_derived_update_job_impl,
)
from options_helper.pipelines.visibility_jobs_briefing_legacy import run_briefing_job_impl
from options_helper.pipelines.visibility_jobs_flow_legacy import run_flow_report_job_impl
from options_helper.pipelines.visibility_jobs_ingest_candles_legacy import run_ingest_candles_job_impl
from options_helper.pipelines.visibility_jobs_ingest_options_bars_legacy import (
    run_ingest_options_bars_job_impl,
)
from options_helper.pipelines.visibility_jobs_snapshot_legacy import run_snapshot_options_job_impl
from options_helper.storage import load_portfolio
from options_helper.ui.dashboard import render_dashboard
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    pass


class VisibilityJobParameterError(ValueError):
    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


class VisibilityJobExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestCandlesJobResult:
    warnings: list[str]
    symbols: list[str]
    results: list[CandleIngestResult]
    no_symbols: bool
    endpoint_stats: EndpointStats | None = None


@dataclass(frozen=True)
class IngestOptionsBarsJobResult:
    warnings: list[str]
    underlyings: list[str]
    root_symbols: list[str]
    limited_underlyings: bool
    discovery: ContractDiscoveryOutput | None
    prepared: PreparedContracts | None
    summary: BarsBackfillSummary | None
    dry_run: bool
    contracts_only: bool
    no_symbols: bool
    no_contracts: bool
    no_eligible_contracts: bool
    contracts_endpoint_stats: ContractDiscoveryStats | None = None
    bars_endpoint_stats: BarsEndpointStats | None = None


@dataclass(frozen=True)
class SnapshotOptionsJobResult:
    messages: list[str]
    dates_used: list[date]
    symbols: list[str]
    no_symbols: bool


@dataclass(frozen=True)
class FlowReportJobResult:
    renderables: list[RenderableType]
    no_symbols: bool


@dataclass(frozen=True)
class DerivedUpdateJobResult:
    symbol: str
    as_of_date: date
    output_path: Path


@dataclass(frozen=True)
class BriefingJobResult:
    report_date: str
    markdown: str
    markdown_path: Path
    json_path: Path | None
    renderables: list[RenderableType]


@dataclass(frozen=True)
class DashboardJobResult:
    json_path: Path
    artifact: Any


class _FetchOnlyOptionBarsStore:
    """No-op bars store for benchmark/fetch-only runs."""

    def upsert_bars(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        return

    def mark_meta_success(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        return

    def mark_meta_error(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        return

    def coverage(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:  # noqa: ANN401
        return None

    def coverage_bulk(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:  # noqa: ANN401
        return {}


def _active_snapshot_store(store: Any) -> Any:
    # Snapshot/report flows should use the active storage backend end-to-end.
    return store


def _filesystem_compatible_derived_store(derived_dir: Path, store: Any) -> Any:
    if store.__class__.__name__.startswith("DuckDB"):
        return DerivedStore(derived_dir)
    return store


def _filesystem_compatible_candle_store(candle_cache_dir: Path, store: Any) -> Any:
    if store.__class__.__name__.startswith("DuckDB"):
        return CandleStore(candle_cache_dir)
    return store


def _resolve_quality_run_logger(run_logger: Any | None = None) -> Any | None:
    """
    Resolve the active run logger for quality checks.

    Producer jobs can be invoked directly (without a logger) or via CLI command wrappers
    that keep a local `run_logger` variable. Accept explicit injection first, then
    best-effort stack lookup to avoid widening command call signatures.
    """
    if run_logger is not None:
        return run_logger

    import inspect

    frame = inspect.currentframe()
    try:
        current = frame.f_back if frame is not None else None
        while current is not None:
            candidate = current.f_locals.get("run_logger")
            if candidate is not None and hasattr(candidate, "log_check"):
                return candidate
            current = current.f_back
    finally:
        del frame
    return None


def _persist_quality_results(run_logger: Any | None, checks: list[Any]) -> None:
    if run_logger is None:
        return
    persist_quality_checks(run_logger=run_logger, checks=checks)


def run_ingest_candles_job(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    candle_cache_dir: Path,
    candles_concurrency: int = 1,
    candles_max_requests_per_second: float | None = None,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
    run_logger: Any | None = None,
) -> IngestCandlesJobResult:
    return cast(
        IngestCandlesJobResult,
        run_ingest_candles_job_impl(
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            candle_cache_dir=candle_cache_dir,
            candles_concurrency=candles_concurrency,
            candles_max_requests_per_second=candles_max_requests_per_second,
            provider_builder=provider_builder,
            candle_store_builder=candle_store_builder,
            run_logger=run_logger,
            resolve_quality_run_logger_fn=_resolve_quality_run_logger,
            persist_quality_results_fn=_persist_quality_results,
            run_candle_quality_checks_fn=run_candle_quality_checks,
            result_factory=IngestCandlesJobResult,
        ),
    )


def run_ingest_options_bars_job(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    contracts_root_symbols: list[str] | None,
    contract_symbol_prefix: str | None,
    contracts_exp_start: str,
    contracts_exp_end: str | None,
    lookback_years: int,
    page_limit: int,
    contracts_page_size: int = 10000,
    max_underlyings: int | None,
    max_contracts: int | None,
    max_expiries: int | None,
    contracts_max_requests_per_second: float | None,
    bars_concurrency: int,
    bars_max_requests_per_second: float | None,
    bars_batch_mode: str = "adaptive",
    bars_batch_size: int = 8,
    bars_write_batch_size: int,
    resume: bool,
    dry_run: bool,
    fail_fast: bool,
    contracts_status: str = "all",
    contracts_only: bool = False,
    fetch_only: bool = False,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    contracts_store_builder: Callable[[Path], Any] = cli_deps.build_option_contracts_store,
    bars_store_builder: Callable[[Path], Any] = cli_deps.build_option_bars_store,
    client_factory: Callable[[], AlpacaClient] = AlpacaClient,
    contracts_store_dir: Path = Path("data/option_contracts"),
    bars_store_dir: Path = Path("data/option_bars"),
    today: date | None = None,
    run_logger: Any | None = None,
) -> IngestOptionsBarsJobResult:
    impl_kwargs = {
        **locals(),
        "resolve_quality_run_logger_fn": _resolve_quality_run_logger,
        "persist_quality_results_fn": _persist_quality_results,
        "run_options_bars_quality_checks_fn": run_options_bars_quality_checks,
        "parameter_error_factory": VisibilityJobParameterError,
        "result_factory": IngestOptionsBarsJobResult,
        "fetch_only_store_factory": _FetchOnlyOptionBarsStore,
        "discover_option_contracts_fn": discover_option_contracts,
        "prepare_contracts_for_bars_fn": prepare_contracts_for_bars,
        "backfill_option_bars_fn": backfill_option_bars,
    }
    return cast(IngestOptionsBarsJobResult, run_ingest_options_bars_job_impl(**impl_kwargs))


def run_snapshot_options_job(
    *,
    portfolio_path: Path,
    cache_dir: Path,
    candle_cache_dir: Path,
    window_pct: float,
    spot_period: str,
    require_data_date: str | None,
    require_data_tz: str,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    all_expiries: bool,
    full_chain: bool,
    max_expiries: int | None,
    risk_free_rate: float,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
    portfolio_loader: Callable[[Path], Any] = load_portfolio,
    watchlists_loader: Callable[[Path], Any] = load_watchlists,
    run_logger: Any | None = None,
) -> SnapshotOptionsJobResult:
    return cast(
        SnapshotOptionsJobResult,
        run_snapshot_options_job_impl(
            portfolio_path=portfolio_path,
            cache_dir=cache_dir,
            candle_cache_dir=candle_cache_dir,
            window_pct=window_pct,
            spot_period=spot_period,
            require_data_date=require_data_date,
            require_data_tz=require_data_tz,
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            all_watchlists=all_watchlists,
            all_expiries=all_expiries,
            full_chain=full_chain,
            max_expiries=max_expiries,
            risk_free_rate=risk_free_rate,
            provider_builder=provider_builder,
            snapshot_store_builder=snapshot_store_builder,
            candle_store_builder=candle_store_builder,
            portfolio_loader=portfolio_loader,
            watchlists_loader=watchlists_loader,
            run_logger=run_logger,
            resolve_quality_run_logger_fn=_resolve_quality_run_logger,
            persist_quality_results_fn=_persist_quality_results,
            run_snapshot_quality_checks_fn=run_snapshot_quality_checks,
            active_snapshot_store_fn=_active_snapshot_store,
            parameter_error_factory=VisibilityJobParameterError,
            result_factory=SnapshotOptionsJobResult,
        ),
    )


def run_flow_report_job(
    *,
    portfolio_path: Path,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    cache_dir: Path,
    window: int,
    group_by: str,
    top: int,
    out: Path | None,
    strict: bool,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    flow_store_builder: Callable[[Path], Any] = cli_deps.build_flow_store,
    portfolio_loader: Callable[[Path], Any] = load_portfolio,
    watchlists_loader: Callable[[Path], Any] = load_watchlists,
    run_logger: Any | None = None,
) -> FlowReportJobResult:
    return cast(
        FlowReportJobResult,
        run_flow_report_job_impl(
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
            snapshot_store_builder=snapshot_store_builder,
            flow_store_builder=flow_store_builder,
            portfolio_loader=portfolio_loader,
            watchlists_loader=watchlists_loader,
            run_logger=run_logger,
            resolve_quality_run_logger_fn=_resolve_quality_run_logger,
            persist_quality_results_fn=_persist_quality_results,
            run_flow_quality_checks_fn=run_flow_quality_checks,
            active_snapshot_store_fn=_active_snapshot_store,
            parameter_error_factory=VisibilityJobParameterError,
            result_factory=FlowReportJobResult,
        ),
    )


def run_derived_update_job(
    *,
    symbol: str,
    as_of: str,
    cache_dir: Path,
    derived_dir: Path,
    candle_cache_dir: Path,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    derived_store_builder: Callable[[Path], Any] = cli_deps.build_derived_store,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
    run_logger: Any | None = None,
) -> DerivedUpdateJobResult:
    return cast(
        DerivedUpdateJobResult,
        run_derived_update_job_impl(
            symbol=symbol,
            as_of=as_of,
            cache_dir=cache_dir,
            derived_dir=derived_dir,
            candle_cache_dir=candle_cache_dir,
            snapshot_store_builder=snapshot_store_builder,
            derived_store_builder=derived_store_builder,
            candle_store_builder=candle_store_builder,
            run_logger=run_logger,
            resolve_quality_run_logger_fn=_resolve_quality_run_logger,
            persist_quality_results_fn=_persist_quality_results,
            run_derived_quality_checks_fn=run_derived_quality_checks,
            active_snapshot_store_fn=_active_snapshot_store,
            filesystem_compatible_derived_store_fn=_filesystem_compatible_derived_store,
            filesystem_compatible_candle_store_fn=_filesystem_compatible_candle_store,
            execution_error_factory=VisibilityJobExecutionError,
            result_factory=DerivedUpdateJobResult,
        ),
    )


def run_briefing_job(
    *,
    portfolio_path: Path,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: str | None,
    as_of: str,
    compare: str,
    cache_dir: Path,
    candle_cache_dir: Path,
    technicals_config: Path,
    out: Path | None,
    print_to_console: bool,
    write_json: bool,
    strict: bool,
    update_derived: bool,
    derived_dir: Path,
    top: int,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    derived_store_builder: Callable[[Path], Any] = cli_deps.build_derived_store,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
    earnings_store_builder: Callable[[Path], Any] = cli_deps.build_earnings_store,
    safe_next_earnings_date_fn: Callable[..., date | None] = safe_next_earnings_date,
) -> BriefingJobResult:
    return cast(
        BriefingJobResult,
        run_briefing_job_impl(
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
            snapshot_store_builder=snapshot_store_builder,
            derived_store_builder=derived_store_builder,
            candle_store_builder=candle_store_builder,
            earnings_store_builder=earnings_store_builder,
            safe_next_earnings_date_fn=safe_next_earnings_date_fn,
            portfolio_loader=load_portfolio,
            watchlists_loader=load_watchlists,
            active_snapshot_store_fn=_active_snapshot_store,
            filesystem_compatible_derived_store_fn=_filesystem_compatible_derived_store,
            filesystem_compatible_candle_store_fn=_filesystem_compatible_candle_store,
            execution_error_factory=VisibilityJobExecutionError,
            result_factory=BriefingJobResult,
        ),
    )


def run_dashboard_job(
    *,
    report_date: str,
    reports_dir: Path,
) -> DashboardJobResult:
    return cast(
        DashboardJobResult,
        run_dashboard_job_impl(
            report_date=report_date,
            reports_dir=reports_dir,
            execution_error_factory=VisibilityJobExecutionError,
            result_factory=DashboardJobResult,
        ),
    )


def render_dashboard_report(
    *,
    result: DashboardJobResult,
    reports_dir: Path,
    scanner_run_dir: Path,
    scanner_run_id: str | None,
    max_shortlist_rows: int,
    render_fn: Callable[..., None] = render_dashboard,
    render_console: Any,
) -> None:
    render_dashboard_report_impl(
        result=result,
        reports_dir=reports_dir,
        scanner_run_dir=scanner_run_dir,
        scanner_run_id=scanner_run_id,
        max_shortlist_rows=max_shortlist_rows,
        render_fn=render_fn,
        render_console=render_console,
    )
