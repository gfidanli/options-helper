from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from rich.console import RenderableType
from rich.markdown import Markdown

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, score_confluence
from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.flow import aggregate_flow_window, compute_flow
from options_helper.analysis.portfolio_risk import compute_portfolio_exposure, run_stress
from options_helper.commands.common import _build_stress_scenarios, _spot_from_meta
from options_helper.commands.position_metrics import _extract_float, _mark_price, _position_metrics
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.candles import CandleStore, close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
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
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.models import Position
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
    render_portfolio_table_markdown,
)
from options_helper.pipelines.visibility_jobs_derived_dashboard_legacy import (
    render_dashboard_report_impl,
    run_dashboard_job_impl,
    run_derived_update_job_impl,
)
from options_helper.pipelines.visibility_jobs_flow_legacy import run_flow_report_job_impl
from options_helper.pipelines.visibility_jobs_ingest_candles_legacy import run_ingest_candles_job_impl
from options_helper.pipelines.visibility_jobs_ingest_options_bars_legacy import (
    run_ingest_options_bars_job_impl,
)
from options_helper.pipelines.visibility_jobs_snapshot_legacy import run_snapshot_options_job_impl
from options_helper.schemas.briefing import BriefingArtifact
from options_helper.storage import load_portfolio
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot
from options_helper.ui.dashboard import render_dashboard
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


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
    return cast(
        IngestOptionsBarsJobResult,
        run_ingest_options_bars_job_impl(
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            contracts_root_symbols=contracts_root_symbols,
            contract_symbol_prefix=contract_symbol_prefix,
            contracts_exp_start=contracts_exp_start,
            contracts_exp_end=contracts_exp_end,
            lookback_years=lookback_years,
            page_limit=page_limit,
            contracts_page_size=contracts_page_size,
            max_underlyings=max_underlyings,
            max_contracts=max_contracts,
            max_expiries=max_expiries,
            contracts_max_requests_per_second=contracts_max_requests_per_second,
            bars_concurrency=bars_concurrency,
            bars_max_requests_per_second=bars_max_requests_per_second,
            bars_batch_mode=bars_batch_mode,
            bars_batch_size=bars_batch_size,
            bars_write_batch_size=bars_write_batch_size,
            resume=resume,
            dry_run=dry_run,
            fail_fast=fail_fast,
            contracts_status=contracts_status,
            contracts_only=contracts_only,
            fetch_only=fetch_only,
            provider_builder=provider_builder,
            contracts_store_builder=contracts_store_builder,
            bars_store_builder=bars_store_builder,
            client_factory=client_factory,
            contracts_store_dir=contracts_store_dir,
            bars_store_dir=bars_store_dir,
            today=today,
            run_logger=run_logger,
            resolve_quality_run_logger_fn=_resolve_quality_run_logger,
            persist_quality_results_fn=_persist_quality_results,
            run_options_bars_quality_checks_fn=run_options_bars_quality_checks,
            parameter_error_factory=VisibilityJobParameterError,
            result_factory=IngestOptionsBarsJobResult,
            fetch_only_store_factory=_FetchOnlyOptionBarsStore,
            discover_option_contracts_fn=discover_option_contracts,
            prepare_contracts_for_bars_fn=prepare_contracts_for_bars,
            backfill_option_bars_fn=backfill_option_bars,
        ),
    )


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
    import pandas as pd

    renderables: list[RenderableType] = []
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    positions_by_symbol: dict[str, list[Position]] = {}
    for p in portfolio.positions:
        positions_by_symbol.setdefault(p.symbol.upper(), []).append(p)

    portfolio_symbols = sorted({p.symbol.upper() for p in portfolio.positions})
    watch_symbols: list[str] = []
    watchlist_symbols_by_name: dict[str, list[str]] = {}
    if watchlist:
        try:
            wl = load_watchlists(watchlists_path)
            for name in watchlist:
                wl_symbols = wl.get(name)
                watch_symbols.extend(wl_symbols)
                watchlist_symbols_by_name[name] = wl_symbols
        except Exception as exc:  # noqa: BLE001
            renderables.append(f"[yellow]Warning:[/yellow] failed to load watchlists: {exc}")

    symbols = sorted(set(portfolio_symbols).union({s.upper() for s in watch_symbols if s}))
    if symbol is not None:
        symbols = [symbol.upper().strip()]

    if symbol is not None:
        sym = symbols[0] if symbols else ""
        symbol_sources_map: dict[str, set[str]] = {}
        if sym:
            if sym in portfolio_symbols:
                symbol_sources_map.setdefault(sym, set()).add("portfolio")
            symbol_sources_map.setdefault(sym, set()).add("manual")
        symbol_sources_payload = [
            {"symbol": sym_value, "sources": sorted(symbol_sources_map.get(sym_value, set()))}
            for sym_value in symbols
        ]
        watchlists_payload: list[dict[str, object]] = []
    else:
        symbol_sources_map = {}
        for sym_value in portfolio_symbols:
            symbol_sources_map.setdefault(sym_value, set()).add("portfolio")
        for name, syms in watchlist_symbols_by_name.items():
            for sym_value in syms:
                symbol_sources_map.setdefault(sym_value, set()).add(f"watchlist:{name}")

        symbol_sources_payload = [
            {"symbol": sym_value, "sources": sorted(symbol_sources_map.get(sym_value, set()))}
            for sym_value in symbols
        ]
        watchlists_payload = [
            {"name": name, "symbols": watchlist_symbols_by_name.get(name, [])}
            for name in watchlist
            if name in watchlist_symbols_by_name
        ]

    if not symbols:
        raise VisibilityJobExecutionError("no symbols selected (empty portfolio and no watchlists)")

    store = _active_snapshot_store(snapshot_store_builder(cache_dir))
    derived_store = _filesystem_compatible_derived_store(derived_dir, derived_store_builder(derived_dir))
    candle_store = _filesystem_compatible_candle_store(candle_cache_dir, candle_store_builder(candle_cache_dir))
    earnings_store = earnings_store_builder(Path("data/earnings"))

    technicals_cfg: dict | None = None
    technicals_cfg_error: str | None = None
    try:
        technicals_cfg = load_technical_backtesting_config(technicals_config)
    except Exception as exc:  # noqa: BLE001
        technicals_cfg_error = str(exc)
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        renderables.append(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")

    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}

    sections: list[BriefingSymbolSection] = []
    resolved_to_dates: list[date] = []
    compare_norm = compare.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    def _trend_from_weekly_flag(flag: bool | None) -> str | None:
        if flag is True:
            return "up"
        if flag is False:
            return "down"
        return None

    def _extension_percentile_from_snapshot(snapshot: TechnicalSnapshot | None) -> float | None:
        if snapshot is None or snapshot.extension_percentiles is None:
            return None
        daily = snapshot.extension_percentiles.daily
        if daily is None or not daily.current_percentiles:
            return None
        items: list[tuple[float, float]] = []
        for key, value in daily.current_percentiles.items():
            try:
                items.append((float(key), float(value)))
            except Exception:  # noqa: BLE001
                continue
        if not items:
            return None
        return sorted(items, key=lambda t: t[0])[-1][1]

    def _net_flow_delta_oi_notional(flow_net: pd.DataFrame | None) -> float | None:
        if flow_net is None or flow_net.empty:
            return None
        if "deltaOI_notional" not in flow_net.columns or "optionType" not in flow_net.columns:
            return None
        df_local = flow_net.copy()
        df_local["deltaOI_notional"] = pd.to_numeric(df_local["deltaOI_notional"], errors="coerce")
        df_local["optionType"] = df_local["optionType"].astype(str).str.lower()
        calls = df_local[df_local["optionType"] == "call"]["deltaOI_notional"].dropna()
        puts = df_local[df_local["optionType"] == "put"]["deltaOI_notional"].dropna()
        if calls.empty and puts.empty:
            return None
        return float(calls.sum()) - float(puts.sum())

    for sym in symbols:
        errors: list[str] = []
        warnings: list[str] = []
        chain = None
        compare_report = None
        flow_net = None
        technicals = None
        candles = None
        derived_updated = False
        derived_row = None
        confluence_score = None
        quote_quality = None
        next_earnings_date = safe_next_earnings_date_fn(earnings_store, sym)
        next_earnings_by_symbol[sym] = next_earnings_date

        try:
            to_date = store.resolve_date(sym, as_of)
            resolved_to_dates.append(to_date)

            event_warnings: set[str] = set()
            base_risk = earnings_event_risk(
                today=to_date,
                expiry=None,
                next_earnings_date=next_earnings_date,
                warn_days=rp.earnings_warn_days,
                avoid_days=rp.earnings_avoid_days,
            )
            event_warnings.update(base_risk["warnings"])
            for pos in positions_by_symbol.get(sym, []):
                pos_risk = earnings_event_risk(
                    today=to_date,
                    expiry=pos.expiry,
                    next_earnings_date=next_earnings_date,
                    warn_days=rp.earnings_warn_days,
                    avoid_days=rp.earnings_avoid_days,
                )
                event_warnings.update(pos_risk["warnings"])
            if event_warnings:
                warnings.extend(sorted(event_warnings))

            df_to = store.load_day(sym, to_date)
            meta_to = store.load_meta(sym, to_date)
            spot_to = _spot_from_meta(meta_to)
            quote_quality = meta_to.get("quote_quality") if isinstance(meta_to, dict) else None
            if spot_to is None:
                raise VisibilityJobExecutionError("missing spot price in meta.json (run snapshot-options first)")

            day_cache[sym] = (to_date, df_to)

            if technicals_cfg is None:
                if technicals_cfg_error is not None:
                    warnings.append(f"technicals unavailable: {technicals_cfg_error}")
            else:
                try:
                    candles = candle_store.load(sym)
                    if candles.empty:
                        warnings.append("technicals unavailable: missing candle cache (run refresh-candles)")
                    else:
                        technicals = compute_technical_snapshot(candles, technicals_cfg)
                        if technicals is None:
                            warnings.append("technicals unavailable: insufficient candle history / warmup")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"technicals unavailable: {exc}")

            if candles is None:
                candles = pd.DataFrame()
            candles_by_symbol[sym] = candles

            chain = compute_chain_report(
                df_to,
                symbol=sym,
                as_of=to_date,
                spot=spot_to,
                expiries_mode="near",
                top=10,
                best_effort=True,
            )

            if update_derived:
                if candles is None:
                    candles = candle_store.load(sym)
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain, candles=candles, derived_history=history)
                try:
                    derived_store.upsert(sym, row)
                    derived_updated = True
                    derived_row = row
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"derived update failed: {exc}")

            if compare_enabled:
                from_date: date | None = None
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    try:
                        from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"compare unavailable: {exc}")
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                if from_date is not None and from_date != to_date:
                    df_from = store.load_day(sym, from_date)
                    meta_from = store.load_meta(sym, from_date)
                    spot_from = _spot_from_meta(meta_from)
                    if spot_from is None:
                        warnings.append("compare unavailable: missing spot in from-date meta.json")
                    elif df_from.empty or df_to.empty:
                        warnings.append("compare unavailable: missing snapshot CSVs for from/to date")
                    else:
                        compare_report, _, _ = compute_compare_report(
                            symbol=sym,
                            from_date=from_date,
                            to_date=to_date,
                            from_df=df_from,
                            to_df=df_to,
                            spot_from=spot_from,
                            spot_to=spot_to,
                            top=top,
                        )

                        try:
                            flow = compute_flow(df_to, df_from, spot=spot_to)
                            flow_net = aggregate_flow_window([flow], group_by="strike")
                        except Exception:  # noqa: BLE001
                            warnings.append("flow unavailable: compute failed")

            try:
                trend = _trend_from_weekly_flag(technicals.weekly_trend_up if technicals is not None else None)
                ext_pct = _extension_percentile_from_snapshot(technicals)
                flow_notional = _net_flow_delta_oi_notional(flow_net)
                iv_rv = derived_row.iv_rv_20d if derived_row is not None else None
                inputs = ConfluenceInputs(
                    weekly_trend=trend,
                    extension_percentile=ext_pct,
                    rsi_divergence=None,
                    flow_delta_oi_notional=flow_notional,
                    iv_rv_20d=iv_rv,
                )
                confluence_score = score_confluence(inputs, cfg=confluence_cfg)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"confluence unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        as_of_label = "-" if sym not in day_cache else day_cache[sym][0].isoformat()
        sections.append(
            BriefingSymbolSection(
                symbol=sym,
                as_of=as_of_label,
                chain=chain,
                compare=compare_report,
                flow_net=flow_net,
                technicals=technicals,
                confluence=confluence_score,
                errors=errors,
                warnings=warnings,
                quote_quality=quote_quality,
                derived_updated=derived_updated,
                derived=derived_row,
                next_earnings_date=next_earnings_date,
            )
        )

    if not resolved_to_dates:
        raise VisibilityJobExecutionError("no snapshots found for selected symbols")
    report_date = max(resolved_to_dates).isoformat()
    portfolio_rows: list[dict[str, str]] = []
    portfolio_rows_payload: list[dict[str, object]] = []
    portfolio_rows_with_pnl: list[tuple[float, dict[str, str]]] = []
    portfolio_metrics: list[Any] = []
    for p in portfolio.positions:
        sym = p.symbol.upper()
        to_date, df_to = day_cache.get(sym, (None, pd.DataFrame()))

        mark = None
        spr_pct = None
        snapshot_row = None
        if not df_to.empty:
            sub = df_to.copy()
            if "expiry" in sub.columns:
                sub = sub[sub["expiry"].astype(str) == p.expiry.isoformat()]
            if "optionType" in sub.columns:
                sub = sub[sub["optionType"].astype(str).str.lower() == p.option_type]
            if "strike" in sub.columns:
                strike = pd.to_numeric(sub["strike"], errors="coerce")
                sub = sub.assign(_strike=strike)
                sub = sub[(sub["_strike"] - float(p.strike)).abs() < 1e-9]
            if not sub.empty:
                snapshot_row = sub.iloc[0]
                bid = _extract_float(snapshot_row, "bid")
                ask = _extract_float(snapshot_row, "ask")
                mark = _mark_price(bid=bid, ask=ask, last=_extract_float(snapshot_row, "lastPrice"))
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    if mid > 0:
                        spr_pct = (ask - bid) / mid

        history = candles_by_symbol.get(sym)
        if history is None or history.empty:
            try:
                history = candle_store.load(sym)
            except Exception:  # noqa: BLE001
                history = pd.DataFrame()
            candles_by_symbol[sym] = history

        try:
            last_price = close_asof(history, to_date) if to_date is not None else last_close(history)
            metrics = _position_metrics(
                None,
                p,
                risk_profile=rp,
                underlying_history=history,
                underlying_last_price=last_price,
                as_of=to_date,
                next_earnings_date=next_earnings_by_symbol.get(sym),
                snapshot_row=snapshot_row if snapshot_row is not None else {},
            )
            portfolio_metrics.append(metrics)
        except Exception as exc:  # noqa: BLE001
            renderables.append(f"[yellow]Warning:[/yellow] portfolio exposure skipped for {p.id}: {exc}")

        pnl_abs = None
        pnl_pct = None
        if mark is not None:
            pnl_abs = (mark - p.cost_basis) * 100.0 * p.contracts
            pnl_pct = ((mark / p.cost_basis) - 1.0) if p.cost_basis > 0 else None

        portfolio_rows.append(
            {
                "id": p.id,
                "symbol": sym,
                "type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": f"{p.strike:g}",
                "ct": str(p.contracts),
                "cost": f"{p.cost_basis:.2f}",
                "mark": "-" if mark is None else f"{mark:.2f}",
                "pnl_$": "-" if pnl_abs is None else f"{pnl_abs:+.0f}",
                "pnl_%": "-" if pnl_pct is None else f"{pnl_pct * 100.0:+.1f}%",
                "spr_%": "-" if spr_pct is None else f"{spr_pct * 100.0:.1f}%",
                "as_of": "-" if to_date is None else to_date.isoformat(),
            }
        )
        portfolio_rows_payload.append(
            {
                "id": p.id,
                "symbol": sym,
                "option_type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": float(p.strike),
                "contracts": int(p.contracts),
                "cost_basis": float(p.cost_basis),
                "mark": None if mark is None else float(mark),
                "pnl": None if pnl_abs is None else float(pnl_abs),
                "pnl_pct": None if pnl_pct is None else float(pnl_pct),
                "spr_pct": None if spr_pct is None else float(spr_pct),
                "as_of": None if to_date is None else to_date.isoformat(),
            }
        )
        pnl_sort = float(pnl_pct) if pnl_pct is not None else float("-inf")
        portfolio_rows_with_pnl.append((pnl_sort, portfolio_rows[-1]))

    portfolio_table_md = None
    if portfolio_rows:
        portfolio_rows_sorted = [row for _, row in sorted(portfolio_rows_with_pnl, key=lambda r: r[0], reverse=True)]
        include_spread = any(r.get("spr_%") not in (None, "-") for r in portfolio_rows_sorted)
        portfolio_table_md = render_portfolio_table_markdown(portfolio_rows_sorted, include_spread=include_spread)

    portfolio_exposure = None
    portfolio_stress = None
    if portfolio_metrics:
        portfolio_exposure = compute_portfolio_exposure(portfolio_metrics)
        portfolio_stress = run_stress(
            portfolio_exposure,
            _build_stress_scenarios(stress_spot_pct=[], stress_vol_pp=5.0, stress_days=7),
        )

    md = render_briefing_markdown(
        report_date=report_date,
        portfolio_path=str(portfolio_path),
        symbol_sections=sections,
        portfolio_table_md=portfolio_table_md,
        top=top,
    )

    if out is None:
        out_path = Path("data/reports/daily") / f"{report_date}.md"
    else:
        out_path = out
        if out_path.suffix.lower() != ".md":
            out_path = out_path / f"{report_date}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    renderables.append(f"Saved: {out_path}")

    json_path: Path | None = None
    if write_json:
        payload = build_briefing_payload(
            report_date=report_date,
            as_of=report_date,
            portfolio_path=str(portfolio_path),
            symbol_sections=sections,
            top=top,
            technicals_config=str(technicals_config),
            portfolio_exposure=portfolio_exposure,
            portfolio_stress=portfolio_stress,
            portfolio_rows=portfolio_rows_payload,
            symbol_sources=symbol_sources_payload,
            watchlists=watchlists_payload,
        )
        if strict:
            BriefingArtifact.model_validate(payload)
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        renderables.append(f"Saved: {json_path}")

    if print_to_console:
        renderables.append(Markdown(md))

    return BriefingJobResult(
        report_date=report_date,
        markdown=md,
        markdown_path=out_path,
        json_path=json_path,
        renderables=renderables,
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
