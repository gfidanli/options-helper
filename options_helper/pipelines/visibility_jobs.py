from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast
from zoneinfo import ZoneInfo

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.table import Table

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, score_confluence
from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.portfolio_risk import compute_portfolio_exposure, run_stress
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.commands.common import _build_stress_scenarios, _spot_from_meta
from options_helper.commands.position_metrics import _extract_float, _mark_price, _position_metrics
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.candles import CandleStore, close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.derived import DerivedStore
from options_helper.data.earnings import safe_next_earnings_date
from options_helper.data.ingestion.candles import CandleIngestResult, ingest_candles
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS, parse_date, resolve_symbols, shift_years
from options_helper.data.ingestion.options_bars import (
    BarsBackfillSummary,
    ContractDiscoveryOutput,
    PreparedContracts,
    backfill_option_bars,
    discover_option_contracts,
    prepare_contracts_for_bars,
)
from options_helper.data.market_types import DataFetchError
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.models import Position
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
    render_portfolio_table_markdown,
)
from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.common import utc_now
from options_helper.schemas.flow import FlowArtifact
from options_helper.storage import load_portfolio
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot
from options_helper.ui.dashboard import load_briefing_artifact, render_dashboard, resolve_briefing_paths
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


@dataclass(frozen=True)
class IngestOptionsBarsJobResult:
    warnings: list[str]
    underlyings: list[str]
    limited_underlyings: bool
    discovery: ContractDiscoveryOutput | None
    prepared: PreparedContracts | None
    summary: BarsBackfillSummary | None
    dry_run: bool
    no_symbols: bool
    no_contracts: bool
    no_eligible_contracts: bool


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


def _filesystem_compatible_snapshot_store(cache_dir: Path, store: Any) -> Any:
    # The CLI snapshot/report flows are offline-file-first and tests exercise CSV/meta.json
    # directories. If storage runtime resolved a DuckDB snapshot store, switch to filesystem
    # store to preserve existing command behavior.
    if store.__class__.__name__.startswith("DuckDB"):
        return OptionsSnapshotStore(cache_dir)
    return store


def _filesystem_compatible_derived_store(derived_dir: Path, store: Any) -> Any:
    if store.__class__.__name__.startswith("DuckDB"):
        return DerivedStore(derived_dir)
    return store


def _filesystem_compatible_candle_store(candle_cache_dir: Path, store: Any) -> Any:
    if store.__class__.__name__.startswith("DuckDB"):
        return CandleStore(candle_cache_dir)
    return store


def run_ingest_candles_job(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    candle_cache_dir: Path,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
) -> IngestCandlesJobResult:
    selection = resolve_symbols(
        watchlists_path=watchlists_path,
        watchlists=watchlist,
        symbols=symbol,
        default_watchlists=DEFAULT_WATCHLISTS,
    )

    if not selection.symbols:
        return IngestCandlesJobResult(
            warnings=list(selection.warnings),
            symbols=[],
            results=[],
            no_symbols=True,
        )

    provider = provider_builder()
    store = candle_store_builder(candle_cache_dir, provider=provider)
    results = ingest_candles(store, selection.symbols, period="max", best_effort=True)
    return IngestCandlesJobResult(
        warnings=list(selection.warnings),
        symbols=list(selection.symbols),
        results=results,
        no_symbols=False,
    )


def run_ingest_options_bars_job(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    contracts_exp_start: str,
    contracts_exp_end: str | None,
    lookback_years: int,
    page_limit: int,
    max_underlyings: int | None,
    max_contracts: int | None,
    max_expiries: int | None,
    resume: bool,
    dry_run: bool,
    fail_fast: bool,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    contracts_store_builder: Callable[[Path], Any] = cli_deps.build_option_contracts_store,
    bars_store_builder: Callable[[Path], Any] = cli_deps.build_option_bars_store,
    client_factory: Callable[[], AlpacaClient] = AlpacaClient,
    contracts_store_dir: Path = Path("data/option_contracts"),
    bars_store_dir: Path = Path("data/option_bars"),
    today: date | None = None,
) -> IngestOptionsBarsJobResult:
    selection = resolve_symbols(
        watchlists_path=watchlists_path,
        watchlists=watchlist,
        symbols=symbol,
        default_watchlists=DEFAULT_WATCHLISTS,
    )

    if not selection.symbols:
        return IngestOptionsBarsJobResult(
            warnings=list(selection.warnings),
            underlyings=[],
            limited_underlyings=False,
            discovery=None,
            prepared=None,
            summary=None,
            dry_run=dry_run,
            no_symbols=True,
            no_contracts=False,
            no_eligible_contracts=False,
        )

    underlyings = list(selection.symbols)
    limited_underlyings = False
    if max_underlyings is not None:
        underlyings = underlyings[:max_underlyings]
        limited_underlyings = True

    provider = provider_builder()
    provider_name = getattr(provider, "name", None)
    if provider_name != "alpaca":
        raise VisibilityJobParameterError("Options bars ingestion requires --provider alpaca.")

    run_day = today or date.today()
    try:
        exp_start = parse_date(contracts_exp_start, label="contracts-exp-start")
    except ValueError as exc:
        raise VisibilityJobParameterError(str(exc), param_hint="--contracts-exp-start") from exc

    if contracts_exp_end:
        try:
            exp_end = parse_date(contracts_exp_end, label="contracts-exp-end")
        except ValueError as exc:
            raise VisibilityJobParameterError(str(exc), param_hint="--contracts-exp-end") from exc
    else:
        exp_end = shift_years(run_day, 5)

    if exp_end < exp_start:
        raise VisibilityJobParameterError("contracts-exp-end must be >= contracts-exp-start")

    contracts_store = contracts_store_builder(contracts_store_dir)
    bars_store = bars_store_builder(bars_store_dir)

    client = client_factory()
    discovery = discover_option_contracts(
        client,
        underlyings=underlyings,
        exp_start=exp_start,
        exp_end=exp_end,
        page_limit=page_limit,
        max_contracts=max_contracts,
        fail_fast=fail_fast,
    )

    if discovery.contracts.empty:
        return IngestOptionsBarsJobResult(
            warnings=list(selection.warnings),
            underlyings=underlyings,
            limited_underlyings=limited_underlyings,
            discovery=discovery,
            prepared=None,
            summary=None,
            dry_run=dry_run,
            no_symbols=False,
            no_contracts=True,
            no_eligible_contracts=False,
        )

    if not dry_run:
        contracts_store.upsert_contracts(
            discovery.contracts,
            provider="alpaca",
            as_of_date=run_day,
            raw_by_contract_symbol=discovery.raw_by_symbol,
        )

    prepared = prepare_contracts_for_bars(
        discovery.contracts,
        max_expiries=max_expiries,
        max_contracts=max_contracts,
    )

    if prepared.contracts.empty:
        return IngestOptionsBarsJobResult(
            warnings=list(selection.warnings),
            underlyings=underlyings,
            limited_underlyings=limited_underlyings,
            discovery=discovery,
            prepared=prepared,
            summary=None,
            dry_run=dry_run,
            no_symbols=False,
            no_contracts=False,
            no_eligible_contracts=True,
        )

    summary = backfill_option_bars(
        client,
        bars_store,
        prepared.contracts,
        provider="alpaca",
        lookback_years=lookback_years,
        page_limit=page_limit,
        resume=resume,
        dry_run=dry_run,
        fail_fast=fail_fast,
        today=run_day,
    )

    return IngestOptionsBarsJobResult(
        warnings=list(selection.warnings),
        underlyings=underlyings,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=prepared,
        summary=summary,
        dry_run=dry_run,
        no_symbols=False,
        no_contracts=False,
        no_eligible_contracts=False,
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
) -> SnapshotOptionsJobResult:
    import numpy as np
    import pandas as pd

    portfolio = portfolio_loader(portfolio_path)
    store = _filesystem_compatible_snapshot_store(cache_dir, snapshot_store_builder(cache_dir))
    provider = provider_builder()
    candle_store = candle_store_builder(candle_cache_dir, provider=provider)
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )

    want_full_chain = full_chain
    want_all_expiries = all_expiries
    use_watchlists = bool(watchlist) or all_watchlists

    watchlists_used: list[str] = []
    expiries_by_symbol: dict[str, set[date]] = {}
    symbols: list[str]
    messages: list[str] = []

    if use_watchlists:
        wl = watchlists_loader(watchlists_path)
        if all_watchlists:
            watchlists_used = sorted(wl.watchlists.keys())
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                return SnapshotOptionsJobResult(
                    messages=[f"No watchlists in {watchlists_path}"],
                    dates_used=[],
                    symbols=[],
                    no_symbols=True,
                )
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise VisibilityJobParameterError(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
            watchlists_used = sorted(set(watchlist))
    else:
        if not portfolio.positions:
            return SnapshotOptionsJobResult(
                messages=["No positions."],
                dates_used=[],
                symbols=[],
                no_symbols=True,
            )
        for p in portfolio.positions:
            expiries_by_symbol.setdefault(p.symbol, set()).add(p.expiry)
        symbols = sorted(expiries_by_symbol.keys())

    dates_used: set[date] = set()

    required_date: date | None = None
    if require_data_date is not None:
        spec = require_data_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_data_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise VisibilityJobParameterError(
                f"Invalid --require-data-date/--require-data-tz: {exc}",
                param_hint="--require-data-date",
            ) from exc

    mode = "watchlists" if use_watchlists else "portfolio"
    messages.append(
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full-chain' if want_full_chain else 'windowed'})..."
    )

    effective_max_expiries = max_expiries
    if use_watchlists and not want_all_expiries and effective_max_expiries is None:
        effective_max_expiries = 2

    for symbol_value in symbols:
        history = candle_store.get_daily_history(symbol_value, period=spot_period)
        spot = last_close(history)
        data_date: date | None = history.index.max().date() if not history.empty else None
        if spot is None:
            try:
                underlying = provider.get_underlying(symbol_value, period=spot_period, interval="1d")
                spot = underlying.last_price
                if data_date is None and underlying.history is not None and not underlying.history.empty:
                    try:
                        data_date = underlying.history.index.max().date()
                    except Exception:  # noqa: BLE001
                        pass
            except DataFetchError:
                spot = None

        if spot is None or spot <= 0:
            messages.append(f"[yellow]Warning:[/yellow] {symbol_value}: missing spot price; skipping snapshot.")
            continue

        if required_date is not None and data_date != required_date:
            got = "-" if data_date is None else data_date.isoformat()
            messages.append(
                f"[yellow]Warning:[/yellow] {symbol_value}: candle date {got} != required {required_date.isoformat()}; "
                "skipping snapshot to avoid mis-dated overwrite."
            )
            continue

        effective_snapshot_date = data_date or date.today()
        dates_used.add(effective_snapshot_date)

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        meta = {
            "spot": spot,
            "spot_period": spot_period,
            "full_chain": want_full_chain,
            "all_expiries": want_all_expiries,
            "risk_free_rate": risk_free_rate,
            "window_pct": None if want_full_chain else window_pct,
            "strike_min": None if want_full_chain else strike_min,
            "strike_max": None if want_full_chain else strike_max,
            "snapshot_date": effective_snapshot_date.isoformat(),
            "symbol_source": mode,
            "watchlists": watchlists_used,
            "provider": provider_name,
        }
        if provider_version:
            meta["provider_version"] = provider_version

        expiries: list[date]
        if not use_watchlists and not want_all_expiries:
            expiries = sorted(expiries_by_symbol.get(symbol_value, set()))
        else:
            expiries = provider.list_option_expiries(symbol_value)
            if not expiries:
                messages.append(
                    f"[yellow]Warning:[/yellow] {symbol_value}: no listed option expiries; skipping snapshot."
                )
                continue
            if effective_max_expiries is not None:
                expiries = expiries[:effective_max_expiries]

        chain_frames: list[pd.DataFrame] = []
        quality_frames: list[pd.DataFrame] = []
        saved_expiries: list[date] = []
        raw_by_expiry: dict[date, dict[str, object]] = {}
        underlying_payload: dict[str, object] | None = None

        for exp in expiries:
            if want_full_chain:
                try:
                    raw = provider.get_options_chain_raw(symbol_value, exp)
                except DataFetchError as exc:
                    messages.append(
                        f"[yellow]Warning:[/yellow] {symbol_value} {exp.isoformat()}: {exc}; skipping snapshot."
                    )
                    continue

                underlying = raw.get("underlying")
                if not isinstance(underlying, dict):
                    underlying = {}
                if underlying_payload is None or underlying:
                    underlying_payload = underlying

                calls = pd.DataFrame(raw.get("calls", []))
                puts = pd.DataFrame(raw.get("puts", []))
                calls["optionType"] = "call"
                puts["optionType"] = "put"
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=effective_snapshot_date,
                    r=risk_free_rate,
                )

                chain_frames.append(df)
                quality_frames.append(df)
                raw_by_expiry[exp] = raw
                saved_expiries.append(exp)
                messages.append(f"{symbol_value} {exp.isoformat()}: saved {len(df)} contracts (full)")
                continue

            try:
                chain = provider.get_options_chain(symbol_value, exp)
            except DataFetchError as exc:
                messages.append(
                    f"[yellow]Warning:[/yellow] {symbol_value} {exp.isoformat()}: {exc}; skipping snapshot."
                )
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["optionType"] = "call"
            puts["optionType"] = "put"
            calls["expiry"] = exp.isoformat()
            puts["expiry"] = exp.isoformat()

            df = pd.concat([calls, puts], ignore_index=True)
            if not want_full_chain and "strike" in df.columns:
                df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

            df = add_black_scholes_greeks_to_chain(
                df,
                spot=spot,
                expiry=exp,
                as_of=effective_snapshot_date,
                r=risk_free_rate,
            )

            quality_frames.append(df)

            keep = [
                "contractSymbol",
                "optionType",
                "expiry",
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "change",
                "percentChange",
                "volume",
                "openInterest",
                "impliedVolatility",
                "inTheMoney",
                "bs_price",
                "bs_delta",
                "bs_gamma",
                "bs_theta_per_day",
                "bs_vega",
            ]
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

            chain_frames.append(df)
            saved_expiries.append(exp)
            messages.append(f"{symbol_value} {exp.isoformat()}: saved {len(df)} contracts")

        if not saved_expiries:
            continue

        chain_df = pd.concat(chain_frames, ignore_index=True) if chain_frames else pd.DataFrame()
        quality_df = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()

        total_contracts = int(len(quality_df))
        if total_contracts > 0:
            quality = compute_quote_quality(
                quality_df,
                min_volume=0,
                min_open_interest=0,
                as_of=effective_snapshot_date,
            )
            missing_bid_ask = 0
            stale_quotes = 0
            spread_pcts: list[float] = []
            if not quality.empty:
                q_warn = quality["quality_warnings"].tolist()
                missing_bid_ask = sum("quote_missing_bid_ask" in w for w in q_warn if isinstance(w, list))
                stale_quotes = sum("quote_stale" in w for w in q_warn if isinstance(w, list))
                spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
                spread_series = spread_series.where(spread_series >= 0)
                spread_pcts.extend(spread_series.dropna().tolist())
            spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
            spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
            meta["quote_quality"] = {
                "contracts": total_contracts,
                "missing_bid_ask_count": int(missing_bid_ask),
                "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
                "spread_pct_median": spread_median,
                "spread_pct_worst": spread_worst,
                "stale_quotes": int(stale_quotes),
                "stale_pct": float(stale_quotes / total_contracts),
            }

        if want_full_chain and underlying_payload is not None:
            meta["underlying"] = underlying_payload

        store.save_day_snapshot(
            symbol_value,
            effective_snapshot_date,
            chain=chain_df,
            expiries=saved_expiries,
            raw_by_expiry=raw_by_expiry if raw_by_expiry else None,
            meta=meta,
        )

    if dates_used:
        days = ", ".join(sorted({d.isoformat() for d in dates_used}))
        messages.append(f"Snapshot complete. Data date(s): {days}.")

    return SnapshotOptionsJobResult(
        messages=messages,
        dates_used=sorted(dates_used),
        symbols=symbols,
        no_symbols=False,
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
    portfolio_loader: Callable[[Path], Any] = load_portfolio,
    watchlists_loader: Callable[[Path], Any] = load_watchlists,
) -> FlowReportJobResult:
    import pandas as pd

    portfolio = portfolio_loader(portfolio_path)
    renderables: list[RenderableType] = []

    store = _filesystem_compatible_snapshot_store(cache_dir, snapshot_store_builder(cache_dir))
    use_watchlists = bool(watchlist) or all_watchlists
    if use_watchlists:
        wl = watchlists_loader(watchlists_path)
        if all_watchlists:
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                return FlowReportJobResult(
                    renderables=[f"No watchlists in {watchlists_path}"],
                    no_symbols=True,
                )
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise VisibilityJobParameterError(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
    else:
        symbols = sorted({p.symbol for p in portfolio.positions})
        if not symbols and symbol is None:
            return FlowReportJobResult(renderables=["No positions."], no_symbols=True)

    if symbol is not None:
        symbols = [symbol.upper()]

    pos_keys = {(p.symbol, p.expiry.isoformat(), float(p.strike), p.option_type) for p in portfolio.positions}

    group_by_norm = group_by.strip().lower()
    valid_group_by = {"contract", "strike", "expiry", "expiry-strike"}
    if group_by_norm not in valid_group_by:
        raise VisibilityJobParameterError(
            f"Invalid --group-by (use {', '.join(sorted(valid_group_by))})",
            param_hint="--group-by",
        )
    group_by_val = cast(FlowGroupBy, group_by_norm)

    for sym in symbols:
        need = window + 1
        dates = store.latest_dates(sym, n=need)
        if len(dates) < need:
            renderables.append(f"[yellow]No flow data for {sym}:[/yellow] need at least {need} snapshots.")
            continue

        pair_flows: list[pd.DataFrame] = []
        for prev_date, today_date in zip(dates[:-1], dates[1:], strict=False):
            today_df = store.load_day(sym, today_date)
            prev_df = store.load_day(sym, prev_date)
            if today_df.empty or prev_df.empty:
                renderables.append(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s) in window.")
                pair_flows = []
                break

            spot = _spot_from_meta(store.load_meta(sym, today_date))
            pair_flows.append(compute_flow(today_df, prev_df, spot=spot))

        if not pair_flows:
            continue

        start_date, end_date = dates[0], dates[-1]

        if window == 1 and group_by_norm == "contract":
            prev_date, today_date = dates[-2], dates[-1]
            flow = pair_flows[-1]
            summary = summarize_flow(flow)

            renderables.append(
                f"\n[bold]{sym}[/bold] flow {prev_date.isoformat()} → {today_date.isoformat()} | "
                f"calls ΔOI$={summary['calls_delta_oi_notional']:,.0f} | puts ΔOI$={summary['puts_delta_oi_notional']:,.0f}"
            )

            if flow.empty:
                renderables.append("No flow rows.")
                continue

            if "deltaOI_notional" in flow.columns:
                flow = flow.assign(_abs=flow["deltaOI_notional"].abs())
                flow = flow.sort_values("_abs", ascending=False).drop(columns=["_abs"])

            table = Table(title=f"{sym} top {top} contracts by |ΔOI_notional|")
            table.add_column("*")
            table.add_column("Expiry")
            table.add_column("Type")
            table.add_column("Strike", justify="right")
            table.add_column("ΔOI", justify="right")
            table.add_column("OI", justify="right")
            table.add_column("Vol", justify="right")
            table.add_column("ΔOI$", justify="right")
            table.add_column("Class")

            for _, row in flow.head(top).iterrows():
                expiry = str(row.get("expiry", "-"))
                opt_type = str(row.get("optionType", "-"))
                strike = row.get("strike")
                strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
                key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
                in_port = key in pos_keys if strike_val is not None else False

                table.add_row(
                    "*" if in_port else "",
                    expiry,
                    opt_type,
                    "-" if strike_val is None else f"{strike_val:g}",
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("openInterest")) else f"{row.get('openInterest'):.0f}",
                    "-" if pd.isna(row.get("volume")) else f"{row.get('volume'):.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    str(row.get("flow_class", "-")),
                )

            renderables.append(table)

            if out is not None:
                net = aggregate_flow_window(pair_flows, group_by="contract")
                net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
                sort_cols = ["_abs"]
                ascending = [False]
                for c in ["expiry", "strike", "optionType", "contractSymbol"]:
                    if c in net.columns:
                        sort_cols.append(c)
                        ascending.append(True)
                net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

                base = out / "flow" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / f"{prev_date.isoformat()}_to_{today_date.isoformat()}_w1_contract.json"
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
                    as_of=today_date.isoformat(),
                    symbol=sym.upper(),
                    from_date=prev_date.isoformat(),
                    to_date=today_date.isoformat(),
                    window=1,
                    group_by="contract",
                    snapshot_dates=[prev_date.isoformat(), today_date.isoformat()],
                    net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                ).to_dict()
                if strict:
                    FlowArtifact.model_validate(payload)
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                renderables.append(f"\nSaved: {out_path}")
            continue

        net = aggregate_flow_window(pair_flows, group_by=group_by_val)
        if net.empty:
            renderables.append(
                f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()})"
            )
            renderables.append("No net flow rows.")
            continue

        calls_premium = (
            float(net[net["optionType"] == "call"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        )
        puts_premium = (
            float(net[net["optionType"] == "put"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        )

        renderables.append(
            f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()}) | "
            f"group-by={group_by_norm} | calls ΔOI$={calls_premium:,.0f} | puts ΔOI$={puts_premium:,.0f}"
        )

        net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
        sort_cols = ["_abs"]
        ascending = [False]
        for c in ["expiry", "strike", "optionType", "contractSymbol"]:
            if c in net.columns:
                sort_cols.append(c)
                ascending.append(True)

        net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

        def _render_zone_table(title: str) -> Table:
            t = Table(title=title)
            if group_by_norm == "contract":
                t.add_column("*")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                t.add_column("Expiry")
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                t.add_column("Strike", justify="right")
            t.add_column("Type")
            t.add_column("Net ΔOI", justify="right")
            t.add_column("Net ΔOI$", justify="right")
            t.add_column("Net Δ$", justify="right")
            t.add_column("N", justify="right")
            return t

        def _add_zone_row(t: Table, row: pd.Series) -> None:
            expiry = str(row.get("expiry", "-"))
            opt_type = str(row.get("optionType", "-"))
            strike = row.get("strike")
            strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
            key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
            in_port = key in pos_keys if strike_val is not None else False

            cells: list[str] = []
            if group_by_norm == "contract":
                cells.append("*" if in_port else "")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                cells.append(expiry)
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                cells.append("-" if strike_val is None else f"{strike_val:g}")
            cells.extend(
                [
                    opt_type,
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    "-" if pd.isna(row.get("delta_notional")) else f"{row.get('delta_notional'):+.0f}",
                    "-" if pd.isna(row.get("size")) else f"{int(row.get('size')):d}",
                ]
            )
            t.add_row(*cells)

        building = net[net["deltaOI_notional"] > 0].head(top)
        unwinding = net[net["deltaOI_notional"] < 0].head(top)

        t_build = _render_zone_table(f"{sym} building zones (top {top} by |net ΔOI$|)")
        for _, row in building.iterrows():
            _add_zone_row(t_build, row)
        renderables.append(t_build)

        t_unwind = _render_zone_table(f"{sym} unwinding zones (top {top} by |net ΔOI$|)")
        for _, row in unwinding.iterrows():
            _add_zone_row(t_unwind, row)
        renderables.append(t_unwind)

        if out is not None:
            base = out / "flow" / sym.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{start_date.isoformat()}_to_{end_date.isoformat()}_w{window}_{group_by_norm}.json"
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
                as_of=end_date.isoformat(),
                symbol=sym.upper(),
                from_date=start_date.isoformat(),
                to_date=end_date.isoformat(),
                window=window,
                group_by=group_by_norm,
                snapshot_dates=[d.isoformat() for d in dates],
                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
            ).to_dict()
            if strict:
                FlowArtifact.model_validate(payload)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            renderables.append(f"\nSaved: {out_path}")

    return FlowReportJobResult(renderables=renderables, no_symbols=False)


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
) -> DerivedUpdateJobResult:
    store = _filesystem_compatible_snapshot_store(cache_dir, snapshot_store_builder(cache_dir))
    derived = _filesystem_compatible_derived_store(derived_dir, derived_store_builder(derived_dir))
    candle_store = _filesystem_compatible_candle_store(candle_cache_dir, candle_store_builder(candle_cache_dir))

    as_of_date = store.resolve_date(symbol, as_of)
    df = store.load_day(symbol, as_of_date)
    meta = store.load_meta(symbol, as_of_date)
    spot = _spot_from_meta(meta)
    if spot is None:
        raise VisibilityJobExecutionError("missing spot price in meta.json (run snapshot-options first)")

    report = compute_chain_report(
        df,
        symbol=symbol,
        as_of=as_of_date,
        spot=spot,
        expiries_mode="near",
        top=10,
        best_effort=True,
    )

    candles = candle_store.load(symbol)
    history = derived.load(symbol)
    row = DerivedRow.from_chain_report(report, candles=candles, derived_history=history)
    out_path = derived.upsert(symbol, row)
    return DerivedUpdateJobResult(symbol=symbol.upper(), as_of_date=as_of_date, output_path=out_path)


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

    store = _filesystem_compatible_snapshot_store(cache_dir, snapshot_store_builder(cache_dir))
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
    try:
        paths = resolve_briefing_paths(reports_dir, report_date)
    except Exception as exc:  # noqa: BLE001
        raise VisibilityJobExecutionError(str(exc)) from exc

    try:
        artifact = load_briefing_artifact(paths.json_path)
    except Exception as exc:  # noqa: BLE001
        raise VisibilityJobExecutionError(f"failed to load briefing JSON: {exc}") from exc

    return DashboardJobResult(json_path=paths.json_path, artifact=artifact)


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
    render_console.print(f"Briefing JSON: {result.json_path}")
    render_fn(
        artifact=result.artifact,
        console=render_console,
        reports_dir=reports_dir,
        scanner_run_dir=scanner_run_dir,
        scanner_run_id=scanner_run_id,
        max_shortlist_rows=max_shortlist_rows,
    )
