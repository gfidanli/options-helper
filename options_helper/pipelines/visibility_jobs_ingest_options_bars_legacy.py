from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import options_helper.cli_deps as cli_deps
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.common import (
    DEFAULT_WATCHLISTS,
    SymbolSelection,
    parse_date,
    resolve_symbols,
    shift_years,
)
from options_helper.data.ingestion.options_bars import (
    BarsBackfillSummary,
    ContractDiscoveryOutput,
    PreparedContracts,
    backfill_option_bars,
    discover_option_contracts,
    prepare_contracts_for_bars,
)


def _normalize_root_symbols(contracts_root_symbols: list[str] | None) -> list[str]:
    return [
        str(sym).strip().upper()
        for sym in (contracts_root_symbols or [])
        if str(sym or "").strip()
    ]


def _resolve_symbol_selection(
    *,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: list[str],
    root_symbols: list[str],
) -> SymbolSelection:
    try:
        return resolve_symbols(
            watchlists_path=watchlists_path,
            watchlists=watchlist,
            symbols=symbol,
            default_watchlists=DEFAULT_WATCHLISTS,
        )
    except Exception as exc:  # noqa: BLE001
        if root_symbols:
            return SymbolSelection(
                symbols=[],
                watchlists_used=[],
                warnings=[f"Watchlists unavailable: {exc}"],
            )
        raise


def _limit_underlyings(selection: SymbolSelection, max_underlyings: int | None) -> tuple[list[str], bool]:
    underlyings = list(selection.symbols)
    limited_underlyings = False
    if max_underlyings is not None:
        underlyings = underlyings[:max_underlyings]
        limited_underlyings = True
    return underlyings, limited_underlyings


def _resolve_expiry_window(
    *,
    contracts_exp_start: str,
    contracts_exp_end: str | None,
    run_day: date,
    parameter_error_factory: Callable[..., Exception],
) -> tuple[date, date]:
    try:
        exp_start = parse_date(contracts_exp_start, label="contracts-exp-start")
    except ValueError as exc:
        raise parameter_error_factory(str(exc), param_hint="--contracts-exp-start") from exc

    if contracts_exp_end:
        try:
            exp_end = parse_date(contracts_exp_end, label="contracts-exp-end")
        except ValueError as exc:
            raise parameter_error_factory(str(exc), param_hint="--contracts-exp-end") from exc
    else:
        exp_end = shift_years(run_day, 5)

    if exp_end < exp_start:
        raise parameter_error_factory("contracts-exp-end must be >= contracts-exp-start")
    return exp_start, exp_end


def _resolve_contract_status(
    contracts_status: str,
    *,
    parameter_error_factory: Callable[..., Exception],
) -> str:
    contract_status = str(contracts_status or "").strip().lower()
    if contract_status not in {"active", "inactive", "all"}:
        raise parameter_error_factory(
            "contracts-status must be one of: active, inactive, all",
            param_hint="--contracts-status",
        )
    return contract_status


def _persist_options_bars_quality(
    *,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    bars_store: Any,
    contract_symbols: list[str],
    dry_run: bool,
    skip_reason: str | None = None,
) -> None:
    persist_quality_results_fn(
        quality_logger,
        run_options_bars_quality_checks_fn(
            bars_store=bars_store,
            contract_symbols=contract_symbols,
            dry_run=dry_run,
            skip_reason=skip_reason,
        ),
    )


def _build_ingest_options_bars_result(
    *,
    result_factory: Callable[..., Any],
    selection: SymbolSelection,
    underlyings: list[str],
    root_symbols: list[str],
    limited_underlyings: bool,
    discovery: ContractDiscoveryOutput | None,
    prepared: PreparedContracts | None,
    summary: BarsBackfillSummary | None,
    dry_run: bool,
    contracts_only: bool,
    no_symbols: bool,
    no_contracts: bool,
    no_eligible_contracts: bool,
    contracts_endpoint_stats: Any | None = None,
    bars_endpoint_stats: Any | None = None,
) -> Any:
    return result_factory(
        warnings=list(selection.warnings),
        underlyings=underlyings,
        root_symbols=root_symbols,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=prepared,
        summary=summary,
        dry_run=dry_run,
        contracts_only=contracts_only,
        no_symbols=no_symbols,
        no_contracts=no_contracts,
        no_eligible_contracts=no_eligible_contracts,
        contracts_endpoint_stats=contracts_endpoint_stats,
        bars_endpoint_stats=bars_endpoint_stats,
    )


def _persist_discovered_contracts(
    *,
    contracts_store: Any,
    discovery: ContractDiscoveryOutput,
    run_day: date,
    contracts_page_size: int,
) -> None:
    if contracts_store is None:
        return
    contracts_frame = discovery.contracts.reset_index(drop=True)
    contract_write_batch_size = max(1, int(contracts_page_size or 10000))
    for offset in range(0, len(contracts_frame), contract_write_batch_size):
        chunk = contracts_frame.iloc[offset : offset + contract_write_batch_size].copy()
        if chunk.empty:
            continue
        chunk_symbols: set[str] = set()
        if "contractSymbol" in chunk.columns:
            chunk_symbols = {
                str(value).strip().upper()
                for value in chunk["contractSymbol"].tolist()
                if str(value or "").strip()
            }
        chunk_raw = (
            {key: val for key, val in discovery.raw_by_symbol.items() if key in chunk_symbols}
            if chunk_symbols
            else None
        )
        contracts_store.upsert_contracts(
            chunk,
            provider="alpaca",
            as_of_date=run_day,
            raw_by_contract_symbol=chunk_raw,
        )


def _prepared_contract_symbols(prepared: PreparedContracts) -> list[str]:
    if "contractSymbol" not in prepared.contracts.columns:
        return []
    return [
        str(value).strip().upper()
        for value in prepared.contracts["contractSymbol"].tolist()
        if str(value or "").strip()
    ]


def _bars_store_for_mode(
    *,
    fetch_only: bool,
    bars_store_builder: Callable[[Path], Any],
    bars_store_dir: Path,
    fetch_only_store_factory: Callable[[], Any],
) -> Any:
    if fetch_only:
        return fetch_only_store_factory()
    return bars_store_builder(bars_store_dir)


def _no_symbols_result(
    *,
    selection: SymbolSelection,
    result_factory: Callable[..., Any],
    contracts_only: bool,
    dry_run: bool,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    quality_dry_run: bool,
) -> Any:
    _persist_options_bars_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_options_bars_quality_checks_fn=run_options_bars_quality_checks_fn,
        quality_logger=quality_logger,
        bars_store=None,
        contract_symbols=[],
        dry_run=quality_dry_run,
        skip_reason="no_symbols",
    )
    return _build_ingest_options_bars_result(
        result_factory=result_factory,
        selection=selection,
        underlyings=[],
        root_symbols=[],
        limited_underlyings=False,
        discovery=None,
        prepared=None,
        summary=None,
        dry_run=dry_run,
        contracts_only=contracts_only,
        no_symbols=True,
        no_contracts=False,
        no_eligible_contracts=False,
    )


def _no_contracts_result(
    *,
    selection: SymbolSelection,
    result_factory: Callable[..., Any],
    underlyings: list[str],
    root_symbols: list[str],
    limited_underlyings: bool,
    discovery: ContractDiscoveryOutput,
    contracts_only: bool,
    dry_run: bool,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    quality_dry_run: bool,
    bars_store: Any,
) -> Any:
    _persist_options_bars_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_options_bars_quality_checks_fn=run_options_bars_quality_checks_fn,
        quality_logger=quality_logger,
        bars_store=bars_store,
        contract_symbols=[],
        dry_run=quality_dry_run,
        skip_reason="no_contracts",
    )
    return _build_ingest_options_bars_result(
        result_factory=result_factory,
        selection=selection,
        underlyings=underlyings,
        root_symbols=root_symbols,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=None,
        summary=None,
        dry_run=dry_run,
        contracts_only=contracts_only,
        no_symbols=False,
        no_contracts=True,
        no_eligible_contracts=False,
    )


def _contracts_only_result(
    *,
    selection: SymbolSelection,
    result_factory: Callable[..., Any],
    underlyings: list[str],
    root_symbols: list[str],
    limited_underlyings: bool,
    discovery: ContractDiscoveryOutput,
    dry_run: bool,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    quality_dry_run: bool,
    bars_store: Any,
) -> Any:
    _persist_options_bars_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_options_bars_quality_checks_fn=run_options_bars_quality_checks_fn,
        quality_logger=quality_logger,
        bars_store=bars_store,
        contract_symbols=[],
        dry_run=quality_dry_run,
        skip_reason="contracts_only",
    )
    summary = BarsBackfillSummary(
        total_contracts=0,
        total_expiries=0,
        planned_contracts=0,
        skipped_contracts=0,
        ok_contracts=0,
        error_contracts=0,
        bars_rows=0,
        requests_attempted=0,
        endpoint_stats=None,
    )
    return _build_ingest_options_bars_result(
        result_factory=result_factory,
        selection=selection,
        underlyings=underlyings,
        root_symbols=root_symbols,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=None,
        summary=summary,
        dry_run=dry_run,
        contracts_only=True,
        no_symbols=False,
        no_contracts=False,
        no_eligible_contracts=False,
        contracts_endpoint_stats=discovery.endpoint_stats,
        bars_endpoint_stats=None,
    )


def _no_eligible_contracts_result(
    *,
    selection: SymbolSelection,
    result_factory: Callable[..., Any],
    underlyings: list[str],
    root_symbols: list[str],
    limited_underlyings: bool,
    discovery: ContractDiscoveryOutput,
    prepared: PreparedContracts,
    contracts_only: bool,
    dry_run: bool,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    quality_dry_run: bool,
    bars_store: Any,
) -> Any:
    _persist_options_bars_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_options_bars_quality_checks_fn=run_options_bars_quality_checks_fn,
        quality_logger=quality_logger,
        bars_store=bars_store,
        contract_symbols=[],
        dry_run=quality_dry_run,
        skip_reason="no_eligible_contracts",
    )
    return _build_ingest_options_bars_result(
        result_factory=result_factory,
        selection=selection,
        underlyings=underlyings,
        root_symbols=root_symbols,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=prepared,
        summary=None,
        dry_run=dry_run,
        contracts_only=contracts_only,
        no_symbols=False,
        no_contracts=False,
        no_eligible_contracts=True,
    )


def _run_backfill_and_build_result(
    *,
    selection: SymbolSelection,
    result_factory: Callable[..., Any],
    underlyings: list[str],
    root_symbols: list[str],
    limited_underlyings: bool,
    discovery: ContractDiscoveryOutput,
    prepared: PreparedContracts,
    contracts_only: bool,
    dry_run: bool,
    quality_dry_run: bool,
    bars_store: Any,
    quality_logger: Any | None,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    backfill_option_bars_fn: Callable[..., BarsBackfillSummary],
    client: Any,
    lookback_years: int,
    bars_concurrency: int,
    bars_max_requests_per_second: float | None,
    bars_batch_mode: str,
    bars_batch_size: int,
    bars_write_batch_size: int,
    effective_resume: bool,
    fail_fast: bool,
    run_day: date,
) -> Any:
    summary = backfill_option_bars_fn(
        client,
        bars_store,
        prepared.contracts,
        provider="alpaca",
        lookback_years=lookback_years,
        page_limit=None,
        bars_concurrency=bars_concurrency,
        bars_max_requests_per_second=bars_max_requests_per_second,
        bars_batch_mode=bars_batch_mode,
        bars_batch_size=bars_batch_size,
        bars_write_batch_size=bars_write_batch_size,
        resume=effective_resume,
        dry_run=dry_run,
        fail_fast=fail_fast,
        today=run_day,
    )
    contract_symbols = _prepared_contract_symbols(prepared)
    _persist_options_bars_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_options_bars_quality_checks_fn=run_options_bars_quality_checks_fn,
        quality_logger=quality_logger,
        bars_store=bars_store,
        contract_symbols=contract_symbols,
        dry_run=quality_dry_run,
    )
    return _build_ingest_options_bars_result(
        result_factory=result_factory,
        selection=selection,
        underlyings=underlyings,
        root_symbols=root_symbols,
        limited_underlyings=limited_underlyings,
        discovery=discovery,
        prepared=prepared,
        summary=summary,
        dry_run=dry_run,
        contracts_only=contracts_only,
        no_symbols=False,
        no_contracts=False,
        no_eligible_contracts=False,
        contracts_endpoint_stats=discovery.endpoint_stats,
        bars_endpoint_stats=summary.endpoint_stats if summary is not None else None,
    )


@dataclass(frozen=True)
class _OptionsBarsRunState:
    selection: SymbolSelection
    quality_logger: Any | None
    quality_dry_run: bool
    effective_resume: bool
    root_symbols: list[str]
    underlyings: list[str]
    limited_underlyings: bool
    run_day: date
    exp_start: date
    exp_end: date
    contract_status: str
    contracts_store: Any
    bars_store: Any
    client: Any


def _prepare_options_bars_run_state(params: dict[str, Any]) -> tuple[_OptionsBarsRunState | None, Any | None]:
    dry_run = bool(params["dry_run"])
    fetch_only = bool(params["fetch_only"])
    parameter_error_factory = params["parameter_error_factory"]
    if dry_run and fetch_only:
        raise parameter_error_factory("fetch-only and dry-run are mutually exclusive.")

    quality_logger = params["resolve_quality_run_logger_fn"](params["run_logger"])
    quality_dry_run = dry_run or fetch_only
    effective_resume = bool(params["resume"]) and not fetch_only
    root_symbols = _normalize_root_symbols(params["contracts_root_symbols"])
    selection = _resolve_symbol_selection(
        watchlists_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        symbol=params["symbol"],
        root_symbols=root_symbols,
    )
    if not selection.symbols and not root_symbols:
        return None, _no_symbols_result(
            selection=selection,
            result_factory=params["result_factory"],
            contracts_only=bool(params["contracts_only"]),
            dry_run=dry_run,
            persist_quality_results_fn=params["persist_quality_results_fn"],
            run_options_bars_quality_checks_fn=params["run_options_bars_quality_checks_fn"],
            quality_logger=quality_logger,
            quality_dry_run=quality_dry_run,
        )

    underlyings, limited_underlyings = _limit_underlyings(selection, params["max_underlyings"])
    provider = params["provider_builder"]()
    if getattr(provider, "name", None) != "alpaca":
        raise parameter_error_factory("Options bars ingestion requires --provider alpaca.")
    run_day = params["today"] or date.today()
    exp_start, exp_end = _resolve_expiry_window(
        contracts_exp_start=params["contracts_exp_start"],
        contracts_exp_end=params["contracts_exp_end"],
        run_day=run_day,
        parameter_error_factory=parameter_error_factory,
    )
    contract_status = _resolve_contract_status(
        params["contracts_status"],
        parameter_error_factory=parameter_error_factory,
    )
    contracts_store = (
        params["contracts_store_builder"](params["contracts_store_dir"])
        if not dry_run and not fetch_only
        else None
    )
    bars_store = _bars_store_for_mode(
        fetch_only=fetch_only,
        bars_store_builder=params["bars_store_builder"],
        bars_store_dir=params["bars_store_dir"],
        fetch_only_store_factory=params["fetch_only_store_factory"],
    )
    client = params["client_factory"]()
    return (
        _OptionsBarsRunState(
            selection=selection,
            quality_logger=quality_logger,
            quality_dry_run=quality_dry_run,
            effective_resume=effective_resume,
            root_symbols=root_symbols,
            underlyings=underlyings,
            limited_underlyings=limited_underlyings,
            run_day=run_day,
            exp_start=exp_start,
            exp_end=exp_end,
            contract_status=contract_status,
            contracts_store=contracts_store,
            bars_store=bars_store,
            client=client,
        ),
        None,
    )


def _discover_options_bars_contracts(
    state: _OptionsBarsRunState,
    params: dict[str, Any],
) -> ContractDiscoveryOutput:
    return params["discover_option_contracts_fn"](
        state.client,
        underlyings=state.underlyings,
        root_symbols=state.root_symbols or None,
        exp_start=state.exp_start,
        exp_end=state.exp_end,
        contract_symbol_prefix=params["contract_symbol_prefix"],
        limit=params["contracts_page_size"],
        page_limit=params["page_limit"],
        max_contracts=params["max_contracts"],
        max_requests_per_second=params["contracts_max_requests_per_second"],
        contract_status=state.contract_status,
        fail_fast=params["fail_fast"],
    )


def _prepare_discovery_result(
    state: _OptionsBarsRunState,
    params: dict[str, Any],
    discovery: ContractDiscoveryOutput,
) -> tuple[PreparedContracts | None, Any | None]:
    if discovery.contracts.empty:
        return None, _no_contracts_result(
            selection=state.selection,
            result_factory=params["result_factory"],
            underlyings=state.underlyings,
            root_symbols=state.root_symbols,
            limited_underlyings=state.limited_underlyings,
            discovery=discovery,
            contracts_only=bool(params["contracts_only"]),
            dry_run=bool(params["dry_run"]),
            persist_quality_results_fn=params["persist_quality_results_fn"],
            run_options_bars_quality_checks_fn=params["run_options_bars_quality_checks_fn"],
            quality_logger=state.quality_logger,
            quality_dry_run=state.quality_dry_run,
            bars_store=state.bars_store,
        )
    _persist_discovered_contracts(
        contracts_store=state.contracts_store,
        discovery=discovery,
        run_day=state.run_day,
        contracts_page_size=params["contracts_page_size"],
    )
    if bool(params["contracts_only"]):
        return None, _contracts_only_result(
            selection=state.selection,
            result_factory=params["result_factory"],
            underlyings=state.underlyings,
            root_symbols=state.root_symbols,
            limited_underlyings=state.limited_underlyings,
            discovery=discovery,
            dry_run=bool(params["dry_run"]),
            persist_quality_results_fn=params["persist_quality_results_fn"],
            run_options_bars_quality_checks_fn=params["run_options_bars_quality_checks_fn"],
            quality_logger=state.quality_logger,
            quality_dry_run=state.quality_dry_run,
            bars_store=state.bars_store,
        )
    prepared = params["prepare_contracts_for_bars_fn"](
        discovery.contracts,
        max_expiries=params["max_expiries"],
        max_contracts=params["max_contracts"],
    )
    if prepared.contracts.empty:
        return None, _no_eligible_contracts_result(
            selection=state.selection,
            result_factory=params["result_factory"],
            underlyings=state.underlyings,
            root_symbols=state.root_symbols,
            limited_underlyings=state.limited_underlyings,
            discovery=discovery,
            prepared=prepared,
            contracts_only=bool(params["contracts_only"]),
            dry_run=bool(params["dry_run"]),
            persist_quality_results_fn=params["persist_quality_results_fn"],
            run_options_bars_quality_checks_fn=params["run_options_bars_quality_checks_fn"],
            quality_logger=state.quality_logger,
            quality_dry_run=state.quality_dry_run,
            bars_store=state.bars_store,
        )
    return prepared, None


def _run_options_bars_with_state(state: _OptionsBarsRunState, params: dict[str, Any]) -> Any:
    discovery = _discover_options_bars_contracts(state, params)
    prepared, early_result = _prepare_discovery_result(state, params, discovery)
    if early_result is not None:
        return early_result
    if prepared is None:
        raise RuntimeError("prepared contracts missing for options bars run")
    return _run_backfill_and_build_result(
        selection=state.selection,
        result_factory=params["result_factory"],
        underlyings=state.underlyings,
        root_symbols=state.root_symbols,
        limited_underlyings=state.limited_underlyings,
        discovery=discovery,
        prepared=prepared,
        contracts_only=bool(params["contracts_only"]),
        dry_run=bool(params["dry_run"]),
        quality_dry_run=state.quality_dry_run,
        bars_store=state.bars_store,
        quality_logger=state.quality_logger,
        persist_quality_results_fn=params["persist_quality_results_fn"],
        run_options_bars_quality_checks_fn=params["run_options_bars_quality_checks_fn"],
        backfill_option_bars_fn=params["backfill_option_bars_fn"],
        client=state.client,
        lookback_years=params["lookback_years"],
        bars_concurrency=params["bars_concurrency"],
        bars_max_requests_per_second=params["bars_max_requests_per_second"],
        bars_batch_mode=params["bars_batch_mode"],
        bars_batch_size=params["bars_batch_size"],
        bars_write_batch_size=params["bars_write_batch_size"],
        effective_resume=state.effective_resume,
        fail_fast=params["fail_fast"],
        run_day=state.run_day,
    )


def run_ingest_options_bars_job_impl(
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
    resolve_quality_run_logger_fn: Callable[[Any | None], Any | None],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_options_bars_quality_checks_fn: Callable[..., list[Any]],
    parameter_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
    fetch_only_store_factory: Callable[[], Any],
    discover_option_contracts_fn: Callable[..., ContractDiscoveryOutput] = discover_option_contracts,
    prepare_contracts_for_bars_fn: Callable[..., PreparedContracts] = prepare_contracts_for_bars,
    backfill_option_bars_fn: Callable[..., BarsBackfillSummary] = backfill_option_bars,
) -> Any:
    params = dict(locals())
    state, early_result = _prepare_options_bars_run_state(params)
    if early_result is not None:
        return early_result
    if state is None:
        raise RuntimeError("options bars run state missing")
    return _run_options_bars_with_state(state, params)
