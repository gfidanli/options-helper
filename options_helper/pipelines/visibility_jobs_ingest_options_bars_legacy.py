from __future__ import annotations

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
    quality_logger = resolve_quality_run_logger_fn(run_logger)
    if dry_run and fetch_only:
        raise parameter_error_factory("fetch-only and dry-run are mutually exclusive.")

    quality_dry_run = dry_run or fetch_only
    effective_resume = resume and not fetch_only
    root_symbols = [
        str(sym).strip().upper()
        for sym in (contracts_root_symbols or [])
        if str(sym or "").strip()
    ]

    try:
        selection = resolve_symbols(
            watchlists_path=watchlists_path,
            watchlists=watchlist,
            symbols=symbol,
            default_watchlists=DEFAULT_WATCHLISTS,
        )
    except Exception as exc:  # noqa: BLE001
        if root_symbols:
            selection = SymbolSelection(
                symbols=[],
                watchlists_used=[],
                warnings=[f"Watchlists unavailable: {exc}"],
            )
        else:
            raise

    if not selection.symbols and not root_symbols:
        persist_quality_results_fn(
            quality_logger,
            run_options_bars_quality_checks_fn(
                bars_store=None,
                contract_symbols=[],
                dry_run=quality_dry_run,
                skip_reason="no_symbols",
            ),
        )
        return result_factory(
            warnings=list(selection.warnings),
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

    underlyings = list(selection.symbols)
    limited_underlyings = False
    if max_underlyings is not None:
        underlyings = underlyings[:max_underlyings]
        limited_underlyings = True

    provider = provider_builder()
    provider_name = getattr(provider, "name", None)
    if provider_name != "alpaca":
        raise parameter_error_factory("Options bars ingestion requires --provider alpaca.")

    run_day = today or date.today()
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
    contract_status = str(contracts_status or "").strip().lower()
    if contract_status not in {"active", "inactive", "all"}:
        raise parameter_error_factory(
            "contracts-status must be one of: active, inactive, all",
            param_hint="--contracts-status",
        )

    contracts_store = contracts_store_builder(contracts_store_dir) if not dry_run and not fetch_only else None
    bars_store: Any
    if fetch_only:
        bars_store = fetch_only_store_factory()
    else:
        bars_store = bars_store_builder(bars_store_dir)

    client = client_factory()
    discovery = discover_option_contracts_fn(
        client,
        underlyings=underlyings,
        root_symbols=root_symbols or None,
        exp_start=exp_start,
        exp_end=exp_end,
        contract_symbol_prefix=contract_symbol_prefix,
        limit=contracts_page_size,
        page_limit=page_limit,
        max_contracts=max_contracts,
        max_requests_per_second=contracts_max_requests_per_second,
        contract_status=contract_status,
        fail_fast=fail_fast,
    )

    if discovery.contracts.empty:
        persist_quality_results_fn(
            quality_logger,
            run_options_bars_quality_checks_fn(
                bars_store=bars_store,
                contract_symbols=[],
                dry_run=quality_dry_run,
                skip_reason="no_contracts",
            ),
        )
        return result_factory(
            warnings=list(selection.warnings),
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

    if contracts_store is not None:
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

    if contracts_only:
        persist_quality_results_fn(
            quality_logger,
            run_options_bars_quality_checks_fn(
                bars_store=bars_store,
                contract_symbols=[],
                dry_run=quality_dry_run,
                skip_reason="contracts_only",
            ),
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
        return result_factory(
            warnings=list(selection.warnings),
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

    prepared = prepare_contracts_for_bars_fn(
        discovery.contracts,
        max_expiries=max_expiries,
        max_contracts=max_contracts,
    )

    if prepared.contracts.empty:
        persist_quality_results_fn(
            quality_logger,
            run_options_bars_quality_checks_fn(
                bars_store=bars_store,
                contract_symbols=[],
                dry_run=quality_dry_run,
                skip_reason="no_eligible_contracts",
            ),
        )
        return result_factory(
            warnings=list(selection.warnings),
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

    contract_symbols: list[str] = []
    if "contractSymbol" in prepared.contracts.columns:
        contract_symbols = [
            str(value).strip().upper()
            for value in prepared.contracts["contractSymbol"].tolist()
            if str(value or "").strip()
        ]
    persist_quality_results_fn(
        quality_logger,
        run_options_bars_quality_checks_fn(
            bars_store=bars_store,
            contract_symbols=contract_symbols,
            dry_run=quality_dry_run,
        ),
    )

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
        no_symbols=False,
        no_contracts=False,
        no_eligible_contracts=False,
        contracts_endpoint_stats=discovery.endpoint_stats,
        bars_endpoint_stats=summary.endpoint_stats if summary is not None else None,
    )
