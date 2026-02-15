from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import options_helper.cli_deps as cli_deps
from options_helper.data.ingestion.candles import CandleIngestOutput, ingest_candles_with_summary
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS, resolve_symbols


def run_ingest_candles_job_impl(
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
    resolve_quality_run_logger_fn: Callable[[Any | None], Any | None],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_candle_quality_checks_fn: Callable[..., list[Any]],
    result_factory: Callable[..., Any],
) -> Any:
    quality_logger = resolve_quality_run_logger_fn(run_logger)
    selection = resolve_symbols(
        watchlists_path=watchlists_path,
        watchlists=watchlist,
        symbols=symbol,
        default_watchlists=DEFAULT_WATCHLISTS,
    )

    if not selection.symbols:
        persist_quality_results_fn(
            quality_logger,
            run_candle_quality_checks_fn(candle_store=None, symbols=[], skip_reason="no_symbols"),
        )
        return result_factory(
            warnings=list(selection.warnings),
            symbols=[],
            results=[],
            no_symbols=True,
            endpoint_stats=None,
        )

    provider = provider_builder()
    store = candle_store_builder(candle_cache_dir, provider=provider)
    output: CandleIngestOutput = ingest_candles_with_summary(
        store,
        selection.symbols,
        period="max",
        best_effort=True,
        concurrency=max(1, int(candles_concurrency)),
        max_requests_per_second=candles_max_requests_per_second,
    )
    persist_quality_results_fn(
        quality_logger,
        run_candle_quality_checks_fn(candle_store=store, symbols=selection.symbols),
    )
    return result_factory(
        warnings=list(selection.warnings),
        symbols=list(selection.symbols),
        results=output.results,
        no_symbols=False,
        endpoint_stats=output.summary.endpoint_stats,
    )

