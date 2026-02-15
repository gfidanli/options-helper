from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.commands.common import _spot_from_meta
from options_helper.ui.dashboard import load_briefing_artifact, render_dashboard, resolve_briefing_paths


def run_derived_update_job_impl(
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
    resolve_quality_run_logger_fn: Callable[[Any | None], Any | None],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_derived_quality_checks_fn: Callable[..., list[Any]],
    active_snapshot_store_fn: Callable[[Any], Any],
    filesystem_compatible_derived_store_fn: Callable[[Path, Any], Any],
    filesystem_compatible_candle_store_fn: Callable[[Path, Any], Any],
    execution_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
) -> Any:
    quality_logger = resolve_quality_run_logger_fn(run_logger)
    store = active_snapshot_store_fn(snapshot_store_builder(cache_dir))
    derived = filesystem_compatible_derived_store_fn(derived_dir, derived_store_builder(derived_dir))
    candle_store = filesystem_compatible_candle_store_fn(candle_cache_dir, candle_store_builder(candle_cache_dir))

    as_of_date = store.resolve_date(symbol, as_of)
    df = store.load_day(symbol, as_of_date)
    meta = store.load_meta(symbol, as_of_date)
    spot = _spot_from_meta(meta)
    if spot is None:
        raise execution_error_factory("missing spot price in meta.json (run snapshot-options first)")

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
    persist_quality_results_fn(
        quality_logger,
        run_derived_quality_checks_fn(derived_store=derived, symbol=symbol.upper()),
    )
    return result_factory(symbol=symbol.upper(), as_of_date=as_of_date, output_path=out_path)


def run_dashboard_job_impl(
    *,
    report_date: str,
    reports_dir: Path,
    execution_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
) -> Any:
    try:
        paths = resolve_briefing_paths(reports_dir, report_date)
    except Exception as exc:  # noqa: BLE001
        raise execution_error_factory(str(exc)) from exc

    try:
        artifact = load_briefing_artifact(paths.json_path)
    except Exception as exc:  # noqa: BLE001
        raise execution_error_factory(f"failed to load briefing JSON: {exc}") from exc

    return result_factory(json_path=paths.json_path, artifact=artifact)


def render_dashboard_report_impl(
    *,
    result: Any,
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

