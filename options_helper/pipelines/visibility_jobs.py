from __future__ import annotations

from typing import Any

from options_helper.pipelines import visibility_jobs_legacy as _legacy

VisibilityJobParameterError = _legacy.VisibilityJobParameterError
VisibilityJobExecutionError = _legacy.VisibilityJobExecutionError
IngestCandlesJobResult = _legacy.IngestCandlesJobResult
IngestOptionsBarsJobResult = _legacy.IngestOptionsBarsJobResult
SnapshotOptionsJobResult = _legacy.SnapshotOptionsJobResult
FlowReportJobResult = _legacy.FlowReportJobResult
DerivedUpdateJobResult = _legacy.DerivedUpdateJobResult
BriefingJobResult = _legacy.BriefingJobResult
DashboardJobResult = _legacy.DashboardJobResult
_FetchOnlyOptionBarsStore = _legacy._FetchOnlyOptionBarsStore

# Compatibility seams expected by tests that monkeypatch this module.
discover_option_contracts = _legacy.discover_option_contracts
prepare_contracts_for_bars = _legacy.prepare_contracts_for_bars
backfill_option_bars = _legacy.backfill_option_bars
run_candle_quality_checks = _legacy.run_candle_quality_checks
run_options_bars_quality_checks = _legacy.run_options_bars_quality_checks
run_snapshot_quality_checks = _legacy.run_snapshot_quality_checks
run_flow_quality_checks = _legacy.run_flow_quality_checks
run_derived_quality_checks = _legacy.run_derived_quality_checks
persist_quality_checks = _legacy.persist_quality_checks


def _sync_legacy_seams() -> None:
    import options_helper.pipelines.visibility_jobs as jobs_pkg

    _legacy.discover_option_contracts = jobs_pkg.discover_option_contracts
    _legacy.prepare_contracts_for_bars = jobs_pkg.prepare_contracts_for_bars
    _legacy.backfill_option_bars = jobs_pkg.backfill_option_bars
    _legacy.run_candle_quality_checks = jobs_pkg.run_candle_quality_checks
    _legacy.run_options_bars_quality_checks = jobs_pkg.run_options_bars_quality_checks
    _legacy.run_snapshot_quality_checks = jobs_pkg.run_snapshot_quality_checks
    _legacy.run_flow_quality_checks = jobs_pkg.run_flow_quality_checks
    _legacy.run_derived_quality_checks = jobs_pkg.run_derived_quality_checks
    _legacy.persist_quality_checks = jobs_pkg.persist_quality_checks


def run_ingest_candles_job(*args: Any, **kwargs: Any) -> IngestCandlesJobResult:
    _sync_legacy_seams()
    return _legacy.run_ingest_candles_job(*args, **kwargs)


def run_ingest_options_bars_job(*args: Any, **kwargs: Any) -> IngestOptionsBarsJobResult:
    _sync_legacy_seams()
    return _legacy.run_ingest_options_bars_job(*args, **kwargs)


def run_snapshot_options_job(*args: Any, **kwargs: Any) -> SnapshotOptionsJobResult:
    _sync_legacy_seams()
    return _legacy.run_snapshot_options_job(*args, **kwargs)


def run_flow_report_job(*args: Any, **kwargs: Any) -> FlowReportJobResult:
    _sync_legacy_seams()
    return _legacy.run_flow_report_job(*args, **kwargs)


def run_derived_update_job(*args: Any, **kwargs: Any) -> DerivedUpdateJobResult:
    _sync_legacy_seams()
    return _legacy.run_derived_update_job(*args, **kwargs)


def run_briefing_job(*args: Any, **kwargs: Any) -> BriefingJobResult:
    _sync_legacy_seams()
    return _legacy.run_briefing_job(*args, **kwargs)


def run_dashboard_job(*args: Any, **kwargs: Any) -> DashboardJobResult:
    _sync_legacy_seams()
    return _legacy.run_dashboard_job(*args, **kwargs)


def render_dashboard_report(*args: Any, **kwargs: Any) -> Any:
    _sync_legacy_seams()
    return _legacy.render_dashboard_report(*args, **kwargs)


__all__ = [
    "VisibilityJobParameterError",
    "VisibilityJobExecutionError",
    "IngestCandlesJobResult",
    "IngestOptionsBarsJobResult",
    "SnapshotOptionsJobResult",
    "FlowReportJobResult",
    "DerivedUpdateJobResult",
    "BriefingJobResult",
    "DashboardJobResult",
    "_FetchOnlyOptionBarsStore",
    "discover_option_contracts",
    "prepare_contracts_for_bars",
    "backfill_option_bars",
    "run_candle_quality_checks",
    "run_options_bars_quality_checks",
    "run_snapshot_quality_checks",
    "run_flow_quality_checks",
    "run_derived_quality_checks",
    "persist_quality_checks",
    "run_ingest_candles_job",
    "run_ingest_options_bars_job",
    "run_snapshot_options_job",
    "run_flow_report_job",
    "run_derived_update_job",
    "run_briefing_job",
    "run_dashboard_job",
    "render_dashboard_report",
]
