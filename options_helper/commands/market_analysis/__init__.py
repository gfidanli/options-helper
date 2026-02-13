from __future__ import annotations

import typer

from options_helper.commands import market_analysis_legacy as _legacy
from options_helper.commands.market_analysis.core import exposure, iv_surface, levels, register as register_core, tail_risk
from options_helper.commands.market_analysis.zero_dte import (
    register as register_zero_dte,
    zero_dte_put_forward_snapshot,
    zero_dte_put_study,
)
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.storage_runtime import get_storage_runtime_config

app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")
register_core(app)
register_zero_dte(app)

# Compatibility seams expected by tests and callers that monkeypatch module globals.
cli_deps = _legacy.cli_deps
pd = _legacy.pd
clean_nan = _legacy.clean_nan
ZeroDteDisclaimerMetadata = _legacy.ZeroDteDisclaimerMetadata
_ZeroDTEStudyResult = _legacy._ZeroDTEStudyResult
_ZeroDTEForwardResult = _legacy._ZeroDTEForwardResult
_run_zero_dte_put_study_workflow = _legacy._run_zero_dte_put_study_workflow
_run_zero_dte_forward_snapshot_workflow = _legacy._run_zero_dte_forward_snapshot_workflow
_assemble_zero_dte_candidate_rows = _legacy._assemble_zero_dte_candidate_rows
_upsert_forward_snapshot_records = _legacy._upsert_forward_snapshot_records

__all__ = [
    "app",
    "cli_deps",
    "pd",
    "clean_nan",
    "get_storage_runtime_config",
    "IntradayStore",
    "ZeroDteDisclaimerMetadata",
    "_ZeroDTEStudyResult",
    "_ZeroDTEForwardResult",
    "_run_zero_dte_put_study_workflow",
    "_run_zero_dte_forward_snapshot_workflow",
    "_assemble_zero_dte_candidate_rows",
    "_upsert_forward_snapshot_records",
    "tail_risk",
    "iv_surface",
    "exposure",
    "levels",
    "zero_dte_put_study",
    "zero_dte_put_forward_snapshot",
]
