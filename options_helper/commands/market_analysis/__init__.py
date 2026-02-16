from __future__ import annotations

import pandas as pd
import typer

import options_helper.cli_deps as cli_deps
from options_helper.commands.market_analysis.core import exposure, iv_surface, levels, register as register_core, tail_risk
from options_helper.commands.market_analysis.zero_dte import (
    register as register_zero_dte,
    zero_dte_put_forward_snapshot,
    zero_dte_put_study,
)
from options_helper.commands.market_analysis.zero_dte_candidates import _assemble_zero_dte_candidate_rows
from options_helper.commands.market_analysis.zero_dte_output import _upsert_forward_snapshot_records
from options_helper.commands.market_analysis.zero_dte_types import _ZeroDTEForwardResult, _ZeroDTEStudyResult
from options_helper.commands.market_analysis.zero_dte_workflows import (
    _run_zero_dte_forward_snapshot_workflow,
    _run_zero_dte_put_study_workflow,
)
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.storage_runtime import get_storage_runtime_config
from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import ZeroDteDisclaimerMetadata

app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")
register_core(app)
register_zero_dte(app)


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
