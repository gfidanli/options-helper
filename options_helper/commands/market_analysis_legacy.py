from __future__ import annotations

import importlib

import pandas as pd
import typer

import options_helper.cli_deps as cli_deps
from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import ZeroDteDisclaimerMetadata


app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")


def _core_impl_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.core_impl")


def _core_helpers_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.core_helpers")


def _core_io_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.core_io")


def _zero_dte_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte")


def _zero_dte_candidates_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_candidates")


def _zero_dte_output_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_output")


def _zero_dte_parsing_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_parsing")


def _zero_dte_serialization_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_serialization")


def _zero_dte_types_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_types")


def _zero_dte_utils_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_utils")


def _zero_dte_workflows_module() -> object:
    return importlib.import_module("options_helper.commands.market_analysis.zero_dte_workflows")


tail_risk = _core_impl_module().tail_risk
iv_surface = _core_impl_module().iv_surface
exposure = _core_impl_module().exposure
levels = _core_impl_module().levels

app.command("tail-risk")(tail_risk)
app.command("iv-surface")(iv_surface)
app.command("exposure")(exposure)
app.command("levels")(levels)

_normalize_symbol = _core_helpers_module()._normalize_symbol
_normalize_output_format = _core_helpers_module()._normalize_output_format
_dedupe = _core_helpers_module()._dedupe
_as_float = _core_helpers_module()._as_float
_normalize_daily_history = _core_io_module()._normalize_daily_history

_zero_dte = _zero_dte_module()
_zero_dte_candidates = _zero_dte_candidates_module()
_zero_dte_output = _zero_dte_output_module()
_zero_dte_parsing = _zero_dte_parsing_module()
_zero_dte_serialization = _zero_dte_serialization_module()
_zero_dte_types = _zero_dte_types_module()
_zero_dte_utils = _zero_dte_utils_module()
_zero_dte_workflows = _zero_dte_workflows_module()

zero_dte_put_study = _zero_dte.zero_dte_put_study
zero_dte_put_forward_snapshot = _zero_dte.zero_dte_put_forward_snapshot

_build_zero_dte_candidates = _zero_dte_candidates._build_zero_dte_candidates
_resolve_zero_dte_study_range = _zero_dte_candidates._resolve_zero_dte_study_range
_resolve_latest_intraday_session = _zero_dte_candidates._resolve_latest_intraday_session
_resolve_decision_times_for_session = _zero_dte_candidates._resolve_decision_times_for_session
_resolve_previous_close = _zero_dte_candidates._resolve_previous_close
_build_strike_snapshot_rows = _zero_dte_candidates._build_strike_snapshot_rows
_assemble_zero_dte_candidate_rows = _zero_dte_candidates._assemble_zero_dte_candidate_rows
_expand_risk_tiers = _zero_dte_candidates._expand_risk_tiers
_apply_fill_model_to_premium = _zero_dte_candidates._apply_fill_model_to_premium

_upsert_forward_snapshot_records = _zero_dte_output._upsert_forward_snapshot_records
_read_jsonl_records = _zero_dte_output._read_jsonl_records
_row_key = _zero_dte_output._row_key
_save_zero_dte_study_artifact = _zero_dte_output._save_zero_dte_study_artifact
_save_zero_dte_active_model = _zero_dte_output._save_zero_dte_active_model
_default_zero_dte_active_model_path = _zero_dte_output._default_zero_dte_active_model_path
_default_zero_dte_forward_snapshot_path = _zero_dte_output._default_zero_dte_forward_snapshot_path
_render_zero_dte_study_console = _zero_dte_output._render_zero_dte_study_console
_render_zero_dte_forward_console = _zero_dte_output._render_zero_dte_forward_console

_parse_decision_mode = _zero_dte_parsing._parse_decision_mode
_parse_fill_model = _zero_dte_parsing._parse_fill_model
_parse_time_csv = _zero_dte_parsing._parse_time_csv
_parse_positive_probability_csv = _zero_dte_parsing._parse_positive_probability_csv
_parse_strike_return_csv = _zero_dte_parsing._parse_strike_return_csv
_parse_float_csv = _zero_dte_parsing._parse_float_csv

_build_forward_snapshot_rows = _zero_dte_serialization._build_forward_snapshot_rows
_build_zero_dte_anchor = _zero_dte_serialization._build_zero_dte_anchor
_build_zero_dte_probability_rows = _zero_dte_serialization._build_zero_dte_probability_rows
_build_zero_dte_simulation_rows = _zero_dte_serialization._build_zero_dte_simulation_rows
_build_zero_dte_strike_ladder_rows = _zero_dte_serialization._build_zero_dte_strike_ladder_rows
_deserialize_tail_model = _zero_dte_serialization._deserialize_tail_model
_fit_and_serialize_active_model = _zero_dte_serialization._fit_and_serialize_active_model
_serialize_tail_model_payload = _zero_dte_serialization._serialize_tail_model_payload

_ZERO_DTE_DEFAULT_STRIKE_GRID = _zero_dte_types._ZERO_DTE_DEFAULT_STRIKE_GRID
_ZERO_DTE_FORWARD_KEY_FIELDS = _zero_dte_types._ZERO_DTE_FORWARD_KEY_FIELDS
_ZeroDTEForwardResult = _zero_dte_types._ZeroDTEForwardResult
_ZeroDTEStudyResult = _zero_dte_types._ZeroDTEStudyResult

_as_clean_text = _zero_dte_utils._as_clean_text
_coerce_quote_quality_status = _zero_dte_utils._coerce_quote_quality_status
_coerce_skip_reason = _zero_dte_utils._coerce_skip_reason
_first_text = _zero_dte_utils._first_text
_frame_records = _zero_dte_utils._frame_records
_hash_zero_dte_assumptions = _zero_dte_utils._hash_zero_dte_assumptions
_resolve_as_of_date = _zero_dte_utils._resolve_as_of_date
_timestamp_to_iso = _zero_dte_utils._timestamp_to_iso

_build_intraday_store = _zero_dte_workflows._build_intraday_store
_run_zero_dte_forward_snapshot_workflow = _zero_dte_workflows._run_zero_dte_forward_snapshot_workflow
_run_zero_dte_put_study_workflow = _zero_dte_workflows._run_zero_dte_put_study_workflow

# Legacy command-name compatibility (current package command implementation lives in
# options_helper.commands.market_analysis.zero_dte).
app.command("zero-dte-put-study")(zero_dte_put_study)
app.command("zero-dte-put-forward-snapshot")(zero_dte_put_forward_snapshot)


__all__ = [
    "app",
    "cli_deps",
    "pd",
    "clean_nan",
    "ZeroDteDisclaimerMetadata",
    "tail_risk",
    "iv_surface",
    "exposure",
    "levels",
    "zero_dte_put_study",
    "zero_dte_put_forward_snapshot",
    "_normalize_symbol",
    "_normalize_output_format",
    "_dedupe",
    "_as_float",
    "_normalize_daily_history",
    "_ZERO_DTE_DEFAULT_STRIKE_GRID",
    "_ZERO_DTE_FORWARD_KEY_FIELDS",
    "_ZeroDTEStudyResult",
    "_ZeroDTEForwardResult",
    "_build_intraday_store",
    "_run_zero_dte_put_study_workflow",
    "_run_zero_dte_forward_snapshot_workflow",
    "_build_zero_dte_candidates",
    "_resolve_zero_dte_study_range",
    "_resolve_latest_intraday_session",
    "_resolve_decision_times_for_session",
    "_resolve_previous_close",
    "_build_strike_snapshot_rows",
    "_assemble_zero_dte_candidate_rows",
    "_expand_risk_tiers",
    "_apply_fill_model_to_premium",
    "_build_zero_dte_probability_rows",
    "_build_zero_dte_strike_ladder_rows",
    "_build_zero_dte_simulation_rows",
    "_build_zero_dte_anchor",
    "_fit_and_serialize_active_model",
    "_serialize_tail_model_payload",
    "_deserialize_tail_model",
    "_build_forward_snapshot_rows",
    "_upsert_forward_snapshot_records",
    "_read_jsonl_records",
    "_row_key",
    "_save_zero_dte_study_artifact",
    "_save_zero_dte_active_model",
    "_default_zero_dte_active_model_path",
    "_default_zero_dte_forward_snapshot_path",
    "_resolve_as_of_date",
    "_render_zero_dte_study_console",
    "_render_zero_dte_forward_console",
    "_parse_decision_mode",
    "_parse_fill_model",
    "_parse_time_csv",
    "_parse_positive_probability_csv",
    "_parse_strike_return_csv",
    "_parse_float_csv",
    "_coerce_quote_quality_status",
    "_coerce_skip_reason",
    "_hash_zero_dte_assumptions",
    "_as_clean_text",
    "_first_text",
    "_timestamp_to_iso",
    "_frame_records",
]
