# ruff: noqa: F401
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.exposure import ExposureSlice, compute_exposure_slices
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow
from options_helper.analysis.iv_surface import compute_iv_surface
from options_helper.analysis.levels import compute_anchored_vwap, compute_levels_summary, compute_volume_profile
from options_helper.analysis.scenarios import compute_position_scenarios
from options_helper.commands.common import _spot_from_meta
from options_helper.commands.position_metrics import _mark_price
from options_helper.data.candles import close_asof, last_close
from options_helper.data.derived import DERIVED_COLUMNS
from options_helper.data.earnings import safe_next_earnings_date as _safe_next_earnings_date
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.options_snapshots import find_snapshot_row
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.models import MultiLegPosition, OptionType, Position
from options_helper.pipelines.technicals_extension_stats import run_extension_stats_for_symbol
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobExecutionError as _VisibilityJobExecutionError,
    VisibilityJobParameterError as _VisibilityJobParameterError,
    render_dashboard_report as _render_dashboard_report,
    run_briefing_job as _run_briefing_job,
    run_dashboard_job as _run_dashboard_job,
    run_flow_report_job as _run_flow_report_job,
)
from options_helper.reporting_chain import render_chain_report_markdown
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.exposure import (
    ExposureArtifact,
    ExposureSliceArtifact,
    ExposureStrikeRow,
    ExposureSummaryRow,
    ExposureTopLevelRow,
)
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.schemas.iv_surface import (
    IvSurfaceArtifact,
    IvSurfaceDeltaBucketChangeRow,
    IvSurfaceDeltaBucketRow,
    IvSurfaceTenorChangeRow,
    IvSurfaceTenorRow,
)
from options_helper.schemas.levels import (
    LevelsAnchoredVwapRow,
    LevelsArtifact,
    LevelsSummaryRow,
    LevelsVolumeProfileRow,
)
from options_helper.schemas.scenarios import ScenarioGridRow, ScenarioSummaryRow, ScenariosArtifact
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


pd: object | None = None

VisibilityJobExecutionError = _VisibilityJobExecutionError
VisibilityJobParameterError = _VisibilityJobParameterError
render_dashboard_report = _render_dashboard_report
run_briefing_job = _run_briefing_job
run_dashboard_job = _run_dashboard_job
run_flow_report_job = _run_flow_report_job
safe_next_earnings_date = _safe_next_earnings_date

JOB_COMPUTE_FLOW = "compute_flow"
JOB_BUILD_BRIEFING = "build_briefing"
JOB_BUILD_DASHBOARD = "build_dashboard"

ASSET_OPTIONS_FLOW = "options_flow"
ASSET_BRIEFING_MARKDOWN = "briefing_markdown"
ASSET_BRIEFING_JSON = "briefing_json"
ASSET_DASHBOARD = "dashboard_view"

NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)

_FLOW_HEADER_RE = re.compile(
    (
        r"^([A-Z0-9._-]+)\s+flow"
        r"(?:\s+net\s+window=\d+\s+\((\d{4}-\d{2}-\d{2})\s+→\s+(\d{4}-\d{2}-\d{2})\)"
        r"|\s+(\d{4}-\d{2}-\d{2})\s+→\s+(\d{4}-\d{2}-\d{2}))"
    )
)
_FLOW_NO_DATA_RE = re.compile(r"^No flow data for\s+([A-Z0-9._-]+):")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _load_reports_command(module_name: str, function_name: str):
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, function_name)


@dataclass(frozen=True)
class _ReportPackScenarioTarget:
    key: str
    symbol: str
    option_type: OptionType
    side: str
    expiry: date
    strike: float
    contracts: int
    basis: float | None


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_file_token(value: str) -> str:
    token = _SAFE_FILENAME_RE.sub("_", str(value or "").strip())
    return token.strip("._") or "item"


def _frame_records(frame: "pd.DataFrame | None") -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [clean_nan(row) for row in frame.to_dict(orient="records")]


def _resolve_snapshot_spot_for_report_pack(
    *,
    meta: dict[str, Any],
    candles: "pd.DataFrame",
    as_of: date,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    from_meta = _spot_from_meta(meta)
    if from_meta is not None and from_meta > 0:
        return float(from_meta), warnings

    from_candles = close_asof(candles, as_of)
    if from_candles is None:
        from_candles = last_close(candles)
    if from_candles is not None and from_candles > 0:
        return float(from_candles), warnings

    warnings.append("missing_spot")
    return 0.0, warnings


def _to_python_datetime(value: object) -> datetime | None:
    _ensure_pandas()
    assert pd is not None
    pandas = cast(Any, pd)

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pandas.Timestamp):
        if pandas.isna(value):
            return None
        return value.to_pydatetime()
    return None


def _normalize_daily_history(history: "pd.DataFrame | None") -> "pd.DataFrame":
    _ensure_pandas()
    assert pd is not None
    pandas = cast(Any, pd)

    if history is None or history.empty:
        return pandas.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = history.copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        date_col = pandas.to_datetime(frame["date"], errors="coerce")
    elif isinstance(frame.index, pandas.DatetimeIndex):
        date_col = pandas.to_datetime(frame.index, errors="coerce")
    else:
        date_col = pandas.Series([pandas.NaT] * len(frame), index=frame.index)

    out = pandas.DataFrame(index=frame.index)
    out["date"] = date_col
    out["open"] = pandas.to_numeric(frame.get("open"), errors="coerce")
    out["high"] = pandas.to_numeric(frame.get("high"), errors="coerce")
    out["low"] = pandas.to_numeric(frame.get("low"), errors="coerce")
    out["close"] = pandas.to_numeric(frame.get("close"), errors="coerce")
    out["volume"] = pandas.to_numeric(frame.get("volume"), errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _slice_history_to_as_of(history: "pd.DataFrame", as_of: date) -> "pd.DataFrame":
    _ensure_pandas()
    assert pd is not None
    pandas = cast(Any, pd)

    if history.empty:
        return history
    sliced = history[history["date"] <= pandas.Timestamp(as_of)].copy()
    if sliced.empty:
        return history.iloc[0:0].copy()
    return sliced.reset_index(drop=True)


def _build_iv_surface_artifact_for_report_pack(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    surface: Any,
    warnings: list[str],
) -> IvSurfaceArtifact:
    return IvSurfaceArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=spot if spot > 0 else None,
        disclaimer="Not financial advice.",
        tenor=[IvSurfaceTenorRow.model_validate(row) for row in _frame_records(surface.tenor)],
        delta_buckets=[IvSurfaceDeltaBucketRow.model_validate(row) for row in _frame_records(surface.delta_buckets)],
        tenor_changes=[IvSurfaceTenorChangeRow.model_validate(row) for row in _frame_records(surface.tenor_changes)],
        delta_bucket_changes=[
            IvSurfaceDeltaBucketChangeRow.model_validate(row) for row in _frame_records(surface.delta_bucket_changes)
        ],
        warnings=_dedupe_strings([*surface.warnings, *warnings]),
    )


def _build_exposure_artifact_for_report_pack(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    slices: dict[str, ExposureSlice],
    warnings: list[str],
) -> ExposureArtifact:
    slice_artifacts: list[ExposureSliceArtifact] = []
    for mode in ("near", "monthly", "all"):
        slice_data = slices[mode]
        summary = ExposureSummaryRow.model_validate(clean_nan(slice_data.summary))
        summary.warnings = _dedupe_strings([*summary.warnings, *warnings])
        slice_artifacts.append(
            ExposureSliceArtifact(
                mode=mode,
                available_expiries=list(slice_data.available_expiries),
                included_expiries=list(slice_data.included_expiries),
                strike_rows=[ExposureStrikeRow.model_validate(clean_nan(row)) for row in slice_data.strike_rows],
                summary=summary,
                top_abs_net_levels=[
                    ExposureTopLevelRow(
                        strike=float(level.strike),
                        net_gex=float(level.net_gex),
                        abs_net_gex=float(level.abs_net_gex),
                    )
                    for level in slice_data.top_abs_net_levels
                ],
            )
        )

    return ExposureArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=float(spot),
        disclaimer="Not financial advice.",
        slices=slice_artifacts,
    )


def _build_levels_artifact_for_report_pack(
    *,
    symbol: str,
    as_of: date,
    summary: Any,
    anchored: Any,
    profile: Any,
) -> LevelsArtifact:
    summary_row = LevelsSummaryRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        spot=summary.spot,
        prev_close=summary.prev_close,
        session_open=summary.session_open,
        gap_pct=summary.gap_pct,
        prior_high=summary.prior_high,
        prior_low=summary.prior_low,
        rolling_high=summary.rolling_high,
        rolling_low=summary.rolling_low,
        rs_ratio=summary.rs_ratio,
        beta_20d=summary.beta_20d,
        corr_20d=summary.corr_20d,
        warnings=list(summary.warnings),
    )
    anchor_row = LevelsAnchoredVwapRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        anchor_id="session_open",
        anchor_type="session_open",
        anchor_ts_utc=_to_python_datetime(anchored.anchor_ts_utc),
        anchor_price=anchored.anchor_price,
        anchored_vwap=anchored.anchored_vwap,
        distance_from_spot_pct=anchored.distance_from_spot_pct,
    )
    profile_rows = [
        LevelsVolumeProfileRow(
            symbol=symbol,
            as_of=as_of.isoformat(),
            price_bin_low=float(row.price_bin_low),
            price_bin_high=float(row.price_bin_high),
            volume=float(row.volume),
            volume_pct=float(row.volume_pct),
            is_poc=bool(row.is_poc),
            is_hvn=bool(row.is_hvn),
            is_lvn=bool(row.is_lvn),
        )
        for row in profile.bins
    ]
    return LevelsArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        disclaimer="Not financial advice.",
        summary=summary_row,
        anchored_vwap=[anchor_row],
        volume_profile=profile_rows,
        volume_profile_poc=profile.poc_price,
        volume_profile_hvn_candidates=[float(value) for value in profile.hvn_candidates],
        volume_profile_lvn_candidates=[float(value) for value in profile.lvn_candidates],
        warnings=_dedupe_strings([*summary.warnings, *anchored.warnings, *profile.warnings]),
    )


def _row_float(row: object, *keys: str) -> float | None:
    if row is None:
        return None
    _ensure_pandas()
    assert pd is not None
    pandas = cast(Any, pd)
    for key in keys:
        try:
            value = row.get(key)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            continue
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if pandas.isna(parsed):
            continue
        return parsed
    return None


def _row_string(row: object, *keys: str) -> str | None:
    if row is None:
        return None
    for key in keys:
        try:
            raw = str(row.get(key) or "").strip()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            continue
        if raw:
            return raw
    return None


def _fallback_contract_symbol(*, symbol: str, expiry: date, option_type: OptionType, strike: float) -> str:
    yymmdd = expiry.strftime("%y%m%d")
    cp = "C" if option_type == "call" else "P"
    strike_token = int(round(float(strike) * 1000))
    return f"{symbol.upper()}{yymmdd}{cp}{strike_token:08d}"


def _build_report_pack_scenario_targets(positions: list[MultiLegPosition | Position]) -> list[_ReportPackScenarioTarget]:
    targets: list[_ReportPackScenarioTarget] = []
    for pos in positions:
        if isinstance(pos, Position):
            targets.append(
                _ReportPackScenarioTarget(
                    key=pos.id,
                    symbol=pos.symbol.upper(),
                    option_type=pos.option_type,
                    side="long",
                    expiry=pos.expiry,
                    strike=float(pos.strike),
                    contracts=int(pos.contracts),
                    basis=float(pos.cost_basis),
                )
            )
            continue
        if not isinstance(pos, MultiLegPosition):
            continue
        for idx, leg in enumerate(pos.legs, start=1):
            targets.append(
                _ReportPackScenarioTarget(
                    key=f"{pos.id}-leg{idx}",
                    symbol=pos.symbol.upper(),
                    option_type=leg.option_type,
                    side=leg.side,
                    expiry=leg.expiry,
                    strike=float(leg.strike),
                    contracts=int(leg.contracts),
                    basis=None,
                )
            )
    return targets


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _coerce_iso_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


def _flow_renderable_statuses(renderables: list[object]) -> tuple[dict[str, date | None], set[str]]:
    success_by_symbol: dict[str, date | None] = {}
    skipped_symbols: set[str] = set()
    for renderable in renderables:
        if not isinstance(renderable, str):
            continue
        plain = _strip_rich_markup(renderable)
        no_data_match = _FLOW_NO_DATA_RE.match(plain)
        if no_data_match:
            sym = no_data_match.group(1).upper()
            if sym not in success_by_symbol:
                skipped_symbols.add(sym)
            continue

        header_match = _FLOW_HEADER_RE.match(plain)
        if not header_match:
            continue
        sym = header_match.group(1).upper()
        end_date = (
            _coerce_iso_date(header_match.group(3))
            or _coerce_iso_date(header_match.group(5))
        )
        success_by_symbol[sym] = end_date
        skipped_symbols.discard(sym)
    return success_by_symbol, skipped_symbols


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


def register(app: typer.Typer) -> None:
    app.command("flow")(flow_report)
    app.command("chain-report")(chain_report)
    app.command("compare")(compare_report)
    app.command("report-pack")(report_pack)
    app.command("briefing")(briefing)
    app.command("dashboard")(dashboard)
    app.command("roll-plan")(roll_plan)


flow_report = _load_reports_command("options_helper.commands.reports.flow", "flow_report")


chain_report = _load_reports_command("options_helper.commands.reports.chain", "chain_report")


compare_report = _load_reports_command("options_helper.commands.reports.chain", "compare_report")


report_pack = _load_reports_command("options_helper.commands.reports.pack_legacy", "report_pack")


briefing = _load_reports_command("options_helper.commands.reports.daily", "briefing")


dashboard = _load_reports_command("options_helper.commands.reports.daily", "dashboard")


roll_plan = _load_reports_command("options_helper.commands.reports.chain", "roll_plan")
