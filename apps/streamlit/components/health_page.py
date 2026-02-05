from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.data.observability_meta import (
    has_observability_tables,
    query_latest_runs,
    query_recent_check_failures,
    query_recent_failures,
    query_watermarks,
)
from options_helper.db.warehouse import DuckDBWarehouse

DEFAULT_DUCKDB_PATH = Path("data/options_helper.duckdb")

LATEST_RUN_COLUMNS = [
    "run_id",
    "job_name",
    "triggered_by",
    "status",
    "started_at",
    "ended_at",
    "duration_ms",
    "provider",
    "storage_backend",
    "error_type",
    "error_message",
    "error_stack_hash",
]

RECURRING_FAILURE_COLUMNS = [
    "error_stack_hash",
    "failure_count",
    "job_count",
    "jobs",
    "latest_started_at",
    "first_started_at",
    "error_type",
    "error_message",
]

WATERMARK_COLUMNS = [
    "asset_key",
    "scope_key",
    "watermark_ts",
    "updated_at",
    "last_run_id",
    "staleness_days",
    "freshness",
]

FAILED_CHECK_COLUMNS = [
    "check_id",
    "checked_at",
    "asset_key",
    "partition_key",
    "check_name",
    "severity",
    "status",
    "metrics_json",
    "message",
    "run_id",
]

_MISSING_STACK_HASH = "(missing)"


@dataclass
class HealthSnapshot:
    database_path: Path
    database_exists: bool
    observability_ready: bool
    guidance: str | None
    latest_runs: pd.DataFrame
    recurring_failures: pd.DataFrame
    watermarks: pd.DataFrame
    failed_checks: pd.DataFrame


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    raw = "" if database_path is None else str(database_path).strip()
    candidate = DEFAULT_DUCKDB_PATH if not raw else Path(raw)
    return candidate.expanduser().resolve()


def load_health_snapshot(
    *,
    database_path: str | Path | None = None,
    days: int = 30,
    stale_days: int = 3,
    limit: int = 200,
) -> HealthSnapshot:
    db_path = resolve_duckdb_path(database_path)
    if not db_path.exists():
        return HealthSnapshot(
            database_path=db_path,
            database_exists=False,
            observability_ready=False,
            guidance=(
                f"DuckDB database not found: {db_path}. "
                "Run `options-helper db init` and at least one producer command first."
            ),
            latest_runs=_empty_latest_runs(),
            recurring_failures=_empty_recurring_failures(),
            watermarks=_empty_watermarks(),
            failed_checks=_empty_failed_checks(),
        )

    warehouse = DuckDBWarehouse(db_path)
    if not has_observability_tables(warehouse):
        return HealthSnapshot(
            database_path=db_path,
            database_exists=True,
            observability_ready=False,
            guidance=(
                "Observability tables are missing (`meta.ingestion_runs`, "
                "`meta.ingestion_run_assets`, `meta.asset_watermarks`, `meta.asset_checks`). "
                "Run `options-helper db init` and producer commands to populate health data."
            ),
            latest_runs=_empty_latest_runs(),
            recurring_failures=_empty_recurring_failures(),
            watermarks=_empty_watermarks(),
            failed_checks=_empty_failed_checks(),
        )

    latest_rows = query_latest_runs(warehouse, days=max(0, int(days)), limit=max(1, int(limit)))
    recent_failure_rows = query_recent_failures(warehouse, days=max(0, int(days)), limit=max(1, int(limit)))
    watermark_rows = query_watermarks(warehouse, stale_days=None, limit=max(1, int(limit)))
    failed_check_rows = query_recent_check_failures(warehouse, days=max(0, int(days)), limit=max(1, int(limit)))

    latest_runs = normalize_latest_runs(latest_rows)
    recurring_failures = build_recurring_failures(recent_failure_rows)
    watermarks = normalize_watermarks(watermark_rows, stale_days=max(0, int(stale_days)))
    failed_checks = normalize_failed_checks(failed_check_rows)

    return HealthSnapshot(
        database_path=db_path,
        database_exists=True,
        observability_ready=True,
        guidance=None,
        latest_runs=latest_runs,
        recurring_failures=recurring_failures,
        watermarks=watermarks,
        failed_checks=failed_checks,
    )


def normalize_latest_runs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_latest_runs()
    out = pd.DataFrame(rows).reindex(columns=LATEST_RUN_COLUMNS)
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    out["ended_at"] = pd.to_datetime(out["ended_at"], errors="coerce")
    out["duration_ms"] = pd.to_numeric(out["duration_ms"], errors="coerce")
    out["status"] = out["status"].astype("string").str.lower().fillna("unknown")
    out = out.sort_values(by=["started_at", "job_name"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)


def build_recurring_failures(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_recurring_failures()

    failures = pd.DataFrame(rows).reindex(columns=LATEST_RUN_COLUMNS)
    failures["started_at"] = pd.to_datetime(failures["started_at"], errors="coerce")
    failures["error_stack_hash"] = failures["error_stack_hash"].astype("string").fillna("")
    failures["error_type"] = failures["error_type"].astype("string").fillna("")
    failures["error_message"] = failures["error_message"].astype("string").fillna("")
    failures["job_name"] = failures["job_name"].astype("string").fillna("")

    failures["stack_group"] = failures["error_stack_hash"].map(_normalize_stack_hash)

    records: list[dict[str, Any]] = []
    for stack_group, group in failures.groupby("stack_group", sort=False):
        jobs = sorted({str(value).strip() for value in group["job_name"].tolist() if str(value).strip()})
        started = pd.to_datetime(group["started_at"], errors="coerce")
        recent_row = group.sort_values(by="started_at", ascending=False, kind="stable").iloc[0]
        records.append(
            {
                "error_stack_hash": stack_group,
                "failure_count": int(len(group)),
                "job_count": int(len(jobs)),
                "jobs": ", ".join(jobs),
                "latest_started_at": started.max(),
                "first_started_at": started.min(),
                "error_type": _coalesce_text([recent_row.get("error_type")]),
                "error_message": _coalesce_text([recent_row.get("error_message")]),
            }
        )

    out = pd.DataFrame(records).reindex(columns=RECURRING_FAILURE_COLUMNS)
    out["failure_count"] = pd.to_numeric(out["failure_count"], errors="coerce")
    out["job_count"] = pd.to_numeric(out["job_count"], errors="coerce")
    out["latest_started_at"] = pd.to_datetime(out["latest_started_at"], errors="coerce")
    out["first_started_at"] = pd.to_datetime(out["first_started_at"], errors="coerce")
    out = out.sort_values(
        by=["failure_count", "latest_started_at", "error_stack_hash"],
        ascending=[False, False, True],
        kind="stable",
    )
    return out.reset_index(drop=True)


def normalize_watermarks(rows: list[dict[str, Any]], *, stale_days: int) -> pd.DataFrame:
    if not rows:
        return _empty_watermarks()

    out = pd.DataFrame(rows).reindex(columns=WATERMARK_COLUMNS[:-1])
    out["watermark_ts"] = pd.to_datetime(out["watermark_ts"], errors="coerce")
    out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
    out["staleness_days"] = pd.to_numeric(out["staleness_days"], errors="coerce")
    threshold = max(0, int(stale_days))
    out["freshness"] = out["staleness_days"].map(lambda value: _freshness(value, stale_days=threshold))
    out = out.reindex(columns=WATERMARK_COLUMNS)
    out = out.sort_values(
        by=["staleness_days", "asset_key", "scope_key"],
        ascending=[False, True, True],
        kind="stable",
    )
    return out.reset_index(drop=True)


def normalize_failed_checks(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_failed_checks()
    out = pd.DataFrame(rows).reindex(columns=FAILED_CHECK_COLUMNS)
    out["checked_at"] = pd.to_datetime(out["checked_at"], errors="coerce")
    out["severity"] = out["severity"].astype("string").str.lower().fillna("")
    out["status"] = out["status"].astype("string").str.lower().fillna("")
    out = out.sort_values(by=["checked_at", "asset_key"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)


def _normalize_stack_hash(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return _MISSING_STACK_HASH
    return raw


def _coalesce_text(values: list[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _freshness(value: Any, *, stale_days: int) -> str:
    if value is None:
        return "unknown"
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "stale" if as_int >= stale_days else "fresh"


def _empty_latest_runs() -> pd.DataFrame:
    return pd.DataFrame(columns=LATEST_RUN_COLUMNS)


def _empty_recurring_failures() -> pd.DataFrame:
    return pd.DataFrame(columns=RECURRING_FAILURE_COLUMNS)


def _empty_watermarks() -> pd.DataFrame:
    return pd.DataFrame(columns=WATERMARK_COLUMNS)


def _empty_failed_checks() -> pd.DataFrame:
    return pd.DataFrame(columns=FAILED_CHECK_COLUMNS)
