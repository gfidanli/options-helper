from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Protocol
from uuid import uuid4

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_utc_naive(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def _coerce_datetime(value: datetime | date | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_utc_naive(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
        return _to_utc_naive(parsed)
    except Exception:  # noqa: BLE001
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def serialize_args_json(args: Any) -> str | None:
    if args is None:
        return None
    payload = args
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except Exception:  # noqa: BLE001
            return json.dumps(stripped)
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    except Exception:  # noqa: BLE001
        return json.dumps(str(payload), sort_keys=True, separators=(",", ":"))


def hash_stack_text(stack_text: str | None) -> str | None:
    if stack_text is None:
        return None
    cleaned = stack_text.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def hash_exception_stack(error: BaseException) -> str | None:
    stack_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return hash_stack_text(stack_text)


def _duration_ms(
    *,
    start_perf: float | None,
    started_at: datetime | None,
    ended_at: datetime,
) -> int | None:
    if start_perf is not None:
        elapsed = max(time.perf_counter() - start_perf, 0.0)
        return int(round(elapsed * 1000.0))
    if started_at is None:
        return None
    return int(max((ended_at - started_at).total_seconds() * 1000.0, 0.0))


class RunLogger(Protocol):
    run_id: str | None
    job_name: str | None

    def start_run(
        self,
        *,
        job_name: str,
        triggered_by: str = "cli",
        parent_run_id: str | None = None,
        provider: str | None = None,
        storage_backend: str | None = None,
        args: Any = None,
        git_sha: str | None = None,
        app_version: str | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
    ) -> str:
        ...

    def finalize_success(self, *, ended_at: datetime | None = None) -> None:
        ...

    def finalize_failure(
        self,
        error: BaseException | str,
        *,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        ...

    def log_asset_event(
        self,
        *,
        asset_key: str,
        asset_kind: str,
        partition_key: str = "ALL",
        status: str,
        rows_inserted: int | None = None,
        rows_updated: int | None = None,
        rows_deleted: int | None = None,
        bytes_written: int | None = None,
        min_event_ts: datetime | date | str | None = None,
        max_event_ts: datetime | date | str | None = None,
        extra: Any = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> None:
        ...

    def upsert_watermark(
        self,
        *,
        asset_key: str,
        scope_key: str = "ALL",
        watermark_ts: datetime | date | str | None,
        updated_at: datetime | None = None,
        last_run_id: str | None = None,
    ) -> None:
        ...

    def persist_check(
        self,
        *,
        asset_key: str,
        check_name: str,
        severity: str,
        status: str,
        partition_key: str | None = None,
        metrics: Any = None,
        message: str | None = None,
        run_id: str | None = None,
        check_id: str | None = None,
        checked_at: datetime | None = None,
    ) -> str:
        ...


class DuckDBRunLogger:
    def __init__(self, warehouse: DuckDBWarehouse) -> None:
        self.warehouse = warehouse
        self.run_id: str | None = None
        self.job_name: str | None = None
        self.triggered_by: str = "cli"
        self.parent_run_id: str | None = None
        self.provider: str | None = None
        self.storage_backend: str | None = None
        self.args: Any = None
        self.git_sha: str | None = None
        self.app_version: str | None = None
        self.started_at: datetime | None = None
        self._start_perf: float | None = None
        self._finalized = False
        self._schema_ready = False

    def __enter__(self) -> DuckDBRunLogger:
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is None:
            self.finalize_success()
        else:
            self.finalize_failure(exc)
        return False

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        ensure_schema(self.warehouse)
        self._schema_ready = True

    def start_run(
        self,
        *,
        job_name: str,
        triggered_by: str = "cli",
        parent_run_id: str | None = None,
        provider: str | None = None,
        storage_backend: str | None = None,
        args: Any = None,
        git_sha: str | None = None,
        app_version: str | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
    ) -> str:
        if self.run_id is not None:
            return self.run_id

        clean_job_name = str(job_name).strip()
        if not clean_job_name:
            raise ValueError("job_name is required")

        self._ensure_schema()
        self.run_id = run_id or str(uuid4())
        self.job_name = clean_job_name
        self.triggered_by = str(triggered_by).strip() or "cli"
        self.parent_run_id = parent_run_id
        self.provider = provider
        self.storage_backend = storage_backend
        self.args = args
        self.git_sha = git_sha
        self.app_version = app_version
        self.started_at = _coerce_datetime(started_at) or _utc_now()
        self._start_perf = time.perf_counter()
        self._finalized = False

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO meta.ingestion_runs(
                  run_id, parent_run_id, job_name, triggered_by, status,
                  started_at, provider, storage_backend, args_json, git_sha, app_version
                )
                VALUES (?, ?, ?, ?, 'started', ?, ?, ?, ?, ?, ?)
                """,
                [
                    self.run_id,
                    self.parent_run_id,
                    self.job_name,
                    self.triggered_by,
                    self.started_at,
                    self.provider,
                    self.storage_backend,
                    serialize_args_json(self.args),
                    self.git_sha,
                    self.app_version,
                ],
            )
        return self.run_id

    def finalize_success(self, *, ended_at: datetime | None = None) -> None:
        if self.run_id is None or self._finalized:
            return
        self._ensure_schema()

        finished_at = _coerce_datetime(ended_at) or _utc_now()
        duration_ms = _duration_ms(
            start_perf=self._start_perf,
            started_at=self.started_at,
            ended_at=finished_at,
        )
        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                UPDATE meta.ingestion_runs
                SET status = 'success',
                    ended_at = ?,
                    duration_ms = ?,
                    error_type = NULL,
                    error_message = NULL,
                    error_stack_hash = NULL
                WHERE run_id = ?
                """,
                [finished_at, duration_ms, self.run_id],
            )
        self._finalized = True

    def finalize_failure(
        self,
        error: BaseException | str,
        *,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        if self.run_id is None or self._finalized:
            return
        self._ensure_schema()

        finished_at = _coerce_datetime(ended_at) or _utc_now()
        duration_ms = _duration_ms(
            start_perf=self._start_perf,
            started_at=self.started_at,
            ended_at=finished_at,
        )

        if isinstance(error, BaseException):
            error_type = type(error).__name__
            error_message = str(error)
            hashed_stack = hash_stack_text(stack_text) or hash_exception_stack(error)
        else:
            error_type = "RuntimeError"
            error_message = str(error)
            hashed_stack = hash_stack_text(stack_text)

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                UPDATE meta.ingestion_runs
                SET status = 'failed',
                    ended_at = ?,
                    duration_ms = ?,
                    error_type = ?,
                    error_message = ?,
                    error_stack_hash = ?
                WHERE run_id = ?
                """,
                [finished_at, duration_ms, error_type, error_message, hashed_stack, self.run_id],
            )
        self._finalized = True

    def finalize(
        self,
        *,
        status: str = "success",
        error: BaseException | str | None = None,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        if error is not None or str(status).strip().lower() == "failed":
            self.finalize_failure(error or "run failed", ended_at=ended_at, stack_text=stack_text)
            return
        self.finalize_success(ended_at=ended_at)

    def mark_success(self, *, ended_at: datetime | None = None) -> None:
        self.finalize_success(ended_at=ended_at)

    def mark_failure(
        self,
        error: BaseException | str,
        *,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        self.finalize_failure(error, ended_at=ended_at, stack_text=stack_text)

    def _require_started(self) -> str:
        if self.run_id is None:
            raise RuntimeError("start_run must be called before writing run assets/checks")
        return self.run_id

    @staticmethod
    def _normalize_asset_event_fields(
        *,
        asset_key: str,
        asset_kind: str,
        partition_key: str,
        status: str,
    ) -> tuple[str, str, str, str]:
        clean_partition_key = str(partition_key).strip() or "ALL"
        clean_status = str(status).strip() or "success"
        clean_asset_key = str(asset_key).strip()
        clean_asset_kind = str(asset_kind).strip()
        if not clean_asset_key:
            raise ValueError("asset_key is required")
        if not clean_asset_kind:
            raise ValueError("asset_kind is required")
        return clean_asset_key, clean_asset_kind, clean_partition_key, clean_status

    @staticmethod
    def _resolve_asset_event_timing(
        *,
        started_at: datetime | None,
        ended_at: datetime | None,
        duration_ms: int | None,
    ) -> tuple[datetime | None, datetime | None, int | None]:
        event_started_at = _coerce_datetime(started_at)
        event_ended_at = _coerce_datetime(ended_at)
        computed_duration_ms = duration_ms
        if computed_duration_ms is None and event_started_at is not None and event_ended_at is not None:
            computed_duration_ms = int(
                max((event_ended_at - event_started_at).total_seconds() * 1000.0, 0.0)
            )
        return event_started_at, event_ended_at, computed_duration_ms

    def log_asset_event(
        self,
        *,
        asset_key: str,
        asset_kind: str,
        partition_key: str = "ALL",
        status: str,
        rows_inserted: int | None = None,
        rows_updated: int | None = None,
        rows_deleted: int | None = None,
        bytes_written: int | None = None,
        min_event_ts: datetime | date | str | None = None,
        max_event_ts: datetime | date | str | None = None,
        extra: Any = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self._ensure_schema()
        run_id = self._require_started()
        clean_asset_key, clean_asset_kind, clean_partition_key, clean_status = self._normalize_asset_event_fields(
            asset_key=asset_key,
            asset_kind=asset_kind,
            partition_key=partition_key,
            status=status,
        )
        event_started_at, event_ended_at, computed_duration_ms = self._resolve_asset_event_timing(
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
        )
        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO meta.ingestion_run_assets(
                  run_id, asset_key, asset_kind, partition_key, status,
                  rows_inserted, rows_updated, rows_deleted, bytes_written,
                  min_event_ts, max_event_ts,
                  extra_json, started_at, ended_at, duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, asset_key, partition_key)
                DO UPDATE SET
                  asset_kind = excluded.asset_kind,
                  status = excluded.status,
                  rows_inserted = excluded.rows_inserted,
                  rows_updated = excluded.rows_updated,
                  rows_deleted = excluded.rows_deleted,
                  bytes_written = excluded.bytes_written,
                  min_event_ts = excluded.min_event_ts,
                  max_event_ts = excluded.max_event_ts,
                  extra_json = excluded.extra_json,
                  started_at = excluded.started_at,
                  ended_at = excluded.ended_at,
                  duration_ms = excluded.duration_ms
                """,
                [
                    run_id,
                    clean_asset_key,
                    clean_asset_kind,
                    clean_partition_key,
                    clean_status,
                    rows_inserted,
                    rows_updated,
                    rows_deleted,
                    bytes_written,
                    _coerce_datetime(min_event_ts),
                    _coerce_datetime(max_event_ts),
                    serialize_args_json(extra),
                    event_started_at,
                    event_ended_at,
                    computed_duration_ms,
                ],
            )

    def log_asset_success(self, **kwargs: Any) -> None:
        self.log_asset_event(status="success", **kwargs)

    def log_asset_failure(self, **kwargs: Any) -> None:
        self.log_asset_event(status="failed", **kwargs)

    def log_asset_skipped(self, **kwargs: Any) -> None:
        self.log_asset_event(status="skipped", **kwargs)

    def upsert_watermark(
        self,
        *,
        asset_key: str,
        scope_key: str = "ALL",
        watermark_ts: datetime | date | str | None,
        updated_at: datetime | None = None,
        last_run_id: str | None = None,
    ) -> None:
        self._ensure_schema()
        clean_asset_key = str(asset_key).strip()
        clean_scope_key = str(scope_key).strip() or "ALL"
        if not clean_asset_key:
            raise ValueError("asset_key is required")

        normalized_watermark_ts = _coerce_datetime(watermark_ts)
        if normalized_watermark_ts is None:
            return

        effective_run_id = last_run_id if last_run_id is not None else self.run_id
        watermark_updated_at = _coerce_datetime(updated_at) or _utc_now()

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO meta.asset_watermarks(
                  asset_key, scope_key, watermark_ts, updated_at, last_run_id
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asset_key, scope_key)
                DO UPDATE SET
                  watermark_ts = GREATEST(meta.asset_watermarks.watermark_ts, excluded.watermark_ts),
                  updated_at = CASE
                    WHEN excluded.watermark_ts >= meta.asset_watermarks.watermark_ts
                      THEN excluded.updated_at
                    ELSE meta.asset_watermarks.updated_at
                  END,
                  last_run_id = CASE
                    WHEN excluded.watermark_ts >= meta.asset_watermarks.watermark_ts
                      THEN excluded.last_run_id
                    ELSE meta.asset_watermarks.last_run_id
                  END
                """,
                [
                    clean_asset_key,
                    clean_scope_key,
                    normalized_watermark_ts,
                    watermark_updated_at,
                    effective_run_id,
                ],
            )

    def log_watermark(
        self,
        *,
        asset_key: str,
        scope_key: str = "ALL",
        watermark_ts: datetime | date | str | None,
        updated_at: datetime | None = None,
        last_run_id: str | None = None,
    ) -> None:
        self.upsert_watermark(
            asset_key=asset_key,
            scope_key=scope_key,
            watermark_ts=watermark_ts,
            updated_at=updated_at,
            last_run_id=last_run_id,
        )

    def persist_check(
        self,
        *,
        asset_key: str,
        check_name: str,
        severity: str,
        status: str,
        partition_key: str | None = None,
        metrics: Any = None,
        message: str | None = None,
        run_id: str | None = None,
        check_id: str | None = None,
        checked_at: datetime | None = None,
    ) -> str:
        self._ensure_schema()
        clean_asset_key = str(asset_key).strip()
        clean_check_name = str(check_name).strip()
        clean_severity = str(severity).strip()
        clean_status = str(status).strip()
        if not clean_asset_key:
            raise ValueError("asset_key is required")
        if not clean_check_name:
            raise ValueError("check_name is required")
        if not clean_severity:
            raise ValueError("severity is required")
        if not clean_status:
            raise ValueError("status is required")

        effective_check_id = check_id or str(uuid4())
        effective_run_id = run_id if run_id is not None else self.run_id
        check_time = _coerce_datetime(checked_at) or _utc_now()

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO meta.asset_checks(
                  check_id, asset_key, partition_key, check_name,
                  severity, status, checked_at, metrics_json, message, run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(check_id)
                DO UPDATE SET
                  asset_key = excluded.asset_key,
                  partition_key = excluded.partition_key,
                  check_name = excluded.check_name,
                  severity = excluded.severity,
                  status = excluded.status,
                  checked_at = excluded.checked_at,
                  metrics_json = excluded.metrics_json,
                  message = excluded.message,
                  run_id = excluded.run_id
                """,
                [
                    effective_check_id,
                    clean_asset_key,
                    partition_key,
                    clean_check_name,
                    clean_severity,
                    clean_status,
                    check_time,
                    serialize_args_json(metrics),
                    message,
                    effective_run_id,
                ],
            )
        return effective_check_id

    def log_check(self, **kwargs: Any) -> str:
        return self.persist_check(**kwargs)


class NoopRunLogger:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.job_name: str | None = None
        self.started_at: datetime | None = None
        self._finalized = False

    def __enter__(self) -> NoopRunLogger:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False

    def start_run(
        self,
        *,
        job_name: str,
        triggered_by: str = "cli",
        parent_run_id: str | None = None,
        provider: str | None = None,
        storage_backend: str | None = None,
        args: Any = None,
        git_sha: str | None = None,
        app_version: str | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
    ) -> str:
        del triggered_by, parent_run_id, provider, storage_backend, args, git_sha, app_version
        self.job_name = str(job_name).strip() or None
        self.run_id = run_id or str(uuid4())
        self.started_at = started_at or _utc_now()
        self._finalized = False
        return self.run_id

    def finalize_success(self, *, ended_at: datetime | None = None) -> None:
        del ended_at
        self._finalized = True

    def finalize_failure(
        self,
        error: BaseException | str,
        *,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        del error, ended_at, stack_text
        self._finalized = True

    def finalize(
        self,
        *,
        status: str = "success",
        error: BaseException | str | None = None,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        if error is not None or str(status).strip().lower() == "failed":
            self.finalize_failure(error or "run failed", ended_at=ended_at, stack_text=stack_text)
            return
        self.finalize_success(ended_at=ended_at)

    def mark_success(self, *, ended_at: datetime | None = None) -> None:
        self.finalize_success(ended_at=ended_at)

    def mark_failure(
        self,
        error: BaseException | str,
        *,
        ended_at: datetime | None = None,
        stack_text: str | None = None,
    ) -> None:
        self.finalize_failure(error, ended_at=ended_at, stack_text=stack_text)

    def log_asset_event(
        self,
        *,
        asset_key: str,
        asset_kind: str,
        partition_key: str = "ALL",
        status: str,
        rows_inserted: int | None = None,
        rows_updated: int | None = None,
        rows_deleted: int | None = None,
        bytes_written: int | None = None,
        min_event_ts: datetime | date | str | None = None,
        max_event_ts: datetime | date | str | None = None,
        extra: Any = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> None:
        del (
            asset_key,
            asset_kind,
            partition_key,
            status,
            rows_inserted,
            rows_updated,
            rows_deleted,
            bytes_written,
            min_event_ts,
            max_event_ts,
            extra,
            started_at,
            ended_at,
            duration_ms,
        )

    def log_asset_success(self, **kwargs: Any) -> None:
        del kwargs

    def log_asset_failure(self, **kwargs: Any) -> None:
        del kwargs

    def log_asset_skipped(self, **kwargs: Any) -> None:
        del kwargs

    def upsert_watermark(
        self,
        *,
        asset_key: str,
        scope_key: str = "ALL",
        watermark_ts: datetime | date | str | None,
        updated_at: datetime | None = None,
        last_run_id: str | None = None,
    ) -> None:
        del asset_key, scope_key, watermark_ts, updated_at, last_run_id

    def log_watermark(
        self,
        *,
        asset_key: str,
        scope_key: str = "ALL",
        watermark_ts: datetime | date | str | None,
        updated_at: datetime | None = None,
        last_run_id: str | None = None,
    ) -> None:
        self.upsert_watermark(
            asset_key=asset_key,
            scope_key=scope_key,
            watermark_ts=watermark_ts,
            updated_at=updated_at,
            last_run_id=last_run_id,
        )

    def persist_check(
        self,
        *,
        asset_key: str,
        check_name: str,
        severity: str,
        status: str,
        partition_key: str | None = None,
        metrics: Any = None,
        message: str | None = None,
        run_id: str | None = None,
        check_id: str | None = None,
        checked_at: datetime | None = None,
    ) -> str:
        del asset_key, check_name, severity, status, partition_key, metrics, message, run_id, checked_at
        return check_id or str(uuid4())

    def log_check(self, **kwargs: Any) -> str:
        return self.persist_check(**kwargs)


def _safe_rows(
    warehouse: DuckDBWarehouse,
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        df = warehouse.fetch_df(sql, params)
    except Exception:  # noqa: BLE001
        return []
    if df is None or df.empty:
        return []
    return list(df.to_dict(orient="records"))


def has_observability_tables(warehouse: DuckDBWarehouse) -> bool:
    rows = _safe_rows(
        warehouse,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'meta'
          AND table_name IN ('ingestion_runs', 'ingestion_run_assets', 'asset_watermarks', 'asset_checks')
        """,
    )
    found = {str(row.get("table_name") or "") for row in rows}
    return {
        "ingestion_runs",
        "ingestion_run_assets",
        "asset_watermarks",
        "asset_checks",
    }.issubset(found)


def query_latest_runs(
    warehouse: DuckDBWarehouse,
    *,
    days: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_clauses: list[str] = []
    if days is not None:
        where_clauses.append("started_at >= ?")
        params.append(_utc_now() - timedelta(days=max(int(days), 0)))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(max(int(limit), 1))

    return _safe_rows(
        warehouse,
        f"""
        SELECT
          run_id,
          job_name,
          triggered_by,
          status,
          started_at,
          ended_at,
          duration_ms,
          provider,
          storage_backend,
          error_type,
          error_message,
          error_stack_hash
        FROM meta.ingestion_runs
        {where_sql}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY job_name ORDER BY started_at DESC) = 1
        ORDER BY started_at DESC
        LIMIT ?
        """,
        params,
    )


def query_recent_failures(
    warehouse: DuckDBWarehouse,
    *,
    days: int = 7,
    limit: int = 200,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_clauses = ["status = 'failed'"]
    if days >= 0:
        where_clauses.append("started_at >= ?")
        params.append(_utc_now() - timedelta(days=days))
    params.append(max(int(limit), 1))

    return _safe_rows(
        warehouse,
        f"""
        SELECT
          run_id,
          parent_run_id,
          job_name,
          triggered_by,
          status,
          started_at,
          ended_at,
          duration_ms,
          provider,
          storage_backend,
          error_type,
          error_message,
          error_stack_hash
        FROM meta.ingestion_runs
        WHERE {' AND '.join(where_clauses)}
        ORDER BY started_at DESC
        LIMIT ?
        """,
        params,
    )


def query_watermarks(
    warehouse: DuckDBWarehouse,
    *,
    stale_days: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_clauses: list[str] = []
    if stale_days is not None:
        where_clauses.append("DATE_DIFF('day', watermark_ts, NOW()) >= ?")
        params.append(max(int(stale_days), 0))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(max(int(limit), 1))

    return _safe_rows(
        warehouse,
        f"""
        SELECT
          asset_key,
          scope_key,
          watermark_ts,
          updated_at,
          last_run_id,
          DATE_DIFF('day', watermark_ts, NOW()) AS staleness_days
        FROM meta.asset_watermarks
        {where_sql}
        ORDER BY staleness_days DESC, asset_key ASC, scope_key ASC
        LIMIT ?
        """,
        params,
    )


def query_recent_check_failures(
    warehouse: DuckDBWarehouse,
    *,
    days: int = 30,
    limit: int = 200,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_clauses = ["status = 'fail'"]
    if days >= 0:
        where_clauses.append("checked_at >= ?")
        params.append(_utc_now() - timedelta(days=days))
    params.append(max(int(limit), 1))

    return _safe_rows(
        warehouse,
        f"""
        SELECT
          check_id,
          checked_at,
          asset_key,
          partition_key,
          check_name,
          severity,
          status,
          metrics_json,
          message,
          run_id
        FROM meta.asset_checks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY checked_at DESC
        LIMIT ?
        """,
        params,
    )


latest_runs = query_latest_runs
recent_failures = query_recent_failures
watermarks = query_watermarks
recent_check_failures = query_recent_check_failures
