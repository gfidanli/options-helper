from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_PARTITION_TZ = ZoneInfo("America/Chicago")
_LEGACY_LOG_FILENAME_RE = re.compile(r"_(\d{8}T\d{6}Z)_\d+\.log$")
_LOG_LINE_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class RunLogger:
    logger: logging.Logger
    log_path: Path | None
    started_at: datetime
    start_perf: float
    command_name: str


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "options_helper"


def build_log_path(log_dir: Path, command_name: str, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_name = _safe_name(command_name)
    pid = os.getpid()
    return log_dir / f"{safe_name}_{timestamp}_{pid}.log"


def _partition_date_for_legacy_log(path: Path) -> str:
    match = _LEGACY_LOG_FILENAME_RE.search(path.name)
    if match is not None:
        try:
            run_utc = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return run_utc.astimezone(_LOG_PARTITION_TZ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline()
        line_match = _LOG_LINE_TIMESTAMP_RE.match(first_line)
        if line_match is not None:
            run_local = datetime.strptime(line_match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_LOG_PARTITION_TZ)
            return run_local.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        pass

    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=_LOG_PARTITION_TZ)
    return modified_at.strftime("%Y-%m-%d")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_{int(time.time() * 1000)}{path.suffix}")


def _migrate_legacy_log_files(log_dir: Path) -> None:
    try:
        legacy_logs = [path for path in log_dir.glob("*.log") if path.is_file()]
    except Exception:  # noqa: BLE001
        return

    for legacy_path in legacy_logs:
        try:
            run_day = _partition_date_for_legacy_log(legacy_path)
            target_dir = log_dir / run_day
            target_dir.mkdir(parents=True, exist_ok=True)
            destination = _next_available_path(target_dir / legacy_path.name)
            legacy_path.replace(destination)
        except Exception:  # noqa: BLE001
            continue


def setup_run_logger(
    log_dir: Path,
    command_name: str,
    *,
    level: int = logging.INFO,
    log_path: Path | None = None,
) -> RunLogger | None:
    effective_log_path = log_path
    if effective_log_path is None:
        _migrate_legacy_log_files(log_dir)
        now_utc = datetime.now(timezone.utc)
        run_day = now_utc.astimezone(_LOG_PARTITION_TZ).strftime("%Y-%m-%d")
        effective_log_dir = log_dir / run_day
        try:
            effective_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return None
        effective_log_path = build_log_path(effective_log_dir, command_name, now=now_utc)
    else:
        try:
            effective_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return None

    logger = logging.getLogger("options_helper.cli")
    logger.setLevel(level)
    logger.propagate = False
    _reset_file_handlers(logger)

    handler = logging.FileHandler(effective_log_path, mode="a")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    logger.info("Start %s", command_name)
    return RunLogger(
        logger=logger,
        log_path=effective_log_path,
        started_at=started_at,
        start_perf=start_perf,
        command_name=command_name,
    )


def finalize_run_logger(run_logger: RunLogger) -> None:
    elapsed = time.perf_counter() - run_logger.start_perf
    run_logger.logger.info("End %s duration=%.2fs", run_logger.command_name, elapsed)
    for handler in list(run_logger.logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()


def _reset_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass
            logger.removeHandler(handler)
