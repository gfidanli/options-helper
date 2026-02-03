from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


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


def setup_run_logger(
    log_dir: Path,
    command_name: str,
    *,
    level: int = logging.INFO,
) -> RunLogger | None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return None

    log_path = build_log_path(log_dir, command_name)
    logger = logging.getLogger("options_helper.cli")
    logger.setLevel(level)
    logger.propagate = False
    _reset_file_handlers(logger)

    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    logger.info("Start %s", command_name)
    return RunLogger(
        logger=logger,
        log_path=log_path,
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
