from __future__ import annotations

import logging
from pathlib import Path


def setup_technicals_logging(cfg: dict) -> None:
    level = cfg["logging"]["level"].upper()
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "technical_backtesting.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path)]
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
