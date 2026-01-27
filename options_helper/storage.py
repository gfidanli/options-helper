from __future__ import annotations

import json
from pathlib import Path

from options_helper.models import Portfolio, Position, RiskProfile


def load_portfolio(path: Path) -> Portfolio:
    raw = path.read_text(encoding="utf-8")
    try:
        return Portfolio.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 - surface validation errors to CLI
        raise ValueError(f"Invalid portfolio JSON at {path}") from exc


def save_portfolio(path: Path, portfolio: Portfolio) -> None:
    path.write_text(
        portfolio.model_dump_json(indent=2, by_alias=True, exclude_none=True),
        encoding="utf-8",
    )


def portfolio_template() -> Portfolio:
    return Portfolio(
        cash=0.0,
        risk_profile=RiskProfile(),
        positions=[
            Position(
                id="example-2026-04-17-5c",
                symbol="UROY",
                option_type="call",
                expiry="2026-04-17",
                strike=5.0,
                contracts=1,
                cost_basis=0.45,
                opened_at="2026-01-10",
            )
        ],
    )


def write_template(path: Path, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    save_portfolio(path, portfolio_template())


def dump_snapshot_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
