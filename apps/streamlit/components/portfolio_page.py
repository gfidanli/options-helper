from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.models import Leg, MultiLegPosition, Portfolio, Position, PositionLike
from options_helper.storage import load_portfolio
from options_helper.ui.dashboard import load_briefing_artifact, resolve_briefing_json

DEFAULT_PORTFOLIO_PATH = Path("portfolio.json")
DEFAULT_REPORTS_DIR = Path("reports")


def resolve_portfolio_path(portfolio_path: str | Path | None = None) -> Path:
    raw = "" if portfolio_path is None else str(portfolio_path).strip()
    candidate = DEFAULT_PORTFOLIO_PATH if not raw else Path(raw)
    return candidate.expanduser().resolve()


def resolve_reports_path(reports_path: str | Path | None = None) -> Path:
    raw = "" if reports_path is None else str(reports_path).strip()
    candidate = DEFAULT_REPORTS_DIR if not raw else Path(raw)
    return candidate.expanduser().resolve()


def load_portfolio_safe(
    portfolio_path: str | Path | None = None,
) -> tuple[Portfolio | None, Path, str | None]:
    path = resolve_portfolio_path(portfolio_path)
    if not path.exists():
        return None, path, f"Portfolio JSON not found: {path}"
    try:
        return load_portfolio(path), path, None
    except Exception as exc:  # noqa: BLE001
        return None, path, f"Failed to parse portfolio JSON at {path}: {exc}"


def build_positions_dataframe(portfolio: Portfolio) -> pd.DataFrame:
    columns = [
        "id",
        "symbol",
        "structure",
        "option_type",
        "expiry",
        "strike",
        "contracts",
        "cost_basis",
        "premium_at_risk",
        "opened_at",
        "notes",
    ]
    if not portfolio.positions:
        return pd.DataFrame(columns=columns)

    rows = [_position_to_row(position) for position in portfolio.positions]
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns].sort_values(by=["symbol", "id"], kind="stable").reset_index(drop=True)


def build_computed_risk_summary(portfolio: Portfolio) -> dict[str, Any]:
    symbols = sorted({position.symbol for position in portfolio.positions})
    capital_cost_basis = float(portfolio.capital_cost_basis())
    risk_profile = portfolio.risk_profile
    max_portfolio_pct = _safe_float(risk_profile.max_portfolio_risk_pct)
    max_single_pct = _safe_float(risk_profile.max_single_position_risk_pct)
    return {
        "source": "computed",
        "cash": float(portfolio.cash),
        "position_count": len(portfolio.positions),
        "symbol_count": len(symbols),
        "premium_at_risk": float(portfolio.premium_at_risk()),
        "capital_cost_basis": capital_cost_basis,
        "next_expiry": _nearest_expiry_iso(portfolio.positions),
        "max_portfolio_risk_pct": max_portfolio_pct,
        "max_portfolio_risk_dollars": None
        if max_portfolio_pct is None
        else capital_cost_basis * max_portfolio_pct,
        "max_single_position_risk_pct": max_single_pct,
        "max_single_position_risk_dollars": None
        if max_single_pct is None
        else capital_cost_basis * max_single_pct,
        "take_profit_pct": _safe_float(risk_profile.take_profit_pct),
        "stop_loss_pct": _safe_float(risk_profile.stop_loss_pct),
        "total_delta_shares": None,
        "total_theta_dollars_per_day": None,
        "total_vega_dollars_per_iv": None,
        "missing_greeks": None,
        "stress": [],
        "warnings": [],
    }


def load_latest_briefing_risk_summary(
    reports_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    resolved_reports = resolve_reports_path(reports_path)
    try:
        briefing_path = resolve_briefing_json(resolved_reports, "latest")
    except Exception as exc:  # noqa: BLE001
        return None, f"Briefing summary unavailable ({exc})"

    try:
        artifact = load_briefing_artifact(briefing_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to load briefing artifact {briefing_path}: {exc}"

    exposure = artifact.portfolio.exposure if isinstance(artifact.portfolio.exposure, dict) else {}
    stress_rows = list(artifact.portfolio.stress or [])

    if not exposure and not stress_rows:
        return None, f"Latest briefing {briefing_path.name} does not include portfolio exposure."

    positions_value = exposure.get("positions") if isinstance(exposure, dict) else None
    warnings_value = exposure.get("warnings") if isinstance(exposure, dict) else None
    summary = {
        "source": "briefing",
        "report_date": artifact.report_date,
        "as_of": artifact.as_of,
        "generated_at": artifact.generated_at.isoformat(),
        "portfolio_path": artifact.portfolio_path,
        "position_count": len(positions_value) if isinstance(positions_value, list) else None,
        "symbol_count": None,
        "premium_at_risk": None,
        "capital_cost_basis": None,
        "next_expiry": None,
        "max_portfolio_risk_pct": None,
        "max_portfolio_risk_dollars": None,
        "max_single_position_risk_pct": None,
        "max_single_position_risk_dollars": None,
        "take_profit_pct": None,
        "stop_loss_pct": None,
        "total_delta_shares": _safe_float(exposure.get("total_delta_shares")),
        "total_theta_dollars_per_day": _safe_float(exposure.get("total_theta_dollars_per_day")),
        "total_vega_dollars_per_iv": _safe_float(exposure.get("total_vega_dollars_per_iv")),
        "missing_greeks": _safe_int(exposure.get("missing_greeks")),
        "stress": stress_rows,
        "warnings": []
        if not isinstance(warnings_value, list)
        else [str(w) for w in warnings_value if str(w).strip()],
    }
    return summary, None


def build_portfolio_risk_summary(
    portfolio: Portfolio,
    *,
    reports_path: str | Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    computed = build_computed_risk_summary(portfolio)
    briefing_summary, briefing_note = load_latest_briefing_risk_summary(reports_path=reports_path)
    if briefing_summary is None:
        return computed, briefing_note

    merged = dict(computed)
    for key, value in briefing_summary.items():
        if key in {"source", "report_date", "as_of", "generated_at", "portfolio_path", "stress", "warnings"}:
            merged[key] = value
            continue
        if value is not None:
            merged[key] = value
    return merged, None


def _position_to_row(position: PositionLike) -> dict[str, Any]:
    if isinstance(position, Position):
        return {
            "id": position.id,
            "symbol": position.symbol,
            "structure": "single",
            "option_type": position.option_type,
            "expiry": position.expiry.isoformat(),
            "strike": float(position.strike),
            "contracts": int(position.contracts),
            "cost_basis": float(position.cost_basis),
            "premium_at_risk": float(position.premium_paid),
            "opened_at": None if position.opened_at is None else position.opened_at.isoformat(),
            "notes": "",
        }

    expiry_dates = sorted({leg.expiry.isoformat() for leg in position.legs})
    expiry_text = "-"
    if len(expiry_dates) == 1:
        expiry_text = expiry_dates[0]
    elif expiry_dates:
        expiry_text = f"{expiry_dates[0]}..{expiry_dates[-1]}"
    return {
        "id": position.id,
        "symbol": position.symbol,
        "structure": "multi-leg",
        "option_type": "multi",
        "expiry": expiry_text,
        "strike": None,
        "contracts": sum(leg.signed_contracts for leg in position.legs),
        "cost_basis": None if position.net_debit is None else float(position.net_debit),
        "premium_at_risk": float(position.premium_paid),
        "opened_at": None if position.opened_at is None else position.opened_at.isoformat(),
        "notes": " | ".join(_format_leg(leg) for leg in position.legs),
    }


def _format_leg(leg: Leg) -> str:
    side = "L" if leg.side == "long" else "S"
    option_type = "C" if leg.option_type == "call" else "P"
    ratio = "" if leg.ratio is None else f"x{leg.ratio:g}"
    return f"{side}{option_type} {leg.expiry.isoformat()} {leg.strike:g} ({leg.contracts}{ratio})"


def _nearest_expiry_iso(positions: list[PositionLike]) -> str | None:
    expiries: list[date] = []
    for position in positions:
        if isinstance(position, Position):
            expiries.append(position.expiry)
            continue
        expiries.extend(leg.expiry for leg in position.legs)
    if not expiries:
        return None
    return min(expiries).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
