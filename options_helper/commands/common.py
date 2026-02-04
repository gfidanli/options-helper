from __future__ import annotations

from datetime import date, datetime

import typer

from options_helper.analysis.portfolio_risk import StressScenario


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


def _spot_from_meta(meta: dict) -> float | None:
    if not meta:
        return None
    candidates = [
        meta.get("spot"),
        (meta.get("underlying") or {}).get("regularMarketPrice"),
        (meta.get("underlying") or {}).get("regularMarketPreviousClose"),
        (meta.get("underlying") or {}).get("regularMarketOpen"),
    ]
    for v in candidates:
        try:
            if v is None:
                continue
            spot = float(v)
            if spot > 0:
                return spot
        except Exception:  # noqa: BLE001
            continue
    return None


def _normalize_pct(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _normalize_pp(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _build_stress_scenarios(
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> list[StressScenario]:
    scenarios: list[StressScenario] = []
    spot_values = stress_spot_pct or [0.05]
    seen_spot: set[float] = set()
    for raw in spot_values:
        pct = abs(_normalize_pct(raw))
        if pct <= 0 or pct in seen_spot:
            continue
        seen_spot.add(pct)
        scenarios.append(StressScenario(name=f"Spot {pct:+.0%}", spot_pct=pct))
        scenarios.append(StressScenario(name=f"Spot {-pct:+.0%}", spot_pct=-pct))

    vol_pp = _normalize_pp(stress_vol_pp)
    if vol_pp != 0:
        pp_label = vol_pp * 100.0
        scenarios.append(StressScenario(name=f"IV {pp_label:+.1f}pp", vol_pp=vol_pp))
        scenarios.append(StressScenario(name=f"IV {-pp_label:+.1f}pp", vol_pp=-vol_pp))

    if stress_days > 0:
        scenarios.append(StressScenario(name=f"Time +{stress_days}d", days=stress_days))

    return scenarios
