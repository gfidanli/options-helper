from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class FlowClass(str, Enum):
    BUILDING = "building"
    UNWINDING = "unwinding"
    CHURN = "churn"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FlowRow:
    contract_symbol: str
    symbol: str
    option_type: str
    expiry: str
    strike: float | None
    last_price: float | None
    volume: float | None
    oi_today: float | None
    oi_prev: float | None
    delta_oi: float | None
    delta_oi_notional: float | None
    volume_notional: float | None
    vol_oi_ratio: float | None
    flow_class: FlowClass


def _as_float(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def classify_flow(
    *,
    oi_prev: float | None,
    oi_today: float | None,
    delta_oi: float | None,
    volume: float | None,
) -> FlowClass:
    """
    Heuristic classifier for day-to-day OI/volume behavior.

    Notes:
    - OI generally updates once/day; volume is same-day.
    - This is not a definitive "smart money" label; it's a positioning proxy.
    """
    if oi_prev is None or oi_today is None or delta_oi is None:
        return FlowClass.UNKNOWN

    volume = volume or 0.0

    # Adaptive threshold: require a meaningful OI change to classify build/unwind.
    threshold = max(10.0, 0.10 * max(oi_prev, 0.0))

    if delta_oi >= threshold:
        return FlowClass.BUILDING
    if delta_oi <= -threshold:
        return FlowClass.UNWINDING

    # High trading activity but not much net positioning change.
    churn_threshold = max(50.0, 0.50 * max(oi_prev, 0.0))
    if volume >= churn_threshold:
        return FlowClass.CHURN

    return FlowClass.UNKNOWN


def compute_flow(today: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Compute day-to-day flow metrics from two option chain snapshots.

    Required columns in `today`:
    - contractSymbol, optionType, expiry, lastPrice, volume, openInterest, strike (optional)

    Required columns in `prev`:
    - contractSymbol, openInterest
    """
    if today.empty:
        return pd.DataFrame()

    required_today = {"contractSymbol", "optionType", "expiry"}
    missing = required_today - set(today.columns)
    if missing:
        raise ValueError(f"today snapshot missing columns: {sorted(missing)}")

    if "contractSymbol" not in prev.columns or "openInterest" not in prev.columns:
        raise ValueError("prev snapshot missing required columns: contractSymbol, openInterest")

    t = today.copy()
    p = prev[["contractSymbol", "openInterest"]].copy()
    p = p.rename(columns={"openInterest": "openInterest_prev"})

    merged = t.merge(p, on="contractSymbol", how="left")

    # Normalize types
    merged["lastPrice"] = merged.get("lastPrice").map(_as_float) if "lastPrice" in merged.columns else None
    merged["volume"] = merged.get("volume").map(_as_float) if "volume" in merged.columns else None
    merged["openInterest"] = merged.get("openInterest").map(_as_float) if "openInterest" in merged.columns else None
    merged["openInterest_prev"] = merged["openInterest_prev"].map(_as_float)

    merged["deltaOI"] = merged["openInterest"] - merged["openInterest_prev"]

    merged["deltaOI_notional"] = merged["deltaOI"] * merged["lastPrice"] * 100.0
    merged["volume_notional"] = merged["volume"] * merged["lastPrice"] * 100.0

    def _vol_oi(row) -> float | None:
        oi_prev = row["openInterest_prev"]
        vol = row["volume"]
        if oi_prev is None or vol is None:
            return None
        denom = max(float(oi_prev), 1.0)
        return float(vol) / denom

    merged["vol_oi_ratio"] = merged.apply(_vol_oi, axis=1)

    def _classify(row) -> str:
        return classify_flow(
            oi_prev=row["openInterest_prev"],
            oi_today=row["openInterest"],
            delta_oi=row["deltaOI"],
            volume=row["volume"],
        ).value

    merged["flow_class"] = merged.apply(_classify, axis=1)

    return merged


def summarize_flow(flow: pd.DataFrame) -> dict[str, float]:
    """
    Produce simple aggregate summaries for a symbol/day.
    """
    if flow.empty:
        return {"calls_delta_oi_notional": 0.0, "puts_delta_oi_notional": 0.0}

    def _sum_where(option_type: str) -> float:
        sub = flow[flow["optionType"] == option_type]
        val = sub["deltaOI_notional"].dropna()
        return float(val.sum()) if not val.empty else 0.0

    return {
        "calls_delta_oi_notional": _sum_where("call"),
        "puts_delta_oi_notional": _sum_where("put"),
    }

