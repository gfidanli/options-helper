from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

from options_helper.analysis.chain_metrics import compute_mark_price


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


def _clean_key_series(series: pd.Series) -> pd.Series:
    def _clean(value: object) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            return None
        text = str(value).strip()
        return text or None

    return series.map(_clean)


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
    def _missing(x: float | None) -> bool:
        return x is None or pd.isna(x)

    if _missing(oi_prev) or _missing(oi_today) or _missing(delta_oi):
        return FlowClass.UNKNOWN

    volume = 0.0 if volume is None or pd.isna(volume) else float(volume)

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


def compute_flow(today: pd.DataFrame, prev: pd.DataFrame, *, spot: float | None = None) -> pd.DataFrame:
    """
    Compute day-to-day flow metrics from two option chain snapshots.

    Required columns in `today`:
    - contractSymbol, optionType, expiry, lastPrice, volume, openInterest, strike (optional)
    - osi (optional, used as join key when available)

    Required columns in `prev`:
    - contractSymbol, openInterest (or osi + openInterest if available)
    """
    if today.empty:
        return pd.DataFrame()

    required_today = {"contractSymbol", "optionType", "expiry"}
    missing = required_today - set(today.columns)
    if missing:
        raise ValueError(f"today snapshot missing columns: {sorted(missing)}")

    if "openInterest" not in prev.columns:
        raise ValueError("prev snapshot missing required columns: openInterest")
    if "contractSymbol" not in prev.columns and "osi" not in prev.columns:
        raise ValueError("prev snapshot missing required columns: contractSymbol or osi")

    t = today.copy()
    t["_join_key"] = _clean_key_series(t["contractSymbol"])
    if "osi" in t.columns:
        osi_key = _clean_key_series(t["osi"])
        t["_join_key"] = osi_key.where(osi_key.notna(), t["_join_key"])

    p = prev.copy()
    base_key = _clean_key_series(p["contractSymbol"]) if "contractSymbol" in p.columns else pd.Series([None] * len(p), index=p.index)
    if "osi" in p.columns:
        osi_key = _clean_key_series(p["osi"])
        base_key = osi_key.where(osi_key.notna(), base_key)
    p = p.assign(_join_key=base_key)[["_join_key", "openInterest"]].copy()
    p = p.rename(columns={"openInterest": "openInterest_prev"})

    merged = t.merge(p, on="_join_key", how="left")
    merged = merged.drop(columns=["_join_key"])

    # Normalize types (prefer numeric dtype + NaN for missing)
    merged["lastPrice"] = (
        pd.to_numeric(merged.get("lastPrice"), errors="coerce") if "lastPrice" in merged.columns else float("nan")
    )
    merged["volume"] = pd.to_numeric(merged.get("volume"), errors="coerce") if "volume" in merged.columns else float("nan")
    merged["openInterest"] = (
        pd.to_numeric(merged.get("openInterest"), errors="coerce") if "openInterest" in merged.columns else float("nan")
    )
    merged["openInterest_prev"] = pd.to_numeric(merged.get("openInterest_prev"), errors="coerce")

    merged["deltaOI"] = merged["openInterest"] - merged["openInterest_prev"]

    # Deterministic mark price for notional computations.
    merged["mark"] = compute_mark_price(merged)

    merged["deltaOI_notional"] = merged["deltaOI"] * merged["mark"] * 100.0
    merged["volume_notional"] = merged["volume"] * merged["mark"] * 100.0

    denom = merged["openInterest_prev"].clip(lower=1.0)
    merged["vol_oi_ratio"] = merged["volume"] / denom

    # Best-effort delta-notional:
    #   ΔOI * delta * spot * 100
    if spot is not None and spot > 0 and "bs_delta" in merged.columns:
        merged["bs_delta"] = pd.to_numeric(merged.get("bs_delta"), errors="coerce")
        merged["delta_notional"] = merged["deltaOI"] * merged["bs_delta"] * float(spot) * 100.0
    else:
        merged["delta_notional"] = float("nan")

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


FlowGroupBy = Literal["contract", "strike", "expiry", "expiry-strike"]


def aggregate_flow_window(flows: list[pd.DataFrame], *, group_by: FlowGroupBy) -> pd.DataFrame:
    """
    Net and aggregate a list of per-day flow frames (e.g., from multiple snapshot pairs).

    This is a pure aggregation step. The caller is responsible for computing per-day flow frames
    (including spot-aware delta-notional) via `compute_flow`.
    """
    non_empty = [f for f in flows if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame()

    df = pd.concat(non_empty, ignore_index=True)

    group_cols: list[str]
    if group_by == "contract":
        if "contractSymbol" not in df.columns:
            raise ValueError("flow missing contractSymbol column for group_by=contract")
        df = df.copy()
        contract_key = _clean_key_series(df["contractSymbol"])
        if "osi" in df.columns:
            osi_key = _clean_key_series(df["osi"])
            contract_key = osi_key.where(osi_key.notna(), contract_key)
        df["_contract_key"] = contract_key
        group_cols = ["_contract_key", "expiry", "optionType", "strike"]
    elif group_by == "strike":
        group_cols = ["optionType", "strike"]
    elif group_by == "expiry":
        group_cols = ["optionType", "expiry"]
    elif group_by == "expiry-strike":
        group_cols = ["optionType", "expiry", "strike"]
    else:
        raise ValueError(f"unsupported group_by: {group_by}")

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"flow missing columns required for group_by={group_by}: {missing}")

    metric_cols = ["deltaOI", "deltaOI_notional", "volume_notional", "delta_notional"]
    for c in metric_cols:
        if c not in df.columns:
            df[c] = float("nan")

    sub = df[group_cols + metric_cols].copy()
    for c in metric_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    agg = {c: "sum" for c in metric_cols}
    grouped = sub.groupby(group_cols, as_index=False, sort=True).agg(agg)
    grouped = grouped.merge(sub.groupby(group_cols, as_index=False, sort=True).size(), on=group_cols, how="left")
    if group_by == "contract" and "_contract_key" in grouped.columns:
        grouped = grouped.rename(columns={"_contract_key": "contractSymbol"})
    return grouped
