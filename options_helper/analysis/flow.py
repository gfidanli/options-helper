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
    _validate_flow_inputs(today=today, prev=prev)
    merged = _merge_prev_open_interest(today=today, prev=prev)
    merged = _normalize_flow_numeric_columns(merged)
    merged = _compute_flow_notional_columns(merged, spot=spot)
    merged["flow_class"] = merged.apply(_classify_flow_row, axis=1)
    return merged


def _validate_flow_inputs(*, today: pd.DataFrame, prev: pd.DataFrame) -> None:
    required_today = {"contractSymbol", "optionType", "expiry"}
    missing = required_today - set(today.columns)
    if missing:
        raise ValueError(f"today snapshot missing columns: {sorted(missing)}")
    if "openInterest" not in prev.columns:
        raise ValueError("prev snapshot missing required columns: openInterest")
    if "contractSymbol" not in prev.columns and "osi" not in prev.columns:
        raise ValueError("prev snapshot missing required columns: contractSymbol or osi")


def _merge_prev_open_interest(*, today: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    merged = today.copy()
    merged["_contract_key"] = _clean_key_series(merged["contractSymbol"])
    merged["_osi_key"] = (
        _clean_key_series(merged["osi"])
        if "osi" in merged.columns
        else pd.Series([None] * len(merged), index=merged.index)
    )
    merged["openInterest_prev"] = float("nan")
    prior = prev.copy()
    if "contractSymbol" in prior.columns:
        by_contract = pd.DataFrame(
            {
                "_contract_key": _clean_key_series(prior["contractSymbol"]),
                "openInterest_prev": prior["openInterest"],
            }
        )
        by_contract = by_contract.dropna(subset=["_contract_key"]).drop_duplicates(subset=["_contract_key"], keep="last")
        merged = merged.merge(by_contract, on="_contract_key", how="left", suffixes=("", "_from_contract"))
        if "openInterest_prev_from_contract" in merged.columns:
            merged["openInterest_prev"] = merged["openInterest_prev_from_contract"]
            merged = merged.drop(columns=["openInterest_prev_from_contract"])
    if "osi" in prior.columns:
        by_osi = pd.DataFrame(
            {
                "_osi_key": _clean_key_series(prior["osi"]),
                "openInterest_prev_from_osi": prior["openInterest"],
            }
        )
        by_osi = by_osi.dropna(subset=["_osi_key"]).drop_duplicates(subset=["_osi_key"], keep="last")
        merged = merged.merge(by_osi, on="_osi_key", how="left")
        merged["openInterest_prev"] = merged["openInterest_prev"].where(
            merged["openInterest_prev"].notna(),
            merged["openInterest_prev_from_osi"],
        )
        merged = merged.drop(columns=["openInterest_prev_from_osi"])
    return merged.drop(columns=["_contract_key", "_osi_key"])


def _normalize_flow_numeric_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["lastPrice"] = pd.to_numeric(out.get("lastPrice"), errors="coerce") if "lastPrice" in out.columns else float("nan")
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce") if "volume" in out.columns else float("nan")
    out["openInterest"] = (
        pd.to_numeric(out.get("openInterest"), errors="coerce")
        if "openInterest" in out.columns
        else float("nan")
    )
    out["openInterest_prev"] = pd.to_numeric(out.get("openInterest_prev"), errors="coerce")
    out["deltaOI"] = out["openInterest"] - out["openInterest_prev"]
    return out


def _compute_flow_notional_columns(merged: pd.DataFrame, *, spot: float | None) -> pd.DataFrame:
    out = merged.copy()
    out["mark"] = compute_mark_price(out)
    out["deltaOI_notional"] = out["deltaOI"] * out["mark"] * 100.0
    out["volume_notional"] = out["volume"] * out["mark"] * 100.0
    out["vol_oi_ratio"] = out["volume"] / out["openInterest_prev"].clip(lower=1.0)
    if spot is not None and spot > 0 and "bs_delta" in out.columns:
        out["bs_delta"] = pd.to_numeric(out.get("bs_delta"), errors="coerce")
        out["delta_notional"] = out["deltaOI"] * out["bs_delta"] * float(spot) * 100.0
    else:
        out["delta_notional"] = float("nan")
    return out


def _classify_flow_row(row: pd.Series) -> str:
    return classify_flow(
        oi_prev=row["openInterest_prev"],
        oi_today=row["openInterest"],
        delta_oi=row["deltaOI"],
        volume=row["volume"],
    ).value


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
