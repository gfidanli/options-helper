from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Literal

import pandas as pd

from options_helper.analysis.chain_metrics import select_expiries
from options_helper.schemas.research_metrics_contracts import (
    EXPOSURE_STRIKE_FIELDS,
    EXPOSURE_SUMMARY_FIELDS,
    SIGNED_EXPOSURE_CONVENTION,
    SignedExposureConvention,
)


ExposureMode = Literal["near", "monthly", "all"]


@dataclass(frozen=True)
class NetExposureLevel:
    strike: float
    net_gex: float
    abs_net_gex: float


@dataclass(frozen=True)
class ExposureSlice:
    schema_version: int
    signed_exposure_convention: SignedExposureConvention
    symbol: str
    as_of: str
    spot: float
    mode: ExposureMode
    available_expiries: list[str]
    included_expiries: list[str]
    strike_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    top_abs_net_levels: list[NetExposureLevel]


def _summarize_exposure_rows(
    strike_rows: list[dict[str, Any]],
    *,
    can_compute_gex: bool,
    top_n: int,
) -> tuple[float | None, float | None, float | None, float | None, list[NetExposureLevel]]:
    if not can_compute_gex:
        return None, None, None, None, []
    total_call_gex = sum(_as_float_or_zero(row.get("call_gex")) for row in strike_rows)
    total_put_gex = sum(_as_float_or_zero(row.get("put_gex")) for row in strike_rows)
    total_net_gex = sum(_as_float_or_zero(row.get("net_gex")) for row in strike_rows)
    net_by_strike = aggregate_net_exposure_by_strike(strike_rows)
    flip_strike = compute_flip_strike(net_by_strike)
    top_abs_net_levels = rank_top_abs_net_levels(net_by_strike, top_n=top_n)
    return total_call_gex, total_put_gex, total_net_gex, flip_strike, top_abs_net_levels


def compute_exposure_slice(
    snapshot: pd.DataFrame,
    *,
    symbol: str,
    as_of: date | str,
    spot: float,
    mode: ExposureMode = "all",
    include_expiries: Iterable[date | str] | None = None,
    near_n: int = 4,
    top_n: int = 10,
) -> ExposureSlice:
    if mode not in {"near", "monthly", "all"}:
        raise ValueError(f"unsupported mode: {mode}")
    symbol_norm = symbol.strip().upper()
    as_of_str = _normalize_as_of(as_of)
    warnings: list[str] = []
    cleaned = _normalize_chain_rows(snapshot, warnings=warnings)
    available_dates = sorted(set(cleaned["expiry_date"].tolist())) if not cleaned.empty else []
    available_expiries = [exp.isoformat() for exp in available_dates]
    include_dates = _parse_include_expiries(include_expiries, warnings=warnings)
    if include_expiries is not None and include_dates == []:
        selected_dates: list[date] = []
    else:
        selected_dates = select_expiries(
            available_dates,
            mode=mode,
            include=include_dates,
            near_n=near_n,
        )
    if not selected_dates:
        warnings.append("no_expiries_selected")
    included_expiries = [exp.isoformat() for exp in selected_dates]
    if cleaned.empty or not selected_dates:
        selected = cleaned.iloc[0:0].copy()
    else:
        selected = cleaned[cleaned["expiry_date"].isin(set(selected_dates))].copy()
    can_compute_gex = spot > 0
    if not can_compute_gex:
        warnings.append("non_positive_spot")
    strike_rows = _build_strike_rows(
        selected,
        symbol=symbol_norm,
        as_of=as_of_str,
        spot=spot,
        can_compute_gex=can_compute_gex,
    )
    total_call_gex, total_put_gex, total_net_gex, flip_strike, top_abs_net_levels = _summarize_exposure_rows(
        strike_rows,
        can_compute_gex=can_compute_gex,
        top_n=top_n,
    )
    summary_values: dict[str, Any] = {
        "symbol": symbol_norm,
        "as_of": as_of_str,
        "spot": float(spot),
        "flip_strike": flip_strike,
        "total_call_gex": total_call_gex,
        "total_put_gex": total_put_gex,
        "total_net_gex": total_net_gex,
        "warnings": _dedupe_preserve_order(warnings),
    }
    summary = {field: summary_values[field] for field in EXPOSURE_SUMMARY_FIELDS}
    return ExposureSlice(
        schema_version=1,
        signed_exposure_convention=SIGNED_EXPOSURE_CONVENTION,
        symbol=symbol_norm,
        as_of=as_of_str,
        spot=float(spot),
        mode=mode,
        available_expiries=available_expiries,
        included_expiries=included_expiries,
        strike_rows=strike_rows,
        summary=summary,
        top_abs_net_levels=top_abs_net_levels,
    )


def compute_exposure_slices(
    snapshot: pd.DataFrame,
    *,
    symbol: str,
    as_of: date | str,
    spot: float,
    include_expiries: Iterable[date | str] | None = None,
    near_n: int = 4,
    top_n: int = 10,
) -> dict[ExposureMode, ExposureSlice]:
    return {
        "near": compute_exposure_slice(
            snapshot,
            symbol=symbol,
            as_of=as_of,
            spot=spot,
            mode="near",
            include_expiries=include_expiries,
            near_n=near_n,
            top_n=top_n,
        ),
        "monthly": compute_exposure_slice(
            snapshot,
            symbol=symbol,
            as_of=as_of,
            spot=spot,
            mode="monthly",
            include_expiries=include_expiries,
            near_n=near_n,
            top_n=top_n,
        ),
        "all": compute_exposure_slice(
            snapshot,
            symbol=symbol,
            as_of=as_of,
            spot=spot,
            mode="all",
            include_expiries=include_expiries,
            near_n=near_n,
            top_n=top_n,
        ),
    }


def aggregate_net_exposure_by_strike(strike_rows: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    totals: dict[float, float] = {}
    for row in strike_rows:
        strike = _to_float(row.get("strike"))
        net_gex = _to_float(row.get("net_gex"))
        if strike is None or net_gex is None:
            continue
        totals[strike] = totals.get(strike, 0.0) + net_gex
    return sorted(totals.items(), key=lambda item: item[0])


def compute_flip_strike(net_by_strike: Iterable[tuple[float, float]]) -> float | None:
    ladder = sorted((float(strike), float(net)) for strike, net in net_by_strike)
    cumulative = 0.0
    prev_cumulative: float | None = None
    epsilon = 1e-12

    for strike, net in ladder:
        cumulative += net
        if abs(cumulative) <= epsilon:
            return strike
        if prev_cumulative is not None:
            crossed = (prev_cumulative < -epsilon and cumulative > epsilon) or (
                prev_cumulative > epsilon and cumulative < -epsilon
            )
            if crossed:
                return strike
        prev_cumulative = cumulative

    return None


def rank_top_abs_net_levels(
    net_by_strike: Iterable[tuple[float, float]],
    *,
    top_n: int = 10,
) -> list[NetExposureLevel]:
    if top_n <= 0:
        return []
    ordered = sorted(
        ((float(strike), float(net)) for strike, net in net_by_strike),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return [
        NetExposureLevel(strike=strike, net_gex=net, abs_net_gex=abs(net))
        for strike, net in ordered[:top_n]
    ]


def _build_strike_rows(
    cleaned: pd.DataFrame,
    *,
    symbol: str,
    as_of: str,
    spot: float,
    can_compute_gex: bool,
) -> list[dict[str, Any]]:
    if cleaned.empty:
        return []

    out = cleaned.copy()
    out["_call_oi"] = out["open_interest"].where(out["option_type"] == "call", 0.0)
    out["_put_oi"] = out["open_interest"].where(out["option_type"] == "put", 0.0)

    if can_compute_gex:
        gex_factor = (spot**2) * 0.01 * 100.0
        out["_gamma_notional"] = out["bs_gamma"] * out["open_interest"] * gex_factor
        out["_call_gex"] = out["_gamma_notional"].where(out["option_type"] == "call", 0.0)
        out["_put_gex"] = out["_gamma_notional"].where(out["option_type"] == "put", 0.0)
    else:
        out["_call_gex"] = 0.0
        out["_put_gex"] = 0.0

    grouped = (
        out.groupby(["expiry", "strike"], as_index=False)[["_call_oi", "_put_oi", "_call_gex", "_put_gex"]]
        .sum()
        .sort_values(["expiry", "strike"], ascending=[True, True], kind="mergesort")
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        call_gex = float(row["_call_gex"]) if can_compute_gex else None
        put_gex = float(row["_put_gex"]) if can_compute_gex else None
        net_gex = (call_gex - put_gex) if call_gex is not None and put_gex is not None else None

        values: dict[str, Any] = {
            "symbol": symbol,
            "as_of": as_of,
            "expiry": str(row["expiry"]),
            "strike": float(row["strike"]),
            "call_oi": float(row["_call_oi"]),
            "put_oi": float(row["_put_oi"]),
            "call_gex": call_gex,
            "put_gex": put_gex,
            "net_gex": net_gex,
        }
        rows.append({field: values[field] for field in EXPOSURE_STRIKE_FIELDS})

    return rows


def _normalize_chain_rows(snapshot: pd.DataFrame, *, warnings: list[str]) -> pd.DataFrame:
    if snapshot is None or snapshot.empty:
        warnings.append("empty_snapshot")
        return pd.DataFrame(
            columns=["expiry_date", "expiry", "strike", "option_type", "open_interest", "bs_gamma"]
        )

    out = snapshot.copy()
    if "expiry" not in out.columns:
        warnings.append("missing_expiry")
        out["expiry"] = pd.NA
    if "strike" not in out.columns:
        warnings.append("missing_strike")
        out["strike"] = pd.NA
    if "optionType" not in out.columns:
        warnings.append("missing_optionType")
        out["optionType"] = pd.NA
    if "openInterest" not in out.columns:
        warnings.append("missing_openInterest")
        out["openInterest"] = 0.0
    if "bs_gamma" not in out.columns:
        warnings.append("missing_bs_gamma")
        out["bs_gamma"] = 0.0

    expiry_ts = pd.to_datetime(out["expiry"], errors="coerce")
    expiry_date = expiry_ts.dt.date
    expiry_str = expiry_date.map(lambda value: value.isoformat() if isinstance(value, date) else None)

    strike = pd.to_numeric(out["strike"], errors="coerce")
    option_type = out["optionType"].astype(str).str.lower().str.strip()
    option_type = option_type.where(option_type.isin({"call", "put"}))

    open_interest = pd.to_numeric(out["openInterest"], errors="coerce")
    if open_interest.isna().any():
        warnings.append("invalid_open_interest_coerced_to_zero")
    open_interest = open_interest.fillna(0.0)
    if (open_interest < 0).any():
        warnings.append("negative_open_interest_clamped_to_zero")
    open_interest = open_interest.clip(lower=0.0)

    gamma = pd.to_numeric(out["bs_gamma"], errors="coerce")
    if gamma.isna().any():
        warnings.append("invalid_bs_gamma_coerced_to_zero")
    gamma = gamma.fillna(0.0)

    cleaned = pd.DataFrame(
        {
            "expiry_date": expiry_date,
            "expiry": expiry_str,
            "strike": strike,
            "option_type": option_type,
            "open_interest": open_interest,
            "bs_gamma": gamma,
        }
    )

    invalid_expiry = int(cleaned["expiry_date"].isna().sum())
    invalid_strike = int(cleaned["strike"].isna().sum())
    invalid_option_type = int(cleaned["option_type"].isna().sum())
    if invalid_expiry > 0:
        warnings.append("invalid_expiry_dropped")
    if invalid_strike > 0:
        warnings.append("invalid_strike_dropped")
    if invalid_option_type > 0:
        warnings.append("invalid_option_type_dropped")

    cleaned = cleaned.dropna(subset=["expiry_date", "expiry", "strike", "option_type"]).copy()
    if cleaned.empty:
        warnings.append("no_valid_contract_rows")
    return cleaned


def _parse_include_expiries(
    include_expiries: Iterable[date | str] | None,
    *,
    warnings: list[str],
) -> list[date] | None:
    if include_expiries is None:
        return None

    out: list[date] = []
    invalid_count = 0
    for raw in include_expiries:
        if isinstance(raw, date):
            out.append(raw)
            continue
        parsed = pd.to_datetime(raw, errors="coerce")
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            out.append(parsed.date())
        else:
            invalid_count += 1

    if invalid_count > 0:
        warnings.append("invalid_include_expiry_ignored")
    return sorted(set(out))


def _normalize_as_of(as_of: date | str) -> str:
    if isinstance(as_of, date):
        return as_of.isoformat()
    return str(as_of)


def _to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _as_float_or_zero(value: object) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return 0.0
    return numeric


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "ExposureMode",
    "ExposureSlice",
    "NetExposureLevel",
    "aggregate_net_exposure_by_strike",
    "compute_exposure_slice",
    "compute_exposure_slices",
    "compute_flip_strike",
    "rank_top_abs_net_levels",
]
