from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class LiquidityResult:
    is_liquid: bool
    eligible_contracts: int
    eligible_expiries: list[str]
    max_dte: int | None


def evaluate_liquidity(
    df: pd.DataFrame,
    *,
    snapshot_date: date,
    min_dte: int,
    min_volume: int,
    min_open_interest: int,
) -> LiquidityResult:
    if df is None or df.empty:
        return LiquidityResult(is_liquid=False, eligible_contracts=0, eligible_expiries=[], max_dte=None)

    expiry_raw = df.get("expiry")
    if expiry_raw is None:
        return LiquidityResult(is_liquid=False, eligible_contracts=0, eligible_expiries=[], max_dte=None)

    expiry_dt = pd.to_datetime(expiry_raw, errors="coerce")
    dte = (expiry_dt - pd.Timestamp(snapshot_date)).dt.days

    volume_raw = df.get("volume")
    oi_raw = df.get("openInterest")
    volume = pd.to_numeric(volume_raw, errors="coerce").fillna(0.0) if volume_raw is not None else pd.Series(0.0, index=df.index)
    oi = pd.to_numeric(oi_raw, errors="coerce").fillna(0.0) if oi_raw is not None else pd.Series(0.0, index=df.index)

    mask = (dte >= int(min_dte)) & (volume >= float(min_volume)) & (oi >= float(min_open_interest))
    if not mask.any():
        return LiquidityResult(is_liquid=False, eligible_contracts=0, eligible_expiries=[], max_dte=None)

    eligible_expiry_dt = expiry_dt.loc[mask]
    elig_exp = eligible_expiry_dt.dt.date.dropna().astype(str).unique().tolist()
    elig_exp_sorted = sorted({e for e in elig_exp if e})

    max_dte = int(dte[mask].max()) if dte[mask].notna().any() else None
    return LiquidityResult(
        is_liquid=True,
        eligible_contracts=int(mask.sum()),
        eligible_expiries=elig_exp_sorted,
        max_dte=max_dte,
    )
