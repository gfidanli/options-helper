from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

FillMode = Literal["worst_case", "mark_slippage"]
FillSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class FillResult:
    price: float | None
    reason: str
    used: str | None = None
    warnings: list[str] = field(default_factory=list)


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _positive_float(value: object) -> float | None:
    val = _as_float(value)
    if val is None or val <= 0:
        return None
    return val


def _best_effort_mark(
    *,
    bid: float | None,
    ask: float | None,
    last: float | None,
    mark: float | None,
) -> float | None:
    if mark is not None:
        return mark
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    if ask is not None:
        return ask
    if bid is not None:
        return bid
    return None


def _spread_pct_from_bid_ask(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def fill_price(
    *,
    side: FillSide,
    fill_mode: FillMode,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    mark: float | None = None,
    spread_pct: float | None = None,
    slippage_factor: float = 0.5,
    allow_worst_case_mark_fallback: bool = False,
) -> FillResult:
    bid_val = _positive_float(bid)
    ask_val = _positive_float(ask)
    last_val = _positive_float(last)
    mark_val = _positive_float(mark)

    if fill_mode == "worst_case":
        price = ask_val if side == "buy" else bid_val
        if price is not None:
            return FillResult(price=price, reason="ok", used="ask" if side == "buy" else "bid")
        if not allow_worst_case_mark_fallback:
            return FillResult(price=None, reason="missing_bid_ask", warnings=["missing_bid_ask"])
        mark_fallback = _best_effort_mark(
            bid=bid_val,
            ask=ask_val,
            last=last_val,
            mark=mark_val,
        )
        if mark_fallback is None:
            return FillResult(price=None, reason="missing_quote", warnings=["missing_bid_ask", "missing_mark"])
        return FillResult(
            price=mark_fallback,
            reason="fallback_mark",
            used="mark",
            warnings=["worst_case_fallback"],
        )

    mark_final = _best_effort_mark(bid=bid_val, ask=ask_val, last=last_val, mark=mark_val)
    if mark_final is None:
        return FillResult(price=None, reason="missing_mark", warnings=["missing_mark"])

    spread_final = _as_float(spread_pct)
    if spread_final is None:
        spread_final = _spread_pct_from_bid_ask(bid_val, ask_val)
    if spread_final is None:
        spread_final = 0.0

    warnings: list[str] = []
    if spread_final < 0:
        warnings.append("negative_spread_pct")
        spread_final = 0.0

    slip = max(0.0, float(slippage_factor)) * float(spread_final)
    if side == "buy":
        price = mark_final * (1.0 + slip)
    else:
        price = mark_final * (1.0 - slip)

    if price <= 0:
        warnings.append("non_positive_fill")
        return FillResult(price=None, reason="invalid_fill", warnings=warnings)

    used = "mark_slippage" if slip > 0 else "mark"
    return FillResult(price=price, reason="ok", used=used, warnings=warnings)

