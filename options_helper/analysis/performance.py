from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyPerformanceQuote:
    last_price: float | None
    change: float | None
    percent_change: float | None  # percent points, e.g. 5.4 means +5.4%
    prev_close_price: float | None
    daily_pnl: float | None


def compute_daily_performance_quote(
    *,
    last_price: float | None,
    change: float | None,
    percent_change_raw: float | None,
    contracts: int,
) -> DailyPerformanceQuote:
    """
    Compute a best-effort daily performance snapshot for an option contract.

    Conventions:
    - `percent_change` is returned as percent points (not a 0-1 fraction).
    - If `last_price` + `change` are known, `percent_change` is derived from those values.
    - Otherwise, `percent_change_raw` is passed through (assumed percent points as commonly returned by yfinance).
    """
    prev_close_price = None
    pct_points = percent_change_raw
    daily_pnl = None

    if last_price is not None and change is not None:
        prev_close_price = last_price - change
        daily_pnl = change * 100.0 * contracts
        if prev_close_price and prev_close_price > 0:
            pct_points = 100.0 * change / prev_close_price

    return DailyPerformanceQuote(
        last_price=last_price,
        change=change,
        percent_change=pct_points,
        prev_close_price=prev_close_price,
        daily_pnl=daily_pnl,
    )

