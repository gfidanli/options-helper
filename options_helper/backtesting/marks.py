from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row


def _filter_dates(
    dates: Iterable[date],
    *,
    start: date | None,
    end: date | None,
) -> list[date]:
    out: list[date] = []
    for d in dates:
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        out.append(d)
    return out


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _positive_float(value: object) -> float | None:
    val = _as_float(value)
    if val is None:
        return None
    if val <= 0:
        return None
    return val


def _best_effort_mark(*, bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    if ask is not None:
        return ask
    if bid is not None:
        return bid
    return None


def _resolve_row(
    df: pd.DataFrame,
    *,
    contract_symbol: str | None,
    expiry: date | None,
    strike: float | None,
    option_type: str | None,
) -> dict | None:
    if df is None or df.empty:
        return None
    if contract_symbol and "contractSymbol" in df.columns:
        mask = df["contractSymbol"].astype(str) == str(contract_symbol)
        if mask.any():
            row = df.loc[mask].iloc[0]
            return row.to_dict()

    if expiry is None or strike is None or option_type is None:
        return None

    match = find_snapshot_row(
        df,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        contract_symbol=contract_symbol,
    )
    if match is None:
        return None
    return match.to_dict()


@dataclass(frozen=True)
class MarkSeries:
    symbol: str
    contract_symbol: str | None
    expiry: date | None
    strike: float | None
    option_type: str | None
    series: pd.DataFrame


def build_mark_series(
    snapshot_store: OptionsSnapshotStore,
    *,
    symbol: str,
    contract_symbol: str | None = None,
    expiry: date | None = None,
    strike: float | None = None,
    option_type: str | None = None,
    start: date | None = None,
    end: date | None = None,
    dates: Iterable[date] | None = None,
) -> MarkSeries:
    if contract_symbol is None and (expiry is None or strike is None or option_type is None):
        raise ValueError("contract_symbol or (expiry, strike, option_type) is required")

    if dates is None:
        dates = snapshot_store.list_dates(symbol)
    dates_list = _filter_dates(dates, start=start, end=end)

    rows: list[dict] = []
    index: list[pd.Timestamp] = []

    for snapshot_date in dates_list:
        day_df = snapshot_store.load_day(symbol, snapshot_date)
        row = _resolve_row(
            day_df,
            contract_symbol=contract_symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
        )
        has_contract = row is not None
        bid = ask = last = mark = spread = spread_pct = None

        if row is not None:
            bid = _positive_float(row.get("bid"))
            ask = _positive_float(row.get("ask"))
            last = _positive_float(row.get("lastPrice") or row.get("last"))
            mark = _positive_float(row.get("mark"))
            if mark is None:
                mark = _best_effort_mark(bid=bid, ask=ask, last=last)
            if bid is not None and ask is not None:
                spread = ask - bid
                mid = (ask + bid) / 2.0
                if mid > 0:
                    spread_pct = spread / mid

        rows.append(
            {
                "bid": bid,
                "ask": ask,
                "last": last,
                "mark": mark,
                "spread": spread,
                "spread_pct": spread_pct,
                "has_contract": has_contract,
                "has_quote": mark is not None,
            }
        )
        index.append(pd.Timestamp(snapshot_date))

    series = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="snapshot_date"))
    return MarkSeries(
        symbol=symbol,
        contract_symbol=contract_symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        series=series,
    )
