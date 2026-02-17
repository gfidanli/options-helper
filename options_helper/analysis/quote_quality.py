from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from options_helper.analysis.chain_metrics import compute_spread, compute_spread_pct


def _col_as_float(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def _parse_last_trade_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    numeric = pd.to_numeric(series, errors="coerce")
    needs_numeric = parsed.isna() & numeric.notna()
    if needs_numeric.any():
        max_val = numeric[needs_numeric].max()
        if pd.isna(max_val):
            return parsed
        if max_val < 1e11:
            unit = "s"
        elif max_val < 1e14:
            unit = "ms"
        else:
            unit = "ns"
        parsed_numeric = pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit)
        parsed = parsed.where(~needs_numeric, parsed_numeric)
    return parsed


def _business_days_since(last_trade: pd.Series, *, as_of: date) -> tuple[pd.Series, pd.Series]:
    age = pd.Series([float("nan")] * len(last_trade), index=last_trade.index, dtype="float64")
    future_mask = pd.Series([False] * len(last_trade), index=last_trade.index, dtype="bool")

    if last_trade.empty:
        return age, future_mask

    last_dt = last_trade
    try:
        if last_dt.dt.tz is not None:
            last_dt = last_dt.dt.tz_convert(None)
    except Exception:  # noqa: BLE001
        last_dt = last_dt

    last_dt = last_dt.dt.normalize()
    valid = last_dt.notna()
    if not valid.any():
        return age, future_mask

    last_days = last_dt[valid].values.astype("datetime64[D]")
    as_of_np = np.datetime64(as_of)
    raw = np.busday_count(last_days, as_of_np)
    future = raw < 0
    raw = np.where(future, 0, raw)

    age.loc[valid] = raw.astype(float)
    future_mask.loc[valid] = future
    return age, future_mask


@dataclass(frozen=True)
class _QuoteQualityState:
    spread: pd.Series
    spread_pct: pd.Series
    last_trade_age_days: pd.Series
    has_bid_ask: pd.Series
    has_last: pd.Series
    volume: pd.Series
    open_interest: pd.Series
    low_liquidity: pd.Series
    stale_mask: pd.Series
    invalid_spread: pd.Series
    future_trade: pd.Series


def _empty_quote_quality(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spread": pd.Series(dtype="float64"),
            "spread_pct": pd.Series(dtype="float64"),
            "last_trade_age_days": pd.Series(dtype="float64"),
            "quality_score": pd.Series(dtype="float64"),
            "quality_label": pd.Series(dtype="object"),
            "quality_warnings": pd.Series(dtype="object"),
        },
        index=df.index if df is not None else None,
    )


def _build_quote_quality_state(
    df: pd.DataFrame,
    *,
    min_volume: int,
    min_open_interest: int,
    as_of: date,
) -> _QuoteQualityState:
    bid = _col_as_float(df, "bid")
    ask = _col_as_float(df, "ask")
    last = _col_as_float(df, "lastPrice")
    volume = _col_as_float(df, "volume")
    open_interest = _col_as_float(df, "openInterest")
    spread = compute_spread(df)
    spread_pct = compute_spread_pct(df)
    has_bid_ask = (bid > 0) & (ask > 0)
    has_last = last > 0
    if "lastTradeDate" in df.columns:
        last_trade = _parse_last_trade_dates(df["lastTradeDate"])
    else:
        last_trade = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
    last_trade_age_days, future_trade = _business_days_since(last_trade, as_of=as_of)
    vol_low = volume.notna() & (volume < min_volume)
    oi_low = open_interest.notna() & (open_interest < min_open_interest)
    low_liquidity = vol_low | oi_low
    stale_mask = last_trade_age_days > 5
    invalid_spread = spread_pct.notna() & (spread_pct < 0)
    return _QuoteQualityState(
        spread=spread,
        spread_pct=spread_pct,
        last_trade_age_days=last_trade_age_days,
        has_bid_ask=has_bid_ask,
        has_last=has_last,
        volume=volume,
        open_interest=open_interest,
        low_liquidity=low_liquidity,
        stale_mask=stale_mask,
        invalid_spread=invalid_spread,
        future_trade=future_trade,
    )


def _compute_quality_scores(state: _QuoteQualityState) -> tuple[pd.Series, pd.Series]:
    scores = pd.Series(100.0, index=state.spread.index, dtype="float64")
    missing_bid_ask = ~state.has_bid_ask
    scores = scores.where(~missing_bid_ask, 40.0)
    spread_penalty = pd.Series(0.0, index=state.spread.index, dtype="float64")
    mid_spread = (state.spread_pct > 0.15) & (state.spread_pct <= 0.35)
    wide_spread = state.spread_pct > 0.35
    spread_penalty = spread_penalty.where(~mid_spread, 20.0)
    spread_penalty = spread_penalty.where(~wide_spread, 40.0)
    spread_penalty = spread_penalty.where(~state.invalid_spread, 40.0)
    scores = scores - spread_penalty
    scores = scores - (state.stale_mask.astype("float64") * 30.0)
    scores = scores - (state.low_liquidity.astype("float64") * 20.0)
    return scores.clip(lower=0.0, upper=100.0), missing_bid_ask


def _compute_quality_labels(
    *,
    state: _QuoteQualityState,
    scores: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    has_signal = (
        state.has_bid_ask
        | state.spread_pct.notna()
        | state.last_trade_age_days.notna()
        | state.volume.notna()
        | state.open_interest.notna()
    )
    unknown_mask = ~has_signal & ~state.has_last
    labels = pd.Series(["unknown"] * len(scores), index=scores.index, dtype="object")
    known = ~unknown_mask
    labels = labels.mask(known & (scores >= 80), "good")
    labels = labels.mask(known & (scores >= 50) & (scores < 80), "ok")
    labels = labels.mask(known & (scores < 50), "bad")
    return labels, unknown_mask


def _compute_quality_warnings(
    index: pd.Index,
    *,
    missing_bid_ask: pd.Series,
    invalid_spread: pd.Series,
    future_trade: pd.Series,
    stale_mask: pd.Series,
) -> list[list[str]]:
    warnings_list: list[list[str]] = []
    for idx in index:
        row_warnings: list[str] = []
        if missing_bid_ask.loc[idx]:
            row_warnings.append("quote_missing_bid_ask")
        if invalid_spread.loc[idx] or future_trade.loc[idx]:
            row_warnings.append("quote_invalid")
        if stale_mask.loc[idx]:
            row_warnings.append("quote_stale")
        warnings_list.append(row_warnings)
    return warnings_list


def compute_quote_quality(
    df: pd.DataFrame,
    *,
    min_volume: int,
    min_open_interest: int,
    as_of: date | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_quote_quality(df)
    as_of = as_of or date.today()
    state = _build_quote_quality_state(
        df,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        as_of=as_of,
    )
    scores, missing_bid_ask = _compute_quality_scores(state)
    labels, unknown_mask = _compute_quality_labels(state=state, scores=scores)
    scores = scores.where(~unknown_mask)
    warnings_list = _compute_quality_warnings(
        df.index,
        missing_bid_ask=missing_bid_ask,
        invalid_spread=state.invalid_spread,
        future_trade=state.future_trade,
        stale_mask=state.stale_mask,
    )
    return pd.DataFrame(
        {
            "spread": state.spread,
            "spread_pct": state.spread_pct,
            "last_trade_age_days": state.last_trade_age_days,
            "quality_score": scores,
            "quality_label": labels,
            "quality_warnings": warnings_list,
        },
        index=df.index,
    )
