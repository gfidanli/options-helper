from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_yfinance_price_adjustment(df: pd.DataFrame, *, auto_adjust: bool, back_adjust: bool) -> pd.DataFrame:
    """
    Apply yfinance-style price adjustment locally.

    This is a pure transformation used to ensure indicators/backtests operate on the
    intended price series even when the upstream cache stored unadjusted OHLC.

    Behavior:
    - auto_adjust=True: adjust Open/High/Low/Close by (Adj Close / Close), replace Close with Adj Close, drop Adj Close
      (mirrors yfinance.utils.auto_adjust)
    - back_adjust=True: adjust Open/High/Low by (Adj Close / Close), keep Close, drop Adj Close
      (mirrors yfinance.utils.back_adjust)
    - If Adj Close is missing, returns df unchanged.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    auto_adjust = bool(auto_adjust)
    back_adjust = bool(back_adjust)
    if auto_adjust and back_adjust:
        raise ValueError("Invalid price adjustment config: auto_adjust and back_adjust cannot both be true")

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    close_col = lower_cols.get("close")
    adj_col = None
    for key in ("adj close", "adj_close", "adjclose"):
        if key in lower_cols:
            adj_col = lower_cols[key]
            break

    if close_col is None or adj_col is None:
        return df.copy()

    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    adj_close = pd.to_numeric(out[adj_col], errors="coerce")

    ratio = adj_close / close
    ratio = ratio.where(ratio.notna() & (ratio != 0.0), other=1.0)

    if auto_adjust:
        for col_key in ("open", "high", "low", "close"):
            col = lower_cols.get(col_key)
            if col is None:
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
        # Replace Close with Adj Close to match yfinance.auto_adjust output.
        out[close_col] = adj_close
        out = out.drop(columns=[adj_col])
        return out

    if back_adjust:
        for col_key in ("open", "high", "low"):
            col = lower_cols.get(col_key)
            if col is None:
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
        out = out.drop(columns=[adj_col])
        return out

    return df.copy()


def standardize_ohlc(
    df: pd.DataFrame,
    *,
    required_cols: Iterable[str],
    optional_cols: Iterable[str] | None = None,
    dropna_ohlc: bool = True,
) -> pd.DataFrame:
    if df is None:
        raise ValueError("OHLC input is None")
    if df.empty:
        return df.copy()

    out = df.copy()

    # Normalize column names.
    col_map: dict[str, str] = {}
    lower_cols = {str(c).strip().lower(): c for c in out.columns}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            col_map[col] = key.title()
        elif key in {"adj close", "adj_close", "adjclose"} and "close" not in lower_cols:
            col_map[col] = "Close"
    out = out.rename(columns=col_map)

    required = [c.title() for c in required_cols]
    optional = [c.title() for c in (optional_cols or [])]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    out = _normalize_ohlc_datetime_index(out)

    if out.index.has_duplicates:
        dupes = out.index.duplicated(keep="last")
        logger.warning("Dropping %s duplicate OHLC rows", dupes.sum())
        out = out.loc[~dupes]

    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    # Drop rows with missing OHLC if configured.
    if dropna_ohlc:
        before = len(out)
        out = out.dropna(subset=required)
        dropped = before - len(out)
        if dropped:
            logger.warning("Dropped %s OHLC rows with missing values", dropped)

    # Enforce numeric dtypes for price columns.
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in optional:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    ordered = [c for c in required + optional if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    out = out[ordered]

    # Basic sanity checks.
    high = out["High"]
    low = out["Low"]
    open_ = out["Open"]
    close = out["Close"]
    valid = ~(high.isna() | low.isna() | open_.isna() | close.isna())
    if valid.any():
        high_bad = (high[valid] < np.maximum(open_[valid], close[valid])).sum()
        low_bad = (low[valid] > np.minimum(open_[valid], close[valid])).sum()
        if high_bad:
            logger.warning("High < max(Open, Close) on %s rows", high_bad)
        if low_bad:
            logger.warning("Low > min(Open, Close) on %s rows", low_bad)

    return out


def _normalize_ohlc_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    if not isinstance(out.index, pd.DatetimeIndex):
        lower_names = {str(c).lower() for c in out.columns}
        for candidate in ("date", "datetime", "timestamp"):
            if candidate not in lower_names:
                continue
            col_name = next(c for c in out.columns if str(c).lower() == candidate)
            idx = pd.to_datetime(out[col_name], errors="coerce", utc=True)
            out = out.drop(columns=[col_name])
            out.index = idx
            break
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLC DataFrame must have a DatetimeIndex or a date-like column")
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    if idx.isna().all():
        raise ValueError("Unable to parse OHLC index as datetimes")
    out = out.loc[~idx.isna()].copy()
    out.index = idx[~idx.isna()].tz_localize(None)
    return out
