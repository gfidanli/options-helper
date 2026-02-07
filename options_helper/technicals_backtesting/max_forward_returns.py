from __future__ import annotations

import pandas as pd


def forward_max_up_return(
    *,
    open_series: pd.Series,
    high_series: pd.Series,
    start_iloc: int,
    horizon_bars: int,
) -> float | None:
    """
    Max upside (MFE-style) return within the next `horizon_bars` bars, using High.

    Definition (anchored on next-bar open after signal at bar i):
    - entry = Open[i+1]
    - max_up = max(High[i+1 : i+horizon]) / entry - 1

    Returns None if:
    - horizon is invalid,
    - series are too short to cover the full horizon,
    - entry open is missing/0.
    """
    horizon_bars = int(horizon_bars)
    start_iloc = int(start_iloc)
    if horizon_bars <= 0:
        return None
    if start_iloc < 0 or start_iloc >= len(open_series):
        return None
    entry_iloc = start_iloc + 1
    if entry_iloc >= len(open_series):
        return None
    # Require the full horizon (matches how forward-returns are treated elsewhere).
    end_iloc = start_iloc + horizon_bars
    if end_iloc >= len(high_series):
        return None

    c0 = open_series.iloc[entry_iloc]
    try:
        if c0 is None or pd.isna(c0) or float(c0) == 0.0:
            return None
        c0_f = float(c0)
    except Exception:  # noqa: BLE001
        return None

    window = high_series.iloc[entry_iloc : end_iloc + 1]
    if window.empty:
        return None
    hi = window.max()
    try:
        if hi is None or pd.isna(hi):
            return None
        hi_f = float(hi)
    except Exception:  # noqa: BLE001
        return None

    return hi_f / c0_f - 1.0


def forward_max_up_move(
    *,
    open_series: pd.Series,
    high_series: pd.Series,
    start_iloc: int,
    horizon_bars: int,
) -> float | None:
    """
    Max *favorable* upside move within the next `horizon_bars` bars (non-negative), using High.

    This is `max(0, forward_max_up_return(...))`.
    """
    r = forward_max_up_return(
        open_series=open_series,
        high_series=high_series,
        start_iloc=start_iloc,
        horizon_bars=horizon_bars,
    )
    if r is None:
        return None
    return max(0.0, float(r))


def forward_max_down_move(
    *,
    open_series: pd.Series,
    low_series: pd.Series,
    start_iloc: int,
    horizon_bars: int,
) -> float | None:
    """
    Max *favorable* downside move within the next `horizon_bars` bars (non-negative), using Low.

    Definition (anchored on next-bar open after signal at bar i):
    - entry = Open[i+1]
    - min_low = min(Low[i+1 : i+horizon])
    - down_move = max(0, -(min_low / entry - 1))

    This returns a positive number for pullbacks (e.g., 0.10 = 10% drop) and 0 when
    price never traded below the entry open in the lookahead window.
    """
    horizon_bars = int(horizon_bars)
    start_iloc = int(start_iloc)
    if horizon_bars <= 0:
        return None
    if start_iloc < 0 or start_iloc >= len(open_series):
        return None
    entry_iloc = start_iloc + 1
    if entry_iloc >= len(open_series):
        return None
    end_iloc = start_iloc + horizon_bars
    if end_iloc >= len(low_series):
        return None

    c0 = open_series.iloc[entry_iloc]
    try:
        if c0 is None or pd.isna(c0) or float(c0) == 0.0:
            return None
        c0_f = float(c0)
    except Exception:  # noqa: BLE001
        return None

    window = low_series.iloc[entry_iloc : end_iloc + 1]
    if window.empty:
        return None
    lo = window.min()
    try:
        if lo is None or pd.isna(lo):
            return None
        lo_f = float(lo)
    except Exception:  # noqa: BLE001
        return None

    r = lo_f / c0_f - 1.0
    return max(0.0, -float(r))
