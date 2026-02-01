from __future__ import annotations

import pandas as pd


def forward_max_up_return(
    *,
    close_series: pd.Series,
    high_series: pd.Series,
    start_iloc: int,
    horizon_bars: int,
) -> float | None:
    """
    Max upside (MFE-style) return within the next `horizon_bars` bars, using High.

    Definition (anchored on entry close at bar i):
    - max_up = max(High[i+1 : i+horizon]) / Close[i] - 1

    Returns None if:
    - horizon is invalid,
    - series are too short to cover the full horizon,
    - entry close is missing/0.
    """
    horizon_bars = int(horizon_bars)
    start_iloc = int(start_iloc)
    if horizon_bars <= 0:
        return None
    if start_iloc < 0 or start_iloc >= len(close_series):
        return None
    # Require the full horizon (matches how forward-returns are treated elsewhere).
    end_iloc = start_iloc + horizon_bars
    if end_iloc >= len(high_series):
        return None

    c0 = close_series.iloc[start_iloc]
    try:
        if c0 is None or pd.isna(c0) or float(c0) == 0.0:
            return None
        c0_f = float(c0)
    except Exception:  # noqa: BLE001
        return None

    window = high_series.iloc[start_iloc + 1 : end_iloc + 1]
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

