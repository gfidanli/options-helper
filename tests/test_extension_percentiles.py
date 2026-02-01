from __future__ import annotations

import pandas as pd

from options_helper.technicals_backtesting.extension_percentiles import (
    compute_extension_percentiles,
    rolling_percentile_rank,
)


def test_rolling_percentile_rank_increasing_series() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    s = pd.Series(range(1, 11), index=idx, dtype="float64")
    pct = rolling_percentile_rank(s, window=10)
    assert round(float(pct.iloc[-1]), 1) == 95.0  # (9 + 0.5) / 10 * 100


def test_extension_percentiles_tail_events_toggle() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    ext = pd.Series(range(1, 31), index=idx, dtype="float64")
    close = pd.Series(range(100, 130), index=idx, dtype="float64")

    report = compute_extension_percentiles(
        extension_series=ext,
        close_series=close,
        windows_years=[1],
        days_per_year=10,
        tail_high_pct=90,
        tail_low_pct=10,
        forward_days=[1, 3],
        include_tail_events=True,
    )
    assert report.tail_events

    report_no_tail = compute_extension_percentiles(
        extension_series=ext,
        close_series=close,
        windows_years=[1],
        days_per_year=10,
        tail_high_pct=90,
        tail_low_pct=10,
        forward_days=[1, 3],
        include_tail_events=False,
    )
    assert report_no_tail.tail_events == []
