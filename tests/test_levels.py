from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_helper.analysis.levels import (
    compute_anchored_vwap,
    compute_gap_and_daily_levels,
    compute_levels_summary,
    compute_relative_strength_beta_corr,
    compute_volume_profile,
)


def _intraday_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 14:30:00+00:00", periods=3, freq="min")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [10.0, 10.8, 11.8],
            "high": [10.2, 11.5, 12.2],
            "low": [9.8, 10.5, 11.7],
            "close": [10.0, 11.0, 12.0],
            "vwap": [10.0, np.nan, 12.0],
            "volume": [100.0, 200.0, 100.0],
        }
    )


def _daily_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "open": [100.0, 101.0, 102.0, 103.0, 110.0],
            "high": [101.0, 102.0, 103.0, 104.0, 112.0],
            "low": [99.0, 100.0, 101.0, 102.0, 108.0],
            "close": [100.0, 101.0, 102.0, 104.0, 111.0],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_500],
        }
    )


def _make_close_daily(close: np.ndarray, *, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 1_000.0),
        }
    )


def test_compute_anchored_vwap_session_open_uses_vwap_then_typical_proxy() -> None:
    bars = _intraday_frame()

    result = compute_anchored_vwap(bars, anchor_type="session_open", spot=11.0)

    assert result.anchor_ts_utc == pd.Timestamp("2026-01-05 14:30:00+00:00")
    assert result.anchor_price == pytest.approx(10.0)
    assert result.anchored_vwap == pytest.approx(11.0)
    assert result.distance_from_spot_pct == pytest.approx(0.0)
    assert result.warnings == []


def test_compute_anchored_vwap_timestamp_anchor_starts_at_next_bar() -> None:
    bars = _intraday_frame()

    result = compute_anchored_vwap(
        bars,
        anchor_type="timestamp",
        anchor_timestamp="2026-01-05 14:31:30+00:00",
    )

    assert result.anchor_ts_utc == pd.Timestamp("2026-01-05 14:32:00+00:00")
    assert result.anchored_vwap == pytest.approx(12.0)


def test_compute_anchored_vwap_supports_date_and_breakout_day_anchor() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-05 14:30:00+00:00",
                    "2026-01-05 14:31:00+00:00",
                    "2026-01-06 14:30:00+00:00",
                    "2026-01-06 14:31:00+00:00",
                ]
            ),
            "close": [100.0, 101.0, 103.0, 104.0],
            "high": [100.5, 101.5, 103.5, 104.5],
            "low": [99.5, 100.5, 102.5, 103.5],
            "vwap": [100.0, 101.0, 103.0, 104.0],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )

    breakout_daily = pd.DataFrame(
        {
            "date": pd.date_range("2025-12-30", periods=8, freq="D"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 110.0],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 110.0],
            "high": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 110.0],
            "low": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 110.0],
            "volume": [1_000] * 8,
        }
    )

    date_result = compute_anchored_vwap(bars, anchor_type="date", anchor_date=date(2026, 1, 6))
    breakout_result = compute_anchored_vwap(
        bars,
        anchor_type="breakout_day",
        breakout_daily=breakout_daily,
        breakout_lookback=5,
    )

    assert date_result.anchor_ts_utc == pd.Timestamp("2026-01-06 14:30:00+00:00")
    assert date_result.anchored_vwap == pytest.approx((103.0 * 30.0 + 104.0 * 40.0) / 70.0)

    assert breakout_result.anchor_ts_utc == pd.Timestamp("2026-01-06 14:30:00+00:00")
    assert breakout_result.anchored_vwap == pytest.approx((103.0 * 30.0 + 104.0 * 40.0) / 70.0)


def test_compute_anchored_vwap_zero_volume_returns_none_with_warning() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 14:30:00+00:00", periods=3, freq="min"),
            "close": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "volume": [0.0, np.nan, 0.0],
        }
    )

    result = compute_anchored_vwap(bars)

    assert result.anchored_vwap is None
    assert "zero_volume_after_anchor" in result.warnings


def test_compute_volume_profile_identifies_poc_hvn_lvn() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 14:30:00+00:00", periods=5, freq="min"),
            "close": [100.0, 100.4, 100.6, 101.2, 101.6],
            "high": [100.1, 100.5, 100.7, 101.3, 101.7],
            "low": [99.9, 100.3, 100.5, 101.1, 101.5],
            "volume": [100.0, 500.0, 200.0, 50.0, 25.0],
        }
    )

    result = compute_volume_profile(bars, num_bins=4)

    assert len(result.bins) == 4
    assert result.poc_price is not None
    assert sum(1 for row in result.bins if row.is_poc) == 1
    assert result.hvn_candidates
    assert result.lvn_candidates
    assert sum(row.volume_pct for row in result.bins) == pytest.approx(1.0)


def test_compute_volume_profile_zero_volume_returns_warning() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 14:30:00+00:00", periods=3, freq="min"),
            "close": [100.0, 101.0, np.nan],
            "high": [100.0, 101.0, np.nan],
            "low": [100.0, 101.0, np.nan],
            "volume": [0.0, np.nan, 0.0],
        }
    )

    result = compute_volume_profile(bars, num_bins=5)

    assert result.bins == []
    assert result.poc_price is None
    assert "zero_volume_profile" in result.warnings


def test_compute_gap_and_daily_levels_returns_gap_prior_and_rolling_levels() -> None:
    result = compute_gap_and_daily_levels(_daily_frame(), rolling_window=3)

    assert result.spot == pytest.approx(111.0)
    assert result.prev_close == pytest.approx(104.0)
    assert result.session_open == pytest.approx(110.0)
    assert result.gap_pct == pytest.approx((110.0 - 104.0) / 104.0)
    assert result.prior_high == pytest.approx(104.0)
    assert result.prior_low == pytest.approx(102.0)
    assert result.rolling_high == pytest.approx(104.0)
    assert result.rolling_low == pytest.approx(100.0)


def test_compute_relative_strength_beta_corr_returns_expected_linear_metrics() -> None:
    benchmark_returns = np.array(
        [0.01, -0.004, 0.006, -0.003, 0.005, -0.002, 0.004, -0.001, 0.003, -0.002] * 4,
        dtype="float64",
    )
    asset_returns = benchmark_returns * 2.0

    benchmark_close = 100.0 * np.cumprod(1.0 + benchmark_returns)
    asset_close = 50.0 * np.cumprod(1.0 + asset_returns)

    benchmark_daily = _make_close_daily(benchmark_close)
    asset_daily = _make_close_daily(asset_close)

    result = compute_relative_strength_beta_corr(asset_daily, benchmark_daily, window=20)

    assert result.rs_ratio == pytest.approx(float(asset_close[-1] / benchmark_close[-1]))
    assert result.beta is not None
    assert result.corr is not None
    assert result.beta == pytest.approx(2.0, rel=1e-6)
    assert result.corr == pytest.approx(1.0, rel=1e-6)


def test_compute_relative_strength_beta_corr_handles_short_history() -> None:
    benchmark_close = np.array([100.0, 101.0, 100.5, 101.2, 101.0], dtype="float64")
    asset_close = np.array([50.0, 51.0, 50.0, 51.5, 51.0], dtype="float64")

    result = compute_relative_strength_beta_corr(
        _make_close_daily(asset_close),
        _make_close_daily(benchmark_close),
        window=20,
    )

    assert result.beta is None
    assert result.corr is None
    assert "insufficient_return_history" in result.warnings


def test_compute_levels_summary_combines_daily_and_relative_strength() -> None:
    daily = _daily_frame()
    benchmark = daily.copy()
    benchmark["close"] = benchmark["close"] * 0.9

    result = compute_levels_summary(daily, benchmark_daily=benchmark, rolling_window=3, rs_window=3)

    assert result.spot == pytest.approx(111.0)
    assert result.prev_close == pytest.approx(104.0)
    assert result.gap_pct == pytest.approx((110.0 - 104.0) / 104.0)
    assert result.rs_ratio == pytest.approx(111.0 / (111.0 * 0.9))
    assert result.beta_20d is not None
    assert result.corr_20d is not None
