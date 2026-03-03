from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


def _rolling_max(values: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be > 0")
    series = pd.Series(values, copy=False)
    return series.rolling(window=window, min_periods=window).max().to_numpy(dtype=float, copy=False)


def _rolling_mean(values: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be > 0")
    series = pd.Series(values, copy=False)
    return series.rolling(window=window, min_periods=window).mean().to_numpy(dtype=float, copy=False)


def _compute_ibs(close: np.ndarray, low: np.ndarray, high: np.ndarray, *, zero_range_value: float) -> np.ndarray:
    out = np.full(close.shape[0], np.nan, dtype=float)
    spread = high - low
    finite = np.isfinite(close) & np.isfinite(low) & np.isfinite(high)
    positive_range = finite & (spread > 0.0)
    out[positive_range] = (close[positive_range] - low[positive_range]) / spread[positive_range]
    zero_or_negative = finite & (spread <= 0.0)
    out[zero_or_negative] = float(zero_range_value)
    return out


class MeanReversionIBS(Strategy):
    """
    Long-only mean reversion strategy using IBS and volatility-adjusted pullback.

    Canonical defaults:
    - Entry: close < (rolling_high_10 - 2.5 * avg(high-low, 25)) and IBS < 0.3
    - Exit: close > yesterday_high
    - IBS fallback for high == low: 0.5
    """

    lookback_high = 10
    range_window = 25
    range_mult = 2.5
    ibs_threshold = 0.30
    exit_lookback = 1
    ibs_zero_range_value = 0.5

    def init(self) -> None:
        self._lookback_high = max(1, int(self.lookback_high))
        self._range_window = max(1, int(self.range_window))
        self._range_mult = float(self.range_mult)
        self._ibs_threshold = float(self.ibs_threshold)
        self._exit_lookback = max(1, int(self.exit_lookback))
        self._ibs_zero_range_value = float(self.ibs_zero_range_value)

        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)
        close = np.asarray(self.data.Close, dtype=float)
        intrabar_range = high - low

        self.rolling_high = self.I(
            lambda x: x,
            _rolling_max(high, window=self._lookback_high),
            name=f"rolling_high_{self._lookback_high}",
        )
        self.avg_range = self.I(
            lambda x: x,
            _rolling_mean(intrabar_range, window=self._range_window),
            name=f"avg_range_{self._range_window}",
        )
        self.ibs = self.I(
            lambda x: x,
            _compute_ibs(close, low, high, zero_range_value=self._ibs_zero_range_value),
            name="ibs",
        )

    def next(self) -> None:
        if len(self.data) <= self._exit_lookback:
            return

        close_now = float(self.data.Close[-1])
        rolling_high_now = float(self.rolling_high[-1]) if np.isfinite(self.rolling_high[-1]) else np.nan
        avg_range_now = float(self.avg_range[-1]) if np.isfinite(self.avg_range[-1]) else np.nan
        ibs_now = float(self.ibs[-1]) if np.isfinite(self.ibs[-1]) else np.nan

        if not self.position:
            if np.isnan(rolling_high_now) or np.isnan(avg_range_now) or np.isnan(ibs_now):
                return
            threshold = rolling_high_now - self._range_mult * avg_range_now
            if close_now < threshold and ibs_now < self._ibs_threshold:
                self.buy()
            return

        yesterday_high = float(self.data.High[-(self._exit_lookback + 1)])
        if np.isfinite(yesterday_high) and close_now > yesterday_high:
            self.position.close()
