from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


def _coerce_bool_param(name: str, value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 1") from exc
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer >= 1")
    out = int(numeric)
    if out < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return out


def _coerce_gate_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


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
    use_sma_trend_gate = False
    sma_trend_window = 200
    use_weekly_trend_gate = False
    weekly_trend_col = "weekly_trend_up"
    use_ma_direction_gate = False
    ma_direction_window = 200
    ma_direction_lookback = 1

    def init(self) -> None:
        self._lookback_high = max(1, int(self.lookback_high))
        self._range_window = max(1, int(self.range_window))
        self._range_mult = float(self.range_mult)
        self._ibs_threshold = float(self.ibs_threshold)
        self._exit_lookback = max(1, int(self.exit_lookback))
        self._ibs_zero_range_value = float(self.ibs_zero_range_value)
        self._use_sma_trend_gate = _coerce_bool_param("use_sma_trend_gate", self.use_sma_trend_gate)
        self._use_weekly_trend_gate = _coerce_bool_param("use_weekly_trend_gate", self.use_weekly_trend_gate)
        self._use_ma_direction_gate = _coerce_bool_param("use_ma_direction_gate", self.use_ma_direction_gate)
        self._sma_trend_window = _coerce_positive_int("sma_trend_window", self.sma_trend_window)
        self._ma_direction_window = _coerce_positive_int("ma_direction_window", self.ma_direction_window)
        self._ma_direction_lookback = _coerce_positive_int("ma_direction_lookback", self.ma_direction_lookback)

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
        self.sma_trend = None
        if self._use_sma_trend_gate:
            self.sma_trend = self.I(
                lambda x: x,
                _rolling_mean(close, window=self._sma_trend_window),
                name=f"sma_trend_{self._sma_trend_window}",
            )

        self.weekly_trend = None
        self._weekly_trend_col = str(self.weekly_trend_col).strip()
        if self._use_weekly_trend_gate:
            if not self._weekly_trend_col:
                raise ValueError("weekly_trend_col must be a non-empty string when use_weekly_trend_gate is true")
            if not hasattr(self.data, "df") or self._weekly_trend_col not in self.data.df.columns:
                raise ValueError(
                    f"weekly trend gate requires column '{self._weekly_trend_col}' in the strategy input frame"
                )
            self.weekly_trend = self.I(
                lambda x: x,
                self.data[self._weekly_trend_col],
                name=self._weekly_trend_col,
            )

        self.ma_direction = None
        if self._use_ma_direction_gate:
            self.ma_direction = self.I(
                lambda x: x,
                _rolling_mean(close, window=self._ma_direction_window),
                name=f"ma_direction_{self._ma_direction_window}",
            )

    def _sma_trend_gate_ok(self, close_now: float) -> bool:
        if not self._use_sma_trend_gate:
            return True
        if self.sma_trend is None:
            return False
        sma_now = float(self.sma_trend[-1]) if np.isfinite(self.sma_trend[-1]) else np.nan
        return bool(np.isfinite(sma_now) and close_now > sma_now)

    def _weekly_trend_gate_ok(self) -> bool:
        if not self._use_weekly_trend_gate:
            return True
        if self.weekly_trend is None:
            return False
        return _coerce_gate_bool(self.weekly_trend[-1])

    def _ma_direction_gate_ok(self) -> bool:
        if not self._use_ma_direction_gate:
            return True
        if self.ma_direction is None or len(self.data) <= self._ma_direction_lookback:
            return False
        ma_now = float(self.ma_direction[-1]) if np.isfinite(self.ma_direction[-1]) else np.nan
        ma_prev_idx = -(self._ma_direction_lookback + 1)
        ma_prev = float(self.ma_direction[ma_prev_idx]) if np.isfinite(self.ma_direction[ma_prev_idx]) else np.nan
        return bool(np.isfinite(ma_now) and np.isfinite(ma_prev) and ma_now > ma_prev)

    def _entry_overlays_ok(self, close_now: float) -> bool:
        return self._sma_trend_gate_ok(close_now) and self._weekly_trend_gate_ok() and self._ma_direction_gate_ok()

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
            if close_now < threshold and ibs_now < self._ibs_threshold and self._entry_overlays_ok(close_now):
                self.buy()
            return

        yesterday_high = float(self.data.High[-(self._exit_lookback + 1)])
        if np.isfinite(yesterday_high) and close_now > yesterday_high:
            self.position.close()
