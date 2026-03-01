from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesting import Strategy


@dataclass(frozen=True)
class _ActiveSetup:
    pivot2_idx: int
    pivot2_low: float
    break_level: float
    break_level_idx: int
    created_idx: int


class CvdDivergenceMSB(Strategy):
    """
    Long-only hidden bullish divergence + market-structure-break strategy.

    Pivot handling is lookahead-safe: swing pivots are only consumed at
    `pivot_idx + pivot_right`.
    """

    atr_window = 14
    stop_mult_atr = 2.0
    take_profit_mult_atr = 3.0
    max_holding_bars = 0
    use_weekly_filter = True

    cvd_smooth_span = 20
    cvd_z_window = 30

    pivot_left = 3
    pivot_right = 3
    divergence_window_bars = 80
    min_separation_bars = 4
    min_price_delta_pct = 0.1
    min_cvd_z_delta = 0.1
    max_setup_age_bars = 30
    msb_min_distance_bars = 1

    def init(self) -> None:
        self._atr_col = f"atr_{int(self.atr_window)}"
        self._weekly_col = "weekly_trend_up"

        self._pivot_left = max(1, int(self.pivot_left))
        self._pivot_right = max(1, int(self.pivot_right))
        self._divergence_window_bars = max(1, int(self.divergence_window_bars))
        self._min_separation_bars = max(1, int(self.min_separation_bars))
        self._max_setup_age_bars = max(0, int(self.max_setup_age_bars))
        self._msb_min_distance_bars = max(0, int(self.msb_min_distance_bars))
        self._cvd_smooth_span = max(1, int(self.cvd_smooth_span))
        self._cvd_z_window = max(2, int(self.cvd_z_window))
        self._min_price_delta_pct = float(self.min_price_delta_pct)
        self._min_cvd_z_delta = float(self.min_cvd_z_delta)
        self._stop_mult_atr = float(self.stop_mult_atr)
        self._take_profit_mult_atr = float(self.take_profit_mult_atr)
        self._max_holding_bars = int(self.max_holding_bars) if int(self.max_holding_bars) > 0 else None
        self._use_weekly_filter = bool(self.use_weekly_filter)

        self.atr = self.I(lambda x: x, self.data[self._atr_col], name=self._atr_col)
        self.weekly_trend = None
        if hasattr(self.data, "df") and self._weekly_col in self.data.df.columns:
            self.weekly_trend = self.I(lambda x: x, self.data[self._weekly_col], name=self._weekly_col)

        self._open = np.asarray(self.data.Open, dtype=float)
        self._high = np.asarray(self.data.High, dtype=float)
        self._low = np.asarray(self.data.Low, dtype=float)
        self._close = np.asarray(self.data.Close, dtype=float)

        self._volume_ok, self._cvd_z = self._build_cvd_z()
        self.cvd_z = self.I(lambda x: x, self._cvd_z, name="cvd_z")

        self._confirm_to_pivots = self._build_confirmed_lows()
        self._confirmed_low_pivots: list[int] = []
        self._active_setup: _ActiveSetup | None = None
        self._entry_bar_idx: int | None = None

    def _build_cvd_z(self) -> tuple[bool, np.ndarray]:
        empty = np.full(len(self.data), np.nan, dtype=float)
        if not hasattr(self.data, "df") or "Volume" not in self.data.df.columns:
            warnings.warn("CvdDivergenceMSB disabled: missing Volume column.", UserWarning, stacklevel=2)
            return False, empty

        volume = np.asarray(self.data.Volume, dtype=float)
        finite = np.isfinite(volume)
        if not bool(finite.any()):
            warnings.warn("CvdDivergenceMSB disabled: missing Volume column.", UserWarning, stacklevel=2)
            return False, empty

        valid = finite & (volume >= 0)
        if not bool(valid.all()):
            warnings.warn(
                "CvdDivergenceMSB disabled: invalid/non-finite Volume values.",
                UserWarning,
                stacklevel=2,
            )
            return False, empty

        delta = np.zeros(len(volume), dtype=float)
        up = self._close > self._open
        down = self._close < self._open
        delta[up] = volume[up]
        delta[down] = -volume[down]
        cvd = np.cumsum(delta, dtype=float)

        osc = self._cvd_oscillator(cvd)
        return True, self._zscore(osc, window=self._cvd_z_window)

    def _cvd_oscillator(self, cvd: np.ndarray) -> np.ndarray:
        cvd_series = pd.Series(cvd, copy=False)
        ema = cvd_series.ewm(span=self._cvd_smooth_span, adjust=False).mean()
        return (cvd_series - ema).to_numpy(dtype=float, copy=False)

    def _zscore(self, values: np.ndarray, *, window: int) -> np.ndarray:
        series = pd.Series(values, copy=False)
        roll_mean = series.rolling(window=window, min_periods=window).mean()
        roll_std = series.rolling(window=window, min_periods=window).std(ddof=0)
        z = (series - roll_mean) / roll_std
        z = z.where(roll_std > 0, np.nan)
        return z.to_numpy(dtype=float, copy=False)

    def _build_confirmed_lows(self) -> dict[int, list[int]]:
        confirms: dict[int, list[int]] = {}
        stop = len(self._low) - self._pivot_right
        for pivot_idx in range(self._pivot_left, stop):
            if not self._is_pivot_low(pivot_idx):
                continue
            confirm_idx = pivot_idx + self._pivot_right
            confirms.setdefault(confirm_idx, []).append(pivot_idx)
        return confirms

    def _is_pivot_low(self, pivot_idx: int) -> bool:
        start = pivot_idx - self._pivot_left
        stop = pivot_idx + self._pivot_right + 1
        window = self._low[start:stop]
        if window.size == 0 or not bool(np.isfinite(window).all()):
            return False
        return float(self._low[pivot_idx]) == float(np.min(window))

    def _weekly_ok(self) -> bool:
        if not self._use_weekly_filter:
            return True
        if self.weekly_trend is None:
            return True
        val = self.weekly_trend[-1]
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        try:
            if np.isnan(val):
                return False
        except TypeError:
            return bool(val)
        return bool(val)

    def _refresh_entry_tracking(self, idx: int) -> None:
        if self.position and self._entry_bar_idx is None:
            self._entry_bar_idx = idx
        if not self.position:
            self._entry_bar_idx = None

    def _consume_newly_confirmed_pivots(self, idx: int) -> None:
        for pivot2_idx in self._confirm_to_pivots.get(idx, []):
            setup = self._build_setup_from_pivot(pivot2_idx=pivot2_idx, confirm_idx=idx)
            self._confirmed_low_pivots.append(pivot2_idx)
            if setup is not None:
                self._active_setup = setup

    def _build_setup_from_pivot(self, *, pivot2_idx: int, confirm_idx: int) -> _ActiveSetup | None:
        pivot1_idx = self._select_pivot1(pivot2_idx)
        if pivot1_idx is None:
            return None
        if not self._is_hidden_bullish_divergence(pivot1_idx, pivot2_idx):
            return None

        break_level, break_level_idx = self._break_level_between(pivot1_idx, pivot2_idx)
        if break_level_idx is None:
            return None
        return _ActiveSetup(
            pivot2_idx=pivot2_idx,
            pivot2_low=float(self._low[pivot2_idx]),
            break_level=break_level,
            break_level_idx=break_level_idx,
            created_idx=confirm_idx,
        )

    def _select_pivot1(self, pivot2_idx: int) -> int | None:
        for pivot1_idx in reversed(self._confirmed_low_pivots):
            bars = pivot2_idx - pivot1_idx
            if bars < self._min_separation_bars:
                continue
            if bars > self._divergence_window_bars:
                break
            return pivot1_idx
        return None

    def _is_hidden_bullish_divergence(self, pivot1_idx: int, pivot2_idx: int) -> bool:
        low1 = float(self._low[pivot1_idx])
        low2 = float(self._low[pivot2_idx])
        price_floor = low1 * (1.0 + self._min_price_delta_pct / 100.0)
        if low2 < price_floor:
            return False

        cvd1 = float(self._cvd_z[pivot1_idx]) if np.isfinite(self._cvd_z[pivot1_idx]) else np.nan
        cvd2 = float(self._cvd_z[pivot2_idx]) if np.isfinite(self._cvd_z[pivot2_idx]) else np.nan
        if np.isnan(cvd1) or np.isnan(cvd2):
            return False
        return cvd2 <= (cvd1 - self._min_cvd_z_delta)

    def _break_level_between(self, pivot1_idx: int, pivot2_idx: int) -> tuple[float, int | None]:
        highs = self._high[pivot1_idx:pivot2_idx]
        if highs.size == 0 or not bool(np.isfinite(highs).any()):
            return np.nan, None
        rel_idx = int(np.nanargmax(highs))
        return float(highs[rel_idx]), pivot1_idx + rel_idx

    def _expire_or_invalidate_setup(self, idx: int) -> None:
        if self._active_setup is None:
            return
        if float(self._low[idx]) < self._active_setup.pivot2_low:
            self._active_setup = None
            return
        if idx - self._active_setup.created_idx > self._max_setup_age_bars:
            self._active_setup = None

    def _should_enter_on_bar(self, idx: int) -> bool:
        if not self._volume_ok or self._active_setup is None or not self._weekly_ok():
            return False
        if idx - self._active_setup.break_level_idx < self._msb_min_distance_bars:
            return False
        close_now = float(self._close[idx])
        return bool(np.isfinite(close_now) and close_now > self._active_setup.break_level)

    def _risk_levels(self, close_now: float, atr_now: float) -> tuple[float | None, float | None]:
        stop = None
        target = None
        if self._stop_mult_atr > 0:
            stop = close_now - self._stop_mult_atr * atr_now
        if self._take_profit_mult_atr > 0:
            target = close_now + self._take_profit_mult_atr * atr_now
        return stop, target

    def _maybe_enter(self, idx: int) -> None:
        if not self._should_enter_on_bar(idx):
            return
        atr_now = float(self.atr[-1]) if np.isfinite(self.atr[-1]) else np.nan
        if np.isnan(atr_now):
            return
        close_now = float(self._close[idx])
        stop, target = self._risk_levels(close_now, atr_now)
        self.buy(sl=stop, tp=target)

    def _maybe_time_stop(self, idx: int) -> None:
        if self._max_holding_bars is None or self._entry_bar_idx is None:
            return
        if idx - self._entry_bar_idx >= self._max_holding_bars:
            self.position.close()

    def next(self) -> None:
        if len(self.data) < 2:
            return
        idx = len(self.data) - 1
        self._refresh_entry_tracking(idx)
        self._consume_newly_confirmed_pivots(idx)
        self._expire_or_invalidate_setup(idx)

        if self.position:
            self._maybe_time_stop(idx)
            return
        self._maybe_enter(idx)
