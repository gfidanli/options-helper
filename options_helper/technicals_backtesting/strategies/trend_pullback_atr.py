from __future__ import annotations

import numpy as np
from backtesting import Strategy


class TrendPullbackATR(Strategy):
    """
    Long-only pullback strategy with ATR-based stop.

    Stop logic:
    - On entry, place a fixed stop at entry_price - stop_mult_atr * ATR (ATR at entry).
    """

    atr_window = 14
    sma_window = 20
    z_window = 20
    add_z = -1.0
    trim_ext_atr = 1.8
    stop_mult_atr = 2.5
    use_weekly_filter = True

    def init(self) -> None:
        self._add_z = float(self.add_z)
        self._trim_ext_atr = float(self.trim_ext_atr)
        self._stop_mult_atr = float(self.stop_mult_atr)

        self._atr_col = f"atr_{int(self.atr_window)}"
        self._z_col = f"zscore_{int(self.z_window)}"
        self._ext_col = f"extension_atr_{int(self.sma_window)}_{int(self.atr_window)}"
        self._weekly_col = "weekly_trend_up"

        self.atr = self.I(lambda x: x, self.data[self._atr_col], name=self._atr_col)
        self.zscore = self.I(lambda x: x, self.data[self._z_col], name=self._z_col)
        self.extension = self.I(lambda x: x, self.data[self._ext_col], name=self._ext_col)

        self.weekly_trend = None
        if hasattr(self.data, "df") and self._weekly_col in self.data.df.columns:
            self.weekly_trend = self.I(lambda x: x, self.data[self._weekly_col], name=self._weekly_col)

    def _weekly_ok(self) -> bool:
        if not self.use_weekly_filter:
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

    def next(self) -> None:
        if len(self.data) < 2:
            return

        close = float(self.data.Close[-1])
        atr = float(self.atr[-1]) if not np.isnan(self.atr[-1]) else np.nan
        zscore = float(self.zscore[-1]) if not np.isnan(self.zscore[-1]) else np.nan
        extension = (
            float(self.extension[-1]) if not np.isnan(self.extension[-1]) else np.nan
        )

        if not self.position:
            if self._weekly_ok() and not np.isnan(zscore) and zscore <= self._add_z:
                stop_price = None
                if self._stop_mult_atr and self._stop_mult_atr > 0 and not np.isnan(atr):
                    stop_price = close - self._stop_mult_atr * atr
                if stop_price is not None:
                    self.buy(sl=stop_price)
                else:
                    self.buy()
            return

        if not np.isnan(extension) and extension >= self._trim_ext_atr:
            self.position.close()
