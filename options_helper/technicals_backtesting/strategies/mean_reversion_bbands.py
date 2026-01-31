from __future__ import annotations

import numpy as np
from backtesting import Strategy


def _format_dev(dev: float) -> str:
    if float(dev).is_integer():
        return str(int(dev))
    return str(dev).replace(".", "p")


class MeanReversionBollinger(Strategy):
    """
    Long-only mean reversion strategy using Bollinger Bands.

    Optional ATR stop:
    - stop_mult_atr == 0 disables stop.
    - Otherwise stop at entry_price - stop_mult_atr * ATR (ATR at entry).
    """

    bb_window = 20
    bb_dev = 2.0
    p_entry = 0.10
    p_exit = 0.50
    atr_window = 14
    stop_mult_atr = 2.0
    use_weekly_filter = False

    def init(self) -> None:
        if self.p_entry >= self.p_exit:
            raise ValueError("p_entry must be less than p_exit")
        dev_label = _format_dev(self.bb_dev)
        self._mavg_col = f"bb_mavg_{int(self.bb_window)}"
        self._lband_col = f"bb_lband_{int(self.bb_window)}_{dev_label}"
        self._pband_col = f"bb_pband_{int(self.bb_window)}_{dev_label}"
        self._atr_col = f"atr_{int(self.atr_window)}"
        self._weekly_col = "weekly_trend_up"

        self.mavg = self.I(lambda x: x, self.data[self._mavg_col], name=self._mavg_col)
        self.lband = self.I(lambda x: x, self.data[self._lband_col], name=self._lband_col)
        self.pband = self.I(lambda x: x, self.data[self._pband_col], name=self._pband_col)
        self.atr = self.I(lambda x: x, self.data[self._atr_col], name=self._atr_col)

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
        prev_close = float(self.data.Close[-2])
        lband = float(self.lband[-1]) if not np.isnan(self.lband[-1]) else np.nan
        prev_lband = float(self.lband[-2]) if not np.isnan(self.lband[-2]) else np.nan
        pband = float(self.pband[-1]) if not np.isnan(self.pband[-1]) else np.nan
        mavg = float(self.mavg[-1]) if not np.isnan(self.mavg[-1]) else np.nan
        atr = float(self.atr[-1]) if not np.isnan(self.atr[-1]) else np.nan

        cross_below = (
            not np.isnan(prev_lband)
            and not np.isnan(lband)
            and prev_close >= prev_lband
            and close < lband
        )
        pband_entry = not np.isnan(pband) and pband <= self.p_entry

        if not self.position:
            if self._weekly_ok() and (cross_below or pband_entry):
                stop_price = None
                if self.stop_mult_atr and self.stop_mult_atr > 0 and not np.isnan(atr):
                    stop_price = close - self.stop_mult_atr * atr
                if stop_price is not None:
                    self.buy(sl=stop_price)
                else:
                    self.buy()
            return

        pband_exit = not np.isnan(pband) and pband >= self.p_exit
        if (not np.isnan(mavg) and close >= mavg) or pband_exit:
            self.position.close()
