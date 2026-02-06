from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from options_helper.technicals_backtesting.indicators.provider_base import FeatureFrame


def _format_dev(dev: float) -> str:
    if float(dev).is_integer():
        return str(int(dev))
    return str(dev).replace(".", "p")


def _unique_sorted(values: set[int | float]) -> list[int | float]:
    return sorted(values, key=lambda x: float(x))


def _collect_strategy_values(cfg: dict, param_name: str) -> set[int | float | bool]:
    values: set[int | float | bool] = set()
    strategies = cfg.get("strategies", {})
    for strat_cfg in strategies.values():
        defaults = strat_cfg.get("defaults", {})
        if param_name in defaults:
            values.add(defaults[param_name])
        search = strat_cfg.get("search_space", {})
        for v in search.get(param_name, []):
            values.add(v)
    return values


@dataclass(frozen=True)
class TaIndicatorProvider:
    def compute_indicators(self, df: pd.DataFrame, cfg: dict) -> FeatureFrame:
        try:
            from ta.momentum import RSIIndicator
            from ta.volatility import AverageTrueRange, BollingerBands
        except Exception as exc:  # noqa: BLE001
            raise ImportError("ta library is required for technical indicators") from exc

        if df.empty:
            return df.copy()

        out = df.copy()
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        out["Close"] = close
        out["High"] = high
        out["Low"] = low
        if "Open" in out.columns:
            out["Open"] = pd.to_numeric(out["Open"], errors="coerce")
        if "Volume" in out.columns:
            out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

        atr_values = set(cfg["indicators"]["atr"]["window_grid"])
        atr_values.add(cfg["indicators"]["atr"]["window_default"])
        atr_values.update(_collect_strategy_values(cfg, "atr_window"))
        atr_windows = [int(v) for v in _unique_sorted({int(v) for v in atr_values})]

        sma_values = set(cfg["indicators"]["sma"]["window_grid"])
        sma_values.add(cfg["indicators"]["sma"]["window_default"])
        sma_values.update(_collect_strategy_values(cfg, "sma_window"))
        sma_windows = [int(v) for v in _unique_sorted({int(v) for v in sma_values})]

        z_values = set(cfg["indicators"]["zscore"]["window_grid"])
        z_values.add(cfg["indicators"]["zscore"]["window_default"])
        z_values.update(_collect_strategy_values(cfg, "z_window"))
        z_windows = [int(v) for v in _unique_sorted({int(v) for v in z_values})]

        bb_windows = set(cfg["indicators"]["bollinger"]["window_grid"])
        bb_windows.add(cfg["indicators"]["bollinger"]["window_default"])
        bb_windows.update(_collect_strategy_values(cfg, "bb_window"))
        bb_window_vals = [int(v) for v in _unique_sorted({int(v) for v in bb_windows})]

        bb_devs = set(cfg["indicators"]["bollinger"]["dev_grid"])
        bb_devs.add(cfg["indicators"]["bollinger"]["dev_default"])
        bb_devs.update(_collect_strategy_values(cfg, "bb_dev"))
        bb_dev_vals = _unique_sorted({float(v) for v in bb_devs})

        rsi_cfg = cfg["indicators"].get("rsi", {})
        rsi_enabled = bool(rsi_cfg.get("enabled", False))
        rsi_windows: list[int] = []
        if rsi_enabled:
            rsi_values = set(rsi_cfg.get("window_grid", []) or [])
            if "window_default" in rsi_cfg:
                rsi_values.add(rsi_cfg["window_default"])
            rsi_windows = [int(v) for v in _unique_sorted({int(v) for v in rsi_values})]

        # ATR + ATR%
        for window in atr_windows:
            if len(close) < window:
                out[f"atr_{window}"] = pd.Series(np.nan, index=out.index)
                out[f"atrp_{window}"] = pd.Series(np.nan, index=out.index)
                continue
            try:
                atr = AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()
                out[f"atr_{window}"] = atr
                out[f"atrp_{window}"] = atr / close
            except Exception:  # noqa: BLE001
                out[f"atr_{window}"] = pd.Series(np.nan, index=out.index)
                out[f"atrp_{window}"] = pd.Series(np.nan, index=out.index)

        # SMA
        for window in sma_windows:
            out[f"sma_{window}"] = close.rolling(window=window).mean()

        # Z-score
        for window in z_windows:
            mean = close.rolling(window=window).mean()
            std = close.rolling(window=window).std()
            std = std.replace(0.0, np.nan)
            out[f"zscore_{window}"] = (close - mean) / std

        # Bollinger Bands
        for window in bb_window_vals:
            for dev in bb_dev_vals:
                dev_label = _format_dev(dev)
                if len(close) < window:
                    out[f"bb_mavg_{window}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_hband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_lband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_pband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_wband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    continue
                try:
                    bb = BollingerBands(close=close, window=window, window_dev=dev)
                    out[f"bb_mavg_{window}"] = bb.bollinger_mavg()
                    out[f"bb_hband_{window}_{dev_label}"] = bb.bollinger_hband()
                    out[f"bb_lband_{window}_{dev_label}"] = bb.bollinger_lband()
                    out[f"bb_pband_{window}_{dev_label}"] = bb.bollinger_pband()
                    out[f"bb_wband_{window}_{dev_label}"] = bb.bollinger_wband()
                except Exception:  # noqa: BLE001
                    out[f"bb_mavg_{window}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_hband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_lband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_pband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)
                    out[f"bb_wband_{window}_{dev_label}"] = pd.Series(np.nan, index=out.index)

        # RSI
        if rsi_enabled:
            for window in rsi_windows:
                if len(close) < window:
                    out[f"rsi_{window}"] = pd.Series(np.nan, index=out.index)
                    continue
                try:
                    rsi = RSIIndicator(close=close, window=window).rsi()
                    out[f"rsi_{window}"] = rsi
                except Exception:  # noqa: BLE001
                    out[f"rsi_{window}"] = pd.Series(np.nan, index=out.index)

        # Extension in ATR units.
        for sma_window in sma_windows:
            sma_col = f"sma_{sma_window}"
            for atr_window in atr_windows:
                atr_col = f"atr_{atr_window}"
                out[f"extension_atr_{sma_window}_{atr_window}"] = (
                    (close - out[sma_col]) / out[atr_col]
                )

        # Pandas can emit "highly fragmented" PerformanceWarnings after many column inserts.
        # A copy defragments blocks and avoids warning spam in logs.
        out = out.copy()

        # Weekly regime.
        weekly_cfg = cfg.get("weekly_regime", {})
        if weekly_cfg.get("enabled", False):
            rule = weekly_cfg["resample_rule"]
            fast = int(weekly_cfg["fast_ma"])
            slow = int(weekly_cfg["slow_ma"])
            ma_type = str(weekly_cfg.get("ma_type", "sma")).strip().lower()
            logic = str(weekly_cfg.get("logic", "close_above_fast_and_fast_above_slow")).strip().lower()
            weekly = (
                out[["Open", "High", "Low", "Close"]]
                .resample(rule)
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            )
            weekly_close = weekly["Close"]
            if ma_type == "ema":
                weekly_fast = weekly_close.ewm(span=fast, adjust=False, min_periods=fast).mean()
                weekly_slow = weekly_close.ewm(span=slow, adjust=False, min_periods=slow).mean()
            else:
                weekly_fast = weekly_close.rolling(window=fast).mean()
                weekly_slow = weekly_close.rolling(window=slow).mean()

            if logic in {"fast_above_slow", "ma_fast_above_slow"}:
                trend = weekly_fast > weekly_slow
            elif logic in {"close_above_fast", "close_above_ma_fast"}:
                trend = weekly_close > weekly_fast
            else:
                # Default: close_above_fast_and_fast_above_slow
                trend = (weekly_close > weekly_fast) & (weekly_fast > weekly_slow)

            out[f"weekly_sma_{fast}"] = weekly_fast.reindex(out.index, method="ffill")
            out[f"weekly_sma_{slow}"] = weekly_slow.reindex(out.index, method="ffill")
            # Avoid FutureWarning about downcasting object dtype arrays during fillna/ffill.
            weekly_trend = trend.reindex(out.index, method="ffill").astype("boolean")
            out["weekly_trend_up"] = weekly_trend.fillna(False).astype(bool)

        return out
