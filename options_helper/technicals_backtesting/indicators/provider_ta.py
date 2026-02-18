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


def _load_ta_indicators() -> tuple[object, object, object]:
    try:
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange, BollingerBands
    except Exception as exc:  # noqa: BLE001
        raise ImportError("ta library is required for technical indicators") from exc
    return RSIIndicator, AverageTrueRange, BollingerBands


def _normalize_price_columns(out: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
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
    return close, high, low


def _resolve_indicator_windows(cfg: dict, indicator_key: str, param_name: str) -> list[int]:
    indicator_cfg = cfg["indicators"][indicator_key]
    values = set(indicator_cfg["window_grid"])
    values.add(indicator_cfg["window_default"])
    values.update(_collect_strategy_values(cfg, param_name))
    return [int(v) for v in _unique_sorted({int(v) for v in values})]


def _resolve_bollinger_windows(cfg: dict) -> tuple[list[int], list[float]]:
    bb_cfg = cfg["indicators"]["bollinger"]
    window_values = set(bb_cfg["window_grid"])
    window_values.add(bb_cfg["window_default"])
    window_values.update(_collect_strategy_values(cfg, "bb_window"))
    windows = [int(v) for v in _unique_sorted({int(v) for v in window_values})]

    dev_values = set(bb_cfg["dev_grid"])
    dev_values.add(bb_cfg["dev_default"])
    dev_values.update(_collect_strategy_values(cfg, "bb_dev"))
    devs = _unique_sorted({float(v) for v in dev_values})
    return windows, devs


def _resolve_rsi_windows(cfg: dict) -> tuple[bool, list[int]]:
    rsi_cfg = cfg["indicators"].get("rsi", {})
    enabled = bool(rsi_cfg.get("enabled", False))
    if not enabled:
        return False, []
    values = set(rsi_cfg.get("window_grid", []) or [])
    if "window_default" in rsi_cfg:
        values.add(rsi_cfg["window_default"])
    windows = [int(v) for v in _unique_sorted({int(v) for v in values})]
    return True, windows


def _assign_nan_columns(out: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        out[column] = pd.Series(np.nan, index=out.index)


def _compute_atr_features(
    out: pd.DataFrame,
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    windows: list[int],
    average_true_range_cls: object,
) -> None:
    for window in windows:
        atr_col = f"atr_{window}"
        atrp_col = f"atrp_{window}"
        if len(close) < window:
            _assign_nan_columns(out, [atr_col, atrp_col])
            continue
        try:
            atr = average_true_range_cls(high=high, low=low, close=close, window=window).average_true_range()
            out[atr_col] = atr
            out[atrp_col] = atr / close
        except Exception:  # noqa: BLE001
            _assign_nan_columns(out, [atr_col, atrp_col])


def _compute_sma_features(out: pd.DataFrame, *, close: pd.Series, windows: list[int]) -> None:
    for window in windows:
        out[f"sma_{window}"] = close.rolling(window=window).mean()


def _compute_zscore_features(out: pd.DataFrame, *, close: pd.Series, windows: list[int]) -> None:
    for window in windows:
        mean = close.rolling(window=window).mean()
        std = close.rolling(window=window).std().replace(0.0, np.nan)
        out[f"zscore_{window}"] = (close - mean) / std


def _compute_bollinger_features(
    out: pd.DataFrame,
    *,
    close: pd.Series,
    windows: list[int],
    devs: list[float],
    bollinger_cls: object,
) -> None:
    for window in windows:
        for dev in devs:
            dev_label = _format_dev(dev)
            _compute_single_bollinger_window(
                out=out,
                close=close,
                window=window,
                dev=dev,
                dev_label=dev_label,
                bollinger_cls=bollinger_cls,
            )


def _compute_single_bollinger_window(
    out: pd.DataFrame,
    *,
    close: pd.Series,
    window: int,
    dev: float,
    dev_label: str,
    bollinger_cls: object,
) -> None:
    mavg_col = f"bb_mavg_{window}"
    hband_col = f"bb_hband_{window}_{dev_label}"
    lband_col = f"bb_lband_{window}_{dev_label}"
    pband_col = f"bb_pband_{window}_{dev_label}"
    wband_col = f"bb_wband_{window}_{dev_label}"
    columns = [mavg_col, hband_col, lband_col, pband_col, wband_col]
    if len(close) < window:
        _assign_nan_columns(out, columns)
        return
    try:
        bb = bollinger_cls(close=close, window=window, window_dev=dev)
        out[mavg_col] = bb.bollinger_mavg()
        out[hband_col] = bb.bollinger_hband()
        out[lband_col] = bb.bollinger_lband()
        out[pband_col] = bb.bollinger_pband()
        out[wband_col] = bb.bollinger_wband()
    except Exception:  # noqa: BLE001
        _assign_nan_columns(out, columns)


def _compute_rsi_features(
    out: pd.DataFrame,
    *,
    close: pd.Series,
    enabled: bool,
    windows: list[int],
    rsi_cls: object,
) -> None:
    if not enabled:
        return
    for window in windows:
        column = f"rsi_{window}"
        if len(close) < window:
            _assign_nan_columns(out, [column])
            continue
        try:
            out[column] = rsi_cls(close=close, window=window).rsi()
        except Exception:  # noqa: BLE001
            _assign_nan_columns(out, [column])


def _compute_extension_features(
    out: pd.DataFrame,
    *,
    close: pd.Series,
    sma_windows: list[int],
    atr_windows: list[int],
) -> None:
    for sma_window in sma_windows:
        sma_col = f"sma_{sma_window}"
        for atr_window in atr_windows:
            atr_col = f"atr_{atr_window}"
            out[f"extension_atr_{sma_window}_{atr_window}"] = (close - out[sma_col]) / out[atr_col]


def _compute_weekly_regime_features(out: pd.DataFrame, *, cfg: dict) -> None:
    weekly_cfg = cfg.get("weekly_regime", {})
    if not weekly_cfg.get("enabled", False):
        return
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
        trend = (weekly_close > weekly_fast) & (weekly_fast > weekly_slow)

    out[f"weekly_sma_{fast}"] = weekly_fast.reindex(out.index, method="ffill")
    out[f"weekly_sma_{slow}"] = weekly_slow.reindex(out.index, method="ffill")
    weekly_trend = trend.reindex(out.index, method="ffill").astype("boolean")
    out["weekly_trend_up"] = weekly_trend.fillna(False).astype(bool)


@dataclass(frozen=True)
class TaIndicatorProvider:
    def compute_indicators(self, df: pd.DataFrame, cfg: dict) -> FeatureFrame:
        rsi_cls, average_true_range_cls, bollinger_cls = _load_ta_indicators()
        if df.empty:
            return df.copy()

        out = df.copy()
        close, high, low = _normalize_price_columns(out)

        atr_windows = _resolve_indicator_windows(cfg, "atr", "atr_window")
        sma_windows = _resolve_indicator_windows(cfg, "sma", "sma_window")
        z_windows = _resolve_indicator_windows(cfg, "zscore", "z_window")
        bb_windows, bb_devs = _resolve_bollinger_windows(cfg)
        rsi_enabled, rsi_windows = _resolve_rsi_windows(cfg)

        _compute_atr_features(
            out,
            close=close,
            high=high,
            low=low,
            windows=atr_windows,
            average_true_range_cls=average_true_range_cls,
        )
        _compute_sma_features(out, close=close, windows=sma_windows)
        _compute_zscore_features(out, close=close, windows=z_windows)
        _compute_bollinger_features(
            out,
            close=close,
            windows=bb_windows,
            devs=bb_devs,
            bollinger_cls=bollinger_cls,
        )
        _compute_rsi_features(out, close=close, enabled=rsi_enabled, windows=rsi_windows, rsi_cls=rsi_cls)
        _compute_extension_features(out, close=close, sma_windows=sma_windows, atr_windows=atr_windows)

        out = out.copy()
        _compute_weekly_regime_features(out, cfg=cfg)
        return out
