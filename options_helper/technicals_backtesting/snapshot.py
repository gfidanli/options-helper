from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars


@dataclass(frozen=True)
class TechnicalSnapshot:
    asof: str
    close: float | None

    weekly_trend_up: bool | None

    atr_window: int
    atr: float | None
    atrp: float | None

    sma_window: int
    z_window: int
    zscore: float | None
    extension_atr: float | None

    bb_window: int
    bb_dev: float
    bb_pband: float | None
    bb_wband: float | None

    rsi_window: int | None
    rsi: float | None


def _format_dev(dev: float) -> str:
    dev_f = float(dev)
    if dev_f.is_integer():
        return str(int(dev_f))
    return f"{dev_f:g}".replace(".", "p")


def _as_float(val: object) -> float | None:
    try:
        if val is None:
            return None
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return None
        if isinstance(val, pd.Timestamp):
            return None
        if pd.isna(val):
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def compute_technical_snapshot(df_ohlc: pd.DataFrame, cfg: dict) -> TechnicalSnapshot | None:
    """
    Compute a compact "technical snapshot" from an OHLCV DataFrame using the configured
    `technicals_backtesting` indicator provider (canonical indicator source).
    """
    if df_ohlc is None or df_ohlc.empty:
        return None

    features = compute_features(df_ohlc, cfg)
    if features.empty:
        return None

    ind_cfg = cfg["indicators"]
    atr_window = int(ind_cfg["atr"]["window_default"])
    sma_window = int(ind_cfg["sma"]["window_default"])
    z_window = int(ind_cfg["zscore"]["window_default"])

    bb_window = int(ind_cfg["bollinger"]["window_default"])
    bb_dev = float(ind_cfg["bollinger"]["dev_default"])
    dev_label = _format_dev(bb_dev)

    rsi_cfg = ind_cfg.get("rsi", {}) or {}
    rsi_window = None
    if bool(rsi_cfg.get("enabled", False)) and "window_default" in rsi_cfg:
        rsi_window = int(rsi_cfg["window_default"])

    cols = [
        "Close",
        "weekly_trend_up",
        f"atr_{atr_window}",
        f"atrp_{atr_window}",
        f"zscore_{z_window}",
        f"extension_atr_{sma_window}_{atr_window}",
        f"bb_pband_{bb_window}_{dev_label}",
        f"bb_wband_{bb_window}_{dev_label}",
    ]
    if rsi_window is not None:
        cols.append(f"rsi_{rsi_window}")

    w = warmup_bars(cfg)
    data = features.iloc[w:] if w > 0 else features
    if data.empty:
        return None

    subset = [c for c in cols if c in data.columns]
    if not subset:
        return None

    valid = data.dropna(subset=[c for c in subset if c != "weekly_trend_up"])
    if valid.empty:
        return None

    row = valid.iloc[-1]
    idx = valid.index[-1]
    asof = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)

    weekly = row.get("weekly_trend_up")
    weekly_trend_up = None
    if weekly is not None and not pd.isna(weekly):
        weekly_trend_up = bool(weekly)

    rsi_val = None
    if rsi_window is not None:
        rsi_val = _as_float(row.get(f"rsi_{rsi_window}"))

    return TechnicalSnapshot(
        asof=asof,
        close=_as_float(row.get("Close")),
        weekly_trend_up=weekly_trend_up,
        atr_window=atr_window,
        atr=_as_float(row.get(f"atr_{atr_window}")),
        atrp=_as_float(row.get(f"atrp_{atr_window}")),
        sma_window=sma_window,
        z_window=z_window,
        zscore=_as_float(row.get(f"zscore_{z_window}")),
        extension_atr=_as_float(row.get(f"extension_atr_{sma_window}_{atr_window}")),
        bb_window=bb_window,
        bb_dev=bb_dev,
        bb_pband=_as_float(row.get(f"bb_pband_{bb_window}_{dev_label}")),
        bb_wband=_as_float(row.get(f"bb_wband_{bb_window}_{dev_label}")),
        rsi_window=rsi_window,
        rsi=rsi_val,
    )

