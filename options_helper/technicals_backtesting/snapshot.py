from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from options_helper.technicals_backtesting.extension_percentiles import (
    ExtensionPercentilesBundle,
    build_weekly_extension_series,
    compute_extension_percentiles,
)
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

    extension_percentiles: ExtensionPercentilesBundle | None


@dataclass(frozen=True)
class _SnapshotParams:
    atr_window: int
    sma_window: int
    z_window: int
    bb_window: int
    bb_dev: float
    bb_dev_label: str
    rsi_window: int | None


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


def _snapshot_params(cfg: dict) -> _SnapshotParams:
    ind_cfg = cfg["indicators"]
    atr_window = int(ind_cfg["atr"]["window_default"])
    sma_window = int(ind_cfg["sma"]["window_default"])
    z_window = int(ind_cfg["zscore"]["window_default"])
    bb_window = int(ind_cfg["bollinger"]["window_default"])
    bb_dev = float(ind_cfg["bollinger"]["dev_default"])
    bb_dev_label = _format_dev(bb_dev)
    rsi_cfg = ind_cfg.get("rsi", {}) or {}
    rsi_window = None
    if bool(rsi_cfg.get("enabled", False)) and "window_default" in rsi_cfg:
        rsi_window = int(rsi_cfg["window_default"])
    return _SnapshotParams(
        atr_window=atr_window,
        sma_window=sma_window,
        z_window=z_window,
        bb_window=bb_window,
        bb_dev=bb_dev,
        bb_dev_label=bb_dev_label,
        rsi_window=rsi_window,
    )


def _required_snapshot_columns(params: _SnapshotParams) -> list[str]:
    cols = [
        "Close",
        "weekly_trend_up",
        f"atr_{params.atr_window}",
        f"atrp_{params.atr_window}",
        f"zscore_{params.z_window}",
        f"extension_atr_{params.sma_window}_{params.atr_window}",
        f"bb_pband_{params.bb_window}_{params.bb_dev_label}",
        f"bb_wband_{params.bb_window}_{params.bb_dev_label}",
    ]
    if params.rsi_window is not None:
        cols.append(f"rsi_{params.rsi_window}")
    return cols


def _latest_valid_snapshot_row(
    *,
    features: pd.DataFrame,
    cfg: dict,
    required_columns: list[str],
) -> tuple[pd.Series, object] | None:
    w = warmup_bars(cfg)
    data = features.iloc[w:] if w > 0 else features
    if data.empty:
        return None
    subset = [col for col in required_columns if col in data.columns]
    if not subset:
        return None
    valid = data.dropna(subset=[col for col in subset if col != "weekly_trend_up"])
    if valid.empty:
        return None
    return valid.iloc[-1], valid.index[-1]


def _snapshot_asof(idx: object) -> str:
    if isinstance(idx, pd.Timestamp):
        return idx.date().isoformat()
    return str(idx)


def _snapshot_weekly_trend(row: pd.Series) -> bool | None:
    weekly = row.get("weekly_trend_up")
    if weekly is None or pd.isna(weekly):
        return None
    return bool(weekly)


def _snapshot_rsi_value(row: pd.Series, *, rsi_window: int | None) -> float | None:
    if rsi_window is None:
        return None
    return _as_float(row.get(f"rsi_{rsi_window}"))


def _build_extension_percentiles(
    *,
    features: pd.DataFrame,
    cfg: dict,
    sma_window: int,
    atr_window: int,
) -> ExtensionPercentilesBundle | None:
    ext_cfg = cfg.get("extension_percentiles", {})
    weekly_rule = cfg["weekly_regime"].get("resample_rule", "W-FRI")
    try:
        ext_series = features[f"extension_atr_{sma_window}_{atr_window}"]
        close_series = features["Close"]
        daily_report = compute_extension_percentiles(
            extension_series=ext_series,
            close_series=close_series,
            open_series=features["Open"],
            windows_years=ext_cfg.get("windows_years", [3]),
            days_per_year=int(ext_cfg.get("days_per_year", 252)),
            tail_high_pct=float(ext_cfg.get("tail_high_pct", 95)),
            tail_low_pct=float(ext_cfg.get("tail_low_pct", 5)),
            forward_days=ext_cfg.get("forward_days", [1, 3, 5, 10]),
            include_tail_events=False,
        )
        weekly_ext, weekly_close = build_weekly_extension_series(
            features,
            sma_window=sma_window,
            atr_window=atr_window,
            resample_rule=weekly_rule,
        )
        weekly_report = compute_extension_percentiles(
            extension_series=weekly_ext,
            close_series=weekly_close,
            open_series=features["Open"].resample(weekly_rule).first(),
            windows_years=ext_cfg.get("windows_years", [3]),
            days_per_year=int(ext_cfg.get("days_per_year", 252) / 5),
            tail_high_pct=float(ext_cfg.get("tail_high_pct", 95)),
            tail_low_pct=float(ext_cfg.get("tail_low_pct", 5)),
            forward_days=ext_cfg.get("forward_days", [1, 3, 5, 10]),
            include_tail_events=False,
        )
        return ExtensionPercentilesBundle(daily=daily_report, weekly=weekly_report)
    except Exception:  # noqa: BLE001
        return None


def compute_technical_snapshot(df_ohlc: pd.DataFrame, cfg: dict) -> TechnicalSnapshot | None:
    """Compute a compact technical snapshot from OHLCV data."""
    if df_ohlc is None or df_ohlc.empty:
        return None
    features = compute_features(df_ohlc, cfg)
    if features.empty:
        return None
    params = _snapshot_params(cfg)
    latest = _latest_valid_snapshot_row(
        features=features,
        cfg=cfg,
        required_columns=_required_snapshot_columns(params),
    )
    if latest is None:
        return None
    row, idx = latest

    extension_report = _build_extension_percentiles(
        features=features,
        cfg=cfg,
        sma_window=params.sma_window,
        atr_window=params.atr_window,
    )

    return TechnicalSnapshot(
        asof=_snapshot_asof(idx),
        close=_as_float(row.get("Close")),
        weekly_trend_up=_snapshot_weekly_trend(row),
        atr_window=params.atr_window,
        atr=_as_float(row.get(f"atr_{params.atr_window}")),
        atrp=_as_float(row.get(f"atrp_{params.atr_window}")),
        sma_window=params.sma_window,
        z_window=params.z_window,
        zscore=_as_float(row.get(f"zscore_{params.z_window}")),
        extension_atr=_as_float(row.get(f"extension_atr_{params.sma_window}_{params.atr_window}")),
        bb_window=params.bb_window,
        bb_dev=params.bb_dev,
        bb_pband=_as_float(row.get(f"bb_pband_{params.bb_window}_{params.bb_dev_label}")),
        bb_wband=_as_float(row.get(f"bb_wband_{params.bb_window}_{params.bb_dev_label}")),
        rsi_window=params.rsi_window,
        rsi=_snapshot_rsi_value(row, rsi_window=params.rsi_window),
        extension_percentiles=extension_report,
    )
