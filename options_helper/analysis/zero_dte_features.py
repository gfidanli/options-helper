from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DEFAULT_MARKET_TZ = "America/New_York"
_FEATURE_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "decision_ts_market",
    "bar_ts",
    "bar_ts_market",
    "intraday_return",
    "drawdown_from_open",
    "distance_from_vwap",
    "realized_intraday_vol",
    "current_bar_range",
    "bar_range_percentile",
    "minutes_since_open",
    "time_of_day_bucket",
    "bars_observed",
    "iv_regime",
    "feature_status",
)


@dataclass(frozen=True)
class ZeroDTEFeatureConfig:
    market_tz_name: str = DEFAULT_MARKET_TZ
    realized_vol_min_returns: int = 5
    bar_range_percentile_min_bars: int = 5
    time_bucket_cutoffs_minutes: tuple[int, int, int] = (60, 180, 330)

    def __post_init__(self) -> None:
        if self.realized_vol_min_returns < 1:
            raise ValueError("realized_vol_min_returns must be >= 1")
        if self.bar_range_percentile_min_bars < 1:
            raise ValueError("bar_range_percentile_min_bars must be >= 1")
        cuts = tuple(int(value) for value in self.time_bucket_cutoffs_minutes)
        if len(cuts) != 3:
            raise ValueError("time_bucket_cutoffs_minutes must contain exactly 3 values")
        if cuts[0] <= 0 or cuts[1] <= cuts[0] or cuts[2] <= cuts[1]:
            raise ValueError("time_bucket_cutoffs_minutes must be strictly increasing positive values")

    @property
    def market_tz(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz_name)


def compute_zero_dte_features(
    state_rows: pd.DataFrame,
    underlying_bars: pd.DataFrame,
    *,
    previous_close: float,
    iv_context: pd.DataFrame | pd.Series | Mapping[object, object] | None = None,
    config: ZeroDTEFeatureConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ZeroDTEFeatureConfig()
    state = state_rows.copy() if state_rows is not None else pd.DataFrame()
    if state.empty:
        return pd.DataFrame(columns=list(_FEATURE_COLUMNS))

    bars = _normalize_underlying_bars(underlying_bars, market_tz=cfg.market_tz)
    iv_lookup = _prepare_iv_regime_lookup(iv_context)
    prev_close = float(previous_close)
    prev_close_valid = math.isfinite(prev_close) and prev_close > 0.0

    rows: list[dict[str, object]] = []
    for _, state_row in state.iterrows():
        rows.append(
            _build_feature_row(
                state_row=state_row,
                bars=bars,
                previous_close=prev_close,
                previous_close_valid=prev_close_valid,
                iv_lookup=iv_lookup,
                config=cfg,
            )
        )

    return pd.DataFrame(rows, columns=list(_FEATURE_COLUMNS))


def _build_feature_row(
    *,
    state_row: pd.Series,
    bars: pd.DataFrame,
    previous_close: float,
    previous_close_valid: bool,
    iv_lookup: dict[pd.Timestamp | date, object],
    config: ZeroDTEFeatureConfig,
) -> dict[str, object]:
    decision_ts = _coerce_timestamp(state_row.get("decision_ts"), utc=True)
    decision_ts_market = _coerce_timestamp(
        state_row.get("decision_ts_market"),
        utc=False,
        market_tz=config.market_tz,
    )
    decision_bar_ts = _coerce_timestamp(state_row.get("bar_ts"), utc=True)
    decision_bar_market = _coerce_timestamp(
        state_row.get("bar_ts_market"),
        utc=False,
        market_tz=config.market_tz,
    )

    out: dict[str, object] = {
        "session_date": state_row.get("session_date"),
        "decision_ts": decision_ts,
        "decision_ts_market": decision_ts_market,
        "bar_ts": decision_bar_ts,
        "bar_ts_market": decision_bar_market,
        "intraday_return": float("nan"),
        "drawdown_from_open": float("nan"),
        "distance_from_vwap": float("nan"),
        "realized_intraday_vol": float("nan"),
        "current_bar_range": float("nan"),
        "bar_range_percentile": float("nan"),
        "minutes_since_open": pd.NA,
        "time_of_day_bucket": None,
        "bars_observed": 0,
        "iv_regime": _lookup_iv_regime(iv_lookup, decision_ts=decision_ts, session_date=state_row.get("session_date")),
        "feature_status": "state_not_ok",
    }

    state_status = str(state_row.get("status") or "").strip().lower()
    if state_status != "ok":
        out["feature_status"] = f"state_{state_status or 'unknown'}"
        return out

    if bars.empty:
        out["feature_status"] = "no_underlying_bars"
        return out

    cutoff_ts = _resolve_cutoff_ts(decision_ts=decision_ts, decision_bar_ts=decision_bar_ts)
    if cutoff_ts is None:
        out["feature_status"] = "missing_decision_timestamp"
        return out

    history = bars.loc[bars["timestamp"] <= cutoff_ts].copy()
    if history.empty:
        out["feature_status"] = "no_bars_before_decision"
        return out

    current = history.iloc[-1]
    out["bars_observed"] = int(len(history))

    current_close = float(pd.to_numeric(current.get("close"), errors="coerce"))
    if previous_close_valid and math.isfinite(current_close):
        out["intraday_return"] = (current_close / previous_close) - 1.0

    session_open = float(pd.to_numeric(history.iloc[0].get("open"), errors="coerce"))
    min_low = float(pd.to_numeric(history.get("low"), errors="coerce").min())
    if session_open > 0.0 and math.isfinite(min_low):
        out["drawdown_from_open"] = (min_low / session_open) - 1.0

    vwap_value = float(pd.to_numeric(current.get("vwap"), errors="coerce"))
    if not (math.isfinite(vwap_value) and vwap_value > 0.0):
        vwap_value = _cumulative_vwap(history)
    if math.isfinite(current_close) and math.isfinite(vwap_value) and vwap_value > 0.0:
        out["distance_from_vwap"] = (current_close / vwap_value) - 1.0

    close_series = pd.to_numeric(history.get("close"), errors="coerce").astype("float64")
    out["realized_intraday_vol"] = _realized_vol(
        close_series,
        min_returns=config.realized_vol_min_returns,
    )

    bar_range_series = _bar_range_series(history)
    if not bar_range_series.empty:
        current_range = float(bar_range_series.iloc[-1])
        out["current_bar_range"] = current_range
        out["bar_range_percentile"] = _current_percentile_rank(
            bar_range_series,
            current_value=current_range,
            min_points=config.bar_range_percentile_min_bars,
        )

    session_open_market = pd.Timestamp(history.iloc[0]["timestamp_market"])
    bar_market_ts = pd.Timestamp(current["timestamp_market"])
    minutes_since_open = int((bar_market_ts - session_open_market).total_seconds() // 60)
    out["minutes_since_open"] = minutes_since_open
    out["time_of_day_bucket"] = _time_bucket(
        minutes_since_open,
        cutoffs=config.time_bucket_cutoffs_minutes,
    )

    out["feature_status"] = "ok" if previous_close_valid else "invalid_previous_close"
    return out


def _normalize_underlying_bars(df: pd.DataFrame | None, *, market_tz: ZoneInfo) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "high", "low", "close", "volume", "vwap"])

    frame = df.copy()
    frame.columns = [str(col) for col in frame.columns]

    ts_col = _first_present(frame.columns, "timestamp", "ts", "time")
    if ts_col is None:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "high", "low", "close", "volume", "vwap"])

    frame["timestamp"] = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "high", "low", "close", "volume", "vwap"])

    market_col = _first_present(frame.columns, "timestamp_market")
    if market_col is None:
        frame["timestamp_market"] = frame["timestamp"].dt.tz_convert(market_tz)
    else:
        market_ts = pd.to_datetime(frame[market_col], errors="coerce")
        frame["timestamp_market"] = _to_market_timezone(market_ts, market_tz=market_tz)

    for col in ("open", "high", "low", "close", "volume", "vwap"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else float("nan")

    frame = frame.sort_values(by="timestamp", kind="mergesort")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    return frame.reset_index(drop=True)


def _prepare_iv_regime_lookup(
    iv_context: pd.DataFrame | pd.Series | Mapping[object, object] | None,
) -> dict[pd.Timestamp | date, object]:
    if iv_context is None:
        return {}

    if isinstance(iv_context, Mapping):
        out: dict[pd.Timestamp | date, object] = {}
        for key, value in iv_context.items():
            ts = _coerce_timestamp(key, utc=True)
            if ts is not None:
                out[ts] = value
                continue
            if isinstance(key, date):
                out[key] = value
        return out

    if isinstance(iv_context, pd.Series):
        frame = iv_context.rename("iv_regime").reset_index()
    else:
        frame = iv_context.copy()

    if frame.empty:
        return {}

    frame.columns = [str(col) for col in frame.columns]
    regime_col = _first_present(frame.columns, "iv_regime", "regime")
    if regime_col is None:
        return {}

    ts_col = _first_present(frame.columns, "decision_ts", "timestamp", "ts")
    date_col = _first_present(frame.columns, "session_date", "date")

    out: dict[pd.Timestamp | date, object] = {}
    if ts_col is not None:
        ts = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
        enriched = pd.DataFrame({"ts": ts, "regime": frame[regime_col]})
        enriched = enriched.loc[~enriched["ts"].isna()].copy()
        enriched = enriched.sort_values(by="ts", kind="mergesort")
        for _, row in enriched.iterrows():
            out[pd.Timestamp(row["ts"])] = row["regime"]

    if date_col is not None:
        for _, row in frame.iterrows():
            try:
                day = date.fromisoformat(str(row[date_col]))
            except ValueError:
                continue
            out[day] = row[regime_col]

    return out


def _lookup_iv_regime(
    iv_lookup: dict[pd.Timestamp | date, object],
    *,
    decision_ts: pd.Timestamp | None,
    session_date: object,
) -> object:
    if not iv_lookup:
        return None

    parsed_date = _coerce_date(session_date)
    if parsed_date is not None and parsed_date in iv_lookup:
        return iv_lookup[parsed_date]

    if decision_ts is None:
        return None

    ts_keys = sorted(key for key in iv_lookup if isinstance(key, pd.Timestamp))
    if not ts_keys:
        return None

    valid = [ts for ts in ts_keys if ts <= decision_ts]
    if not valid:
        return None
    return iv_lookup[valid[-1]]


def _resolve_cutoff_ts(
    *,
    decision_ts: pd.Timestamp | None,
    decision_bar_ts: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if decision_ts is None and decision_bar_ts is None:
        return None
    if decision_ts is None:
        return decision_bar_ts
    if decision_bar_ts is None:
        return decision_ts
    return min(decision_ts, decision_bar_ts)


def _coerce_timestamp(
    raw: object,
    *,
    utc: bool,
    market_tz: ZoneInfo | None = None,
) -> pd.Timestamp | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    timestamp = pd.Timestamp(ts)
    if utc:
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
    if market_tz is None:
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(market_tz)
    return timestamp.tz_convert(market_tz)


def _coerce_date(raw: object) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _to_market_timezone(series: pd.Series, *, market_tz: ZoneInfo) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if getattr(timestamps.dt, "tz", None) is None:
        return timestamps.dt.tz_localize(market_tz)
    return timestamps.dt.tz_convert(market_tz)


def _cumulative_vwap(history: pd.DataFrame) -> float:
    close = pd.to_numeric(history.get("close"), errors="coerce").astype("float64")
    volume = pd.to_numeric(history.get("volume"), errors="coerce").astype("float64")
    valid = close.notna() & volume.notna() & (volume > 0.0)
    if not valid.any():
        return float("nan")
    weights = volume.loc[valid]
    return float((close.loc[valid] * weights).sum() / weights.sum())


def _realized_vol(close: pd.Series, *, min_returns: int) -> float:
    valid = close.loc[close.notna() & (close > 0.0)]
    if valid.empty:
        return float("nan")
    log_returns = np.log(valid / valid.shift(1)).dropna()
    if len(log_returns) < int(min_returns):
        return float("nan")
    return float(log_returns.std(ddof=0))


def _bar_range_series(history: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(history.get("high"), errors="coerce").astype("float64")
    low = pd.to_numeric(history.get("low"), errors="coerce").astype("float64")
    close = pd.to_numeric(history.get("close"), errors="coerce").astype("float64")
    bar_range = (high - low) / close.replace(0.0, np.nan)
    return bar_range.dropna()


def _current_percentile_rank(series: pd.Series, *, current_value: float, min_points: int) -> float:
    clean = series.dropna().astype("float64")
    if len(clean) < int(min_points) or not math.isfinite(current_value):
        return float("nan")
    return float(100.0 * (clean <= current_value).mean())


def _time_bucket(minutes_since_open: int, *, cutoffs: tuple[int, int, int]) -> str:
    open_cutoff, morning_cutoff, midday_cutoff = cutoffs
    if minutes_since_open < open_cutoff:
        return "open"
    if minutes_since_open < morning_cutoff:
        return "morning"
    if minutes_since_open < midday_cutoff:
        return "midday"
    return "late_day"


def _first_present(columns: list[str] | pd.Index, *candidates: str) -> str | None:
    existing = set(str(col) for col in columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


__all__ = [
    "DEFAULT_MARKET_TZ",
    "ZeroDTEFeatureConfig",
    "compute_zero_dte_features",
]
