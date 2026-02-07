from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd


AnchorType = Literal["session_open", "timestamp", "date", "breakout_day"]


@dataclass(frozen=True)
class AnchoredVwapResult:
    anchor_type: AnchorType
    anchor_ts_utc: pd.Timestamp | None
    anchor_price: float | None
    anchored_vwap: float | None
    distance_from_spot_pct: float | None
    warnings: list[str]


@dataclass(frozen=True)
class VolumeProfileBin:
    price_bin_low: float
    price_bin_high: float
    volume: float
    volume_pct: float
    is_poc: bool
    is_hvn: bool
    is_lvn: bool


@dataclass(frozen=True)
class VolumeProfileResult:
    bins: list[VolumeProfileBin]
    poc_price: float | None
    hvn_candidates: list[float]
    lvn_candidates: list[float]
    warnings: list[str]


@dataclass(frozen=True)
class DailyLevelsResult:
    spot: float | None
    prev_close: float | None
    session_open: float | None
    gap_pct: float | None
    prior_high: float | None
    prior_low: float | None
    rolling_high: float | None
    rolling_low: float | None
    warnings: list[str]


@dataclass(frozen=True)
class RelativeStrengthBetaResult:
    rs_ratio: float | None
    beta: float | None
    corr: float | None
    ratio_series: pd.Series
    beta_series: pd.Series
    corr_series: pd.Series
    warnings: list[str]


@dataclass(frozen=True)
class LevelsSummaryResult:
    spot: float | None
    prev_close: float | None
    session_open: float | None
    gap_pct: float | None
    prior_high: float | None
    prior_low: float | None
    rolling_high: float | None
    rolling_low: float | None
    rs_ratio: float | None
    beta_20d: float | None
    corr_20d: float | None
    warnings: list[str]


def compute_anchored_vwap(
    intraday_bars: pd.DataFrame,
    *,
    anchor_type: AnchorType = "session_open",
    anchor_timestamp: pd.Timestamp | str | None = None,
    anchor_date: pd.Timestamp | date | str | None = None,
    breakout_daily: pd.DataFrame | None = None,
    breakout_lookback: int = 20,
    spot: float | None = None,
) -> AnchoredVwapResult:
    bars, warnings = _normalize_intraday_bars(intraday_bars)
    if bars.empty:
        warnings.append("empty_intraday_bars")
        return AnchoredVwapResult(
            anchor_type=anchor_type,
            anchor_ts_utc=None,
            anchor_price=None,
            anchored_vwap=None,
            distance_from_spot_pct=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    anchor_ts = _resolve_anchor_timestamp(
        bars,
        anchor_type=anchor_type,
        anchor_timestamp=anchor_timestamp,
        anchor_date=anchor_date,
        breakout_daily=breakout_daily,
        breakout_lookback=breakout_lookback,
        warnings=warnings,
    )

    if anchor_ts is None:
        return AnchoredVwapResult(
            anchor_type=anchor_type,
            anchor_ts_utc=None,
            anchor_price=None,
            anchored_vwap=None,
            distance_from_spot_pct=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    anchored = bars.loc[bars["timestamp"] >= anchor_ts].copy()
    if anchored.empty:
        warnings.append("anchor_after_last_bar")
        return AnchoredVwapResult(
            anchor_type=anchor_type,
            anchor_ts_utc=anchor_ts,
            anchor_price=None,
            anchored_vwap=None,
            distance_from_spot_pct=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    anchor_row = anchored.iloc[0]
    anchor_price = _row_price(anchor_row)

    vol = anchored["volume"].astype("float64")
    weighted_price = anchored["vwap"]
    typical = (anchored["high"] + anchored["low"] + anchored["close"]) / 3.0
    fallback_price = typical.where(typical.notna(), anchored["close"])
    weighted_price = weighted_price.where(weighted_price.notna(), fallback_price)

    valid = (vol > 0.0) & weighted_price.notna()
    skipped_price_rows = int(((vol > 0.0) & (~weighted_price.notna())).sum())
    if skipped_price_rows > 0:
        warnings.append("missing_prices_skipped")

    if not bool(valid.any()):
        warnings.append("zero_volume_after_anchor")
        return AnchoredVwapResult(
            anchor_type=anchor_type,
            anchor_ts_utc=anchor_ts,
            anchor_price=anchor_price,
            anchored_vwap=None,
            distance_from_spot_pct=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    denominator = float(vol[valid].sum())
    if denominator <= 0.0:
        warnings.append("zero_volume_after_anchor")
        return AnchoredVwapResult(
            anchor_type=anchor_type,
            anchor_ts_utc=anchor_ts,
            anchor_price=anchor_price,
            anchored_vwap=None,
            distance_from_spot_pct=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    numerator = float((weighted_price[valid] * vol[valid]).sum())
    anchored_vwap = numerator / denominator

    distance_from_spot_pct: float | None = None
    spot_value = _coerce_number(spot)
    if spot_value is not None and anchored_vwap > 0.0:
        distance_from_spot_pct = (spot_value - anchored_vwap) / anchored_vwap

    return AnchoredVwapResult(
        anchor_type=anchor_type,
        anchor_ts_utc=anchor_ts,
        anchor_price=anchor_price,
        anchored_vwap=anchored_vwap,
        distance_from_spot_pct=distance_from_spot_pct,
        warnings=_dedupe_preserve_order(warnings),
    )


def find_breakout_day(daily_candles: pd.DataFrame, *, lookback: int = 20) -> date | None:
    if lookback <= 0:
        return None

    frame, _warnings = _normalize_daily_bars(daily_candles)
    if frame.empty or "close" not in frame.columns:
        return None

    close = frame["close"].astype("float64")
    if close.empty or len(close) <= lookback:
        return None

    breakout_day: date | None = None
    for idx in range(lookback, len(close)):
        current = _coerce_number(close.iloc[idx])
        if current is None:
            continue
        prior_window = close.iloc[idx - lookback : idx].dropna()
        if prior_window.empty:
            continue
        prior_high = float(prior_window.max())
        if current > prior_high:
            ts = frame["date"].iloc[idx]
            breakout_day = ts.date()

    return breakout_day


def compute_volume_profile(
    intraday_bars: pd.DataFrame,
    *,
    num_bins: int = 20,
    hvn_quantile: float = 0.8,
    lvn_quantile: float = 0.2,
) -> VolumeProfileResult:
    warnings: list[str] = []
    bars, normalize_warnings = _normalize_intraday_bars(intraday_bars)
    warnings.extend(normalize_warnings)

    if num_bins <= 0:
        warnings.append("invalid_num_bins")
        return VolumeProfileResult(
            bins=[],
            poc_price=None,
            hvn_candidates=[],
            lvn_candidates=[],
            warnings=_dedupe_preserve_order(warnings),
        )

    if bars.empty:
        warnings.append("empty_intraday_bars")
        return VolumeProfileResult(
            bins=[],
            poc_price=None,
            hvn_candidates=[],
            lvn_candidates=[],
            warnings=_dedupe_preserve_order(warnings),
        )

    profile_price = bars["close"].copy()
    high_low_mid = (bars["high"] + bars["low"]) / 2.0
    profile_price = profile_price.where(profile_price.notna(), high_low_mid)
    profile_price = profile_price.where(profile_price.notna(), bars["open"])

    volume = bars["volume"].astype("float64")
    valid = profile_price.notna() & (volume > 0.0)

    if not bool(valid.any()):
        warnings.append("zero_volume_profile")
        return VolumeProfileResult(
            bins=[],
            poc_price=None,
            hvn_candidates=[],
            lvn_candidates=[],
            warnings=_dedupe_preserve_order(warnings),
        )

    price_values = profile_price[valid].astype("float64").to_numpy()
    vol_values = volume[valid].astype("float64").to_numpy()

    price_min = float(np.min(price_values))
    price_max = float(np.max(price_values))

    if price_min == price_max:
        total = float(np.sum(vol_values))
        if total <= 0.0:
            warnings.append("zero_volume_profile")
            return VolumeProfileResult(
                bins=[],
                poc_price=None,
                hvn_candidates=[],
                lvn_candidates=[],
                warnings=_dedupe_preserve_order(warnings),
            )
        single_bin = VolumeProfileBin(
            price_bin_low=price_min,
            price_bin_high=price_max,
            volume=total,
            volume_pct=1.0,
            is_poc=True,
            is_hvn=True,
            is_lvn=True,
        )
        return VolumeProfileResult(
            bins=[single_bin],
            poc_price=price_min,
            hvn_candidates=[price_min],
            lvn_candidates=[price_min],
            warnings=_dedupe_preserve_order(warnings),
        )

    edges = np.linspace(price_min, price_max, num_bins + 1, dtype="float64")
    bin_idx = np.searchsorted(edges, price_values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    bin_volume = np.zeros(num_bins, dtype="float64")
    np.add.at(bin_volume, bin_idx, vol_values)

    total_volume = float(np.sum(bin_volume))
    if total_volume <= 0.0:
        warnings.append("zero_volume_profile")
        return VolumeProfileResult(
            bins=[],
            poc_price=None,
            hvn_candidates=[],
            lvn_candidates=[],
            warnings=_dedupe_preserve_order(warnings),
        )

    active = bin_volume[bin_volume > 0.0]
    hvn_cutoff = float(np.quantile(active, hvn_quantile)) if active.size else float("nan")
    lvn_cutoff = float(np.quantile(active, lvn_quantile)) if active.size else float("nan")

    poc_idx = int(np.argmax(bin_volume))

    bins: list[VolumeProfileBin] = []
    hvn_candidates: list[float] = []
    lvn_candidates: list[float] = []
    poc_price: float | None = None

    for idx in range(num_bins):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        volume_value = float(bin_volume[idx])
        mid = (low + high) / 2.0
        is_poc = idx == poc_idx and volume_value > 0.0
        is_hvn = volume_value > 0.0 and volume_value >= hvn_cutoff
        is_lvn = volume_value > 0.0 and volume_value <= lvn_cutoff

        if is_poc:
            poc_price = mid
        if is_hvn:
            hvn_candidates.append(mid)
        if is_lvn:
            lvn_candidates.append(mid)

        bins.append(
            VolumeProfileBin(
                price_bin_low=low,
                price_bin_high=high,
                volume=volume_value,
                volume_pct=volume_value / total_volume,
                is_poc=is_poc,
                is_hvn=is_hvn,
                is_lvn=is_lvn,
            )
        )

    return VolumeProfileResult(
        bins=bins,
        poc_price=poc_price,
        hvn_candidates=hvn_candidates,
        lvn_candidates=lvn_candidates,
        warnings=_dedupe_preserve_order(warnings),
    )


def compute_gap_and_daily_levels(
    daily_candles: pd.DataFrame,
    *,
    rolling_window: int = 20,
) -> DailyLevelsResult:
    frame, warnings = _normalize_daily_bars(daily_candles)
    if frame.empty:
        warnings.append("empty_daily_candles")
        return DailyLevelsResult(
            spot=None,
            prev_close=None,
            session_open=None,
            gap_pct=None,
            prior_high=None,
            prior_low=None,
            rolling_high=None,
            rolling_low=None,
            warnings=_dedupe_preserve_order(warnings),
        )

    if rolling_window <= 0:
        warnings.append("invalid_rolling_window")

    spot = _coerce_number(frame["close"].iloc[-1]) if "close" in frame.columns else None
    session_open = _coerce_number(frame["open"].iloc[-1]) if "open" in frame.columns else None

    prev_close: float | None = None
    prior_high: float | None = None
    prior_low: float | None = None

    if len(frame) >= 2:
        prev_close = _coerce_number(frame["close"].iloc[-2]) if "close" in frame.columns else None
        prior_high = _coerce_number(frame["high"].iloc[-2]) if "high" in frame.columns else None
        prior_low = _coerce_number(frame["low"].iloc[-2]) if "low" in frame.columns else None
    else:
        warnings.append("insufficient_daily_history")

    gap_pct: float | None = None
    if prev_close is not None and prev_close > 0.0 and session_open is not None:
        gap_pct = (session_open - prev_close) / prev_close

    rolling_high: float | None = None
    rolling_low: float | None = None
    prior_slice = frame.iloc[:-1]
    if not prior_slice.empty:
        lookback = int(rolling_window) if rolling_window > 0 else len(prior_slice)
        lookback = max(1, lookback)
        lookback_slice = prior_slice.tail(lookback)
        high_series = pd.to_numeric(lookback_slice["high"], errors="coerce").dropna()
        low_series = pd.to_numeric(lookback_slice["low"], errors="coerce").dropna()
        if not high_series.empty:
            rolling_high = float(high_series.max())
        if not low_series.empty:
            rolling_low = float(low_series.min())

    return DailyLevelsResult(
        spot=spot,
        prev_close=prev_close,
        session_open=session_open,
        gap_pct=gap_pct,
        prior_high=prior_high,
        prior_low=prior_low,
        rolling_high=rolling_high,
        rolling_low=rolling_low,
        warnings=_dedupe_preserve_order(warnings),
    )


def compute_relative_strength_beta_corr(
    daily_candles: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    window: int = 20,
) -> RelativeStrengthBetaResult:
    warnings: list[str] = []

    if window <= 1:
        warnings.append("invalid_window")

    underlying_frame, underlying_warnings = _normalize_daily_bars(daily_candles)
    benchmark_frame, benchmark_warnings = _normalize_daily_bars(benchmark_daily)
    warnings.extend(underlying_warnings)
    warnings.extend(benchmark_warnings)

    if underlying_frame.empty or benchmark_frame.empty:
        warnings.append("missing_benchmark_history")
        empty_series = pd.Series(dtype="float64")
        return RelativeStrengthBetaResult(
            rs_ratio=None,
            beta=None,
            corr=None,
            ratio_series=empty_series,
            beta_series=empty_series,
            corr_series=empty_series,
            warnings=_dedupe_preserve_order(warnings),
        )

    aligned = pd.concat(
        [
            underlying_frame[["date", "close"]].set_index("date").rename(columns={"close": "asset"}),
            benchmark_frame[["date", "close"]].set_index("date").rename(columns={"close": "benchmark"}),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    aligned = aligned.dropna(subset=["asset", "benchmark"])
    if aligned.empty:
        warnings.append("missing_benchmark_history")
        empty_series = pd.Series(dtype="float64")
        return RelativeStrengthBetaResult(
            rs_ratio=None,
            beta=None,
            corr=None,
            ratio_series=empty_series,
            beta_series=empty_series,
            corr_series=empty_series,
            warnings=_dedupe_preserve_order(warnings),
        )

    ratio_series = aligned["asset"] / aligned["benchmark"].where(aligned["benchmark"] > 0.0)
    ratio_series = ratio_series.astype("float64")
    rs_ratio = _last_finite(ratio_series)

    asset_ret = aligned["asset"].pct_change()
    benchmark_ret = aligned["benchmark"].pct_change()
    returns = pd.concat([asset_ret.rename("asset"), benchmark_ret.rename("benchmark")], axis=1)
    valid_returns = returns.dropna(subset=["asset", "benchmark"])

    beta_series = pd.Series(index=returns.index, dtype="float64")
    corr_series = pd.Series(index=returns.index, dtype="float64")
    beta_value: float | None = None
    corr_value: float | None = None

    if window > 1 and len(valid_returns) >= window:
        cov_series = returns["asset"].rolling(window=window, min_periods=window).cov(returns["benchmark"])
        var_series = returns["benchmark"].rolling(window=window, min_periods=window).var()
        beta_series = cov_series / var_series.where(var_series != 0.0)
        corr_series = returns["asset"].rolling(window=window, min_periods=window).corr(returns["benchmark"])

        beta_value = _last_finite(beta_series)
        corr_value = _last_finite(corr_series)

        var_last = _last_finite(var_series)
        if var_last is None or var_last == 0.0:
            warnings.append("zero_benchmark_variance")
    else:
        warnings.append("insufficient_return_history")

    return RelativeStrengthBetaResult(
        rs_ratio=rs_ratio,
        beta=beta_value,
        corr=corr_value,
        ratio_series=ratio_series,
        beta_series=beta_series,
        corr_series=corr_series,
        warnings=_dedupe_preserve_order(warnings),
    )


def compute_levels_summary(
    daily_candles: pd.DataFrame,
    *,
    benchmark_daily: pd.DataFrame | None = None,
    rolling_window: int = 20,
    rs_window: int = 20,
) -> LevelsSummaryResult:
    daily = compute_gap_and_daily_levels(daily_candles, rolling_window=rolling_window)
    rs_warnings: list[str] = []

    rs_ratio: float | None = None
    beta_20d: float | None = None
    corr_20d: float | None = None

    if benchmark_daily is not None:
        rs = compute_relative_strength_beta_corr(daily_candles, benchmark_daily, window=rs_window)
        rs_ratio = rs.rs_ratio
        beta_20d = rs.beta
        corr_20d = rs.corr
        rs_warnings.extend(rs.warnings)
    else:
        rs_warnings.append("missing_benchmark_history")

    warnings = _dedupe_preserve_order([*daily.warnings, *rs_warnings])

    return LevelsSummaryResult(
        spot=daily.spot,
        prev_close=daily.prev_close,
        session_open=daily.session_open,
        gap_pct=daily.gap_pct,
        prior_high=daily.prior_high,
        prior_low=daily.prior_low,
        rolling_high=daily.rolling_high,
        rolling_low=daily.rolling_low,
        rs_ratio=rs_ratio,
        beta_20d=beta_20d,
        corr_20d=corr_20d,
        warnings=warnings,
    )


def _resolve_anchor_timestamp(
    bars: pd.DataFrame,
    *,
    anchor_type: AnchorType,
    anchor_timestamp: pd.Timestamp | str | None,
    anchor_date: pd.Timestamp | date | str | None,
    breakout_daily: pd.DataFrame | None,
    breakout_lookback: int,
    warnings: list[str],
) -> pd.Timestamp | None:
    if bars.empty:
        return None

    if anchor_type == "session_open":
        return bars["timestamp"].iloc[0]

    if anchor_type == "timestamp":
        anchor_ts = pd.to_datetime(anchor_timestamp, errors="coerce", utc=True)
        if pd.isna(anchor_ts):
            warnings.append("missing_anchor_timestamp")
            return None
        anchored = bars.loc[bars["timestamp"] >= anchor_ts]
        if anchored.empty:
            warnings.append("anchor_after_last_bar")
            return None
        return anchored["timestamp"].iloc[0]

    if anchor_type == "date":
        anchor_day = _coerce_date(anchor_date)
        if anchor_day is None:
            warnings.append("missing_anchor_date")
            return None
        anchored = bars.loc[bars["timestamp"].dt.date == anchor_day]
        if anchored.empty:
            warnings.append("anchor_date_not_found")
            return None
        return anchored["timestamp"].iloc[0]

    if anchor_type == "breakout_day":
        if breakout_daily is None:
            warnings.append("missing_breakout_daily")
            return None
        breakout_day = find_breakout_day(breakout_daily, lookback=breakout_lookback)
        if breakout_day is None:
            warnings.append("breakout_day_not_found")
            return None
        anchored = bars.loc[bars["timestamp"].dt.date == breakout_day]
        if anchored.empty:
            warnings.append("breakout_day_not_found")
            return None
        return anchored["timestamp"].iloc[0]

    warnings.append("unsupported_anchor_type")
    return None


def _normalize_intraday_bars(intraday_bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if intraday_bars is None or intraday_bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "vwap", "volume"]), warnings

    frame = intraday_bars.copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        ts = pd.to_datetime(frame.index, errors="coerce", utc=True)
    else:
        warnings.append("missing_timestamp")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "vwap", "volume"]), warnings

    frame = frame.assign(timestamp=ts)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True, kind="mergesort")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

    for column in ("open", "high", "low", "close", "vwap"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "volume" not in frame.columns:
        warnings.append("missing_volume")
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["volume"] = frame["volume"].where(frame["volume"] > 0.0, 0.0)

    return frame[["timestamp", "open", "high", "low", "close", "vwap", "volume"]], warnings


def _normalize_daily_bars(daily_candles: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if daily_candles is None or daily_candles.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), warnings

    frame = daily_candles.copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        idx = pd.to_datetime(frame["date"], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        idx = pd.to_datetime(frame.index, errors="coerce")
    else:
        warnings.append("missing_date")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), warnings

    frame = frame.assign(date=idx)
    frame = frame.dropna(subset=["date"]).sort_values("date", ascending=True, kind="mergesort")
    frame = frame.drop_duplicates(subset=["date"], keep="last")

    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "volume" not in frame.columns:
        warnings.append("missing_volume")
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["volume"] = frame["volume"].where(frame["volume"] > 0.0, 0.0)

    return frame[["date", "open", "high", "low", "close", "volume"]], warnings


def _coerce_date(value: pd.Timestamp | date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _row_price(row: pd.Series) -> float | None:
    vwap = _coerce_number(row.get("vwap"))
    if vwap is not None:
        return vwap

    close = _coerce_number(row.get("close"))
    if close is not None:
        return close

    high = _coerce_number(row.get("high"))
    low = _coerce_number(row.get("low"))
    if high is not None and low is not None:
        return (high + low) / 2.0

    return _coerce_number(row.get("open"))


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _last_finite(series: pd.Series) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    value = float(cleaned.iloc[-1])
    if not np.isfinite(value):
        return None
    return value


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[Any] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "AnchorType",
    "AnchoredVwapResult",
    "DailyLevelsResult",
    "LevelsSummaryResult",
    "RelativeStrengthBetaResult",
    "VolumeProfileBin",
    "VolumeProfileResult",
    "compute_anchored_vwap",
    "compute_gap_and_daily_levels",
    "compute_levels_summary",
    "compute_relative_strength_beta_corr",
    "compute_volume_profile",
    "find_breakout_day",
]
