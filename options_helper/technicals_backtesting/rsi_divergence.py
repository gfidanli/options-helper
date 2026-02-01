from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi_regime_tag(*, rsi_value: float, rsi_overbought: float, rsi_oversold: float) -> str:
    """
    Tag an RSI value as overbought/oversold/neutral.

    Kept intentionally simple and deterministic; callers can optionally gate on extremes.
    """
    if float(rsi_value) >= float(rsi_overbought):
        return "overbought"
    if float(rsi_value) <= float(rsi_oversold):
        return "oversold"
    return "neutral"


@dataclass(frozen=True)
class RsiDivergenceEvent:
    date: str
    divergence: str  # "bearish" | "bullish"
    window_days: int
    swing1_date: str
    swing2_date: str
    close1: float
    close2: float
    rsi1: float
    rsi2: float
    price_delta_pct: float
    rsi_delta: float
    rsi_regime: str  # "overbought" | "oversold" | "neutral"


def compute_rsi_divergence_flags(
    *,
    close_series: pd.Series,
    rsi_series: pd.Series,
    extension_percentile_series: pd.Series | None,
    window_days: int = 14,
    min_extension_days: int = 5,
    min_extension_percentile: float = 95.0,
    max_extension_percentile: float = 5.0,
    min_price_delta_pct: float = 0.0,
    min_rsi_delta: float = 0.0,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    require_rsi_extreme: bool = False,
    min_separation_bars: int = 2,
) -> pd.DataFrame:
    """
    Detect bearish/bullish RSI divergences (daily bars) within a trailing window.

    Divergence definition (anchored on the *newer* swing point at index i):
    - Bearish: Close makes a higher high while RSI makes a lower high.
    - Bullish: Close makes a lower low while RSI makes a higher low.

    Extension gating:
    - If `extension_percentile_series` is provided, require at least `min_extension_days` in the
      trailing `window_days` where extension percentile is >= `min_extension_percentile` (bearish)
      or <= `max_extension_percentile` (bullish).
    - If not provided, no extension gating is applied (divergence is still computed).

    Returns a DataFrame indexed like the aligned input with:
    - `bearish_divergence`, `bullish_divergence` booleans
    - event metadata populated only on signal days
    """
    window_days = int(window_days)
    min_extension_days = int(min_extension_days)
    min_separation_bars = int(min_separation_bars)

    if window_days <= 1 or min_separation_bars < 1:
        return pd.DataFrame(index=pd.Index([], name=close_series.index.name))

    aligned = pd.concat(
        [
            close_series.rename("close"),
            rsi_series.rename("rsi"),
            (extension_percentile_series.rename("ext_pct") if extension_percentile_series is not None else None),
        ],
        axis=1,
    )
    aligned = aligned.dropna(subset=["close", "rsi"])
    if aligned.empty or len(aligned) < 3:
        return pd.DataFrame(index=aligned.index)

    close = aligned["close"].astype("float64")
    rsi = aligned["rsi"].astype("float64")
    ext_pct = aligned["ext_pct"].astype("float64") if "ext_pct" in aligned.columns else None

    n = len(aligned)
    c = close.to_numpy(dtype=float)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    swing_high[1:-1] = (c[1:-1] >= c[:-2]) & (c[1:-1] >= c[2:])
    swing_low[1:-1] = (c[1:-1] <= c[:-2]) & (c[1:-1] <= c[2:])

    # Extension gating: require sustained extension inside the trailing window.
    ext_high_ok = pd.Series(True, index=aligned.index, dtype=bool)
    ext_low_ok = pd.Series(True, index=aligned.index, dtype=bool)
    if ext_pct is not None and min_extension_days > 0:
        high_mask = (ext_pct >= float(min_extension_percentile)).fillna(False)
        low_mask = (ext_pct <= float(max_extension_percentile)).fillna(False)
        # Require a full window for stability (avoids early-history false positives).
        high_count = high_mask.rolling(window_days, min_periods=window_days).sum()
        low_count = low_mask.rolling(window_days, min_periods=window_days).sum()
        ext_high_ok = (high_count >= float(min_extension_days)).fillna(False)
        ext_low_ok = (low_count >= float(min_extension_days)).fillna(False)

    out = pd.DataFrame(index=aligned.index)
    out["window_days"] = window_days
    out["bearish_divergence"] = False
    out["bullish_divergence"] = False
    out["divergence"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["rsi_regime"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["swing1_date"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["swing2_date"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["close1"] = np.nan
    out["close2"] = np.nan
    out["rsi1"] = np.nan
    out["rsi2"] = np.nan
    out["price_delta_pct"] = np.nan
    out["rsi_delta"] = np.nan

    def _idx_date(i: int) -> str:
        idx = aligned.index[i]
        if isinstance(idx, pd.Timestamp):
            return idx.date().isoformat()
        return str(idx)

    # Scan swing points; each signal is anchored at the newer swing (index i).
    for i in range(1, n - 1):
        # Bearish divergence at swing highs.
        if swing_high[i] and bool(ext_high_ok.iloc[i]):
            start = max(1, i - window_days)
            j = None
            for k in range(i - min_separation_bars, start - 1, -1):
                if swing_high[k]:
                    j = k
                    break
            if j is not None:
                c1 = float(c[j])
                c2 = float(c[i])
                r1 = float(rsi.iloc[j])
                r2 = float(rsi.iloc[i])
                if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(r1) and np.isfinite(r2)):
                    pass
                else:
                    price_delta_pct = (c2 / c1 - 1.0) * 100.0 if c1 else float("nan")
                    rsi_delta = r2 - r1
                    if price_delta_pct >= float(min_price_delta_pct) and rsi_delta <= -float(min_rsi_delta):
                        regime = rsi_regime_tag(rsi_value=r2, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold)
                        if require_rsi_extreme and regime != "overbought":
                            pass
                        else:
                            out.iloc[i, out.columns.get_loc("bearish_divergence")] = True
                            out.iloc[i, out.columns.get_loc("divergence")] = "bearish"
                            out.iloc[i, out.columns.get_loc("rsi_regime")] = regime
                            out.iloc[i, out.columns.get_loc("swing1_date")] = _idx_date(j)
                            out.iloc[i, out.columns.get_loc("swing2_date")] = _idx_date(i)
                            out.iloc[i, out.columns.get_loc("close1")] = c1
                            out.iloc[i, out.columns.get_loc("close2")] = c2
                            out.iloc[i, out.columns.get_loc("rsi1")] = r1
                            out.iloc[i, out.columns.get_loc("rsi2")] = r2
                            out.iloc[i, out.columns.get_loc("price_delta_pct")] = price_delta_pct
                            out.iloc[i, out.columns.get_loc("rsi_delta")] = rsi_delta

        # Bullish divergence at swing lows.
        if swing_low[i] and bool(ext_low_ok.iloc[i]):
            start = max(1, i - window_days)
            j = None
            for k in range(i - min_separation_bars, start - 1, -1):
                if swing_low[k]:
                    j = k
                    break
            if j is not None:
                c1 = float(c[j])
                c2 = float(c[i])
                r1 = float(rsi.iloc[j])
                r2 = float(rsi.iloc[i])
                if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(r1) and np.isfinite(r2)):
                    continue
                price_delta_pct = (c2 / c1 - 1.0) * 100.0 if c1 else float("nan")
                rsi_delta = r2 - r1
                # Bullish: price makes a lower low (negative delta) while RSI makes a higher low.
                if (-price_delta_pct) >= float(min_price_delta_pct) and rsi_delta >= float(min_rsi_delta):
                    regime = rsi_regime_tag(rsi_value=r2, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold)
                    if require_rsi_extreme and regime != "oversold":
                        continue
                    out.iloc[i, out.columns.get_loc("bullish_divergence")] = True
                    out.iloc[i, out.columns.get_loc("divergence")] = "bullish"
                    out.iloc[i, out.columns.get_loc("rsi_regime")] = regime
                    out.iloc[i, out.columns.get_loc("swing1_date")] = _idx_date(j)
                    out.iloc[i, out.columns.get_loc("swing2_date")] = _idx_date(i)
                    out.iloc[i, out.columns.get_loc("close1")] = c1
                    out.iloc[i, out.columns.get_loc("close2")] = c2
                    out.iloc[i, out.columns.get_loc("rsi1")] = r1
                    out.iloc[i, out.columns.get_loc("rsi2")] = r2
                    out.iloc[i, out.columns.get_loc("price_delta_pct")] = price_delta_pct
                    out.iloc[i, out.columns.get_loc("rsi_delta")] = rsi_delta

    return out
