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


def _empty_divergence_output(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def _aligned_divergence_inputs(
    *,
    close_series: pd.Series,
    rsi_series: pd.Series,
    extension_percentile_series: pd.Series | None,
) -> pd.DataFrame:
    columns = [close_series.rename("close"), rsi_series.rename("rsi")]
    if extension_percentile_series is not None:
        columns.append(extension_percentile_series.rename("ext_pct"))
    aligned = pd.concat(columns, axis=1)
    return aligned.dropna(subset=["close", "rsi"])


def _compute_swing_masks(close_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(close_values)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if n > 2:
        swing_high[1:-1] = (close_values[1:-1] >= close_values[:-2]) & (close_values[1:-1] >= close_values[2:])
        swing_low[1:-1] = (close_values[1:-1] <= close_values[:-2]) & (close_values[1:-1] <= close_values[2:])
    return swing_high, swing_low


def _compute_extension_gates(
    *,
    ext_pct: pd.Series | None,
    index: pd.Index,
    window_days: int,
    min_extension_days: int,
    min_extension_percentile: float,
    max_extension_percentile: float,
) -> tuple[pd.Series, pd.Series]:
    ext_high_ok = pd.Series(True, index=index, dtype=bool)
    ext_low_ok = pd.Series(True, index=index, dtype=bool)
    if ext_pct is None or min_extension_days <= 0:
        return ext_high_ok, ext_low_ok
    high_mask = (ext_pct >= float(min_extension_percentile)).fillna(False)
    low_mask = (ext_pct <= float(max_extension_percentile)).fillna(False)
    high_count = high_mask.rolling(window_days, min_periods=window_days).sum()
    low_count = low_mask.rolling(window_days, min_periods=window_days).sum()
    return (high_count >= float(min_extension_days)).fillna(False), (low_count >= float(min_extension_days)).fillna(
        False
    )


def _initialize_output_frame(index: pd.Index, *, window_days: int) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
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
    return out


def _index_date(index: pd.Index, i: int) -> str:
    idx = index[i]
    if isinstance(idx, pd.Timestamp):
        return idx.date().isoformat()
    return str(idx)


def _find_prior_swing(
    *,
    swing_mask: np.ndarray,
    i: int,
    window_days: int,
    min_separation_bars: int,
) -> int | None:
    start = max(1, i - window_days)
    for k in range(i - min_separation_bars, start - 1, -1):
        if swing_mask[k]:
            return k
    return None


def _write_divergence_row(
    out: pd.DataFrame,
    *,
    i: int,
    divergence: str,
    regime: str,
    swing1_date: str,
    swing2_date: str,
    close1: float,
    close2: float,
    rsi1: float,
    rsi2: float,
    price_delta_pct: float,
    rsi_delta: float,
) -> None:
    flag_column = "bearish_divergence" if divergence == "bearish" else "bullish_divergence"
    out.iloc[i, out.columns.get_loc(flag_column)] = True
    out.iloc[i, out.columns.get_loc("divergence")] = divergence
    out.iloc[i, out.columns.get_loc("rsi_regime")] = regime
    out.iloc[i, out.columns.get_loc("swing1_date")] = swing1_date
    out.iloc[i, out.columns.get_loc("swing2_date")] = swing2_date
    out.iloc[i, out.columns.get_loc("close1")] = close1
    out.iloc[i, out.columns.get_loc("close2")] = close2
    out.iloc[i, out.columns.get_loc("rsi1")] = rsi1
    out.iloc[i, out.columns.get_loc("rsi2")] = rsi2
    out.iloc[i, out.columns.get_loc("price_delta_pct")] = price_delta_pct
    out.iloc[i, out.columns.get_loc("rsi_delta")] = rsi_delta


def _try_bearish_divergence(
    *,
    i: int,
    close_values: np.ndarray,
    rsi_series: pd.Series,
    swing_high: np.ndarray,
    ext_high_ok: pd.Series,
    window_days: int,
    min_separation_bars: int,
    min_price_delta_pct: float,
    min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> tuple[int, float, float, float, float, float, float, str] | None:
    if not swing_high[i] or not bool(ext_high_ok.iloc[i]):
        return None
    j = _find_prior_swing(
        swing_mask=swing_high,
        i=i,
        window_days=window_days,
        min_separation_bars=min_separation_bars,
    )
    if j is None:
        return None
    c1 = float(close_values[j])
    c2 = float(close_values[i])
    r1 = float(rsi_series.iloc[j])
    r2 = float(rsi_series.iloc[i])
    if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(r1) and np.isfinite(r2)):
        return None
    price_delta_pct = (c2 / c1 - 1.0) * 100.0 if c1 else float("nan")
    rsi_delta = r2 - r1
    if price_delta_pct < float(min_price_delta_pct) or rsi_delta > -float(min_rsi_delta):
        return None
    regime = rsi_regime_tag(rsi_value=r2, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold)
    if require_rsi_extreme and regime != "overbought":
        return None
    return j, c1, c2, r1, r2, price_delta_pct, rsi_delta, regime


def _try_bullish_divergence(
    *,
    i: int,
    close_values: np.ndarray,
    rsi_series: pd.Series,
    swing_low: np.ndarray,
    ext_low_ok: pd.Series,
    window_days: int,
    min_separation_bars: int,
    min_price_delta_pct: float,
    min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> tuple[int, float, float, float, float, float, float, str] | None:
    if not swing_low[i] or not bool(ext_low_ok.iloc[i]):
        return None
    j = _find_prior_swing(
        swing_mask=swing_low,
        i=i,
        window_days=window_days,
        min_separation_bars=min_separation_bars,
    )
    if j is None:
        return None
    c1 = float(close_values[j])
    c2 = float(close_values[i])
    r1 = float(rsi_series.iloc[j])
    r2 = float(rsi_series.iloc[i])
    if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(r1) and np.isfinite(r2)):
        return None
    price_delta_pct = (c2 / c1 - 1.0) * 100.0 if c1 else float("nan")
    rsi_delta = r2 - r1
    if (-price_delta_pct) < float(min_price_delta_pct) or rsi_delta < float(min_rsi_delta):
        return None
    regime = rsi_regime_tag(rsi_value=r2, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold)
    if require_rsi_extreme and regime != "oversold":
        return None
    return j, c1, c2, r1, r2, price_delta_pct, rsi_delta, regime


@dataclass(frozen=True)
class _DivergenceState:
    index: pd.Index
    close_values: np.ndarray
    rsi_series: pd.Series
    swing_high: np.ndarray
    swing_low: np.ndarray
    ext_high_ok: pd.Series
    ext_low_ok: pd.Series


def _prepare_divergence_state(
    *,
    close_series: pd.Series,
    rsi_series: pd.Series,
    extension_percentile_series: pd.Series | None,
    window_days: int,
    min_extension_days: int,
    min_extension_percentile: float,
    max_extension_percentile: float,
) -> tuple[_DivergenceState | None, pd.Index]:
    aligned = _aligned_divergence_inputs(
        close_series=close_series,
        rsi_series=rsi_series,
        extension_percentile_series=extension_percentile_series,
    )
    if aligned.empty or len(aligned) < 3:
        return None, aligned.index
    close = aligned["close"].astype("float64")
    rsi = aligned["rsi"].astype("float64")
    ext_pct = aligned["ext_pct"].astype("float64") if "ext_pct" in aligned.columns else None
    close_values = close.to_numpy(dtype=float)
    swing_high, swing_low = _compute_swing_masks(close_values)
    ext_high_ok, ext_low_ok = _compute_extension_gates(
        ext_pct=ext_pct,
        index=aligned.index,
        window_days=window_days,
        min_extension_days=min_extension_days,
        min_extension_percentile=min_extension_percentile,
        max_extension_percentile=max_extension_percentile,
    )
    return (
        _DivergenceState(
            index=aligned.index,
            close_values=close_values,
            rsi_series=rsi,
            swing_high=swing_high,
            swing_low=swing_low,
            ext_high_ok=ext_high_ok,
            ext_low_ok=ext_low_ok,
        ),
        aligned.index,
    )


def _maybe_write_bearish_divergence(
    out: pd.DataFrame,
    *,
    state: _DivergenceState,
    i: int,
    window_days: int,
    min_separation_bars: int,
    min_price_delta_pct: float,
    min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> None:
    bearish = _try_bearish_divergence(
        i=i,
        close_values=state.close_values,
        rsi_series=state.rsi_series,
        swing_high=state.swing_high,
        ext_high_ok=state.ext_high_ok,
        window_days=window_days,
        min_separation_bars=min_separation_bars,
        min_price_delta_pct=min_price_delta_pct,
        min_rsi_delta=min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    if bearish is None:
        return
    j, c1, c2, r1, r2, price_delta_pct, rsi_delta, regime = bearish
    _write_divergence_row(
        out,
        i=i,
        divergence="bearish",
        regime=regime,
        swing1_date=_index_date(state.index, j),
        swing2_date=_index_date(state.index, i),
        close1=c1,
        close2=c2,
        rsi1=r1,
        rsi2=r2,
        price_delta_pct=price_delta_pct,
        rsi_delta=rsi_delta,
    )


def _maybe_write_bullish_divergence(
    out: pd.DataFrame,
    *,
    state: _DivergenceState,
    i: int,
    window_days: int,
    min_separation_bars: int,
    min_price_delta_pct: float,
    min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> None:
    bullish = _try_bullish_divergence(
        i=i,
        close_values=state.close_values,
        rsi_series=state.rsi_series,
        swing_low=state.swing_low,
        ext_low_ok=state.ext_low_ok,
        window_days=window_days,
        min_separation_bars=min_separation_bars,
        min_price_delta_pct=min_price_delta_pct,
        min_rsi_delta=min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    if bullish is None:
        return
    j, c1, c2, r1, r2, price_delta_pct, rsi_delta, regime = bullish
    _write_divergence_row(
        out,
        i=i,
        divergence="bullish",
        regime=regime,
        swing1_date=_index_date(state.index, j),
        swing2_date=_index_date(state.index, i),
        close1=c1,
        close2=c2,
        rsi1=r1,
        rsi2=r2,
        price_delta_pct=price_delta_pct,
        rsi_delta=rsi_delta,
    )


def _populate_divergence_rows(
    out: pd.DataFrame,
    *,
    state: _DivergenceState,
    window_days: int,
    min_separation_bars: int,
    min_price_delta_pct: float,
    min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> None:
    for i in range(1, len(state.index) - 1):
        _maybe_write_bearish_divergence(
            out,
            state=state,
            i=i,
            window_days=window_days,
            min_separation_bars=min_separation_bars,
            min_price_delta_pct=min_price_delta_pct,
            min_rsi_delta=min_rsi_delta,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            require_rsi_extreme=require_rsi_extreme,
        )
        _maybe_write_bullish_divergence(
            out,
            state=state,
            i=i,
            window_days=window_days,
            min_separation_bars=min_separation_bars,
            min_price_delta_pct=min_price_delta_pct,
            min_rsi_delta=min_rsi_delta,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            require_rsi_extreme=require_rsi_extreme,
        )


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
    """Detect bearish/bullish RSI divergences with optional extension gating."""
    window_days = int(window_days)
    min_extension_days = int(min_extension_days)
    min_separation_bars = int(min_separation_bars)

    if window_days <= 1 or min_separation_bars < 1:
        return _empty_divergence_output(pd.Index([], name=close_series.index.name))

    state, aligned_index = _prepare_divergence_state(
        close_series=close_series,
        rsi_series=rsi_series,
        extension_percentile_series=extension_percentile_series,
        window_days=window_days,
        min_extension_days=min_extension_days,
        min_extension_percentile=min_extension_percentile,
        max_extension_percentile=max_extension_percentile,
    )
    if state is None:
        return _empty_divergence_output(aligned_index)
    out = _initialize_output_frame(state.index, window_days=window_days)
    _populate_divergence_rows(
        out,
        state=state,
        window_days=window_days,
        min_separation_bars=min_separation_bars,
        min_price_delta_pct=min_price_delta_pct,
        min_rsi_delta=min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    return out
