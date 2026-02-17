from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path
from apps.streamlit.components.queries import run_query
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag

_CANDLES_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_HORIZONS = (1, 5, 10)


def _compute_supports_ignore_swept_swings() -> bool:
    return "ignore_swept_swings" in inspect.signature(compute_sfp_signals).parameters


def _compute_sfp_signals_compat(
    ohlc: pd.DataFrame,
    *,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    ignore_swept_swings: bool,
    timeframe: str,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "swing_left_bars": int(swing_left_bars),
        "swing_right_bars": int(swing_right_bars),
        "min_swing_distance_bars": int(min_swing_distance_bars),
        "timeframe": timeframe,
    }
    if _compute_supports_ignore_swept_swings():
        kwargs["ignore_swept_swings"] = bool(ignore_swept_swings)
    return compute_sfp_signals(ohlc, **kwargs)


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def list_sfp_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    df, note = _run_query_safe(
        """
        SELECT DISTINCT UPPER(symbol) AS symbol
        FROM candles_daily
        WHERE symbol IS NOT NULL
          AND TRIM(symbol) <> ''
          AND interval = '1d'
        ORDER BY symbol ASC
        """,
        database_path=database_path,
    )
    if note is not None:
        return [], note
    if df.empty:
        return [], None
    symbols = [normalize_symbol(value) for value in df["symbol"].tolist()]
    return sorted(symbol for symbol in symbols if symbol), None


def load_candles_history(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    limit: int = 3000,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH ranked AS (
          SELECT
            ts::TIMESTAMP AS ts,
            open,
            high,
            low,
            close,
            volume,
            ROW_NUMBER() OVER (
              PARTITION BY ts
              ORDER BY CAST(auto_adjust AS INTEGER) DESC, CAST(back_adjust AS INTEGER) ASC
            ) AS row_num
          FROM candles_daily
          WHERE UPPER(symbol) = ?
            AND interval = '1d'
        )
        SELECT ts, open, high, low, close, volume
        FROM ranked
        WHERE row_num = 1
        ORDER BY ts DESC
        LIMIT ?
        """,
        params=[sym, max(1, int(limit))],
        database_path=database_path,
    )
    if note is not None:
        return pd.DataFrame(columns=_CANDLES_COLUMNS), note
    if df.empty:
        return pd.DataFrame(columns=_CANDLES_COLUMNS), None

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    out = out.sort_values(by="ts", kind="stable").reset_index(drop=True)
    return out.reindex(columns=_CANDLES_COLUMNS), None


@st.cache_data(ttl=60, show_spinner=False)
def _build_sfp_payload_cached(
    *,
    symbol: str,
    lookback_days: int,
    tail_low_pct: float,
    tail_high_pct: float,
    rsi_overbought: float,
    rsi_oversold: float,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    ignore_swept_swings: bool,
    compute_supports_ignore_swept_swings: bool,
    database_path: str,
) -> dict[str, Any]:
    notes = _sfp_runtime_notes(
        ignore_swept_swings=bool(ignore_swept_swings),
        compute_supports_ignore_swept_swings=bool(compute_supports_ignore_swept_swings),
    )
    normalized_symbol = normalize_symbol(symbol)
    candles, note = load_candles_history(
        symbol=normalized_symbol,
        database_path=database_path,
        limit=max(600, int(lookback_days) + 260),
    )
    if note is not None:
        return _empty_sfp_payload(symbol=normalized_symbol, notes=notes, extra_note=note)
    if candles.empty:
        return _empty_sfp_payload(
            symbol=normalized_symbol,
            notes=notes,
            extra_note=f"No daily candles found for {normalized_symbol}.",
        )

    daily_ohlc = _build_daily_ohlc(candles)
    if daily_ohlc.empty:
        return _empty_sfp_payload(
            symbol=normalized_symbol,
            notes=notes,
            extra_note=f"No usable daily OHLC candles found for {normalized_symbol}.",
        )

    latest_ts, daily_events_filtered, weekly_events_filtered = _build_filtered_sfp_events(
        daily_ohlc=daily_ohlc,
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=bool(ignore_swept_swings),
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        lookback_days=int(lookback_days),
    )
    summary_rows = _build_summary_rows(
        daily_events=daily_events_filtered,
        weekly_events=weekly_events_filtered,
    )
    return {
        "symbol": normalized_symbol,
        "asof": latest_ts.date().isoformat(),
        "daily_events": daily_events_filtered,
        "weekly_events": weekly_events_filtered,
        "summary_rows": summary_rows,
        "counts": _build_sfp_counts(daily_events_filtered, weekly_events_filtered),
        "notes": notes,
    }


def _sfp_runtime_notes(*, ignore_swept_swings: bool, compute_supports_ignore_swept_swings: bool) -> list[str]:
    notes: list[str] = []
    if ignore_swept_swings and not compute_supports_ignore_swept_swings:
        notes.append(
            "Runtime SFP engine does not expose native swept-swing ignore mode; "
            "applied app-layer fallback filter."
        )
    return notes


def _empty_sfp_payload(*, symbol: str, notes: list[str], extra_note: str | None = None) -> dict[str, Any]:
    payload_notes = [*notes]
    if extra_note is not None:
        payload_notes.append(extra_note)
    return {
        "symbol": symbol,
        "asof": None,
        "daily_events": [],
        "weekly_events": [],
        "summary_rows": [],
        "counts": {},
        "notes": payload_notes,
    }


def _build_daily_ohlc(candles: pd.DataFrame) -> pd.DataFrame:
    daily_ohlc = candles.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    ).set_index("ts")[["Open", "High", "Low", "Close"]]
    return daily_ohlc.sort_index()


def _prepare_daily_context(
    *,
    daily_ohlc: pd.DataFrame,
    tail_low_pct: float,
    tail_high_pct: float,
) -> dict[str, Any]:
    daily_close = daily_ohlc["Close"].astype("float64")
    daily_ext, daily_ext_pct = _compute_extension_percentile_series(daily_ohlc)
    week_start_idx = pd.DatetimeIndex([_week_start_monday(ts) for ts in daily_ohlc.index])
    daily_extreme_mask = ((daily_ext_pct <= tail_low_pct) | (daily_ext_pct >= tail_high_pct)).fillna(False)
    daily_week_has_extreme = (
        pd.Series(daily_extreme_mask.to_numpy(dtype=bool), index=week_start_idx).groupby(level=0).max().astype(bool).to_dict()
    )
    return {
        "daily_close": daily_close,
        "daily_open": daily_ohlc["Open"].astype("float64"),
        "daily_high": daily_ohlc["High"].astype("float64"),
        "daily_low": daily_ohlc["Low"].astype("float64"),
        "daily_rsi": _rsi_series(daily_close),
        "daily_ext": daily_ext,
        "daily_ext_pct": daily_ext_pct,
        "daily_week_has_extreme": daily_week_has_extreme,
    }


def _build_filtered_sfp_events(
    *,
    daily_ohlc: pd.DataFrame,
    tail_low_pct: float,
    tail_high_pct: float,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    ignore_swept_swings: bool,
    rsi_overbought: float,
    rsi_oversold: float,
    lookback_days: int,
) -> tuple[pd.Timestamp, list[dict[str, Any]], list[dict[str, Any]]]:
    daily_context = _prepare_daily_context(
        daily_ohlc=daily_ohlc,
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
    )
    daily_events = _build_daily_events(
        daily_ohlc=daily_ohlc,
        daily_context=daily_context,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=ignore_swept_swings,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    weekly_events = _build_weekly_events(
        daily_ohlc=daily_ohlc,
        daily_context=daily_context,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=ignore_swept_swings,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    latest_ts = daily_ohlc.index.max()
    daily_filtered, weekly_filtered = _filter_events_by_lookback(
        daily_events=daily_events,
        weekly_events=weekly_events,
        latest_ts=latest_ts,
        lookback_days=lookback_days,
    )
    return latest_ts, daily_filtered, weekly_filtered


def _build_daily_events(
    *,
    daily_ohlc: pd.DataFrame,
    daily_context: dict[str, Any],
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    ignore_swept_swings: bool,
    rsi_overbought: float,
    rsi_oversold: float,
) -> list[dict[str, Any]]:
    daily_signals = _compute_sfp_signals_compat(
        daily_ohlc,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=ignore_swept_swings,
        timeframe="native",
    )
    return _events_from_signals(
        signals=daily_signals,
        source_daily_close=daily_context["daily_close"],
        source_daily_open=daily_context["daily_open"],
        source_daily_high=daily_context["daily_high"],
        source_daily_low=daily_context["daily_low"],
        rsi_series=daily_context["daily_rsi"],
        extension_series=daily_context["daily_ext"],
        extension_percentile_series=daily_context["daily_ext_pct"],
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        timeframe="daily",
        week_has_daily_extension_map=daily_context["daily_week_has_extreme"],
        weekly_index_labels=False,
        ignore_swept_swings=ignore_swept_swings,
    )


def _build_weekly_events(
    *,
    daily_ohlc: pd.DataFrame,
    daily_context: dict[str, Any],
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    ignore_swept_swings: bool,
    rsi_overbought: float,
    rsi_oversold: float,
) -> list[dict[str, Any]]:
    weekly_ohlc = (
        daily_ohlc.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    weekly_close = weekly_ohlc["Close"].astype("float64")
    weekly_rsi = _rsi_series(weekly_close)
    weekly_ext, weekly_ext_pct = _compute_extension_percentile_series(weekly_ohlc)
    weekly_signals = _compute_sfp_signals_compat(
        weekly_ohlc,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=ignore_swept_swings,
        timeframe="native",
    )
    return _events_from_signals(
        signals=weekly_signals,
        source_daily_close=daily_context["daily_close"],
        source_daily_open=daily_context["daily_open"],
        source_daily_high=daily_context["daily_high"],
        source_daily_low=daily_context["daily_low"],
        rsi_series=weekly_rsi,
        extension_series=weekly_ext,
        extension_percentile_series=weekly_ext_pct,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        timeframe="weekly",
        week_has_daily_extension_map=daily_context["daily_week_has_extreme"],
        weekly_index_labels=True,
        ignore_swept_swings=ignore_swept_swings,
    )


def _filter_events_by_lookback(
    *,
    daily_events: list[dict[str, Any]],
    weekly_events: list[dict[str, Any]],
    latest_ts: pd.Timestamp,
    lookback_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff_ts = latest_ts - pd.Timedelta(days=lookback_days)
    daily_filtered = [row for row in daily_events if pd.Timestamp(row["event_ts"]) >= cutoff_ts]
    weekly_filtered = [row for row in weekly_events if pd.Timestamp(row["event_ts"]) >= cutoff_ts]
    return daily_filtered, weekly_filtered


def _build_sfp_counts(
    daily_events: list[dict[str, Any]],
    weekly_events: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "daily_events": len(daily_events),
        "weekly_events": len(weekly_events),
        "daily_bullish": sum(1 for row in daily_events if row.get("direction") == "bullish"),
        "daily_bearish": sum(1 for row in daily_events if row.get("direction") == "bearish"),
    }


def load_sfp_payload(
    *,
    symbol: str,
    lookback_days: int,
    tail_low_pct: float,
    tail_high_pct: float,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    swing_left_bars: int = 2,
    swing_right_bars: int = 2,
    min_swing_distance_bars: int = 1,
    ignore_swept_swings: bool = False,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    resolved = resolve_duckdb_path(database_path)
    try:
        compute_supports_ignore_swept_swings = _compute_supports_ignore_swept_swings()
        payload = _build_sfp_payload_cached(
            symbol=normalize_symbol(symbol),
            lookback_days=max(1, int(lookback_days)),
            tail_low_pct=float(tail_low_pct),
            tail_high_pct=float(tail_high_pct),
            rsi_overbought=float(rsi_overbought),
            rsi_oversold=float(rsi_oversold),
            swing_left_bars=max(1, int(swing_left_bars)),
            swing_right_bars=max(1, int(swing_right_bars)),
            min_swing_distance_bars=max(1, int(min_swing_distance_bars)),
            ignore_swept_swings=bool(ignore_swept_swings),
            compute_supports_ignore_swept_swings=bool(compute_supports_ignore_swept_swings),
            database_path=str(resolved),
        )
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _run_query_safe(
    sql: str,
    *,
    params: list[Any] | None = None,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    path = resolve_duckdb_path(database_path)
    if not path.exists():
        return pd.DataFrame(), f"DuckDB database not found: {path}"
    try:
        frame = run_query(sql=sql, params=params or [], database_path=str(path))
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), _friendly_error(str(exc))
    return frame, None


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if "candles_daily" in lowered and "does not exist" in lowered:
        return "candles_daily table not found. Run `options-helper ingest candles` first."
    return message


def _rsi_series(close: pd.Series, *, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_extension_percentile_series(ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = ohlc["Close"].astype("float64")
    high = ohlc["High"].astype("float64")
    low = ohlc["Low"].astype("float64")

    sma_window = 20
    atr_window = 14

    sma = close.rolling(window=sma_window, min_periods=sma_window).mean()
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    extension = (close - sma) / atr
    extension = extension.replace([float("inf"), float("-inf")], float("nan"))

    valid_ext = extension.dropna()
    percentile = pd.Series(index=ohlc.index, dtype="float64")
    if len(valid_ext) >= 2:
        def _pct(arr: pd.Series) -> float:
            s = pd.Series(arr).dropna().astype("float64")
            if len(s) < 2:
                return float("nan")
            return float(s.rank(pct=True, method="average").iloc[-1] * 100.0)

        expanding_pct = valid_ext.expanding(min_periods=2).apply(_pct, raw=False)
        percentile.loc[expanding_pct.index] = expanding_pct

    return extension, percentile


def _week_start_monday(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()


def _forward_returns_from_anchor(
    *,
    daily_open: pd.Series,
    daily_high: pd.Series,
    daily_low: pd.Series,
    direction: str,
    anchor_pos: int | None,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for horizon in _HORIZONS:
        if anchor_pos is None:
            out[f"forward_{horizon}d_pct"] = None
            continue
        start = int(anchor_pos) + 1
        end = int(anchor_pos) + int(horizon) + 1
        if start >= len(daily_high):
            out[f"forward_{horizon}d_pct"] = None
            continue
        end = min(end, len(daily_high))
        if end <= start:
            out[f"forward_{horizon}d_pct"] = None
            continue

        entry_open = pd.to_numeric(daily_open.iloc[start : start + 1], errors="coerce").dropna()
        if entry_open.empty:
            out[f"forward_{horizon}d_pct"] = None
            continue
        entry_price = float(entry_open.iloc[0])
        if entry_price == 0.0:
            out[f"forward_{horizon}d_pct"] = None
            continue

        if str(direction).lower() == "bearish":
            window_low = pd.to_numeric(daily_low.iloc[start:end], errors="coerce").dropna()
            if window_low.empty:
                out[f"forward_{horizon}d_pct"] = None
                continue
            move_pct = (float(window_low.min()) / entry_price - 1.0) * 100.0
            out[f"forward_{horizon}d_pct"] = round(min(0.0, move_pct), 2)
        else:
            window_high = pd.to_numeric(daily_high.iloc[start:end], errors="coerce").dropna()
            if window_high.empty:
                out[f"forward_{horizon}d_pct"] = None
                continue
            move_pct = (float(window_high.max()) / entry_price - 1.0) * 100.0
            out[f"forward_{horizon}d_pct"] = round(max(0.0, move_pct), 2)
    return out


def _events_from_signals(
    *,
    signals: pd.DataFrame,
    source_daily_close: pd.Series,
    source_daily_open: pd.Series,
    source_daily_high: pd.Series,
    source_daily_low: pd.Series,
    rsi_series: pd.Series,
    extension_series: pd.Series,
    extension_percentile_series: pd.Series,
    rsi_overbought: float,
    rsi_oversold: float,
    timeframe: str,
    week_has_daily_extension_map: dict[pd.Timestamp, bool],
    weekly_index_labels: bool,
    ignore_swept_swings: bool = False,
) -> list[dict[str, Any]]:
    if signals.empty:
        return []

    events = extract_sfp_events(signals)
    rows: list[dict[str, Any]] = []
    daily_index = source_daily_close.index

    for ev in events:
        row = asdict(ev)
        raw_ts = pd.Timestamp(row["timestamp"])
        raw_swing_ts = pd.Timestamp(row["swept_swing_timestamp"])
        week_start = _week_start_monday(raw_ts)
        display_ts, display_swing_ts, anchor_ts = _resolve_display_and_anchor_timestamps(
            raw_ts=raw_ts,
            raw_swing_ts=raw_swing_ts,
            daily_index=daily_index,
            weekly_index_labels=weekly_index_labels,
        )
        anchor_pos = None if anchor_ts is None else int(daily_index.get_loc(anchor_ts))
        event_close = float(row["candle_close"])
        rsi_value, ext_value, ext_pct = _lookup_event_indicators(
            raw_ts,
            rsi_series=rsi_series,
            extension_series=extension_series,
            extension_percentile_series=extension_percentile_series,
        )
        formatted = _build_sfp_event_row(
            row=row,
            timeframe=timeframe,
            display_ts=display_ts,
            display_swing_ts=display_swing_ts,
            week_start=week_start,
            week_has_daily_extension_map=week_has_daily_extension_map,
            event_close=event_close,
            rsi_value=rsi_value,
            ext_value=ext_value,
            ext_pct=ext_pct,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
        )
        formatted.update(
            _forward_returns_from_anchor(
                daily_open=source_daily_open,
                daily_high=source_daily_high,
                daily_low=source_daily_low,
                direction=str(row["direction"]),
                anchor_pos=anchor_pos,
            )
        )
        rows.append(formatted)

    if ignore_swept_swings and rows:
        rows = _consume_reused_swept_swings(rows)
    rows.sort(key=lambda item: str(item.get("event_ts") or ""), reverse=True)
    return rows


def _build_sfp_event_row(
    *,
    row: dict[str, Any],
    timeframe: str,
    display_ts: pd.Timestamp,
    display_swing_ts: pd.Timestamp,
    week_start: pd.Timestamp,
    week_has_daily_extension_map: dict[pd.Timestamp, bool],
    event_close: float,
    rsi_value: float | None,
    ext_value: float | None,
    ext_pct: float | None,
    rsi_overbought: float,
    rsi_oversold: float,
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "event_ts": display_ts.date().isoformat(),
        "swept_swing_ts": display_swing_ts.date().isoformat(),
        "direction": str(row["direction"]),
        "bars_since_swing": int(row["bars_since_swing"]),
        "candle_open": round(float(row["candle_open"]), 2),
        "candle_high": round(float(row["candle_high"]), 2),
        "candle_low": round(float(row["candle_low"]), 2),
        "candle_close": round(event_close, 2),
        "sweep_level": round(float(row["sweep_level"]), 2),
        "extension_atr": None if ext_value is None else round(ext_value, 2),
        "extension_percentile": None if ext_pct is None else round(ext_pct, 2),
        "rsi": None if rsi_value is None else round(rsi_value, 2),
        "rsi_regime": (
            None
            if rsi_value is None
            else rsi_regime_tag(
                rsi_value=float(rsi_value),
                rsi_overbought=float(rsi_overbought),
                rsi_oversold=float(rsi_oversold),
            )
        ),
        "week_has_daily_extension_extreme": bool(week_has_daily_extension_map.get(week_start, False)),
    }


def _lookup_event_indicators(
    ts: pd.Timestamp,
    *,
    rsi_series: pd.Series,
    extension_series: pd.Series,
    extension_percentile_series: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    rsi_value = float(rsi_series.loc[ts]) if ts in rsi_series.index and pd.notna(rsi_series.loc[ts]) else None
    ext_value = float(extension_series.loc[ts]) if ts in extension_series.index and pd.notna(extension_series.loc[ts]) else None
    ext_pct = (
        float(extension_percentile_series.loc[ts])
        if ts in extension_percentile_series.index and pd.notna(extension_percentile_series.loc[ts])
        else None
    )
    return rsi_value, ext_value, ext_pct


def _resolve_display_and_anchor_timestamps(
    *,
    raw_ts: pd.Timestamp,
    raw_swing_ts: pd.Timestamp,
    daily_index: pd.DatetimeIndex,
    weekly_index_labels: bool,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
    if weekly_index_labels:
        display_ts = _week_start_monday(raw_ts)
        display_swing_ts = _week_start_monday(raw_swing_ts)
        anchor_candidates = daily_index[daily_index <= raw_ts]
        anchor_ts = anchor_candidates.max() if len(anchor_candidates) > 0 else None
        return display_ts, display_swing_ts, anchor_ts
    display_ts = raw_ts
    display_swing_ts = raw_swing_ts
    anchor_ts = raw_ts if raw_ts in daily_index else None
    return display_ts, display_swing_ts, anchor_ts


def _consume_reused_swept_swings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the first SFP event for each (direction, swept swing) pair.

    This mirrors "consume liquidity" behavior and acts as a safety net when an older
    compute_sfp_signals implementation is imported without native ignore support.
    """
    seen: set[tuple[str, str]] = set()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        direction = str(row.get("direction") or "").lower()
        swept_ts = str(row.get("swept_swing_ts") or "")
        key = (direction, swept_ts)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(row)
    return filtered


def _build_summary_rows(*, daily_events: list[dict[str, Any]], weekly_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily_df = pd.DataFrame(daily_events)
    weekly_df = pd.DataFrame(weekly_events)

    def _summary_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {
                "group": name,
                "count": 0,
                "median_1d_pct": None,
                "median_5d_pct": None,
                "median_10d_pct": None,
            }
        return {
            "group": name,
            "count": int(len(frame)),
            "median_1d_pct": _median_or_none(frame.get("forward_1d_pct")),
            "median_5d_pct": _median_or_none(frame.get("forward_5d_pct")),
            "median_10d_pct": _median_or_none(frame.get("forward_10d_pct")),
        }

    rows = []
    if daily_df.empty:
        rows.append(_summary_row("Daily Bullish SFP", daily_df))
        rows.append(_summary_row("Daily Bearish SFP", daily_df))
        rows.append(_summary_row("Daily SFP at RSI Extremes (Bullish)", daily_df))
        rows.append(_summary_row("Daily SFP at RSI Extremes (Bearish)", daily_df))
    else:
        bullish_daily = daily_df[daily_df["direction"] == "bullish"]
        bearish_daily = daily_df[daily_df["direction"] == "bearish"]
        rows.append(_summary_row("Daily Bullish SFP", bullish_daily))
        rows.append(_summary_row("Daily Bearish SFP", bearish_daily))
        rows.append(
            _summary_row(
                "Daily SFP at RSI Extremes (Bullish)",
                bullish_daily[bullish_daily["rsi_regime"].isin(["overbought", "oversold"])],
            )
        )
        rows.append(
            _summary_row(
                "Daily SFP at RSI Extremes (Bearish)",
                bearish_daily[bearish_daily["rsi_regime"].isin(["overbought", "oversold"])],
            )
        )

    if weekly_df.empty:
        rows.append(_summary_row("Weekly SFP + Daily Extension Extreme in Week (Bullish)", weekly_df))
        rows.append(_summary_row("Weekly SFP + Daily Extension Extreme in Week (Bearish)", weekly_df))
    else:
        weekly_extreme = weekly_df[weekly_df["week_has_daily_extension_extreme"] == True]  # noqa: E712
        rows.append(
            _summary_row(
                "Weekly SFP + Daily Extension Extreme in Week (Bullish)",
                weekly_extreme[weekly_extreme["direction"] == "bullish"],
            )
        )
        rows.append(
            _summary_row(
                "Weekly SFP + Daily Extension Extreme in Week (Bearish)",
                weekly_extreme[weekly_extreme["direction"] == "bearish"],
            )
        )
    return rows


def _median_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.median()), 2)
