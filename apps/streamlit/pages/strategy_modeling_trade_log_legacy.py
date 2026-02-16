from __future__ import annotations

import math
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # noqa: BLE001
    alt = None

from apps.streamlit.pages.strategy_modeling_page_helpers_legacy import (
    _coerce_finite_float,
    _to_dict,
)


def _coerce_utc_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
        return None
    return timestamp


def _timeframe_minutes(value: object) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    normalized = text.replace("minutes", "min").replace("minute", "min").replace("mins", "min")
    if normalized.endswith("m") and not normalized.endswith("min"):
        normalized = normalized[:-1] + "min"
    amount = normalized[:-3].strip() if normalized.endswith("min") else normalized
    try:
        minutes = int(amount)
    except ValueError:
        return None
    if minutes <= 0:
        return None
    return minutes


def _dedupe_timeframes(*raw_values: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        value = str(raw or "").strip()
        if not value:
            continue
        token = value.lower()
        if token in seen:
            continue
        seen.add(token)
        out.append(value)
    return out


def _accepted_trade_ids_for_review(run_result: object) -> Sequence[str] | None:
    if run_result is None or not hasattr(run_result, "accepted_trade_ids"):
        return None
    raw = getattr(run_result, "accepted_trade_ids")
    if raw is None:
        return ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(
            token
            for token in (str(item or "").strip() for item in raw)
            if token
        )
    token = str(raw or "").strip()
    return (token,) if token else ()


def _resolve_intraday_root(request: object) -> Path:
    raw = getattr(request, "intraday_dir", Path("data/intraday"))
    if isinstance(raw, Path):
        return raw
    token = str(raw or "").strip()
    if not token:
        return Path("data/intraday")
    return Path(token)


def _trade_markers(trade_row: Mapping[str, Any]) -> pd.DataFrame:
    entry_ts = _coerce_utc_timestamp(trade_row.get("entry_ts"))
    exit_ts = _coerce_utc_timestamp(trade_row.get("exit_ts"))
    entry_price = _coerce_finite_float(trade_row.get("entry_price"))
    stop_price = _coerce_finite_float(trade_row.get("stop_price"))
    target_price = _coerce_finite_float(trade_row.get("target_price"))
    exit_price = _coerce_finite_float(trade_row.get("exit_price"))

    rows: list[dict[str, Any]] = []
    if entry_ts is not None and entry_price is not None:
        rows.append({"label": "entry", "timestamp": entry_ts, "price": entry_price, "note": "Entry"})
    if entry_ts is not None and stop_price is not None:
        rows.append({"label": "stop", "timestamp": entry_ts, "price": stop_price, "note": "Stop"})
    if entry_ts is not None and target_price is not None:
        rows.append({"label": "target", "timestamp": entry_ts, "price": target_price, "note": "Target"})
    if exit_ts is not None and exit_price is not None:
        rows.append({"label": "exit", "timestamp": exit_ts, "price": exit_price, "note": "Exit"})
    return pd.DataFrame(rows, columns=["label", "timestamp", "price", "note"])


def _normalize_datetime_axis_range(
    raw_value: object,
    *,
    min_value: datetime,
    max_value: datetime,
) -> tuple[datetime, datetime]:
    default_range = (min_value, max_value)
    if (
        not isinstance(raw_value, Sequence)
        or isinstance(raw_value, (str, bytes))
        or len(raw_value) != 2
    ):
        return default_range

    start_raw = pd.to_datetime(raw_value[0], errors="coerce")
    end_raw = pd.to_datetime(raw_value[1], errors="coerce")
    if pd.isna(start_raw) or pd.isna(end_raw):
        return default_range

    start = pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw)
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)

    start_py = start.to_pydatetime()
    end_py = end.to_pydatetime()
    if end_py < start_py:
        return default_range

    start_clamped = max(min_value, min(start_py, max_value))
    end_clamped = min(max_value, max(end_py, min_value))
    if end_clamped < start_clamped:
        return default_range
    return (start_clamped, end_clamped)


def _normalize_numeric_axis_range(
    raw_value: object,
    *,
    min_value: float,
    max_value: float,
) -> tuple[float, float]:
    default_range = (min_value, max_value)
    if (
        not isinstance(raw_value, Sequence)
        or isinstance(raw_value, (str, bytes))
        or len(raw_value) != 2
    ):
        return default_range

    start_raw = _coerce_finite_float(raw_value[0])
    end_raw = _coerce_finite_float(raw_value[1])
    if start_raw is None or end_raw is None:
        return default_range
    if end_raw < start_raw:
        return default_range

    start_clamped = max(min_value, min(start_raw, max_value))
    end_clamped = min(max_value, max(end_raw, min_value))
    if end_clamped < start_clamped:
        return default_range
    return (start_clamped, end_clamped)


def _prepare_chart_bars(bars: pd.DataFrame) -> pd.DataFrame:
    chart_bars = bars.copy()
    chart_bars["timestamp"] = pd.to_datetime(chart_bars.get("timestamp"), errors="coerce", utc=True)
    return chart_bars.dropna(subset=["timestamp", "open", "high", "low", "close"])


def _prepare_marker_view(markers: pd.DataFrame) -> pd.DataFrame:
    marker_view = markers.copy()
    marker_view["timestamp"] = pd.to_datetime(marker_view.get("timestamp"), errors="coerce", utc=True)
    return marker_view.dropna(subset=["timestamp", "price"])


def _render_marker_fallback_table(marker_view: pd.DataFrame) -> None:
    if marker_view.empty:
        st.info("No marker data available for the selected trade.")
        return
    marker_fallback = marker_view.copy()
    marker_fallback["timestamp"] = marker_fallback["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption("Markers")
    st.dataframe(marker_fallback[["note", "timestamp", "price"]], hide_index=True, use_container_width=True)


def _render_close_line_fallback(*, chart_bars: pd.DataFrame, marker_view: pd.DataFrame, warning_text: str) -> None:
    st.warning(warning_text)
    st.line_chart(chart_bars.set_index("timestamp")["close"])
    _render_marker_fallback_table(marker_view)


def _build_axis_scales(
    *,
    x_axis_domain: tuple[datetime, datetime] | None,
    y_axis_domain: tuple[float, float] | None,
) -> tuple[object | None, object | None]:
    x_scale = None
    if x_axis_domain is not None:
        x_start, x_end = x_axis_domain
        if x_end > x_start:
            x_scale = alt.Scale(domain=[x_start, x_end])

    y_scale = None
    if y_axis_domain is not None:
        y_min, y_max = y_axis_domain
        if math.isfinite(y_min) and math.isfinite(y_max) and y_max > y_min:
            y_scale = alt.Scale(domain=[float(y_min), float(y_max)])
    return x_scale, y_scale


def _build_base_candlestick_chart(
    *,
    render_bars: pd.DataFrame,
    timeframe: str,
    x_scale: object | None,
    y_scale: object | None,
) -> object:
    x_encoding_kwargs: dict[str, Any] = {"title": f"Timestamp ({timeframe})"}
    if x_scale is not None:
        x_encoding_kwargs["scale"] = x_scale
    price_kwargs: dict[str, Any] = {"title": "Price"}
    if y_scale is not None:
        price_kwargs["scale"] = y_scale

    base = alt.Chart(render_bars).encode(x=alt.X("timestamp:T", **x_encoding_kwargs))
    wicks = base.mark_rule(color="#374151").encode(
        y=alt.Y("low:Q", **price_kwargs),
        y2="high:Q",
        tooltip=[
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("open:Q", format=".4f", title="Open"),
            alt.Tooltip("high:Q", format=".4f", title="High"),
            alt.Tooltip("low:Q", format=".4f", title="Low"),
            alt.Tooltip("close:Q", format=".4f", title="Close"),
            alt.Tooltip("volume:Q", format=".0f", title="Volume"),
        ],
    )
    candles = base.mark_bar().encode(
        y=alt.Y("open:Q", **price_kwargs),
        y2="close:Q",
        color=alt.condition("datum.close >= datum.open", alt.value("#16a34a"), alt.value("#dc2626")),
    )
    return wicks + candles


def _add_marker_layers(*, chart: object, marker_view: pd.DataFrame, y_scale: object | None) -> object:
    if marker_view.empty:
        return chart
    marker_kwargs: dict[str, Any] = {"title": "Price"}
    if y_scale is not None:
        marker_kwargs["scale"] = y_scale
    marker_label_y = alt.Y("price:Q", scale=y_scale) if y_scale is not None else alt.Y("price:Q")
    render_markers = marker_view.copy()
    render_markers["timestamp"] = render_markers["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    marker_points = alt.Chart(render_markers).mark_point(filled=True, size=90).encode(
        x="timestamp:T",
        y=alt.Y("price:Q", **marker_kwargs),
        color=alt.Color("note:N", title="Marker"),
        tooltip=[
            alt.Tooltip("note:N", title="Marker"),
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("price:Q", format=".4f", title="Price"),
        ],
    )
    marker_labels = alt.Chart(render_markers).mark_text(
        align="left",
        dx=7,
        dy=-7,
        fontSize=11,
        color="#111827",
    ).encode(x="timestamp:T", y=marker_label_y, text="note:N")
    return chart + marker_points + marker_labels


def _build_altair_trade_chart(
    *,
    chart_bars: pd.DataFrame,
    marker_view: pd.DataFrame,
    timeframe: str,
    x_axis_domain: tuple[datetime, datetime] | None,
    y_axis_domain: tuple[float, float] | None,
) -> object:
    render_bars = chart_bars.copy()
    render_bars["timestamp"] = render_bars["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    x_scale, y_scale = _build_axis_scales(x_axis_domain=x_axis_domain, y_axis_domain=y_axis_domain)
    chart = _build_base_candlestick_chart(
        render_bars=render_bars,
        timeframe=timeframe,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    return _add_marker_layers(chart=chart, marker_view=marker_view, y_scale=y_scale)


def _render_trade_drilldown_chart(
    bars: pd.DataFrame,
    markers: pd.DataFrame,
    *,
    timeframe: str,
    x_axis_domain: tuple[datetime, datetime] | None = None,
    y_axis_domain: tuple[float, float] | None = None,
) -> None:
    chart_bars = _prepare_chart_bars(bars)
    if chart_bars.empty:
        st.warning("No bars available for drilldown chart rendering.")
        return
    marker_view = _prepare_marker_view(markers)

    if alt is None:
        _render_close_line_fallback(
            chart_bars=chart_bars,
            marker_view=marker_view,
            warning_text="Altair is unavailable in this environment; showing close-line fallback.",
        )
        return

    try:
        chart = _build_altair_trade_chart(
            chart_bars=chart_bars,
            marker_view=marker_view,
            timeframe=timeframe,
            x_axis_domain=x_axis_domain,
            y_axis_domain=y_axis_domain,
        )
        st.altair_chart(chart.properties(height=420).interactive(), use_container_width=True)
    except Exception:  # noqa: BLE001
        _render_close_line_fallback(
            chart_bars=chart_bars,
            marker_view=marker_view,
            warning_text="Altair chart rendering failed; showing close-line fallback.",
        )


def _trade_drilldown_module():
    return import_module("apps.streamlit.components.strategy_modeling_trade_drilldown")


def _trade_review_module():
    return import_module("apps.streamlit.components.strategy_modeling_trade_review")


def _prepare_trade_tables(trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = trade_df.copy()
    if "entry_ts" in trades.columns:
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], errors="coerce", utc=True)
        trades = trades.sort_values(by="entry_ts", ascending=False, kind="stable")
    display_cols = [
        col
        for col in (
            "trade_id",
            "symbol",
            "direction",
            "entry_ts",
            "entry_price",
            "stop_price",
            "target_price",
            "exit_ts",
            "exit_price",
            "exit_reason",
            "status",
            "realized_r",
            "mae_r",
            "mfe_r",
            "gap_fill_applied",
            "reject_code",
        )
        if col in trades.columns
    ]
    return trades, trades[display_cols].reset_index(drop=True)


def _warn_gap_through_trades(trades: pd.DataFrame) -> None:
    realized = pd.to_numeric(trades.get("realized_r"), errors="coerce")
    below_one_r = trades.loc[realized < -1.0]
    if below_one_r.empty:
        return
    st.warning(f"{len(below_one_r)} trade(s) realized below -1.0R under current gap-policy assumptions.")


def _render_trade_review_tables(
    *,
    trades: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    result: object,
    trade_review: object,
    drilldown: object,
    trade_review_best_key: str,
    trade_review_worst_key: str,
    trade_review_log_key: str,
) -> str | None:
    best_trades_df, worst_trades_df, review_scope_label = trade_review.build_trade_review_tables(
        trades,
        accepted_trade_ids=_accepted_trade_ids_for_review(result),
        top_n=20,
    )
    st.markdown("**Top 20 Best Trades (Realized R)**")
    st.caption(f"Ranking scope: {review_scope_label}")
    best_selection_event = st.dataframe(
        best_trades_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=trade_review_best_key,
    )
    st.markdown("**Top 20 Worst Trades (Realized R)**")
    worst_selection_event = st.dataframe(
        worst_trades_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=trade_review_worst_key,
    )
    st.markdown("**Full Trade Log**")
    log_selection_event = st.dataframe(
        trade_log_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=trade_review_log_key,
    )
    return (
        drilldown.selected_trade_id_from_event(best_selection_event, best_trades_df)
        or drilldown.selected_trade_id_from_event(worst_selection_event, worst_trades_df)
        or drilldown.selected_trade_id_from_event(log_selection_event, trade_log_df)
    )


def _resolve_selected_trade(
    *,
    trades: pd.DataFrame,
    selected_trade_id: str,
) -> tuple[dict[str, Any], str, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp] | None:
    if "trade_id" not in trades.columns:
        st.warning("Trade log rows are missing `trade_id`; drilldown is unavailable.")
        return None
    matched = trades.loc[trades["trade_id"].astype(str) == str(selected_trade_id)]
    if matched.empty:
        st.warning(f"Selected trade `{selected_trade_id}` is not present in the current trade log.")
        return None
    selected_trade = _to_dict(matched.iloc[0])
    symbol = str(selected_trade.get("symbol") or "").strip().upper()
    entry_ts = _coerce_utc_timestamp(selected_trade.get("entry_ts"))
    exit_ts = _coerce_utc_timestamp(selected_trade.get("exit_ts"))
    anchor_ts = entry_ts if entry_ts is not None else exit_ts
    if not symbol:
        st.warning(f"Selected trade `{selected_trade_id}` is missing symbol; cannot load intraday bars.")
        return None
    if anchor_ts is None:
        st.warning(f"Selected trade `{selected_trade_id}` is missing entry/exit timestamps; cannot load drilldown.")
        return None
    return selected_trade, symbol, entry_ts, exit_ts, anchor_ts


def _probe_base_timeframe(
    *,
    drilldown: object,
    intraday_root: Path,
    symbol: str,
    anchor_ts: pd.Timestamp,
    base_candidates: Sequence[str],
) -> str:
    probe_base_timeframe = base_candidates[0]
    for candidate in base_candidates:
        probe_minutes = _timeframe_minutes(candidate) or 1
        probe_start = anchor_ts - pd.Timedelta(minutes=probe_minutes * 20)
        probe_end = anchor_ts + pd.Timedelta(minutes=probe_minutes * 20)
        probe_bars = drilldown.load_intraday_window(intraday_root, symbol, candidate, probe_start, probe_end)
        if not probe_bars.empty:
            probe_base_timeframe = candidate
            break
    return probe_base_timeframe


def _select_chart_controls(
    *,
    chart_options: Sequence[str],
    request_intraday_tf: str,
    trade_drilldown_timeframe_key: str,
    trade_drilldown_pre_bars_key: str,
    trade_drilldown_post_bars_key: str,
) -> tuple[str, int, int]:
    default_chart_timeframe = chart_options[0]
    if request_intraday_tf in chart_options:
        default_chart_timeframe = request_intraday_tf
    if st.session_state.get(trade_drilldown_timeframe_key) not in chart_options:
        st.session_state[trade_drilldown_timeframe_key] = default_chart_timeframe
    chart_timeframe = st.selectbox("Chart timeframe", options=chart_options, key=trade_drilldown_timeframe_key)
    pre_context_bars = int(
        st.number_input(
            "Pre-context bars",
            min_value=1,
            max_value=20_000,
            value=int(st.session_state.get(trade_drilldown_pre_bars_key, 120)),
            step=10,
            key=trade_drilldown_pre_bars_key,
        )
    )
    post_context_bars = int(
        st.number_input(
            "Post-context bars",
            min_value=1,
            max_value=20_000,
            value=int(st.session_state.get(trade_drilldown_post_bars_key, 120)),
            step=10,
            key=trade_drilldown_post_bars_key,
        )
    )
    return chart_timeframe, pre_context_bars, post_context_bars


def _resolve_chart_window(
    *,
    chart_timeframe: str,
    probe_base_timeframe: str,
    entry_ts: pd.Timestamp | None,
    exit_ts: pd.Timestamp | None,
    anchor_ts: pd.Timestamp,
    pre_context_bars: int,
    post_context_bars: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    chart_minutes = _timeframe_minutes(chart_timeframe) or (_timeframe_minutes(probe_base_timeframe) or 1)
    entry_anchor = entry_ts if entry_ts is not None else anchor_ts
    exit_anchor = exit_ts if exit_ts is not None else anchor_ts
    if exit_anchor < entry_anchor:
        exit_anchor = entry_anchor
    window_start = entry_anchor - pd.Timedelta(minutes=chart_minutes * pre_context_bars)
    window_end = exit_anchor + pd.Timedelta(minutes=chart_minutes * post_context_bars)
    return window_start, window_end


def _load_base_bars(
    *,
    drilldown: object,
    intraday_root: Path,
    symbol: str,
    probe_base_timeframe: str,
    base_candidates: Sequence[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[pd.DataFrame, str | None]:
    load_candidates = [probe_base_timeframe, *[value for value in base_candidates if value != probe_base_timeframe]]
    for candidate in load_candidates:
        loaded = drilldown.load_intraday_window(intraday_root, symbol, candidate, window_start, window_end)
        if not loaded.empty:
            return loaded, candidate
    return pd.DataFrame(), None


def _resolve_x_axis_domain(
    *,
    bars: pd.DataFrame,
    trade_drilldown_x_range_key: str,
) -> tuple[datetime, datetime] | None:
    if "timestamp" not in bars.columns:
        return None
    axis_timestamps = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True).dropna()
    if axis_timestamps.empty:
        return None
    x_axis_min = axis_timestamps.dt.tz_convert("UTC").dt.tz_localize(None).min().to_pydatetime()
    x_axis_max = axis_timestamps.dt.tz_convert("UTC").dt.tz_localize(None).max().to_pydatetime()
    if x_axis_max <= x_axis_min:
        x_axis_max = x_axis_min + timedelta(minutes=1)
    x_range_default = _normalize_datetime_axis_range(
        st.session_state.get(trade_drilldown_x_range_key),
        min_value=x_axis_min,
        max_value=x_axis_max,
    )
    if st.session_state.get(trade_drilldown_x_range_key) != x_range_default:
        st.session_state[trade_drilldown_x_range_key] = x_range_default
    return tuple(
        st.slider(
            "X-axis range (UTC)",
            min_value=x_axis_min,
            max_value=x_axis_max,
            value=x_range_default,
            format="YYYY-MM-DD HH:mm",
            key=trade_drilldown_x_range_key,
            help="Adjust the visible time window on the drilldown chart.",
        )
    )


def _resolve_y_axis_domain(
    *,
    bars: pd.DataFrame,
    trade_markers: pd.DataFrame,
    trade_drilldown_y_range_key: str,
) -> tuple[float, float] | None:
    low_series = pd.to_numeric(bars.get("low"), errors="coerce")
    high_series = pd.to_numeric(bars.get("high"), errors="coerce")
    marker_prices = pd.to_numeric(trade_markers.get("price"), errors="coerce")
    y_floor_candidates = pd.concat([low_series, marker_prices], ignore_index=True).dropna()
    y_ceiling_candidates = pd.concat([high_series, marker_prices], ignore_index=True).dropna()
    if y_floor_candidates.empty or y_ceiling_candidates.empty:
        return None
    y_axis_min = float(y_floor_candidates.min())
    y_axis_max = float(y_ceiling_candidates.max())
    if not (math.isfinite(y_axis_min) and math.isfinite(y_axis_max)):
        return None
    if y_axis_max <= y_axis_min:
        y_axis_padding = max(abs(y_axis_min) * 0.005, 0.01)
        y_axis_min -= y_axis_padding
        y_axis_max += y_axis_padding
    y_range_default = _normalize_numeric_axis_range(
        st.session_state.get(trade_drilldown_y_range_key),
        min_value=y_axis_min,
        max_value=y_axis_max,
    )
    if st.session_state.get(trade_drilldown_y_range_key) != y_range_default:
        st.session_state[trade_drilldown_y_range_key] = y_range_default
    y_step = max((y_axis_max - y_axis_min) / 500.0, 0.0001)
    return tuple(
        st.slider(
            "Y-axis price range",
            min_value=float(y_axis_min),
            max_value=float(y_axis_max),
            value=(float(y_range_default[0]), float(y_range_default[1])),
            step=float(y_step),
            key=trade_drilldown_y_range_key,
            help="Adjust the visible price range on the drilldown chart.",
        )
    )


def _load_chart_result_for_trade(
    *,
    drilldown: object,
    request_state: object,
    intraday_timeframe: str,
    symbol: str,
    entry_ts: pd.Timestamp | None,
    exit_ts: pd.Timestamp | None,
    anchor_ts: pd.Timestamp,
    trade_drilldown_timeframe_key: str,
    trade_drilldown_pre_bars_key: str,
    trade_drilldown_post_bars_key: str,
) -> tuple[object, str] | None:
    request_intraday_tf = str(getattr(request_state, "intraday_timeframe", intraday_timeframe) or intraday_timeframe)
    intraday_root = _resolve_intraday_root(request_state)
    base_candidates = _dedupe_timeframes("1Min", request_intraday_tf, "5Min") or ["1Min", "5Min"]
    probe_base_timeframe = _probe_base_timeframe(
        drilldown=drilldown,
        intraday_root=intraday_root,
        symbol=symbol,
        anchor_ts=anchor_ts,
        base_candidates=base_candidates,
    )
    chart_options = drilldown.supported_chart_timeframes(probe_base_timeframe)
    if not chart_options:
        st.warning("No supported chart timeframes for drilldown.")
        return None
    chart_timeframe, pre_context_bars, post_context_bars = _select_chart_controls(
        chart_options=chart_options,
        request_intraday_tf=request_intraday_tf,
        trade_drilldown_timeframe_key=trade_drilldown_timeframe_key,
        trade_drilldown_pre_bars_key=trade_drilldown_pre_bars_key,
        trade_drilldown_post_bars_key=trade_drilldown_post_bars_key,
    )
    window_start, window_end = _resolve_chart_window(
        chart_timeframe=chart_timeframe,
        probe_base_timeframe=probe_base_timeframe,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        anchor_ts=anchor_ts,
        pre_context_bars=pre_context_bars,
        post_context_bars=post_context_bars,
    )
    base_bars, base_timeframe_used = _load_base_bars(
        drilldown=drilldown,
        intraday_root=intraday_root,
        symbol=symbol,
        probe_base_timeframe=probe_base_timeframe,
        base_candidates=base_candidates,
        window_start=window_start,
        window_end=window_end,
    )
    if base_bars.empty or base_timeframe_used is None:
        st.warning("No intraday bars found for selected trade/context window. Try a wider context or different symbol.")
        return None
    if base_timeframe_used != "1Min":
        st.warning(f"Using `{base_timeframe_used}` base bars because `1Min` data was unavailable.")
    chart_result = drilldown.resample_for_chart(
        base_bars,
        base_timeframe=base_timeframe_used,
        chart_timeframe=chart_timeframe,
        max_bars=5000,
    )
    if chart_result.warning:
        st.warning(chart_result.warning)
    if chart_result.skipped or chart_result.bars.empty:
        st.warning("No drilldown chart bars available after resampling for the selected context.")
        return None
    return chart_result, base_timeframe_used


def _render_trade_drilldown(
    *,
    drilldown: object,
    request_state: object,
    intraday_timeframe: str,
    selected_trade_id: str,
    selected_trade: dict[str, Any],
    symbol: str,
    entry_ts: pd.Timestamp | None,
    exit_ts: pd.Timestamp | None,
    anchor_ts: pd.Timestamp,
    trade_drilldown_timeframe_key: str,
    trade_drilldown_pre_bars_key: str,
    trade_drilldown_post_bars_key: str,
    trade_drilldown_x_range_key: str,
    trade_drilldown_y_range_key: str,
) -> None:
    loaded_chart = _load_chart_result_for_trade(
        drilldown=drilldown,
        request_state=request_state,
        intraday_timeframe=intraday_timeframe,
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        anchor_ts=anchor_ts,
        trade_drilldown_timeframe_key=trade_drilldown_timeframe_key,
        trade_drilldown_pre_bars_key=trade_drilldown_pre_bars_key,
        trade_drilldown_post_bars_key=trade_drilldown_post_bars_key,
    )
    if loaded_chart is None:
        return
    chart_result, base_timeframe_used = loaded_chart
    trade_markers = _trade_markers(selected_trade)
    x_axis_domain = _resolve_x_axis_domain(bars=chart_result.bars, trade_drilldown_x_range_key=trade_drilldown_x_range_key)
    y_axis_domain = _resolve_y_axis_domain(
        bars=chart_result.bars,
        trade_markers=trade_markers,
        trade_drilldown_y_range_key=trade_drilldown_y_range_key,
    )
    st.caption(
        "Selected trade: "
        f"`{selected_trade_id}` ({symbol}) using base `{base_timeframe_used}` and chart `{chart_result.timeframe}` timeframe."
    )
    _render_trade_drilldown_chart(
        chart_result.bars,
        trade_markers,
        timeframe=chart_result.timeframe,
        x_axis_domain=x_axis_domain,
        y_axis_domain=y_axis_domain,
    )


def render_trade_log_section(
    *,
    trade_df: pd.DataFrame,
    result: object,
    request_state: object,
    intraday_timeframe: str,
    trade_review_best_key: str,
    trade_review_worst_key: str,
    trade_review_log_key: str,
    trade_drilldown_timeframe_key: str,
    trade_drilldown_pre_bars_key: str,
    trade_drilldown_post_bars_key: str,
    trade_drilldown_x_range_key: str,
    trade_drilldown_y_range_key: str,
) -> None:
    st.subheader("Trade Log")
    st.caption(
        "Realized R includes gap-through outcomes. Trade rows can show losses below -1.0R when stop fills occur at open."
    )
    if trade_df.empty:
        st.info("No trade simulations available yet.")
        return

    drilldown = _trade_drilldown_module()
    trade_review = _trade_review_module()

    trades, trade_log_df = _prepare_trade_tables(trade_df)
    _warn_gap_through_trades(trades)
    selected_trade_id = _render_trade_review_tables(
        trades=trades,
        trade_log_df=trade_log_df,
        result=result,
        trade_review=trade_review,
        drilldown=drilldown,
        trade_review_best_key=trade_review_best_key,
        trade_review_worst_key=trade_review_worst_key,
        trade_review_log_key=trade_review_log_key,
    )
    st.subheader("Trade Drilldown")
    if not selected_trade_id:
        st.warning(
            "Select a row from Top 20 Best Trades, Top 20 Worst Trades, or the Full Trade Log to load drilldown."
        )
        return
    resolved_trade = _resolve_selected_trade(
        trades=trades,
        selected_trade_id=str(selected_trade_id),
    )
    if resolved_trade is None:
        return
    selected_trade, symbol, entry_ts, exit_ts, anchor_ts = resolved_trade
    _render_trade_drilldown(
        drilldown=drilldown,
        request_state=request_state,
        intraday_timeframe=intraday_timeframe,
        selected_trade_id=str(selected_trade_id),
        selected_trade=selected_trade,
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        anchor_ts=anchor_ts,
        trade_drilldown_timeframe_key=trade_drilldown_timeframe_key,
        trade_drilldown_pre_bars_key=trade_drilldown_pre_bars_key,
        trade_drilldown_post_bars_key=trade_drilldown_post_bars_key,
        trade_drilldown_x_range_key=trade_drilldown_x_range_key,
        trade_drilldown_y_range_key=trade_drilldown_y_range_key,
    )


__all__ = [
    "_accepted_trade_ids_for_review",
    "_coerce_utc_timestamp",
    "_dedupe_timeframes",
    "_normalize_datetime_axis_range",
    "_normalize_numeric_axis_range",
    "_render_trade_drilldown_chart",
    "_resolve_intraday_root",
    "_timeframe_minutes",
    "_trade_markers",
    "render_trade_log_section",
]
