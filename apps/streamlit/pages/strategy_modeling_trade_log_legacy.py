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


def _render_trade_drilldown_chart(
    bars: pd.DataFrame,
    markers: pd.DataFrame,
    *,
    timeframe: str,
    x_axis_domain: tuple[datetime, datetime] | None = None,
    y_axis_domain: tuple[float, float] | None = None,
) -> None:
    if bars.empty:
        st.warning("No bars available for drilldown chart rendering.")
        return

    chart_bars = bars.copy()
    chart_bars["timestamp"] = pd.to_datetime(chart_bars.get("timestamp"), errors="coerce", utc=True)
    chart_bars = chart_bars.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if chart_bars.empty:
        st.warning("No bars available for drilldown chart rendering.")
        return

    marker_view = markers.copy()
    marker_view["timestamp"] = pd.to_datetime(marker_view.get("timestamp"), errors="coerce", utc=True)
    marker_view = marker_view.dropna(subset=["timestamp", "price"])

    if alt is None:
        st.warning("Altair is unavailable in this environment; showing close-line fallback.")
        st.line_chart(chart_bars.set_index("timestamp")["close"])
        if not marker_view.empty:
            marker_fallback = marker_view.copy()
            marker_fallback["timestamp"] = marker_fallback["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            st.caption("Markers")
            st.dataframe(marker_fallback[["note", "timestamp", "price"]], hide_index=True, use_container_width=True)
        else:
            st.info("No marker data available for the selected trade.")
        return

    try:
        render_bars = chart_bars.copy()
        render_bars["timestamp"] = render_bars["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

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

        x_encoding_kwargs: dict[str, Any] = {"title": f"Timestamp ({timeframe})"}
        if x_scale is not None:
            x_encoding_kwargs["scale"] = x_scale

        price_encoding_kwargs: dict[str, Any] = {"title": "Price"}
        if y_scale is not None:
            price_encoding_kwargs["scale"] = y_scale

        marker_price_kwargs: dict[str, Any] = {"title": "Price"}
        if y_scale is not None:
            marker_price_kwargs["scale"] = y_scale

        marker_label_y = alt.Y("price:Q")
        if y_scale is not None:
            marker_label_y = alt.Y("price:Q", scale=y_scale)

        base = alt.Chart(render_bars).encode(
            x=alt.X("timestamp:T", **x_encoding_kwargs),
        )
        wicks = base.mark_rule(color="#374151").encode(
            y=alt.Y("low:Q", **price_encoding_kwargs),
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
            y=alt.Y("open:Q", **price_encoding_kwargs),
            y2="close:Q",
            color=alt.condition(
                "datum.close >= datum.open",
                alt.value("#16a34a"),
                alt.value("#dc2626"),
            ),
        )
        chart = wicks + candles

        if not marker_view.empty:
            render_markers = marker_view.copy()
            render_markers["timestamp"] = render_markers["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            marker_points = alt.Chart(render_markers).mark_point(filled=True, size=90).encode(
                x="timestamp:T",
                y=alt.Y("price:Q", **marker_price_kwargs),
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
            ).encode(
                x="timestamp:T",
                y=marker_label_y,
                text="note:N",
            )
            chart = chart + marker_points + marker_labels

        st.altair_chart(chart.properties(height=420).interactive(), use_container_width=True)
    except Exception:  # noqa: BLE001
        st.warning("Altair chart rendering failed; showing close-line fallback.")
        st.line_chart(chart_bars.set_index("timestamp")["close"])
        if not marker_view.empty:
            marker_fallback = marker_view.copy()
            marker_fallback["timestamp"] = marker_fallback["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            st.caption("Markers")
            st.dataframe(marker_fallback[["note", "timestamp", "price"]], hide_index=True, use_container_width=True)
        else:
            st.info("No marker data available for the selected trade.")


def _trade_drilldown_module():
    return import_module("apps.streamlit.components.strategy_modeling_trade_drilldown")


def _trade_review_module():
    return import_module("apps.streamlit.components.strategy_modeling_trade_review")


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

    trades = trade_df.copy()
    if "entry_ts" in trades.columns:
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], errors="coerce", utc=True)
        trades = trades.sort_values(by="entry_ts", ascending=False, kind="stable")

    realized = pd.to_numeric(trades.get("realized_r"), errors="coerce")
    below_one_r = trades.loc[realized < -1.0].copy()
    if not below_one_r.empty:
        st.warning(
            f"{len(below_one_r)} trade(s) realized below -1.0R under current gap-policy assumptions."
        )

    display_cols = [
        col
        for col in [
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
        ]
        if col in trades.columns
    ]
    trade_log_df = trades[display_cols].reset_index(drop=True)

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

    selected_trade_id = (
        drilldown.selected_trade_id_from_event(best_selection_event, best_trades_df)
        or drilldown.selected_trade_id_from_event(worst_selection_event, worst_trades_df)
        or drilldown.selected_trade_id_from_event(log_selection_event, trade_log_df)
    )

    st.subheader("Trade Drilldown")
    if not selected_trade_id:
        st.warning(
            "Select a row from Top 20 Best Trades, Top 20 Worst Trades, or the Full Trade Log to load drilldown."
        )
        return

    if "trade_id" not in trades.columns:
        st.warning("Trade log rows are missing `trade_id`; drilldown is unavailable.")
        return

    matched = trades.loc[trades["trade_id"].astype(str) == str(selected_trade_id)]
    if matched.empty:
        st.warning(f"Selected trade `{selected_trade_id}` is not present in the current trade log.")
        return

    selected_trade = _to_dict(matched.iloc[0])
    symbol = str(selected_trade.get("symbol") or "").strip().upper()
    entry_ts = _coerce_utc_timestamp(selected_trade.get("entry_ts"))
    exit_ts = _coerce_utc_timestamp(selected_trade.get("exit_ts"))
    anchor_ts = entry_ts if entry_ts is not None else exit_ts

    if not symbol:
        st.warning(f"Selected trade `{selected_trade_id}` is missing symbol; cannot load intraday bars.")
        return
    if anchor_ts is None:
        st.warning(
            f"Selected trade `{selected_trade_id}` is missing entry/exit timestamps; cannot load drilldown."
        )
        return

    request_intraday_tf = str(getattr(request_state, "intraday_timeframe", intraday_timeframe) or intraday_timeframe)
    intraday_root = _resolve_intraday_root(request_state)
    base_candidates = _dedupe_timeframes("1Min", request_intraday_tf, "5Min")
    if not base_candidates:
        base_candidates = ["1Min", "5Min"]

    probe_base_timeframe = base_candidates[0]
    for candidate in base_candidates:
        probe_minutes = _timeframe_minutes(candidate) or 1
        probe_start = anchor_ts - pd.Timedelta(minutes=probe_minutes * 20)
        probe_end = anchor_ts + pd.Timedelta(minutes=probe_minutes * 20)
        probe_bars = drilldown.load_intraday_window(
            intraday_root,
            symbol,
            candidate,
            probe_start,
            probe_end,
        )
        if not probe_bars.empty:
            probe_base_timeframe = candidate
            break

    chart_options = drilldown.supported_chart_timeframes(probe_base_timeframe)
    if not chart_options:
        st.warning("No supported chart timeframes for drilldown.")
        return

    default_chart_timeframe = chart_options[0]
    if request_intraday_tf in chart_options:
        default_chart_timeframe = request_intraday_tf
    if st.session_state.get(trade_drilldown_timeframe_key) not in chart_options:
        st.session_state[trade_drilldown_timeframe_key] = default_chart_timeframe
    chart_timeframe = st.selectbox(
        "Chart timeframe",
        options=chart_options,
        key=trade_drilldown_timeframe_key,
    )
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

    chart_minutes = _timeframe_minutes(chart_timeframe) or (_timeframe_minutes(probe_base_timeframe) or 1)
    entry_anchor = entry_ts if entry_ts is not None else anchor_ts
    exit_anchor = exit_ts if exit_ts is not None else anchor_ts
    if exit_anchor < entry_anchor:
        exit_anchor = entry_anchor
    window_start = entry_anchor - pd.Timedelta(minutes=chart_minutes * pre_context_bars)
    window_end = exit_anchor + pd.Timedelta(minutes=chart_minutes * post_context_bars)

    load_candidates = [probe_base_timeframe, *[value for value in base_candidates if value != probe_base_timeframe]]
    base_timeframe_used: str | None = None
    base_bars = pd.DataFrame()
    for candidate in load_candidates:
        loaded = drilldown.load_intraday_window(
            intraday_root,
            symbol,
            candidate,
            window_start,
            window_end,
        )
        if loaded.empty:
            continue
        base_timeframe_used = candidate
        base_bars = loaded
        break

    if base_bars.empty or base_timeframe_used is None:
        st.warning(
            "No intraday bars found for selected trade/context window. Try a wider context or different symbol."
        )
        return

    if base_timeframe_used != "1Min":
        st.warning(
            f"Using `{base_timeframe_used}` base bars because `1Min` data was unavailable."
        )
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
        return

    trade_markers = _trade_markers(selected_trade)
    x_axis_domain: tuple[datetime, datetime] | None = None
    y_axis_domain: tuple[float, float] | None = None

    if "timestamp" in chart_result.bars.columns:
        axis_timestamps = pd.to_datetime(
            chart_result.bars["timestamp"],
            errors="coerce",
            utc=True,
        ).dropna()
    else:
        axis_timestamps = pd.Series([], dtype="datetime64[ns, UTC]")
    if not axis_timestamps.empty:
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
        x_axis_domain = tuple(
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

    low_series = (
        pd.to_numeric(chart_result.bars["low"], errors="coerce")
        if "low" in chart_result.bars.columns
        else pd.Series([], dtype=float)
    )
    high_series = (
        pd.to_numeric(chart_result.bars["high"], errors="coerce")
        if "high" in chart_result.bars.columns
        else pd.Series([], dtype=float)
    )
    marker_prices = (
        pd.to_numeric(trade_markers["price"], errors="coerce")
        if "price" in trade_markers.columns
        else pd.Series([], dtype=float)
    )
    y_floor_candidates = pd.concat([low_series, marker_prices], ignore_index=True).dropna()
    y_ceiling_candidates = pd.concat([high_series, marker_prices], ignore_index=True).dropna()
    if not y_floor_candidates.empty and not y_ceiling_candidates.empty:
        y_axis_min = float(y_floor_candidates.min())
        y_axis_max = float(y_ceiling_candidates.max())
        if math.isfinite(y_axis_min) and math.isfinite(y_axis_max):
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
            y_axis_domain = tuple(
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

    st.caption(
        "Selected trade: "
        f"`{selected_trade_id}` ({symbol}) using base `{base_timeframe_used}` and chart "
        f"`{chart_result.timeframe}` timeframe."
    )
    _render_trade_drilldown_chart(
        chart_result.bars,
        trade_markers,
        timeframe=chart_result.timeframe,
        x_axis_domain=x_axis_domain,
        y_axis_domain=y_axis_domain,
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
