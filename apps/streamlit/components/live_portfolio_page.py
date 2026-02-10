from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from options_helper.analysis.live_portfolio_metrics import (
    compute_live_multileg_rows,
    compute_live_position_rows,
)
from options_helper.data.market_types import DataFetchError
from options_helper.data.streaming.live_manager import LiveSnapshot, LiveStreamConfig, LiveStreamManager
from options_helper.data.streaming.subscriptions import SubscriptionPlan, build_subscription_plan
from options_helper.models import Portfolio
from options_helper.storage import load_portfolio

DISCLAIMER_TEXT = "Informational and educational use only. Not financial advice."

_DEFAULT_PORTFOLIO_PATH = "portfolio.json"
_DEFAULT_REFRESH_SECONDS = 3
_DEFAULT_STALE_THRESHOLD_SECONDS = 15.0
_DEFAULT_MAX_OPTION_CONTRACTS = 250

_STREAM_KEYS = ("stocks", "options", "fills")

_STATE_MANAGER_KEY = "live_portfolio_manager"
_STATE_PORTFOLIO_KEY = "live_portfolio_portfolio"
_STATE_PORTFOLIO_ERROR_KEY = "live_portfolio_portfolio_error"
_STATE_PORTFOLIO_PATH_KEY = "live_portfolio_portfolio_path"
_STATE_PORTFOLIO_MTIME_KEY = "live_portfolio_portfolio_mtime"

_WIDGET_PORTFOLIO_PATH_KEY = "live_portfolio_widget_portfolio_path"
_WIDGET_STOCKS_KEY = "live_portfolio_widget_stream_stocks"
_WIDGET_OPTIONS_KEY = "live_portfolio_widget_stream_options"
_WIDGET_FILLS_KEY = "live_portfolio_widget_stream_fills"
_WIDGET_STOCK_FEED_KEY = "live_portfolio_widget_stock_feed"
_WIDGET_OPTIONS_FEED_KEY = "live_portfolio_widget_options_feed"
_WIDGET_REFRESH_SECONDS_KEY = "live_portfolio_widget_refresh_seconds"
_WIDGET_STALE_SECONDS_KEY = "live_portfolio_widget_stale_seconds"
_WIDGET_MAX_CONTRACTS_KEY = "live_portfolio_widget_max_contracts"

_STOCK_FEED_OPTIONS: list[tuple[str, str | None]] = [
    ("Auto (env/default)", None),
    ("IEX", "iex"),
    ("SIP", "sip"),
    ("Delayed SIP", "delayed_sip"),
]
_OPTIONS_FEED_OPTIONS: list[tuple[str, str | None]] = [
    ("Auto (env/default)", None),
    ("OPRA", "opra"),
    ("Indicative", "indicative"),
]


@dataclass(frozen=True)
class LivePageControls:
    portfolio_path_raw: str
    reload_clicked: bool
    stream_stocks: bool
    stream_options: bool
    stream_fills: bool
    stock_feed: str | None
    options_feed: str | None
    refresh_seconds: int
    stale_threshold_seconds: float
    max_option_contracts: int


def render_live_portfolio_page() -> None:
    st.title("Live Portfolio")
    st.caption(DISCLAIMER_TEXT)
    st.info(
        "Read-only live monitoring. Streaming starts only after pressing Start and "
        "runs in background workers without Streamlit calls."
    )

    manager = _ensure_manager()
    controls = _render_sidebar_controls()
    portfolio_path = _resolve_portfolio_path(controls.portfolio_path_raw)
    reload_notice = _sync_portfolio_state(
        portfolio_path=portfolio_path,
        reload_clicked=controls.reload_clicked,
    )
    if reload_notice:
        st.info(reload_notice)

    portfolio = _load_portfolio_from_state()
    portfolio_error = _load_portfolio_error_from_state()

    if portfolio_error:
        st.error(portfolio_error)
    elif portfolio is not None:
        st.caption(f"Portfolio source: `{portfolio_path}`")

    plan = _subscription_plan_from_controls(portfolio=portfolio, controls=controls)
    if plan is not None:
        st.caption(
            "Subscription plan: "
            f"{len(plan.stocks)} stocks, {len(plan.option_contracts)} option contracts."
        )
        for warning in plan.warnings:
            st.warning(warning)

    start_col, stop_col, restart_col = st.columns(3)
    start_clicked = start_col.button(
        "Start",
        type="primary",
        disabled=portfolio is None,
    )
    stop_clicked = stop_col.button("Stop")
    restart_clicked = restart_col.button(
        "Restart",
        disabled=portfolio is None,
    )

    action_note, action_error = _handle_manager_actions(
        manager=manager,
        controls=controls,
        portfolio=portfolio,
        portfolio_error=portfolio_error,
        plan=plan,
        start_clicked=start_clicked,
        stop_clicked=stop_clicked,
        restart_clicked=restart_clicked,
    )
    if action_note:
        st.success(action_note)
    if action_error:
        st.error(action_error)

    st.subheader("Live Snapshot")
    run_every = f"{max(1, int(controls.refresh_seconds))}s"

    @_fragment(run_every=run_every)
    def _live_fragment() -> None:
        snapshot = manager.snapshot()
        _render_health(snapshot)
        _render_fills(snapshot)
        _render_live_portfolio_tables(
            portfolio=portfolio,
            snapshot=snapshot,
            stale_threshold_seconds=controls.stale_threshold_seconds,
        )

    _live_fragment()


def _render_sidebar_controls() -> LivePageControls:
    with st.sidebar:
        st.markdown("### Live Controls")
        portfolio_path_raw = st.text_input(
            "Portfolio JSON path",
            value=_DEFAULT_PORTFOLIO_PATH,
            key=_WIDGET_PORTFOLIO_PATH_KEY,
        )
        reload_clicked = st.button("Reload Portfolio")
        stream_stocks = st.toggle(
            "Stream stocks",
            value=True,
            key=_WIDGET_STOCKS_KEY,
        )
        stream_options = st.toggle(
            "Stream options",
            value=True,
            key=_WIDGET_OPTIONS_KEY,
        )
        stream_fills = st.toggle(
            "Stream fills",
            value=True,
            key=_WIDGET_FILLS_KEY,
        )

        stock_feed = _feed_selectbox(
            "Stock feed",
            options=_STOCK_FEED_OPTIONS,
            key=_WIDGET_STOCK_FEED_KEY,
        )
        options_feed = _feed_selectbox(
            "Options feed",
            options=_OPTIONS_FEED_OPTIONS,
            key=_WIDGET_OPTIONS_FEED_KEY,
        )

        refresh_seconds = int(
            st.number_input(
                "Refresh cadence (seconds)",
                min_value=1,
                max_value=60,
                step=1,
                value=_DEFAULT_REFRESH_SECONDS,
                key=_WIDGET_REFRESH_SECONDS_KEY,
            )
        )
        stale_threshold_seconds = float(
            st.number_input(
                "Stale threshold (seconds)",
                min_value=1.0,
                max_value=3600.0,
                step=1.0,
                value=_DEFAULT_STALE_THRESHOLD_SECONDS,
                key=_WIDGET_STALE_SECONDS_KEY,
            )
        )
        max_option_contracts = int(
            st.number_input(
                "Max option contracts",
                min_value=0,
                max_value=5000,
                step=1,
                value=_DEFAULT_MAX_OPTION_CONTRACTS,
                key=_WIDGET_MAX_CONTRACTS_KEY,
            )
        )

    return LivePageControls(
        portfolio_path_raw=portfolio_path_raw,
        reload_clicked=reload_clicked,
        stream_stocks=stream_stocks,
        stream_options=stream_options,
        stream_fills=stream_fills,
        stock_feed=stock_feed,
        options_feed=options_feed,
        refresh_seconds=refresh_seconds,
        stale_threshold_seconds=stale_threshold_seconds,
        max_option_contracts=max_option_contracts,
    )


def _feed_selectbox(
    label: str,
    *,
    options: list[tuple[str, str | None]],
    key: str,
) -> str | None:
    labels = [item[0] for item in options]
    selected_label = st.selectbox(label, options=labels, key=key)
    mapping = dict(options)
    return mapping.get(selected_label)


def _resolve_portfolio_path(raw_value: str | Path | None) -> Path:
    raw = str(raw_value or "").strip()
    candidate = Path(_DEFAULT_PORTFOLIO_PATH) if not raw else Path(raw)
    return candidate.expanduser().resolve()


def _safe_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def _sync_portfolio_state(*, portfolio_path: Path, reload_clicked: bool) -> str | None:
    state = st.session_state
    current_path = str(portfolio_path)
    previous_path = str(state.get(_STATE_PORTFOLIO_PATH_KEY) or "")
    previous_mtime = state.get(_STATE_PORTFOLIO_MTIME_KEY)
    current_mtime = _safe_mtime(portfolio_path)

    has_cached = _STATE_PORTFOLIO_KEY in state or _STATE_PORTFOLIO_ERROR_KEY in state
    reason: str | None = None
    if reload_clicked:
        reason = "explicit reload"
    elif not has_cached:
        reason = "initial load"
    elif previous_path != current_path:
        reason = "path changed"
    elif previous_path == current_path and current_mtime != previous_mtime:
        reason = "mtime changed"

    if reason is None:
        return None

    portfolio, error = _load_portfolio_safe(portfolio_path)
    state[_STATE_PORTFOLIO_KEY] = portfolio
    state[_STATE_PORTFOLIO_ERROR_KEY] = error
    state[_STATE_PORTFOLIO_PATH_KEY] = current_path
    state[_STATE_PORTFOLIO_MTIME_KEY] = current_mtime

    if error:
        return f"Portfolio {reason}: {error}"
    if reason == "initial load":
        return "Portfolio loaded."
    return f"Portfolio reloaded ({reason})."


def _load_portfolio_safe(path: Path) -> tuple[Portfolio | None, str | None]:
    if not path.exists():
        return None, f"Portfolio JSON not found: {path}"
    try:
        return load_portfolio(path), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to parse portfolio JSON at {path}: {exc}"


def _load_portfolio_from_state() -> Portfolio | None:
    portfolio = st.session_state.get(_STATE_PORTFOLIO_KEY)
    if isinstance(portfolio, Portfolio):
        return portfolio
    return None


def _load_portfolio_error_from_state() -> str | None:
    error = st.session_state.get(_STATE_PORTFOLIO_ERROR_KEY)
    text = str(error or "").strip()
    return text or None


def _ensure_manager() -> LiveStreamManager:
    manager = st.session_state.get(_STATE_MANAGER_KEY)
    if isinstance(manager, LiveStreamManager):
        return manager
    manager = LiveStreamManager()
    st.session_state[_STATE_MANAGER_KEY] = manager
    return manager


def _subscription_plan_from_controls(
    *,
    portfolio: Portfolio | None,
    controls: LivePageControls,
) -> SubscriptionPlan | None:
    if portfolio is None:
        return None
    return build_subscription_plan(
        portfolio,
        stream_stocks=controls.stream_stocks,
        stream_options=controls.stream_options,
        max_option_contracts=controls.max_option_contracts,
    )


def _build_live_stream_config(*, controls: LivePageControls, plan: SubscriptionPlan) -> LiveStreamConfig:
    return LiveStreamConfig(
        stocks=plan.stocks,
        option_contracts=plan.option_contracts,
        stream_stocks=controls.stream_stocks,
        stream_options=controls.stream_options,
        stream_fills=controls.stream_fills,
        stock_feed=controls.stock_feed,
        options_feed=controls.options_feed,
    )


def _validate_start(
    *,
    controls: LivePageControls,
    portfolio: Portfolio | None,
    portfolio_error: str | None,
    plan: SubscriptionPlan | None,
) -> str | None:
    if portfolio_error:
        return portfolio_error
    if portfolio is None or plan is None:
        return "Load a valid portfolio before starting live streams."
    if not any((controls.stream_stocks, controls.stream_options, controls.stream_fills)):
        return "Enable at least one stream toggle before starting."

    has_stock_targets = controls.stream_stocks and bool(plan.stocks)
    has_option_targets = controls.stream_options and bool(plan.option_contracts)
    has_fill_targets = controls.stream_fills
    if not any((has_stock_targets, has_option_targets, has_fill_targets)):
        return "No stock or option subscriptions were derived from the current portfolio."
    return None


def _handle_manager_actions(
    *,
    manager: LiveStreamManager,
    controls: LivePageControls,
    portfolio: Portfolio | None,
    portfolio_error: str | None,
    plan: SubscriptionPlan | None,
    start_clicked: bool,
    stop_clicked: bool,
    restart_clicked: bool,
) -> tuple[str | None, str | None]:
    action_note: str | None = None
    action_error: str | None = None

    if stop_clicked:
        try:
            manager.stop()
            action_note = "Live streams stopped."
        except Exception as exc:  # noqa: BLE001
            action_error = f"Failed to stop live streams: {exc}"

    if start_clicked or restart_clicked:
        start_error = _validate_start(
            controls=controls,
            portfolio=portfolio,
            portfolio_error=portfolio_error,
            plan=plan,
        )
        if start_error:
            return action_note, start_error
        assert plan is not None  # guarded above
        config = _build_live_stream_config(controls=controls, plan=plan)

        try:
            if restart_clicked:
                manager.stop()
            manager.start(config)
            action_note = "Live streams started." if start_clicked else "Live streams restarted."
        except DataFetchError as exc:
            action_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            action_error = f"Failed to start live streams: {exc}"

    return action_note, action_error


def _render_health(snapshot: LiveSnapshot) -> None:
    st.markdown("#### Stream Health")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Running", value="yes" if snapshot.running else "no")
    col2.metric("Queue depth", value=str(snapshot.queue_depth))
    col3.metric("Dropped events", value=str(snapshot.dropped_events))
    col4.metric("As of", value=_fmt_ts(snapshot.as_of))

    if snapshot.last_error:
        st.error(snapshot.last_error)

    rows: list[dict[str, Any]] = []
    for stream_key in _STREAM_KEYS:
        rows.append(
            {
                "stream": stream_key,
                "alive": bool(snapshot.alive.get(stream_key, False)),
                "reconnect_attempts": int(snapshot.reconnect_attempts.get(stream_key, 0)),
                "last_event_ts": _fmt_ts(snapshot.last_event_ts_by_stream.get(stream_key)),
                "dropped_events": int(snapshot.dropped_events_by_stream.get(stream_key, 0)),
                "last_error": snapshot.errors_by_stream.get(stream_key) or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_fills(snapshot: LiveSnapshot) -> None:
    st.markdown("#### Recent Fills / Order Updates")
    fills_df = pd.DataFrame(snapshot.fills)
    if fills_df.empty:
        st.info("No fills or order updates captured yet.")
        return
    display = fills_df.copy()
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"], errors="coerce", utc=True)
        display = display.sort_values(by="timestamp", ascending=False, kind="stable")
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_live_portfolio_tables(
    *,
    portfolio: Portfolio | None,
    snapshot: LiveSnapshot,
    stale_threshold_seconds: float,
) -> None:
    st.markdown("#### Live Position Metrics")
    if portfolio is None:
        st.info("Load a valid portfolio to view live position metrics.")
        return

    single_df = compute_live_position_rows(
        portfolio,
        snapshot,
        stale_after_seconds=stale_threshold_seconds,
    )
    structure_df, legs_df = compute_live_multileg_rows(
        portfolio,
        snapshot,
        stale_after_seconds=stale_threshold_seconds,
    )

    st.markdown("**Single-Leg Positions**")
    if single_df.empty:
        st.caption("No single-leg option positions.")
    else:
        st.dataframe(_prepare_display_df(single_df), hide_index=True, use_container_width=True)

    st.markdown("**Multi-Leg Structures**")
    if structure_df.empty:
        st.caption("No multi-leg positions.")
    else:
        st.dataframe(_prepare_display_df(structure_df), hide_index=True, use_container_width=True)

    st.markdown("**Multi-Leg Legs**")
    if legs_df.empty:
        st.caption("No multi-leg legs.")
    else:
        st.dataframe(_prepare_display_df(legs_df), hide_index=True, use_container_width=True)


def _prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in ("warnings",):
        if column in display.columns:
            display[column] = display[column].map(_fmt_warning_list)
    return display


def _fmt_warning_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        warnings = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(warnings)
    text = str(value or "").strip()
    return text


def _fmt_ts(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = pd.to_datetime(value, errors="coerce", utc=True)
        except Exception:  # noqa: BLE001
            return str(value)
        if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
            return str(value)
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _fragment(*, run_every: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    fragment_fn = getattr(st, "fragment", None)
    if callable(fragment_fn):
        return fragment_fn(run_every=run_every)

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _decorator


__all__ = ["render_live_portfolio_page"]
