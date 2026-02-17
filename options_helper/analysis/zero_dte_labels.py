from __future__ import annotations

from dataclasses import dataclass
import math
from zoneinfo import ZoneInfo

import pandas as pd

from options_helper.schemas.zero_dte_put_study import SkipReason


DEFAULT_MARKET_TZ = "America/New_York"
_LABEL_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "decision_bar_completed_ts",
    "entry_anchor_ts",
    "entry_anchor_price",
    "close_label_ts",
    "close_price",
    "close_return_from_entry",
    "skip_reason",
    "label_status",
)


@dataclass(frozen=True)
class ZeroDTELabelConfig:
    market_tz_name: str = DEFAULT_MARKET_TZ
    max_close_lag_seconds: int = 90

    def __post_init__(self) -> None:
        if self.max_close_lag_seconds < 0:
            raise ValueError("max_close_lag_seconds must be >= 0")

    @property
    def market_tz(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz_name)


def build_zero_dte_labels(
    state_rows: pd.DataFrame,
    underlying_bars: pd.DataFrame,
    *,
    market_close_ts: pd.Timestamp | str,
    config: ZeroDTELabelConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ZeroDTELabelConfig()
    state = state_rows.copy() if state_rows is not None else pd.DataFrame()
    if state.empty:
        return pd.DataFrame(columns=list(_LABEL_COLUMNS))

    bars = _normalize_underlying_bars(underlying_bars, market_tz=cfg.market_tz)
    close_ts = _coerce_utc_timestamp(market_close_ts)
    close_bar = _resolve_close_bar(
        bars,
        market_close_ts=close_ts,
        max_close_lag_seconds=cfg.max_close_lag_seconds,
    )

    out_rows: list[dict[str, object]] = []
    for _, state_row in state.iterrows():
        out_rows.append(
            _build_label_row(
                state_row=state_row,
                bars=bars,
                close_ts=close_ts,
                close_bar=close_bar,
            )
        )

    return pd.DataFrame(out_rows, columns=list(_LABEL_COLUMNS))


def _build_label_row(
    *,
    state_row: pd.Series,
    bars: pd.DataFrame,
    close_ts: pd.Timestamp,
    close_bar: pd.Series | None,
) -> dict[str, object]:
    decision_ts = _coerce_utc_timestamp(state_row.get("decision_ts"))
    decision_bar_ts = _coerce_utc_timestamp(state_row.get("bar_ts"))
    row = _initialize_label_row(
        state_row=state_row,
        decision_ts=decision_ts,
        decision_bar_ts=decision_bar_ts,
        close_bar=close_bar,
    )
    state_status = str(state_row.get("status") or "").strip().lower()
    if state_status != "ok":
        _set_state_status_reject(row, state_status=state_status)
        return row

    if bars.empty:
        row["skip_reason"] = SkipReason.INSUFFICIENT_DATA.value
        row["label_status"] = "no_underlying_bars"
        return row

    decision_bar = _resolve_decision_bar(
        bars,
        decision_ts=decision_ts,
        decision_bar_ts=decision_bar_ts,
    )
    if decision_bar is None:
        row["skip_reason"] = SkipReason.OUTSIDE_DECISION_WINDOW.value
        row["label_status"] = "no_decision_bar"
        return row

    decision_bar_timestamp = pd.Timestamp(decision_bar["timestamp"])
    row["decision_bar_completed_ts"] = decision_bar_timestamp

    anchor_bar = _resolve_entry_anchor_bar(
        bars,
        decision_bar_ts=decision_bar_timestamp,
        market_close_ts=close_ts,
    )
    if anchor_bar is None:
        row["skip_reason"] = SkipReason.NO_ENTRY_ANCHOR.value
        row["label_status"] = "no_entry_anchor"
        return row

    entry_anchor_ts = pd.Timestamp(anchor_bar["timestamp"])
    entry_anchor_price = float(pd.to_numeric(anchor_bar.get("open"), errors="coerce"))
    row["entry_anchor_ts"] = entry_anchor_ts
    row["entry_anchor_price"] = entry_anchor_price

    if close_bar is None:
        row["skip_reason"] = SkipReason.INSUFFICIENT_DATA.value
        row["label_status"] = "missing_close_bar"
        return row

    close_price = float(pd.to_numeric(close_bar.get("close"), errors="coerce"))
    if not math.isfinite(entry_anchor_price) or entry_anchor_price <= 0.0:
        row["skip_reason"] = SkipReason.INSUFFICIENT_DATA.value
        row["label_status"] = "invalid_entry_anchor_price"
        return row
    if not math.isfinite(close_price):
        row["skip_reason"] = SkipReason.INSUFFICIENT_DATA.value
        row["label_status"] = "invalid_close_price"
        return row

    row["close_return_from_entry"] = (close_price / entry_anchor_price) - 1.0
    row["skip_reason"] = None
    row["label_status"] = "ok"
    return row


def _initialize_label_row(
    *,
    state_row: pd.Series,
    decision_ts: pd.Timestamp | None,
    decision_bar_ts: pd.Timestamp | None,
    close_bar: pd.Series | None,
) -> dict[str, object]:
    return {
        "session_date": state_row.get("session_date"),
        "decision_ts": decision_ts,
        "decision_bar_completed_ts": decision_bar_ts,
        "entry_anchor_ts": pd.NaT,
        "entry_anchor_price": float("nan"),
        "close_label_ts": close_bar.get("timestamp") if close_bar is not None else pd.NaT,
        "close_price": float(pd.to_numeric(close_bar.get("close"), errors="coerce")) if close_bar is not None else float("nan"),
        "close_return_from_entry": float("nan"),
        "skip_reason": SkipReason.INSUFFICIENT_DATA.value,
        "label_status": "missing_decision_anchor",
    }


def _set_state_status_reject(row: dict[str, object], *, state_status: str) -> None:
    row["skip_reason"] = SkipReason.OUTSIDE_DECISION_WINDOW.value
    row["label_status"] = f"state_{state_status or 'unknown'}"


def _normalize_underlying_bars(df: pd.DataFrame | None, *, market_tz: ZoneInfo) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "close"])

    frame = df.copy()
    frame.columns = [str(col) for col in frame.columns]

    ts_col = _first_present(frame.columns, "timestamp", "ts", "time")
    if ts_col is None:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "close"])

    frame["timestamp"] = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "timestamp_market", "open", "close"])

    market_col = _first_present(frame.columns, "timestamp_market")
    if market_col is None:
        frame["timestamp_market"] = frame["timestamp"].dt.tz_convert(market_tz)
    else:
        market_ts = pd.to_datetime(frame[market_col], errors="coerce")
        if getattr(market_ts.dt, "tz", None) is None:
            frame["timestamp_market"] = market_ts.dt.tz_localize(market_tz)
        else:
            frame["timestamp_market"] = market_ts.dt.tz_convert(market_tz)

    for col in ("open", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else float("nan")

    frame = frame.sort_values(by="timestamp", kind="mergesort")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    return frame.reset_index(drop=True)


def _resolve_close_bar(
    bars: pd.DataFrame,
    *,
    market_close_ts: pd.Timestamp,
    max_close_lag_seconds: int,
) -> pd.Series | None:
    if bars.empty:
        return None
    eligible = bars.loc[bars["timestamp"] <= market_close_ts]
    if eligible.empty:
        return None

    close_bar = eligible.iloc[-1]
    lag_seconds = float((market_close_ts - pd.Timestamp(close_bar["timestamp"])).total_seconds())
    if lag_seconds < 0:
        return None
    if lag_seconds > float(max_close_lag_seconds):
        return None
    return close_bar


def _resolve_decision_bar(
    bars: pd.DataFrame,
    *,
    decision_ts: pd.Timestamp | None,
    decision_bar_ts: pd.Timestamp | None,
) -> pd.Series | None:
    if bars.empty:
        return None

    cutoff = _resolve_cutoff_ts(decision_ts=decision_ts, decision_bar_ts=decision_bar_ts)
    if cutoff is None:
        return None

    eligible = bars.loc[bars["timestamp"] <= cutoff]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _resolve_entry_anchor_bar(
    bars: pd.DataFrame,
    *,
    decision_bar_ts: pd.Timestamp,
    market_close_ts: pd.Timestamp,
) -> pd.Series | None:
    eligible = bars.loc[
        (bars["timestamp"] > decision_bar_ts)
        & (bars["timestamp"] <= market_close_ts)
    ]
    if eligible.empty:
        return None
    return eligible.iloc[0]


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


def _coerce_utc_timestamp(raw: object) -> pd.Timestamp | None:
    if raw is None or raw is pd.NaT:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _first_present(columns: list[str] | pd.Index, *candidates: str) -> str | None:
    existing = set(str(col) for col in columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


__all__ = [
    "DEFAULT_MARKET_TZ",
    "ZeroDTELabelConfig",
    "build_zero_dte_labels",
]
