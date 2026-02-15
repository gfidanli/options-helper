from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Protocol, Sequence

import pandas as pd

SnapshotLoader = Callable[[str, date], pd.DataFrame | None]


class _SignalContextLike(Protocol):
    value: str


class SignalEventLike(Protocol):
    date: date
    symbol: str
    context: _SignalContextLike
    snapshot_date: date | None
    contract_symbol: str | None
    payload: object | None


@dataclass(frozen=True)
class HorizonOutcome:
    horizon: int
    target_date: date | None
    underlying_start: float | None
    underlying_end: float | None
    underlying_return: float | None
    option_start: float | None
    option_end: float | None
    option_return: float | None


@dataclass(frozen=True)
class EventOutcome:
    event: SignalEventLike
    outcomes: dict[int, HorizonOutcome]
    start_date: date | None
    start_close: float | None
    action: str | None


def _normalize_close_series(history: pd.DataFrame) -> pd.Series:
    if history is None or history.empty or "Close" not in history.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype="float64")
    if not isinstance(close.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(close.index, errors="coerce", utc=True)
    close = close.loc[~idx.isna()].copy()
    close.index = idx[~idx.isna()].tz_localize(None)
    return close.sort_index()


def _locate_start(close: pd.Series, as_of: date) -> tuple[int, date, float] | None:
    if close.empty or not isinstance(close.index, pd.DatetimeIndex):
        return None
    cutoff = pd.Timestamp(as_of)
    eligible = close.loc[close.index <= cutoff]
    if eligible.empty:
        return None
    pos = len(eligible) - 1
    ts = eligible.index[-1]
    return pos, ts.date(), float(eligible.iloc[-1])


def _future_close(close: pd.Series, start_pos: int, horizon: int) -> tuple[date, float] | None:
    target_pos = int(start_pos) + int(horizon)
    if target_pos < 0 or target_pos >= len(close):
        return None
    ts = close.index[target_pos]
    return ts.date(), float(close.iloc[target_pos])


def _mark_from_row(row: pd.Series | dict | None) -> float | None:
    if row is None:
        return None
    def _get(key: str) -> float | None:
        try:
            if isinstance(row, dict):
                val = row.get(key)
            else:
                val = row[key] if key in row else None
            if val is None:
                return None
            return float(val)
        except Exception:  # noqa: BLE001
            return None

    bid = _get("bid")
    ask = _get("ask")
    last = _get("lastPrice")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
        return bid
    return None


def _find_row_by_contract_symbol(df: pd.DataFrame | None, contract_symbol: str) -> pd.Series | None:
    if df is None or df.empty or "contractSymbol" not in df.columns:
        return None
    mask = df["contractSymbol"].astype(str) == str(contract_symbol)
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


def _extract_action(payload: object | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    advice = payload.get("advice")
    if not isinstance(advice, dict):
        return None
    action = advice.get("action")
    return str(action) if action is not None else None


def compute_event_outcomes(
    events: Iterable[SignalEventLike],
    *,
    history_by_symbol: dict[str, pd.DataFrame],
    horizons: Sequence[int] = (1, 5, 20),
    snapshot_loader: SnapshotLoader | None = None,
) -> list[EventOutcome]:
    results: list[EventOutcome] = []
    horizons_list = [int(h) for h in horizons if int(h) > 0]

    for event in events:
        history = history_by_symbol.get(event.symbol.upper(), pd.DataFrame())
        close = _normalize_close_series(history)
        start_info = _locate_start(close, event.date)
        start_pos = None
        start_date = None
        start_close = None
        if start_info is not None:
            start_pos, start_date, start_close = start_info

        mark_start = None
        if event.contract_symbol and event.snapshot_date and snapshot_loader is not None:
            df_start = snapshot_loader(event.symbol, event.snapshot_date)
            row = _find_row_by_contract_symbol(df_start, event.contract_symbol)
            mark_start = _mark_from_row(row)

        horizon_outcomes: dict[int, HorizonOutcome] = {}
        for horizon in horizons_list:
            target_date = None
            end_close = None
            underlying_return = None
            if start_pos is not None:
                future = _future_close(close, start_pos, horizon)
                if future is not None:
                    target_date, end_close = future
                    if start_close is not None and start_close > 0 and end_close is not None:
                        underlying_return = (end_close / start_close) - 1.0

            option_end = None
            option_return = None
            if (
                mark_start is not None
                and mark_start > 0
                and target_date is not None
                and event.contract_symbol
                and snapshot_loader is not None
            ):
                df_end = snapshot_loader(event.symbol, target_date)
                row = _find_row_by_contract_symbol(df_end, event.contract_symbol)
                option_end = _mark_from_row(row)
                if option_end is not None and option_end > 0:
                    option_return = (option_end / mark_start) - 1.0

            horizon_outcomes[horizon] = HorizonOutcome(
                horizon=horizon,
                target_date=target_date,
                underlying_start=start_close,
                underlying_end=end_close,
                underlying_return=underlying_return,
                option_start=mark_start,
                option_end=option_end,
                option_return=option_return,
            )

        results.append(
            EventOutcome(
                event=event,
                outcomes=horizon_outcomes,
                start_date=start_date,
                start_close=start_close,
                action=_extract_action(event.payload),
            )
        )

    return results


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "hit_rate": None,
            "avg": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    series = pd.Series(values, dtype="float64")
    hit_rate = float((series > 0).mean()) if not series.empty else None
    return {
        "count": int(series.shape[0]),
        "hit_rate": hit_rate,
        "avg": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def summarize_outcomes(
    event_outcomes: Iterable[EventOutcome],
    *,
    horizons: Sequence[int],
) -> dict[str, dict[int, dict[str, dict[str, float | int | None]]]]:
    summary: dict[str, dict[int, dict[str, dict[str, float | int | None]]]] = {}
    for outcome in event_outcomes:
        context = outcome.event.context.value
        if context not in summary:
            summary[context] = {}
        for horizon in horizons:
            if horizon not in summary[context]:
                summary[context][horizon] = {"underlying": {}, "option": {}}

    for context in list(summary.keys()):
        for horizon in horizons:
            underlying_values: list[float] = []
            option_values: list[float] = []
            for outcome in event_outcomes:
                if outcome.event.context.value != context:
                    continue
                horizon_outcome = outcome.outcomes.get(int(horizon))
                if horizon_outcome is None:
                    continue
                if horizon_outcome.underlying_return is not None:
                    underlying_values.append(float(horizon_outcome.underlying_return))
                if horizon_outcome.option_return is not None:
                    option_values.append(float(horizon_outcome.option_return))
            summary[context][horizon]["underlying"] = _summary_stats(underlying_values)
            summary[context][horizon]["option"] = _summary_stats(option_values)
    return summary


def _top_bottom(
    event_outcomes: Iterable[EventOutcome],
    *,
    horizon: int,
    key: str,
    top_n: int,
) -> dict[str, list[dict[str, object]]]:
    items: list[tuple[float, EventOutcome]] = []
    for outcome in event_outcomes:
        horizon_outcome = outcome.outcomes.get(int(horizon))
        if horizon_outcome is None:
            continue
        value = getattr(horizon_outcome, key, None)
        if value is None:
            continue
        items.append((float(value), outcome))

    items_sorted = sorted(items, key=lambda t: t[0], reverse=True)
    top = items_sorted[:top_n]
    bottom = list(reversed(items_sorted[-top_n:])) if items_sorted else []

    def _format(row: tuple[float, EventOutcome]) -> dict[str, object]:
        ret, outcome = row
        return {
            "symbol": outcome.event.symbol,
            "context": outcome.event.context.value,
            "date": outcome.event.date.isoformat(),
            "return": ret,
            "action": outcome.action,
        }

    return {
        "top": [_format(row) for row in top],
        "bottom": [_format(row) for row in bottom],
    }


def build_journal_report(
    events: Iterable[SignalEventLike],
    *,
    history_by_symbol: dict[str, pd.DataFrame],
    horizons: Sequence[int] = (1, 5, 20),
    snapshot_loader: SnapshotLoader | None = None,
    top_n: int = 5,
) -> dict:
    horizons_list = [int(h) for h in horizons if int(h) > 0]
    event_outcomes = compute_event_outcomes(
        events,
        history_by_symbol=history_by_symbol,
        horizons=horizons_list,
        snapshot_loader=snapshot_loader,
    )
    summary = summarize_outcomes(event_outcomes, horizons=horizons_list)

    top_bottom: dict[int, dict[str, list[dict[str, object]]]] = {}
    for horizon in horizons_list:
        top_bottom[horizon] = {
            "underlying": _top_bottom(event_outcomes, horizon=horizon, key="underlying_return", top_n=top_n),
            "option": _top_bottom(event_outcomes, horizon=horizon, key="option_return", top_n=top_n),
        }

    serialized_events: list[dict[str, object]] = []
    for outcome in event_outcomes:
        serialized_events.append(
            {
                "date": outcome.event.date.isoformat(),
                "symbol": outcome.event.symbol,
                "context": outcome.event.context.value,
                "snapshot_date": outcome.event.snapshot_date.isoformat()
                if outcome.event.snapshot_date
                else None,
                "contract_symbol": outcome.event.contract_symbol,
                "action": outcome.action,
                "start_date": outcome.start_date.isoformat() if outcome.start_date else None,
                "start_close": outcome.start_close,
                "outcomes": {
                    str(h): {
                        "target_date": o.target_date.isoformat() if o.target_date else None,
                        "underlying_start": o.underlying_start,
                        "underlying_end": o.underlying_end,
                        "underlying_return": o.underlying_return,
                        "option_start": o.option_start,
                        "option_end": o.option_end,
                        "option_return": o.option_return,
                    }
                    for h, o in outcome.outcomes.items()
                },
            }
        )

    return {
        "schema_version": 1,
        "horizons": horizons_list,
        "events": serialized_events,
        "summary": summary,
        "top_bottom": top_bottom,
    }


def render_journal_report_markdown(report: dict) -> str:
    horizons = report.get("horizons", [])
    summary = report.get("summary", {})
    top_bottom = report.get("top_bottom", {})

    lines: list[str] = []
    lines.append("# Journal Evaluation")
    lines.append("")
    lines.append(f"- Events: `{len(report.get('events', []))}`")
    lines.append(f"- Horizons (trading days): `{', '.join(str(h) for h in horizons)}`")
    lines.append("- Not financial advice.")
    lines.append("")

    if not summary:
        lines.append("_No summary data available._")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Summary by Context")
    for context, horizon_map in summary.items():
        lines.append(f"### {context}")
        lines.append("| Horizon | Underlying hit% | Underlying avg | Underlying median | Option hit% | Option avg | Option median |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for horizon in horizons:
            stats = horizon_map.get(int(horizon), {})
            u = stats.get("underlying", {})
            o = stats.get("option", {})

            def _fmt_pct(val) -> str:
                return "-" if val is None else f"{val * 100.0:.0f}%"

            def _fmt_num(val) -> str:
                return "-" if val is None else f"{val:.3f}"

            lines.append(
                f"| {horizon} | {_fmt_pct(u.get('hit_rate'))} | {_fmt_num(u.get('avg'))} | "
                f"{_fmt_num(u.get('median'))} | {_fmt_pct(o.get('hit_rate'))} | "
                f"{_fmt_num(o.get('avg'))} | {_fmt_num(o.get('median'))} |"
            )
        lines.append("")

    lines.append("## Top/Bottom Outcomes")
    for horizon in horizons:
        lines.append(f"### Horizon {horizon}")
        horizon_tb = top_bottom.get(int(horizon), {})
        for key in ("underlying", "option"):
            tb = horizon_tb.get(key, {})
            lines.append(f"#### {key.title()} returns")
            if not tb:
                lines.append("- (no data)")
                continue
            lines.append("- Top:")
            for row in tb.get("top", []):
                lines.append(
                    f"  - {row.get('symbol')} ({row.get('context')}) {row.get('return'):+.2%}"
                )
            lines.append("- Bottom:")
            for row in tb.get("bottom", []):
                lines.append(
                    f"  - {row.get('symbol')} ({row.get('context')}) {row.get('return'):+.2%}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
