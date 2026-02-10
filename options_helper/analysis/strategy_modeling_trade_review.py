from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import math
from typing import Any

_REALIZED_R_METRIC = "realized_r"
_SCOPE_ACCEPTED = "accepted_closed_trades"
_SCOPE_FALLBACK = "closed_nonrejected_trades"
_MISSING_ENTRY_TS = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TradeReviewResult:
    metric: str
    scope: str
    candidate_trade_count: int
    top_best_rows: tuple[dict[str, Any], ...]
    top_worst_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _CandidateTrade:
    row: dict[str, Any]
    metric_value: float
    entry_ts_utc: datetime | None
    trade_id_sort_key: str
    source_index: int


def rank_trades_for_review(
    trade_rows: Iterable[Mapping[str, Any] | object] | None,
    *,
    accepted_trade_ids: Sequence[str] | None,
    top_n: int = 20,
    metric: str = _REALIZED_R_METRIC,
) -> TradeReviewResult:
    metric_key = str(metric).strip().lower()
    if metric_key != _REALIZED_R_METRIC:
        raise ValueError("metric must be 'realized_r'")

    limit = int(top_n)
    if limit < 0:
        raise ValueError("top_n must be >= 0")

    accepted_scope_ids: set[str] | None = None
    scope = _SCOPE_FALLBACK
    if accepted_trade_ids is not None:
        scope = _SCOPE_ACCEPTED
        accepted_scope_ids = {
            trade_id
            for trade_id in (_as_str_or_none(value) for value in accepted_trade_ids)
            if trade_id is not None
        }

    candidates: list[_CandidateTrade] = []
    for source_index, payload in enumerate(_iter_trade_rows(trade_rows)):
        row = _to_mapping(payload)
        if not row:
            continue

        trade_id = _as_str_or_none(row.get("trade_id"))
        if accepted_scope_ids is not None and (trade_id is None or trade_id not in accepted_scope_ids):
            continue

        if _normalize_status(row.get("status")) != "closed":
            continue
        if _as_str_or_none(row.get("reject_code")) is not None:
            continue

        metric_value = _coerce_finite_float(row.get(metric_key))
        if metric_value is None:
            continue

        normalized_row = dict(row)
        normalized_row[metric_key] = metric_value

        candidates.append(
            _CandidateTrade(
                row=normalized_row,
                metric_value=metric_value,
                entry_ts_utc=_coerce_utc_datetime(row.get("entry_ts")),
                trade_id_sort_key=trade_id or "",
                source_index=source_index,
            )
        )

    top_best_rows = _rank_rows(
        sorted(candidates, key=_best_sort_key),
        limit=limit,
    )
    top_worst_rows = _rank_rows(
        sorted(candidates, key=_worst_sort_key),
        limit=limit,
    )

    return TradeReviewResult(
        metric=metric_key,
        scope=scope,
        candidate_trade_count=len(candidates),
        top_best_rows=top_best_rows,
        top_worst_rows=top_worst_rows,
    )


def _iter_trade_rows(
    trade_rows: Iterable[Mapping[str, Any] | object] | None,
) -> Iterable[Mapping[str, Any] | object]:
    if trade_rows is None:
        return ()
    if isinstance(trade_rows, Mapping):
        return (trade_rows,)
    if isinstance(trade_rows, (str, bytes)):
        return ()
    return trade_rows


def _best_sort_key(candidate: _CandidateTrade) -> tuple[float, int, datetime, str, int]:
    missing_flag, entry_ts = _entry_ts_sort_key(candidate.entry_ts_utc)
    return (
        -candidate.metric_value,
        missing_flag,
        entry_ts,
        candidate.trade_id_sort_key,
        candidate.source_index,
    )


def _worst_sort_key(candidate: _CandidateTrade) -> tuple[float, int, datetime, str, int]:
    missing_flag, entry_ts = _entry_ts_sort_key(candidate.entry_ts_utc)
    return (
        candidate.metric_value,
        missing_flag,
        entry_ts,
        candidate.trade_id_sort_key,
        candidate.source_index,
    )


def _entry_ts_sort_key(value: datetime | None) -> tuple[int, datetime]:
    if value is None:
        return (1, _MISSING_ENTRY_TS)
    return (0, value)


def _rank_rows(rows: Sequence[_CandidateTrade], *, limit: int) -> tuple[dict[str, Any], ...]:
    ranked_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(rows[:limit], start=1):
        row_with_rank: dict[str, Any] = {"rank": rank}
        row_with_rank.update(candidate.row)
        ranked_rows.append(row_with_rank)
    return tuple(ranked_rows)


def _normalize_status(value: object) -> str | None:
    status = _as_str_or_none(value)
    if status is None:
        return None
    return status.lower()


def _coerce_finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None

    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime") and callable(value.to_pydatetime):
        try:
            converted = value.to_pydatetime()
        except (TypeError, ValueError):
            return None
        parsed = converted if isinstance(converted, datetime) else None
    else:
        text = _as_str_or_none(value)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _to_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if is_dataclass(value):
        return asdict(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return {str(k): v for k, v in dumped.items()}

    if hasattr(value, "__dict__"):
        return {
            str(k): v
            for k, v in vars(value).items()
            if not str(k).startswith("_") and not callable(v)
        }
    return {}


__all__ = ["TradeReviewResult", "rank_trades_for_review"]
