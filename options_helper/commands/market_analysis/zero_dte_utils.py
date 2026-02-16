from __future__ import annotations

from datetime import date
import hashlib
import json
from typing import Any

import pandas as pd

from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import QuoteQualityStatus, SkipReason, ZeroDteStudyAssumptions


def _coerce_quote_quality_status(raw: object) -> QuoteQualityStatus:
    text = str(raw or "").strip().lower()
    mapping = {item.value: item for item in QuoteQualityStatus}
    return mapping.get(text, QuoteQualityStatus.UNKNOWN)


def _coerce_skip_reason(raw: object) -> SkipReason | None:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    mapping = {item.value: item for item in SkipReason}
    return mapping.get(text)


def _hash_zero_dte_assumptions(assumptions: ZeroDteStudyAssumptions) -> str:
    payload = json.dumps(clean_nan(assumptions.model_dump(mode="json")), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _first_text(*values: object) -> str | None:
    for value in values:
        text = _as_clean_text(value)
        if text:
            return text
    return None


def _timestamp_to_iso(value: object) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [clean_nan(row) for row in frame.to_dict(orient="records")]


def _resolve_as_of_date(frame: pd.DataFrame, *, default: date) -> date:
    if frame is None or frame.empty or "session_date" not in frame.columns:
        return default
    parsed = pd.to_datetime(frame["session_date"], errors="coerce").dropna()
    if parsed.empty:
        return default
    return parsed.max().date()


__all__ = [
    "_coerce_quote_quality_status",
    "_coerce_skip_reason",
    "_hash_zero_dte_assumptions",
    "_as_clean_text",
    "_first_text",
    "_timestamp_to_iso",
    "_frame_records",
    "_resolve_as_of_date",
]
