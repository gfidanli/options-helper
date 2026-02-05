from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class AlpacaRateLimitSnapshot:
    limit: int | None
    remaining: int | None
    reset_at: datetime | None
    reset_epoch: int | None

    def reset_in_seconds(self, *, now: datetime | None = None) -> float | None:
        if self.reset_at is None:
            return None
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        return max((self.reset_at - now_dt.astimezone(timezone.utc)).total_seconds(), 0.0)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_reset_value(value: Any) -> tuple[datetime | None, int | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        num = float(value)
        if num <= 0:
            return None, None
        epoch = int(num)
        if epoch > 10_000_000_000:
            epoch = int(epoch / 1000)
        return datetime.fromtimestamp(epoch, tz=timezone.utc), epoch

    raw = str(value).strip()
    if not raw:
        return None, None

    try:
        num = float(raw)
    except ValueError:
        num = None

    if num is not None and num > 0:
        epoch = int(num)
        if epoch > 10_000_000_000:
            epoch = int(epoch / 1000)
        return datetime.fromtimestamp(epoch, tz=timezone.utc), epoch

    try:
        parsed = parsedate_to_datetime(raw)
    except Exception:  # noqa: BLE001
        return None, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed, int(parsed.timestamp())


def parse_alpaca_rate_limit_headers(headers: Mapping[str, Any] | None) -> AlpacaRateLimitSnapshot | None:
    if not headers:
        return None

    lowered = {str(k).lower(): v for k, v in dict(headers).items()}

    limit = _coerce_int(lowered.get("x-ratelimit-limit") or lowered.get("x-rate-limit-limit"))
    remaining = _coerce_int(lowered.get("x-ratelimit-remaining") or lowered.get("x-rate-limit-remaining"))
    reset_val = lowered.get("x-ratelimit-reset") or lowered.get("x-rate-limit-reset")
    reset_at, reset_epoch = _parse_reset_value(reset_val)

    if limit is None and remaining is None and reset_at is None:
        return None

    return AlpacaRateLimitSnapshot(
        limit=limit,
        remaining=remaining,
        reset_at=reset_at,
        reset_epoch=reset_epoch,
    )

