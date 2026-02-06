from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EndpointStats:
    calls: int = 0
    retries: int = 0
    rate_limit_429: int = 0
    timeout_count: int = 0
    error_count: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    split_count: int = 0
    fallback_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "retries": int(self.retries),
            "rate_limit_429": int(self.rate_limit_429),
            "timeout_count": int(self.timeout_count),
            "error_count": int(self.error_count),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "split_count": int(self.split_count),
            "fallback_count": int(self.fallback_count),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 1:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * p
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return float(ordered[low])
    frac = idx - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def build_endpoint_stats(
    *,
    calls: int,
    retries: int = 0,
    rate_limit_429: int = 0,
    timeout_count: int = 0,
    error_count: int = 0,
    latencies_ms: list[float] | None = None,
    split_count: int = 0,
    fallback_count: int = 0,
) -> EndpointStats:
    values = [float(v) for v in (latencies_ms or []) if v is not None]
    return EndpointStats(
        calls=max(0, int(calls)),
        retries=max(0, int(retries)),
        rate_limit_429=max(0, int(rate_limit_429)),
        timeout_count=max(0, int(timeout_count)),
        error_count=max(0, int(error_count)),
        latency_p50_ms=_percentile(values, 0.50),
        latency_p95_ms=_percentile(values, 0.95),
        split_count=max(0, int(split_count)),
        fallback_count=max(0, int(fallback_count)),
    )


def default_provider_profile() -> dict[str, Any]:
    return {
        "candles": {
            "max_rps": 8.0,
            "concurrency": 1,
        },
        "contracts": {
            "max_rps": 2.5,
            "page_size": 10000,
        },
        "bars": {
            "max_rps": 30.0,
            "concurrency": 8,
            "batch_mode": "adaptive",
            "batch_size": 8,
        },
    }


def _coerce_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if parsed < minimum:
        return float(default)
    return float(parsed)


def _coerce_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001
        return int(default)
    if parsed < minimum:
        return int(default)
    return int(parsed)


def _normalize_profile(raw: dict[str, Any] | None) -> dict[str, Any]:
    defaults = default_provider_profile()
    payload = raw or {}

    candles = dict(payload.get("candles") or {})
    contracts = dict(payload.get("contracts") or {})
    bars = dict(payload.get("bars") or {})

    batch_mode = str(bars.get("batch_mode") or defaults["bars"]["batch_mode"]).strip().lower()
    if batch_mode not in {"adaptive", "per-contract"}:
        batch_mode = defaults["bars"]["batch_mode"]

    return {
        "candles": {
            "max_rps": _coerce_float(candles.get("max_rps"), defaults["candles"]["max_rps"], minimum=0.1),
            "concurrency": _coerce_int(candles.get("concurrency"), defaults["candles"]["concurrency"], minimum=1),
        },
        "contracts": {
            "max_rps": _coerce_float(contracts.get("max_rps"), defaults["contracts"]["max_rps"], minimum=0.1),
            "page_size": _coerce_int(contracts.get("page_size"), defaults["contracts"]["page_size"], minimum=1),
        },
        "bars": {
            "max_rps": _coerce_float(bars.get("max_rps"), defaults["bars"]["max_rps"], minimum=0.1),
            "concurrency": _coerce_int(bars.get("concurrency"), defaults["bars"]["concurrency"], minimum=1),
            "batch_mode": batch_mode,
            "batch_size": _coerce_int(bars.get("batch_size"), defaults["bars"]["batch_size"], minimum=1),
        },
    }


def load_tuning_profile(path: Path, *, provider: str = "alpaca") -> dict[str, Any]:
    provider_name = str(provider or "alpaca").strip().lower() or "alpaca"
    if not path.exists():
        return default_provider_profile()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default_provider_profile()

    if not isinstance(payload, dict):
        return default_provider_profile()

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return default_provider_profile()

    raw_profile = profiles.get(provider_name)
    if not isinstance(raw_profile, dict):
        return default_provider_profile()

    return _normalize_profile(raw_profile)


def _load_all(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": _utc_now_iso(),
            "profiles": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": _utc_now_iso(),
            "profiles": {},
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": _utc_now_iso(),
            "profiles": {},
        }
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("profiles", {})
    if not isinstance(payload.get("profiles"), dict):
        payload["profiles"] = {}
    return payload


def save_tuning_profile(path: Path, *, provider: str, profile: dict[str, Any]) -> None:
    provider_name = str(provider or "alpaca").strip().lower() or "alpaca"
    payload = _load_all(path)
    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _utc_now_iso()
    profiles = dict(payload.get("profiles") or {})
    profiles[provider_name] = _normalize_profile(profile)
    payload["profiles"] = profiles

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _bounded_float(value: float, *, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


def _bounded_int(value: int, *, minimum: int, maximum: int) -> int:
    return int(min(max(value, minimum), maximum))


def _tune_numeric(
    current: float,
    stats: EndpointStats | None,
    *,
    minimum: float,
    maximum: float,
    increase_factor: float = 1.1,
    decrease_factor: float = 0.8,
) -> float:
    if stats is None or stats.calls <= 0:
        return float(current)
    if stats.rate_limit_429 > 0:
        return _bounded_float(float(current) * decrease_factor, minimum=minimum, maximum=maximum)
    error_rate = float(stats.error_count) / float(max(stats.calls, 1))
    if error_rate <= 0.02 and stats.timeout_count == 0:
        return _bounded_float(float(current) * increase_factor, minimum=minimum, maximum=maximum)
    return _bounded_float(float(current), minimum=minimum, maximum=maximum)


def recommend_profile(
    current: dict[str, Any],
    *,
    candles_stats: EndpointStats | None = None,
    contracts_stats: EndpointStats | None = None,
    bars_stats: EndpointStats | None = None,
) -> dict[str, Any]:
    profile = _normalize_profile(current)

    profile["candles"]["max_rps"] = _tune_numeric(
        float(profile["candles"]["max_rps"]),
        candles_stats,
        minimum=0.5,
        maximum=100.0,
    )
    profile["candles"]["concurrency"] = _bounded_int(
        int(round(_tune_numeric(float(profile["candles"]["concurrency"]), candles_stats, minimum=1.0, maximum=32.0))),
        minimum=1,
        maximum=32,
    )

    profile["contracts"]["max_rps"] = _tune_numeric(
        float(profile["contracts"]["max_rps"]),
        contracts_stats,
        minimum=0.5,
        maximum=20.0,
    )
    profile["contracts"]["page_size"] = _bounded_int(int(profile["contracts"]["page_size"]), minimum=100, maximum=10000)

    profile["bars"]["max_rps"] = _tune_numeric(
        float(profile["bars"]["max_rps"]),
        bars_stats,
        minimum=1.0,
        maximum=4000.0,
    )
    profile["bars"]["concurrency"] = _bounded_int(
        int(round(_tune_numeric(float(profile["bars"]["concurrency"]), bars_stats, minimum=1.0, maximum=256.0))),
        minimum=1,
        maximum=256,
    )
    profile["bars"]["batch_size"] = _bounded_int(
        int(round(_tune_numeric(float(profile["bars"]["batch_size"]), bars_stats, minimum=1.0, maximum=128.0))),
        minimum=1,
        maximum=128,
    )

    return profile


__all__ = [
    "EndpointStats",
    "build_endpoint_stats",
    "default_provider_profile",
    "load_tuning_profile",
    "recommend_profile",
    "save_tuning_profile",
]
