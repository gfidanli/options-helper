from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, ClassVar


class CorporateActionsStoreError(RuntimeError):
    pass


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            return raw
    return None


def _coerce_action_type(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    return raw or None


def _action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("symbol"),
        action.get("type"),
        action.get("ex_date"),
        action.get("record_date"),
        action.get("pay_date"),
        action.get("ratio"),
        action.get("cash_amount"),
    )


@dataclass(frozen=True)
class CorporateActionsStore:
    root_dir: Path

    _SCHEMA_VERSION: ClassVar[int] = 1

    def _path(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}.json"

    def load(self, symbol: str) -> dict[str, Any]:
        path = self._path(symbol)
        if not path.exists():
            return {"schema_version": self._SCHEMA_VERSION, "symbol": symbol.upper(), "actions": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise CorporateActionsStoreError(f"Failed to read corporate actions cache: {path}") from exc

    def _normalize_action(self, symbol: str, action: dict[str, Any]) -> dict[str, Any]:
        raw = dict(action or {})
        normalized = {
            "type": _coerce_action_type(
                raw.get("type")
                or raw.get("action_type")
                or raw.get("event_type")
                or raw.get("corporate_action_type")
            ),
            "symbol": (raw.get("symbol") or symbol or "").upper(),
            "ex_date": _coerce_date_string(raw.get("ex_date") or raw.get("exDate") or raw.get("exdate")),
            "record_date": _coerce_date_string(
                raw.get("record_date") or raw.get("recordDate") or raw.get("recorddate")
            ),
            "pay_date": _coerce_date_string(
                raw.get("pay_date") or raw.get("payDate") or raw.get("payment_date") or raw.get("paymentDate")
            ),
            "ratio": _as_float(raw.get("ratio") or raw.get("split_ratio") or raw.get("splitRatio")),
            "cash_amount": _as_float(
                raw.get("cash_amount") or raw.get("cashAmount") or raw.get("dividend") or raw.get("amount")
            ),
            "raw": raw.get("raw") or raw,
        }
        return normalized

    def _dedupe_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        for action in actions:
            seen[_action_key(action)] = action
        return list(seen.values())

    def save(
        self,
        symbol: str,
        actions: list[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
        merge: bool = True,
    ) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(symbol)

        normalized = [self._normalize_action(symbol, action) for action in (actions or [])]
        existing_actions: list[dict[str, Any]] = []
        if merge and path.exists():
            existing = self.load(symbol)
            existing_actions = list(existing.get("actions") or [])

        combined = self._dedupe_actions(existing_actions + normalized)

        ex_dates = [row.get("ex_date") for row in combined if row.get("ex_date")]
        coverage_start = min(ex_dates) if ex_dates else None
        coverage_end = max(ex_dates) if ex_dates else None

        payload: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "symbol": symbol.upper(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "actions": combined,
            "rows": len(combined),
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
        }
        if meta:
            payload["meta"] = meta

        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def query(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
        types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.load(symbol)
        actions = list(payload.get("actions") or [])
        if not actions:
            return []

        types_set = {t.strip().lower() for t in (types or []) if t and str(t).strip()}
        results: list[dict[str, Any]] = []

        for action in actions:
            action_type = str(action.get("type") or "").strip().lower()
            if types_set and action_type not in types_set:
                continue

            date_val = action.get("ex_date") or action.get("record_date") or action.get("pay_date")
            action_date = None
            if isinstance(date_val, str):
                try:
                    action_date = date.fromisoformat(date_val[:10])
                except ValueError:
                    action_date = None
            elif isinstance(date_val, date):
                action_date = date_val

            if start and action_date and action_date < start:
                continue
            if end and action_date and action_date > end:
                continue
            results.append(action)

        return results
