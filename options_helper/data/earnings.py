from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EarningsCacheError(RuntimeError):
    pass


class EarningsRecord(BaseModel):
    schema_version: int = 1
    symbol: str
    fetched_at: datetime
    source: str = Field(default="manual")

    next_earnings_date: date | None = None
    window_start: date | None = None
    window_end: date | None = None

    notes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = None

    @classmethod
    def manual(
        cls,
        *,
        symbol: str,
        next_earnings_date: date | None,
        note: str | None = None,
    ) -> "EarningsRecord":
        notes = []
        if note:
            notes.append(note)
        return cls(
            symbol=symbol.upper(),
            fetched_at=datetime.now(tz=timezone.utc),
            source="manual",
            next_earnings_date=next_earnings_date,
            notes=notes,
        )


@dataclass(frozen=True)
class EarningsStore:
    root_dir: Path

    def _path(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}.json"

    def load(self, symbol: str) -> EarningsRecord | None:
        path = self._path(symbol)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return EarningsRecord.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise EarningsCacheError(f"Failed to read earnings cache: {path}") from exc

    def save(self, record: EarningsRecord) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record.symbol)
        path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def delete(self, symbol: str) -> bool:
        path = self._path(symbol)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except Exception as exc:  # noqa: BLE001
            raise EarningsCacheError(f"Failed to delete earnings cache: {path}") from exc


def safe_next_earnings_date(store: EarningsStore, symbol: str) -> date | None:
    try:
        record = store.load(symbol)
    except EarningsCacheError:
        return None
    if record is None:
        return None
    return record.next_earnings_date
