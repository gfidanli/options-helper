from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator


class JournalStoreError(RuntimeError):
    pass


class SignalContext(str, Enum):
    POSITION = "position"
    RESEARCH = "research"
    SCANNER = "scanner"


class SignalEvent(BaseModel):
    schema_version: int = 1
    date: date
    symbol: str
    context: SignalContext
    payload: dict[str, Any] = Field(default_factory=dict)
    snapshot_date: date | None = None
    contract_symbol: str | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return str(value).strip().upper()


@dataclass(frozen=True)
class JournalReadResult:
    events: list[SignalEvent]
    errors: list[str]


def _normalize_symbols(symbols: Iterable[str] | None) -> set[str] | None:
    if symbols is None:
        return None
    out = {str(s).strip().upper() for s in symbols if s and str(s).strip()}
    return out or None


def _normalize_contexts(contexts: Iterable[SignalContext | str] | None) -> set[str] | None:
    if contexts is None:
        return None
    out: set[str] = set()
    for ctx in contexts:
        if ctx is None:
            continue
        if isinstance(ctx, SignalContext):
            out.add(ctx.value)
        else:
            out.add(str(ctx).strip().lower())
    return out or None


def filter_events(
    events: Iterable[SignalEvent],
    *,
    symbols: Iterable[str] | None = None,
    contexts: Iterable[SignalContext | str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[SignalEvent]:
    symbol_set = _normalize_symbols(symbols)
    context_set = _normalize_contexts(contexts)

    filtered: list[SignalEvent] = []
    for event in events:
        if symbol_set is not None and event.symbol not in symbol_set:
            continue
        if context_set is not None and event.context.value not in context_set:
            continue
        if start is not None and event.date < start:
            continue
        if end is not None and event.date > end:
            continue
        filtered.append(event)
    return filtered


def index_by_symbol(events: Iterable[SignalEvent]) -> dict[str, list[SignalEvent]]:
    out: dict[str, list[SignalEvent]] = {}
    for event in events:
        out.setdefault(event.symbol, []).append(event)
    return out


def index_by_date(events: Iterable[SignalEvent]) -> dict[date, list[SignalEvent]]:
    out: dict[date, list[SignalEvent]] = {}
    for event in events:
        out.setdefault(event.date, []).append(event)
    return out


def index_by_context(events: Iterable[SignalEvent]) -> dict[str, list[SignalEvent]]:
    out: dict[str, list[SignalEvent]] = {}
    for event in events:
        out.setdefault(event.context.value, []).append(event)
    return out


@dataclass(frozen=True)
class JournalStore:
    root_dir: Path
    filename: str = "signal_events.jsonl"

    def path(self) -> Path:
        return self.root_dir / self.filename

    def append_event(self, event: SignalEvent) -> None:
        self.append_events([event])

    def append_events(self, events: Iterable[SignalEvent]) -> int:
        events_list = list(events)
        if not events_list:
            return 0
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in events_list:
                payload = event.model_dump(mode="json")
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        return len(events_list)

    def read_events(self, *, ignore_invalid: bool = True) -> JournalReadResult:
        path = self.path()
        if not path.exists():
            return JournalReadResult(events=[], errors=[])

        events: list[SignalEvent] = []
        errors: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except Exception as exc:  # noqa: BLE001
                    msg = f"line {lineno}: invalid json ({exc})"
                    if not ignore_invalid:
                        raise JournalStoreError(msg) from exc
                    errors.append(msg)
                    continue

                try:
                    event = SignalEvent.model_validate(raw)
                except Exception as exc:  # noqa: BLE001
                    msg = f"line {lineno}: invalid event ({exc})"
                    if not ignore_invalid:
                        raise JournalStoreError(msg) from exc
                    errors.append(msg)
                    continue
                events.append(event)

        return JournalReadResult(events=events, errors=errors)

    def query(
        self,
        *,
        symbols: Iterable[str] | None = None,
        contexts: Iterable[SignalContext | str] | None = None,
        start: date | None = None,
        end: date | None = None,
        ignore_invalid: bool = True,
    ) -> JournalReadResult:
        result = self.read_events(ignore_invalid=ignore_invalid)
        filtered = filter_events(result.events, symbols=symbols, contexts=contexts, start=start, end=end)
        return JournalReadResult(events=filtered, errors=result.errors)
