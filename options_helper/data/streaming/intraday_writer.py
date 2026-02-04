from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.data.intraday_store import IntradayStore


def _timestamp_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = pd.to_datetime(raw, errors="raise", utc=True).to_pydatetime()
            except Exception:  # noqa: BLE001
                return raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = pd.to_datetime(value, errors="raise", utc=True).to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PartitionSpec:
    kind: str
    dataset: str
    timeframe: str
    symbol: str
    day: date


@dataclass
class BufferedIntradayWriter:
    store: IntradayStore
    spec: PartitionSpec
    meta: dict[str, Any] = field(default_factory=dict)
    dedupe_on: tuple[str, ...] = ("timestamp",)
    sort_by: str | None = "timestamp"

    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _flushes: int = field(default=0, init=False)

    def add(self, row: dict[str, Any]) -> None:
        if not row:
            return
        self._rows.append(dict(row))

    def extend(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    @property
    def pending(self) -> int:
        return len(self._rows)

    def flush(self) -> Path | None:
        if not self._rows:
            return None

        new_df = pd.DataFrame(self._rows)
        if not new_df.empty and "timestamp" in new_df.columns:
            new_df["timestamp"] = new_df["timestamp"].map(_timestamp_iso)

        if self.spec.kind == "options" and "contractSymbol" not in new_df.columns:
            new_df["contractSymbol"] = self.spec.symbol

        existing = self.store.load_partition(
            self.spec.kind,
            self.spec.dataset,
            self.spec.timeframe,
            self.spec.symbol,
            self.spec.day,
        )
        if existing is not None and not existing.empty:
            existing_df = existing.copy()
            if "timestamp" in existing_df.columns:
                existing_df["timestamp"] = existing_df["timestamp"].map(_timestamp_iso)
            if self.spec.kind == "options" and "contractSymbol" not in existing_df.columns:
                existing_df["contractSymbol"] = self.spec.symbol
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        if not combined.empty and self.dedupe_on:
            subset = [col for col in self.dedupe_on if col in combined.columns]
            if subset:
                combined = combined.drop_duplicates(subset=subset, keep="last")

        if not combined.empty and self.sort_by and self.sort_by in combined.columns:
            combined = combined.sort_values(self.sort_by, kind="mergesort")

        combined = combined.reset_index(drop=True)

        self._flushes += 1
        meta = dict(self.meta)
        meta["flushes"] = self._flushes

        out_path = self.store.save_partition(
            self.spec.kind,
            self.spec.dataset,
            self.spec.timeframe,
            self.spec.symbol,
            self.spec.day,
            combined,
            meta=meta,
        )
        self._rows.clear()
        return out_path


__all__ = ["BufferedIntradayWriter", "PartitionSpec"]
