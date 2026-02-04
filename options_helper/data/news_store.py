from __future__ import annotations

import gzip
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from options_helper.analysis.osi import normalize_underlying


class NewsStoreError(RuntimeError):
    pass


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            try:
                dt = datetime.strptime(raw[:10], "%Y-%m-%d")
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _item_key(item: dict[str, Any]) -> str:
    if item.get("id"):
        return f"id:{item['id']}"
    headline = str(item.get("headline") or "")
    created = str(item.get("created_at") or "")
    source = str(item.get("source") or "")
    return f"{headline}|{created}|{source}"


@dataclass(frozen=True)
class NewsStore:
    root_dir: Path

    _SCHEMA_VERSION: ClassVar[int] = 1

    def _symbol_dir(self, symbol: str) -> Path:
        return self.root_dir / symbol.upper()

    def partition_path(self, symbol: str, day: date) -> Path:
        return self._symbol_dir(symbol) / f"{day.isoformat()}.jsonl.gz"

    def meta_path(self, symbol: str, day: date) -> Path:
        return self._symbol_dir(symbol) / f"{day.isoformat()}.meta.json"

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item or {})
        created = _parse_datetime(out.get("created_at") or out.get("createdAt") or out.get("published_at"))
        if created is not None:
            out["created_at"] = created.isoformat()
        symbols = out.get("symbols")
        if isinstance(symbols, (list, tuple)):
            cleaned = [normalize_underlying(s) for s in symbols if s]
            out["symbols"] = sorted({s for s in cleaned if s})
        return out

    def load_partition(self, symbol: str, day: date) -> list[dict[str, Any]]:
        path = self.partition_path(symbol, day)
        if not path.exists():
            return []
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        except Exception as exc:  # noqa: BLE001
            raise NewsStoreError(f"Failed to read news partition: {path}") from exc

    def save_partition(
        self,
        symbol: str,
        day: date,
        items: list[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        normalized = [self._normalize_item(item) for item in (items or [])]
        payload = "\n".join(json.dumps(item, default=str) for item in normalized) + "\n"
        compressed = gzip.compress(payload.encode("utf-8"))
        path = self.partition_path(symbol, day)
        self._atomic_write_bytes(path, compressed)

        created_times = []
        for item in normalized:
            dt = _parse_datetime(item.get("created_at"))
            if dt is not None:
                created_times.append(dt)

        meta_payload: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "symbol": symbol.upper(),
            "day": day.isoformat(),
            "rows": len(normalized),
            "coverage_start": min(created_times).isoformat() if created_times else None,
            "coverage_end": max(created_times).isoformat() if created_times else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if meta:
            meta_payload["meta"] = meta
        self.meta_path(symbol, day).write_text(
            json.dumps(meta_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def upsert_items(
        self,
        symbol: str,
        items: list[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> list[Path]:
        grouped: dict[date, list[dict[str, Any]]] = {}
        for item in items or []:
            created = _parse_datetime(
                item.get("created_at")
                or item.get("createdAt")
                or item.get("published_at")
                or item.get("publishedAt")
            )
            if created is None:
                continue
            grouped.setdefault(created.date(), []).append(item)

        paths: list[Path] = []
        for day, day_items in grouped.items():
            existing = self.load_partition(symbol, day)
            merged: dict[str, dict[str, Any]] = {}
            for row in existing:
                merged[_item_key(row)] = row
            for row in day_items:
                merged[_item_key(row)] = row
            paths.append(self.save_partition(symbol, day, list(merged.values()), meta=meta))
        return paths
