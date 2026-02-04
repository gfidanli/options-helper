from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


class OptionContractsStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptionContractsStore:
    root_dir: Path

    def _symbol_dir(self, symbol: str) -> Path:
        return self.root_dir / symbol.upper()

    def _day_dir(self, symbol: str, as_of: date) -> Path:
        return self._symbol_dir(symbol) / as_of.isoformat()

    def _upsert_meta(self, day_dir: Path, meta: dict[str, Any]) -> None:
        meta_path = day_dir / "meta.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {}
        else:
            existing = {}
        existing.update(meta)
        meta_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    def save(
        self,
        symbol: str,
        as_of: date,
        df: pd.DataFrame,
        *,
        raw: dict[str, Any] | None,
        meta: dict[str, Any] | None,
    ) -> Path:
        day_dir = self._day_dir(symbol, as_of)
        day_dir.mkdir(parents=True, exist_ok=True)

        table_path = day_dir / "contracts.csv"
        (df if df is not None else pd.DataFrame()).to_csv(table_path, index=False)

        if raw is not None:
            raw_path = day_dir / "contracts.raw.json"
            raw_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

        meta_payload = dict(meta or {})
        meta_payload.setdefault("symbol", symbol.upper())
        meta_payload.setdefault("as_of", as_of.isoformat())
        meta_payload["rows"] = int(len(df)) if df is not None else 0
        self._upsert_meta(day_dir, meta_payload)

        return table_path

    def load(self, symbol: str, as_of: date) -> pd.DataFrame | None:
        day_dir = self._day_dir(symbol, as_of)
        table_path = day_dir / "contracts.csv"
        if not table_path.exists():
            return None
        try:
            return pd.read_csv(table_path)
        except Exception as exc:  # noqa: BLE001
            raise OptionContractsStoreError(f"Failed to read contracts cache: {table_path}") from exc

    def load_meta(self, symbol: str, as_of: date) -> dict[str, Any]:
        day_dir = self._day_dir(symbol, as_of)
        meta_path = day_dir / "meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise OptionContractsStoreError(f"Failed to read contracts meta: {meta_path}") from exc

