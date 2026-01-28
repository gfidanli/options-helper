from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


class OptionsSnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptionsSnapshotStore:
    root_dir: Path

    def _symbol_dir(self, symbol: str) -> Path:
        return self.root_dir / symbol.upper()

    def _day_dir(self, symbol: str, snapshot_date: date) -> Path:
        return self._symbol_dir(symbol) / snapshot_date.isoformat()

    def list_dates(self, symbol: str) -> list[date]:
        sym_dir = self._symbol_dir(symbol)
        if not sym_dir.exists():
            return []
        out: list[date] = []
        for p in sym_dir.iterdir():
            if not p.is_dir():
                continue
            try:
                out.append(date.fromisoformat(p.name))
            except ValueError:
                continue
        return sorted(out)

    def latest_dates(self, symbol: str, *, n: int = 2) -> list[date]:
        dates = self.list_dates(symbol)
        return dates[-n:]

    def save_expiry_snapshot(
        self,
        symbol: str,
        snapshot_date: date,
        *,
        expiry: date,
        snapshot: pd.DataFrame,
        meta: dict | None = None,
    ) -> Path:
        day_dir = self._day_dir(symbol, snapshot_date)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Save snapshot for this expiry.
        out_path = day_dir / f"{expiry.isoformat()}.csv"
        snapshot.to_csv(out_path, index=False)

        # Optional shared meta file.
        if meta is not None:
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

        return out_path

    def load_day(self, symbol: str, snapshot_date: date) -> pd.DataFrame:
        day_dir = self._day_dir(symbol, snapshot_date)
        if not day_dir.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for p in sorted(day_dir.glob("*.csv")):
            try:
                df = pd.read_csv(p)
            except Exception as exc:  # noqa: BLE001
                raise OptionsSnapshotError(f"Failed to read snapshot file: {p}") from exc
            if df.empty:
                continue
            if "expiry" not in df.columns:
                df["expiry"] = p.stem
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        # De-dupe contracts if present; keep last occurrence.
        if "contractSymbol" in out.columns:
            out = out.drop_duplicates(subset=["contractSymbol"], keep="last")
        return out

