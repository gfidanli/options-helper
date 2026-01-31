from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

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

    def load_meta(self, symbol: str, snapshot_date: date) -> dict[str, Any]:
        day_dir = self._day_dir(symbol, snapshot_date)
        meta_path = day_dir / "meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise OptionsSnapshotError(f"Failed to read meta.json: {meta_path}") from exc

    def resolve_date(self, symbol: str, spec: str) -> date:
        """
        Resolve a snapshot date spec:
        - YYYY-MM-DD
        - latest
        """
        spec = spec.strip().lower()
        if spec == "latest":
            dates = self.latest_dates(symbol, n=1)
            if not dates:
                raise OptionsSnapshotError(f"No snapshots found for {symbol}")
            return dates[-1]
        try:
            return date.fromisoformat(spec)
        except ValueError as exc:
            raise OptionsSnapshotError(f"Invalid date spec: {spec} (use YYYY-MM-DD or 'latest')") from exc

    def resolve_relative_date(self, symbol: str, *, to_date: date, offset: int) -> date:
        """
        Resolve a snapshot date relative to another snapshot date.

        Example: offset=-1 means the snapshot immediately before `to_date`.
        """
        dates = self.list_dates(symbol)
        if not dates:
            raise OptionsSnapshotError(f"No snapshots found for {symbol}")

        try:
            idx = dates.index(to_date)
        except ValueError as exc:
            raise OptionsSnapshotError(
                f"Snapshot date not found for {symbol}: {to_date.isoformat()}"
            ) from exc

        target_idx = idx + offset
        if target_idx < 0 or target_idx >= len(dates):
            raise OptionsSnapshotError(
                f"Relative snapshot out of range for {symbol}: to={to_date.isoformat()} offset={offset}"
            )
        return dates[target_idx]

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
        meta: dict[str, Any] | None = None,
    ) -> Path:
        day_dir = self._day_dir(symbol, snapshot_date)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Save snapshot for this expiry.
        out_path = day_dir / f"{expiry.isoformat()}.csv"
        snapshot.to_csv(out_path, index=False)

        # Optional shared meta file.
        if meta is not None:
            self._upsert_meta(day_dir, meta)

        return out_path

    def save_expiry_snapshot_raw(
        self,
        symbol: str,
        snapshot_date: date,
        *,
        expiry: date,
        raw: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """Save the raw Yahoo options payload for an expiry (best-effort)."""
        day_dir = self._day_dir(symbol, snapshot_date)
        day_dir.mkdir(parents=True, exist_ok=True)

        out_path = day_dir / f"{expiry.isoformat()}.raw.json"
        out_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

        if meta is not None:
            self._upsert_meta(day_dir, meta)

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
