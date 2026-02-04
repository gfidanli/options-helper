from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd


class IntradayStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class IntradayStore:
    root_dir: Path

    _SCHEMA_VERSION: ClassVar[int] = 1
    _TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")

    def _safe_token(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            raise IntradayStoreError("Empty token provided for intraday store path.")
        return self._TOKEN_RE.sub("_", raw)

    def _safe_symbol(self, value: str) -> str:
        return self._safe_token(value).upper()

    def _partition_dir(self, kind: str, dataset: str, timeframe: str, symbol: str) -> Path:
        return (
            self.root_dir
            / self._safe_token(kind)
            / self._safe_token(dataset)
            / self._safe_token(timeframe)
            / self._safe_symbol(symbol)
        )

    def partition_path(self, kind: str, dataset: str, timeframe: str, symbol: str, day: date) -> Path:
        return self._partition_dir(kind, dataset, timeframe, symbol) / f"{day.isoformat()}.csv.gz"

    def meta_path(self, kind: str, dataset: str, timeframe: str, symbol: str, day: date) -> Path:
        return self._partition_dir(kind, dataset, timeframe, symbol) / f"{day.isoformat()}.meta.json"

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _atomic_write_csv(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            (df if df is not None else pd.DataFrame()).to_csv(
                tmp_path, index=False, compression="gzip"
            )
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _coverage_from(self, df: pd.DataFrame | None) -> tuple[str | None, str | None]:
        if df is None or df.empty:
            return None, None
        series = None
        if "timestamp" in df.columns:
            series = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            series = pd.to_datetime(df.index, errors="coerce", utc=True)
        if series is None:
            return None, None
        series = series.dropna()
        if series.empty:
            return None, None
        return series.min().isoformat(), series.max().isoformat()

    def save_partition(
        self,
        kind: str,
        dataset: str,
        timeframe: str,
        symbol: str,
        day: date,
        df: pd.DataFrame | None,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        out_path = self.partition_path(kind, dataset, timeframe, symbol, day)
        table = df if df is not None else pd.DataFrame()
        self._atomic_write_csv(out_path, table)

        coverage_start, coverage_end = self._coverage_from(table)
        payload: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "kind": self._safe_token(kind),
            "dataset": self._safe_token(dataset),
            "timeframe": self._safe_token(timeframe),
            "symbol": self._safe_symbol(symbol),
            "day": day.isoformat(),
            "rows": int(len(table)),
            "columns": list(table.columns),
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        if meta:
            payload.update(meta)
        self._atomic_write_text(
            self.meta_path(kind, dataset, timeframe, symbol, day),
            json.dumps(payload, indent=2, default=str),
        )
        return out_path

    def load_partition(
        self,
        kind: str,
        dataset: str,
        timeframe: str,
        symbol: str,
        day: date,
    ) -> pd.DataFrame:
        path = self.partition_path(kind, dataset, timeframe, symbol, day)
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise IntradayStoreError(f"Failed to read intraday partition: {path}") from exc

    def load_meta(
        self,
        kind: str,
        dataset: str,
        timeframe: str,
        symbol: str,
        day: date,
    ) -> dict[str, Any]:
        path = self.meta_path(kind, dataset, timeframe, symbol, day)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise IntradayStoreError(f"Failed to read intraday meta: {path}") from exc
