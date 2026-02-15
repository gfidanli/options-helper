from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.data.options_snapshots import OptionsSnapshotError, OptionsSnapshotStore, _dedupe_contracts
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.stores_duckdb.common import _coerce_date, _sql_quote
from options_helper.db.warehouse import DuckDBWarehouse


@dataclass(frozen=True)
class DuckDBOptionsSnapshotStore:
    """DuckDB-backed snapshot store with Parquet lake files."""

    lake_root: Path
    warehouse: DuckDBWarehouse
    sync_legacy_files: bool = False

    def _symbol_dir(self, symbol: str) -> Path:
        return self.lake_root / str(symbol).strip().upper()

    def _day_dir(self, symbol: str, snapshot_date: date) -> Path:
        return self._symbol_dir(symbol) / snapshot_date.isoformat()

    def _provider(self) -> str:
        return get_default_provider_name()

    def _filesystem_store(self) -> OptionsSnapshotStore:
        return OptionsSnapshotStore(self.lake_root)

    def _path_under_lake_root(self, value: Any) -> bool:
        if value is None:
            return False
        raw = str(value).strip()
        if not raw:
            return False
        try:
            root = self.lake_root.resolve()
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=False)
            return candidate.is_relative_to(root)
        except Exception:  # noqa: BLE001
            return False

    def _header_rows(self, symbol: str, snapshot_date: date | None = None) -> list[dict[str, Any]]:
        provider = self._provider()
        if snapshot_date is None:
            df = self.warehouse.fetch_df(
                """
                SELECT snapshot_date, chain_path, meta_path
                FROM options_snapshot_headers
                WHERE symbol = ? AND provider = ?
                ORDER BY snapshot_date ASC, updated_at DESC
                """,
                [symbol, provider],
            )
        else:
            df = self.warehouse.fetch_df(
                """
                SELECT snapshot_date, chain_path, meta_path, meta_json
                FROM options_snapshot_headers
                WHERE symbol = ? AND snapshot_date = ? AND provider = ?
                ORDER BY updated_at DESC
                """,
                [symbol, snapshot_date.isoformat(), provider],
            )
        if df is None or df.empty:
            return []
        return [dict(row) for row in df.to_dict(orient="records")]

    def _select_header_row(self, symbol: str, snapshot_date: date) -> dict[str, Any] | None:
        for row in self._header_rows(symbol, snapshot_date):
            if self._path_under_lake_root(row.get("chain_path")) or self._path_under_lake_root(row.get("meta_path")):
                return row
        return None

    def list_dates(self, symbol: str) -> list[date]:
        sym = str(symbol).strip().upper()
        if not sym:
            return []
        fs_dates = self._filesystem_store().list_dates(sym)
        rows = self._header_rows(sym)
        if not rows:
            return fs_dates
        db_dates: list[date] = []
        for row in rows:
            if not (
                self._path_under_lake_root(row.get("chain_path"))
                or self._path_under_lake_root(row.get("meta_path"))
            ):
                continue
            coerced = _coerce_date(row.get("snapshot_date"))
            if coerced is not None:
                db_dates.append(coerced)
        return sorted(set(db_dates).union(fs_dates))

    def latest_dates(self, symbol: str, *, n: int = 2) -> list[date]:
        dates = self.list_dates(symbol)
        return dates[-n:]

    def resolve_date(self, symbol: str, spec: str) -> date:
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
        dates = self.list_dates(symbol)
        if not dates:
            raise OptionsSnapshotError(f"No snapshots found for {symbol}")
        try:
            idx = dates.index(to_date)
        except ValueError as exc:
            raise OptionsSnapshotError(f"Snapshot date not found for {symbol}: {to_date.isoformat()}") from exc
        target_idx = idx + offset
        if target_idx < 0 or target_idx >= len(dates):
            raise OptionsSnapshotError(
                f"Relative snapshot out of range for {symbol}: to={to_date.isoformat()} offset={offset}"
            )
        return dates[target_idx]

    def load_meta(self, symbol: str, snapshot_date: date) -> dict[str, Any]:
        sym = str(symbol).strip().upper()
        if not sym:
            return {}
        fs_store = self._filesystem_store()
        fs_meta = fs_store.load_meta(sym, snapshot_date)
        if fs_meta:
            return fs_meta

        row = self._select_header_row(sym, snapshot_date)
        if row is None:
            return fs_meta
        meta_json = row.get("meta_json")
        if isinstance(meta_json, str) and meta_json.strip():
            try:
                return json.loads(meta_json)
            except Exception:  # noqa: BLE001
                pass

        meta_path = row.get("meta_path")
        if meta_path and self._path_under_lake_root(meta_path):
            p = Path(str(meta_path))
            if p.exists():
                try:
                    if p.suffix.endswith(".gz"):
                        with gzip.open(p, "rt", encoding="utf-8") as handle:
                            return json.load(handle)
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    return fs_meta
        return fs_meta

    def save_day_snapshot(
        self,
        symbol: str,
        snapshot_date: date,
        *,
        chain: pd.DataFrame,
        expiries: list[date],
        raw_by_expiry: dict[date, dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        sym = str(symbol).strip().upper()
        if not sym:
            raise OptionsSnapshotError("symbol required")

        provider = self._provider()

        day_dir = self._day_dir(sym, snapshot_date)
        day_dir.mkdir(parents=True, exist_ok=True)

        chain_path = day_dir / "chain.parquet"
        meta_path = day_dir / "meta.json.gz"
        raw_path = day_dir / "raw.json.gz"

        if chain is None:
            chain = pd.DataFrame()
        if not chain.empty:
            df = chain.copy()
            df.columns = [str(c) for c in df.columns]
            with self.warehouse.transaction() as tx:
                tx.register("tmp_chain", df)
                tx.execute(f"COPY tmp_chain TO '{_sql_quote(str(chain_path))}' (FORMAT PARQUET)")
                tx.unregister("tmp_chain")
        else:
            # Ensure file exists? Prefer not; load_day will handle missing.
            pass

        meta_payload = dict(meta or {})
        meta_payload.setdefault("snapshot_date", snapshot_date.isoformat())
        meta_payload.setdefault("symbol", sym)
        meta_payload.setdefault("provider", provider)
        meta_payload.setdefault("expiries", [d.isoformat() for d in expiries])

        with gzip.open(meta_path, "wt", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, indent=2, default=str)

        if raw_by_expiry:
            raw_payload = {k.isoformat(): v for k, v in raw_by_expiry.items()}
            with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
                json.dump(raw_payload, handle, indent=2, default=str)

        # Upsert header row
        contracts = int(len(chain)) if chain is not None else 0
        now = datetime.now(timezone.utc)

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                DELETE FROM options_snapshot_headers
                WHERE symbol = ? AND snapshot_date = ? AND provider = ?
                """,
                [sym, snapshot_date.isoformat(), provider],
            )
            tx.execute(
                """
                INSERT INTO options_snapshot_headers(
                  symbol, snapshot_date, provider,
                  chain_path, meta_path, raw_path,
                  spot, risk_free_rate, contracts, updated_at, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    sym,
                    snapshot_date.isoformat(),
                    provider,
                    str(chain_path),
                    str(meta_path),
                    str(raw_path) if raw_by_expiry else None,
                    (meta_payload.get("spot") if meta_payload.get("spot") is not None else None),
                    (
                        meta_payload.get("risk_free_rate")
                        if meta_payload.get("risk_free_rate") is not None
                        else None
                    ),
                    contracts,
                    now,
                    json.dumps(meta_payload, default=str),
                ],
            )

        if self.sync_legacy_files:
            # Optional compatibility path for legacy commands/tests that read
            # day-folder CSV/meta artifacts directly.
            self._filesystem_store().save_day_snapshot(
                sym,
                snapshot_date,
                chain=chain,
                expiries=expiries,
                raw_by_expiry=raw_by_expiry,
                meta=meta_payload,
            )

        return chain_path

    def load_day(self, symbol: str, snapshot_date: date) -> pd.DataFrame:
        sym = str(symbol).strip().upper()
        if not sym:
            return pd.DataFrame()
        fs_store = self._filesystem_store()
        fs_day = fs_store.load_day(sym, snapshot_date)
        if not fs_day.empty:
            return _dedupe_contracts(fs_day)

        row = self._select_header_row(sym, snapshot_date)
        if row is None:
            return fs_day
        chain_path_raw = row.get("chain_path")
        if not self._path_under_lake_root(chain_path_raw):
            return fs_day

        chain_path = Path(str(chain_path_raw))
        if not chain_path.exists():
            return fs_day

        try:
            out = self.warehouse.fetch_df(f"SELECT * FROM read_parquet('{_sql_quote(str(chain_path))}')")
        except Exception as exc:  # noqa: BLE001
            try:
                return fs_day
            except Exception:  # noqa: BLE001
                raise OptionsSnapshotError(f"Failed to read Parquet snapshot: {chain_path}") from exc
        if out is None or out.empty:
            return fs_day
        return _dedupe_contracts(out)

__all__ = ["DuckDBOptionsSnapshotStore"]
