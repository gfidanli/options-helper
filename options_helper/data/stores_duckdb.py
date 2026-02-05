from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.data.candles import CandleCacheError, CandleStore
from options_helper.data.derived import DERIVED_COLUMNS, DerivedStoreError
from options_helper.data.journal import (
    JournalReadResult,
    JournalStoreError,
    SignalContext,
    SignalEvent,
    filter_events,
)
from options_helper.data.options_snapshots import OptionsSnapshotError, _dedupe_contracts
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.db.warehouse import DuckDBWarehouse


def _to_utc_naive(ts: object) -> pd.Timestamp | None:
    try:
        out = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(out):
            return None
        return out.tz_convert(None)
    except Exception:  # noqa: BLE001
        try:
            out = pd.to_datetime(ts, errors="coerce")
            if pd.isna(out):
                return None
            if isinstance(out, pd.Timestamp) and out.tzinfo is not None:
                return out.tz_convert(None)
            return out  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            return None


@dataclass(frozen=True)
class DuckDBDerivedStore:
    """DuckDB-backed replacement for DerivedStore (CSV)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def load(self, symbol: str) -> pd.DataFrame:
        sym = str(symbol).strip().upper()
        if not sym:
            return pd.DataFrame(columns=DERIVED_COLUMNS)

        df = self.warehouse.fetch_df(
            """
            SELECT date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike,
                   atm_iv_near, em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d,
                   atm_iv_near_percentile, iv_term_slope
            FROM derived_daily
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            [sym],
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=DERIVED_COLUMNS)

        # Normalize shape/ordering to match DerivedStore.
        df = df.copy()
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        for col in DERIVED_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[DERIVED_COLUMNS].copy()
        return df

    def upsert(self, symbol: str, row: DerivedRow) -> Path:
        sym = str(symbol).strip().upper()
        if not sym:
            raise DerivedStoreError("symbol required")

        payload = row.model_dump()
        date_str = str(payload.get("date") or "").strip()
        if not date_str:
            raise DerivedStoreError("row.date required")

        now = datetime.now(timezone.utc)
        with self.warehouse.transaction() as tx:
            tx.execute("DELETE FROM derived_daily WHERE symbol = ? AND date = ?", [sym, date_str])
            tx.execute(
                """
                INSERT INTO derived_daily(
                  symbol, date, updated_at,
                  spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike,
                  atm_iv_near, em_near_pct, skew_near_pp,
                  rv_20d, rv_60d, iv_rv_20d,
                  atm_iv_near_percentile, iv_term_slope
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    sym,
                    date_str,
                    now,
                    payload.get("spot"),
                    payload.get("pc_oi"),
                    payload.get("pc_vol"),
                    payload.get("call_wall"),
                    payload.get("put_wall"),
                    payload.get("gamma_peak_strike"),
                    payload.get("atm_iv_near"),
                    payload.get("em_near_pct"),
                    payload.get("skew_near_pp"),
                    payload.get("rv_20d"),
                    payload.get("rv_60d"),
                    payload.get("iv_rv_20d"),
                    payload.get("atm_iv_near_percentile"),
                    payload.get("iv_term_slope"),
                ],
            )

        # API compatibility: return a Path like the CSV store.
        return self.warehouse.path


@dataclass(frozen=True)
class DuckDBJournalStore:
    """DuckDB-backed replacement for JournalStore (jsonl)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def path(self) -> Path:
        return self.warehouse.path

    def append_event(self, event: SignalEvent) -> None:
        self.append_events([event])

    def append_events(self, events: Iterable[SignalEvent]) -> int:
        events_list = list(events)
        if not events_list:
            return 0

        now = datetime.now(timezone.utc)
        with self.warehouse.transaction() as tx:
            for event in events_list:
                payload = event.model_dump(mode="json")
                tx.execute(
                    """
                    INSERT INTO signal_events(
                      created_at, date, symbol, context, snapshot_date, contract_symbol, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        now,
                        payload.get("date"),
                        payload.get("symbol"),
                        (payload.get("context") or SignalContext.RESEARCH.value),
                        payload.get("snapshot_date"),
                        payload.get("contract_symbol"),
                        json.dumps(payload.get("payload") or {}, sort_keys=True),
                    ],
                )
        return len(events_list)

    def read_events(self, *, ignore_invalid: bool = True) -> JournalReadResult:
        # Read newest-last for deterministic ordering.
        df = self.warehouse.fetch_df(
            """
            SELECT id, created_at, date, symbol, context, snapshot_date, contract_symbol, payload_json
            FROM signal_events
            ORDER BY id ASC
            """
        )
        if df is None or df.empty:
            return JournalReadResult(events=[], errors=[])

        events: list[SignalEvent] = []
        errors: list[str] = []
        for idx, row in df.iterrows():
            try:
                payload_raw = row.get("payload_json")
                if isinstance(payload_raw, str):
                    payload = json.loads(payload_raw) if payload_raw.strip() else {}
                elif payload_raw is None or (isinstance(payload_raw, float) and pd.isna(payload_raw)):
                    payload = {}
                else:
                    # duckdb may materialize JSON as dict-like already
                    payload = payload_raw  # type: ignore[assignment]

                raw = {
                    "schema_version": 1,
                    "date": row.get("date"),
                    "symbol": row.get("symbol"),
                    "context": row.get("context"),
                    "payload": payload,
                    "snapshot_date": row.get("snapshot_date"),
                    "contract_symbol": row.get("contract_symbol"),
                }
                event = SignalEvent.model_validate(raw)
                events.append(event)
            except Exception as exc:  # noqa: BLE001
                msg = f"row {idx}: invalid event ({exc})"
                if not ignore_invalid:
                    raise JournalStoreError(msg) from exc
                errors.append(msg)

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


@dataclass(frozen=True)
class DuckDBCandleStore(CandleStore):
    """DuckDB-backed CandleStore.

    This is intentionally API-compatible with CandleStore. It uses the same method names and is used
    via the store factory in duckdb mode.
    """

    warehouse: DuckDBWarehouse = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warehouse is None:
            raise ValueError("DuckDBCandleStore requires warehouse")

        # Monkeypatch CandleStore's CSV-based methods with our DuckDB-backed ones.
        object.__setattr__(self, "load", self._load_duckdb)
        object.__setattr__(self, "save", self._save_duckdb)
        object.__setattr__(self, "load_meta", self._load_meta_duckdb)

    def _load_meta_duckdb(self, symbol: str) -> dict | None:
        sym = str(symbol).strip().upper()
        if not sym:
            return None
        df = self.warehouse.fetch_df(
            """
            SELECT updated_at, rows, start_ts, end_ts
            FROM candles_meta
            WHERE symbol = ? AND interval = ? AND auto_adjust = ? AND back_adjust = ?
            """,
            [sym, self.interval, bool(self.auto_adjust), bool(self.back_adjust)],
        )
        if df is None or df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {
            "schema_version": 1,
            "symbol": sym,
            "interval": self.interval,
            "auto_adjust": bool(self.auto_adjust),
            "back_adjust": bool(self.back_adjust),
            "updated_at": str(row.get("updated_at")) if row.get("updated_at") is not None else None,
            "rows": int(row.get("rows") or 0),
            "start": str(row.get("start_ts")) if row.get("start_ts") is not None else None,
            "end": str(row.get("end_ts")) if row.get("end_ts") is not None else None,
            "source": "duckdb",
        }

    def _load_duckdb(self, symbol: str) -> pd.DataFrame:
        sym = str(symbol).strip().upper()
        if not sym:
            return pd.DataFrame()
        df = self.warehouse.fetch_df(
            """
            SELECT ts,
                   open AS "Open",
                   high AS "High",
                   low AS "Low",
                   close AS "Close",
                   volume AS "Volume",
                   dividends AS "Dividends",
                   splits AS "Stock Splits",
                   capital_gains AS "Capital Gains"
            FROM candles_daily
            WHERE symbol = ?
              AND interval = ?
              AND auto_adjust = ?
              AND back_adjust = ?
            ORDER BY ts ASC
            """,
            [sym, self.interval, bool(self.auto_adjust), bool(self.back_adjust)],
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if "ts" not in df.columns:
            return pd.DataFrame()

        ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.loc[~ts.isna()].copy()
        df.index = ts.loc[~ts.isna()].dt.tz_convert(None)
        df = df.drop(columns=["ts"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _save_duckdb(self, symbol: str, history: pd.DataFrame) -> None:
        sym = str(symbol).strip().upper()
        if not sym:
            raise CandleCacheError("symbol required")
        if history is None or history.empty:
            return

        df = history.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise CandleCacheError("history index must be DatetimeIndex")

        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[~df.index.isna()].copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        out = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        out = out.loc[~out["ts"].isna()].copy()

        # Map columns.
        mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
            "Capital Gains": "capital_gains",
        }
        for src, dst in mapping.items():
            if src in out.columns:
                out[dst] = pd.to_numeric(out[src], errors="coerce")
            else:
                out[dst] = pd.Series([None] * len(out))
        out = out[["ts"] + list(mapping.values())]

        now = datetime.now(timezone.utc)
        start_ts = out["ts"].min()
        end_ts = out["ts"].max()

        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                DELETE FROM candles_daily
                WHERE symbol = ? AND interval = ? AND auto_adjust = ? AND back_adjust = ?
                """,
                [sym, self.interval, bool(self.auto_adjust), bool(self.back_adjust)],
            )

            out["symbol"] = sym
            out["interval"] = self.interval
            out["auto_adjust"] = bool(self.auto_adjust)
            out["back_adjust"] = bool(self.back_adjust)

            tx.register("tmp_candles", out)
            tx.execute(
                """
                INSERT INTO candles_daily(
                  symbol, interval, auto_adjust, back_adjust, ts,
                  open, high, low, close, volume, dividends, splits, capital_gains
                )
                SELECT symbol, interval, auto_adjust, back_adjust, ts,
                       open, high, low, close, volume, dividends, splits, capital_gains
                FROM tmp_candles
                """
            )
            tx.unregister("tmp_candles")

            tx.execute(
                """
                DELETE FROM candles_meta
                WHERE symbol = ? AND interval = ? AND auto_adjust = ? AND back_adjust = ?
                """,
                [sym, self.interval, bool(self.auto_adjust), bool(self.back_adjust)],
            )
            tx.execute(
                """
                INSERT INTO candles_meta(symbol, interval, auto_adjust, back_adjust, updated_at, rows, start_ts, end_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    sym,
                    self.interval,
                    bool(self.auto_adjust),
                    bool(self.back_adjust),
                    now,
                    int(len(out)),
                    start_ts,
                    end_ts,
                ],
            )


@dataclass(frozen=True)
class DuckDBOptionsSnapshotStore:
    """DuckDB-backed snapshot store with Parquet lake files."""

    lake_root: Path
    warehouse: DuckDBWarehouse

    def _symbol_dir(self, symbol: str) -> Path:
        return self.lake_root / str(symbol).strip().upper()

    def _day_dir(self, symbol: str, snapshot_date: date) -> Path:
        return self._symbol_dir(symbol) / snapshot_date.isoformat()

    def _provider(self) -> str:
        return get_default_provider_name()

    def list_dates(self, symbol: str) -> list[date]:
        sym = str(symbol).strip().upper()
        if not sym:
            return []
        provider = self._provider()
        df = self.warehouse.fetch_df(
            """
            SELECT snapshot_date
            FROM options_snapshot_headers
            WHERE symbol = ? AND provider = ?
            ORDER BY snapshot_date ASC
            """,
            [sym, provider],
        )
        if df is None or df.empty:
            return []
        out: list[date] = []
        for v in df["snapshot_date"].tolist():
            if isinstance(v, date):
                out.append(v)
            else:
                try:
                    out.append(date.fromisoformat(str(v)))
                except Exception:  # noqa: BLE001
                    continue
        return out

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
        provider = self._provider()
        df = self.warehouse.fetch_df(
            """
            SELECT meta_json, meta_path
            FROM options_snapshot_headers
            WHERE symbol = ? AND snapshot_date = ? AND provider = ?
            """,
            [sym, snapshot_date.isoformat(), provider],
        )
        if df is None or df.empty:
            return {}
        row = df.iloc[0].to_dict()
        meta_json = row.get("meta_json")
        if isinstance(meta_json, str) and meta_json.strip():
            try:
                return json.loads(meta_json)
            except Exception:  # noqa: BLE001
                pass

        meta_path = row.get("meta_path")
        if meta_path:
            p = Path(str(meta_path))
            if p.exists():
                try:
                    if p.suffix.endswith(".gz"):
                        with gzip.open(p, "rt", encoding="utf-8") as handle:
                            return json.load(handle)
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    return {}
        return {}

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

        return chain_path

    def load_day(self, symbol: str, snapshot_date: date) -> pd.DataFrame:
        sym = str(symbol).strip().upper()
        if not sym:
            return pd.DataFrame()
        provider = self._provider()
        df = self.warehouse.fetch_df(
            """
            SELECT chain_path
            FROM options_snapshot_headers
            WHERE symbol = ? AND snapshot_date = ? AND provider = ?
            """,
            [sym, snapshot_date.isoformat(), provider],
        )
        if df is None or df.empty:
            return pd.DataFrame()

        chain_path = Path(str(df.iloc[0]["chain_path"]))
        if not chain_path.exists():
            return pd.DataFrame()

        try:
            out = self.warehouse.fetch_df(f"SELECT * FROM read_parquet('{_sql_quote(str(chain_path))}')")
        except Exception as exc:  # noqa: BLE001
            raise OptionsSnapshotError(f"Failed to read Parquet snapshot: {chain_path}") from exc
        if out is None or out.empty:
            return pd.DataFrame()
        return _dedupe_contracts(out)


def _sql_quote(path: str) -> str:
    return str(path).replace("'", "''")
