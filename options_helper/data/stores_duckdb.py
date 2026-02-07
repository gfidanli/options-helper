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
from options_helper.data.derived import DERIVED_COLUMNS, DerivedStore, DerivedStoreError
from options_helper.data.journal import (
    JournalReadResult,
    JournalStore,
    JournalStoreError,
    SignalContext,
    SignalEvent,
    filter_events,
)
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.data.options_snapshots import OptionsSnapshotError, OptionsSnapshotStore, _dedupe_contracts
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.research_metrics_contracts import (
    EXPOSURE_STRIKE_FIELDS,
    INTRADAY_FLOW_CONTRACT_FIELDS,
    IV_SURFACE_DELTA_BUCKET_FIELDS,
    IV_SURFACE_TENOR_FIELDS,
)


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


def _coerce_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class DuckDBDerivedStore:
    """DuckDB-backed replacement for DerivedStore (CSV)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def _filesystem_store(self) -> DerivedStore:
        return DerivedStore(self.root_dir)

    def load(self, symbol: str) -> pd.DataFrame:
        sym = str(symbol).strip().upper()
        if not sym:
            return pd.DataFrame(columns=DERIVED_COLUMNS)

        fs_store = self._filesystem_store()
        fs_path = self.root_dir / f"{sym}.csv"
        if fs_path.exists():
            return fs_store.load(sym)

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

        # Keep legacy per-symbol CSV artifacts in sync for root_dir-scoped consumers.
        self._filesystem_store().upsert(sym, row)

        # API compatibility: return a Path like the CSV store.
        return self.warehouse.path


@dataclass(frozen=True)
class DuckDBJournalStore:
    """DuckDB-backed replacement for JournalStore (jsonl)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def path(self) -> Path:
        return self.warehouse.path

    def _filesystem_store(self) -> JournalStore:
        return JournalStore(self.root_dir)

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
        # Keep legacy jsonl journal output in sync for filesystem consumers.
        self._filesystem_store().append_events(events_list)
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
            return self._filesystem_store().read_events(ignore_invalid=ignore_invalid)

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
                    "date": _coerce_date(row.get("date")),
                    "symbol": row.get("symbol"),
                    "context": row.get("context"),
                    "payload": payload,
                    "snapshot_date": _coerce_date(row.get("snapshot_date")),
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
            SELECT updated_at, rows, start_ts, end_ts, max_backfill_complete
            FROM candles_meta
            WHERE symbol = ? AND interval = ? AND auto_adjust = ? AND back_adjust = ?
            """,
            [sym, self.interval, bool(self.auto_adjust), bool(self.back_adjust)],
        )
        if df is None or df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {
            "schema_version": 2,
            "symbol": sym,
            "interval": self.interval,
            "auto_adjust": bool(self.auto_adjust),
            "back_adjust": bool(self.back_adjust),
            "updated_at": str(row.get("updated_at")) if row.get("updated_at") is not None else None,
            "rows": int(row.get("rows") or 0),
            "start": str(row.get("start_ts")) if row.get("start_ts") is not None else None,
            "end": str(row.get("end_ts")) if row.get("end_ts") is not None else None,
            "max_backfill_complete": bool(row.get("max_backfill_complete")),
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
                   vwap AS "VWAP",
                   trade_count AS "Trade Count",
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

    def _save_duckdb(
        self,
        symbol: str,
        history: pd.DataFrame,
        *,
        max_backfill_complete: bool | None = None,
    ) -> None:
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
            "VWAP": "vwap",
            "Trade Count": "trade_count",
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
        if max_backfill_complete is None:
            existing_meta = self._load_meta_duckdb(sym)
            effective_max_backfill_complete = bool(
                (existing_meta or {}).get("max_backfill_complete")
            )
        else:
            effective_max_backfill_complete = bool(max_backfill_complete)

        with self.warehouse.transaction() as tx:
            out["symbol"] = sym
            out["interval"] = self.interval
            out["auto_adjust"] = bool(self.auto_adjust)
            out["back_adjust"] = bool(self.back_adjust)

            tx.register("tmp_candles", out)
            tx.execute(
                """
                INSERT INTO candles_daily(
                  symbol, interval, auto_adjust, back_adjust, ts,
                  open, high, low, close, volume, vwap, trade_count, dividends, splits, capital_gains
                )
                SELECT symbol, interval, auto_adjust, back_adjust, ts,
                       open, high, low, close, volume, vwap, trade_count, dividends, splits, capital_gains
                FROM tmp_candles
                ON CONFLICT(symbol, interval, auto_adjust, back_adjust, ts)
                DO UPDATE SET
                  open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low = EXCLUDED.low,
                  close = EXCLUDED.close,
                  volume = EXCLUDED.volume,
                  vwap = EXCLUDED.vwap,
                  trade_count = EXCLUDED.trade_count,
                  dividends = EXCLUDED.dividends,
                  splits = EXCLUDED.splits,
                  capital_gains = EXCLUDED.capital_gains
                """
            )
            tx.unregister("tmp_candles")

            tx.execute(
                """
                INSERT INTO candles_meta(
                  symbol, interval, auto_adjust, back_adjust, updated_at, rows, start_ts, end_ts, max_backfill_complete
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, auto_adjust, back_adjust)
                DO UPDATE SET
                  updated_at = EXCLUDED.updated_at,
                  rows = EXCLUDED.rows,
                  start_ts = EXCLUDED.start_ts,
                  end_ts = EXCLUDED.end_ts,
                  max_backfill_complete = EXCLUDED.max_backfill_complete
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
                    effective_max_backfill_complete,
                ],
            )


@dataclass(frozen=True)
class DuckDBOptionContractsStore:
    """DuckDB-backed option contracts store (dimension + snapshots)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def upsert_contracts(
        self,
        df_contracts: pd.DataFrame,
        *,
        provider: str,
        as_of_date: date,
        raw_by_contract_symbol: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if df_contracts is None or df_contracts.empty:
            return

        provider_name = str(provider or "").strip().lower()
        if not provider_name:
            raise OptionContractsStoreError("provider required")

        as_of = _coerce_date(as_of_date)
        if as_of is None:
            raise OptionContractsStoreError("as_of_date required")

        df = df_contracts.copy()
        if df.empty:
            return
        df.columns = [str(c) for c in df.columns]

        def _get_column(*names: str) -> pd.Series:
            for name in names:
                if name in df.columns:
                    return df[name]
            return pd.Series([None] * len(df), index=df.index)

        def _clean_str(value: Any) -> str | None:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:  # noqa: BLE001
                pass
            text = str(value).strip()
            return text or None

        contract_symbol = _get_column("contractSymbol", "contract_symbol", "symbol", "contract")
        underlying = _get_column("underlying", "underlyingSymbol", "underlying_symbol", "root_symbol")
        expiry = _get_column("expiry", "expiration_date", "expiration", "exp")
        option_type = _get_column("optionType", "option_type", "type")
        strike = _get_column("strike", "strike_price")
        multiplier = _get_column("multiplier")
        open_interest = _get_column("openInterest", "open_interest")
        open_interest_date = _get_column("openInterestDate", "open_interest_date")
        close_price = _get_column("closePrice", "close_price")
        close_price_date = _get_column("closePriceDate", "close_price_date")

        contract_symbol = contract_symbol.map(_clean_str)
        underlying = underlying.map(_clean_str).map(lambda v: v.upper() if v else None)
        option_type = option_type.map(_clean_str).map(lambda v: v.lower() if v else None)

        expiry = pd.to_datetime(expiry, errors="coerce").dt.date
        open_interest_date = pd.to_datetime(open_interest_date, errors="coerce").dt.date
        close_price_date = pd.to_datetime(close_price_date, errors="coerce").dt.date

        strike = pd.to_numeric(strike, errors="coerce")
        close_price = pd.to_numeric(close_price, errors="coerce")
        open_interest = pd.to_numeric(open_interest, errors="coerce").astype("Int64")
        multiplier = pd.to_numeric(multiplier, errors="coerce").astype("Int64")

        normalized = pd.DataFrame(
            {
                "contract_symbol": contract_symbol,
                "underlying": underlying,
                "expiry": expiry,
                "option_type": option_type,
                "strike": strike,
                "multiplier": multiplier,
                "open_interest": open_interest,
                "open_interest_date": open_interest_date,
                "close_price": close_price,
                "close_price_date": close_price_date,
            }
        )
        normalized = normalized.dropna(subset=["contract_symbol"])
        normalized = normalized[normalized["contract_symbol"].map(lambda v: bool(str(v).strip()))]
        if normalized.empty:
            return

        now = datetime.now(timezone.utc)

        contracts_df = normalized[
            ["contract_symbol", "underlying", "expiry", "option_type", "strike", "multiplier"]
        ].copy()
        contracts_df["provider"] = provider_name
        contracts_df["updated_at"] = now

        snapshots_df = normalized[
            ["contract_symbol", "open_interest", "open_interest_date", "close_price", "close_price_date"]
        ].copy()
        snapshots_df["as_of_date"] = as_of
        snapshots_df["provider"] = provider_name
        snapshots_df["updated_at"] = now

        if raw_by_contract_symbol:
            def _raw_payload(symbol: Any) -> str | None:
                if symbol is None:
                    return None
                raw = raw_by_contract_symbol.get(symbol)
                if raw is None:
                    raw = raw_by_contract_symbol.get(str(symbol).upper())
                if raw is None:
                    return None
                try:
                    return json.dumps(raw, default=str)
                except Exception:  # noqa: BLE001
                    return None

            snapshots_df["raw_json"] = snapshots_df["contract_symbol"].map(_raw_payload)
        else:
            snapshots_df["raw_json"] = None

        mask = (
            ~snapshots_df["open_interest"].isna()
            | ~snapshots_df["open_interest_date"].isna()
            | ~snapshots_df["close_price"].isna()
            | ~snapshots_df["close_price_date"].isna()
            | ~snapshots_df["raw_json"].isna()
        )
        snapshots_df = snapshots_df.loc[mask].copy()

        with self.warehouse.transaction() as tx:
            tx.register("tmp_contracts", contracts_df)
            tx.execute(
                """
                INSERT INTO option_contracts(
                  contract_symbol, underlying, expiry, option_type, strike, multiplier, provider, updated_at
                )
                SELECT contract_symbol, underlying, expiry, option_type, strike, multiplier, provider, updated_at
                FROM tmp_contracts
                ON CONFLICT(contract_symbol)
                DO UPDATE SET
                  underlying = EXCLUDED.underlying,
                  expiry = EXCLUDED.expiry,
                  option_type = EXCLUDED.option_type,
                  strike = EXCLUDED.strike,
                  multiplier = EXCLUDED.multiplier,
                  provider = EXCLUDED.provider,
                  updated_at = EXCLUDED.updated_at
                """
            )
            tx.unregister("tmp_contracts")

            if not snapshots_df.empty:
                tx.register("tmp_contract_snapshots", snapshots_df)
                tx.execute(
                    """
                    INSERT INTO option_contract_snapshots(
                      contract_symbol, as_of_date, open_interest, open_interest_date,
                      close_price, close_price_date, provider, updated_at, raw_json
                    )
                    SELECT contract_symbol, as_of_date, open_interest, open_interest_date,
                           close_price, close_price_date, provider, updated_at, raw_json
                    FROM tmp_contract_snapshots
                    ON CONFLICT(contract_symbol, as_of_date, provider)
                    DO UPDATE SET
                      open_interest = EXCLUDED.open_interest,
                      open_interest_date = EXCLUDED.open_interest_date,
                      close_price = EXCLUDED.close_price,
                      close_price_date = EXCLUDED.close_price_date,
                      updated_at = EXCLUDED.updated_at,
                      raw_json = EXCLUDED.raw_json
                    """
                )
                tx.unregister("tmp_contract_snapshots")

    def list_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,
        exp_lte: date | None = None,
    ) -> pd.DataFrame:
        sym = str(underlying).strip().upper()
        cols = ["contractSymbol", "underlying", "expiry", "optionType", "strike", "multiplier", "provider", "updated_at"]
        if not sym:
            return pd.DataFrame(columns=cols)

        gte = _coerce_date(exp_gte)
        lte = _coerce_date(exp_lte)

        sql = """
            SELECT contract_symbol AS "contractSymbol",
                   underlying,
                   expiry,
                   option_type AS "optionType",
                   strike,
                   multiplier,
                   provider,
                   updated_at
            FROM option_contracts
            WHERE underlying = ?
        """
        params: list[Any] = [sym]
        if gte is not None:
            sql += " AND expiry >= ?"
            params.append(gte)
        if lte is not None:
            sql += " AND expiry <= ?"
            params.append(lte)
        sql += " ORDER BY expiry ASC, strike ASC, option_type ASC, contract_symbol ASC"

        df = self.warehouse.fetch_df(sql, params)
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        return df


@dataclass(frozen=True)
class DuckDBOptionBarsStore:
    """DuckDB-backed option bars store (daily bars + meta)."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def _clean_symbol(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        text = str(value).strip()
        if not text:
            return None
        return text.upper()

    def _normalize_provider(self, provider: str) -> str:
        name = str(provider or "").strip().lower()
        if not name:
            raise OptionBarsStoreError("provider required")
        return name

    def _normalize_interval(self, interval: str) -> str:
        name = str(interval or "").strip().lower()
        if not name:
            raise OptionBarsStoreError("interval required")
        return name

    def _normalize_contract_symbols(self, contract_symbols: Iterable[str] | str) -> list[str]:
        if isinstance(contract_symbols, str):
            values = [contract_symbols]
        else:
            values = list(contract_symbols)
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            symbol = self._clean_symbol(value)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            cleaned.append(symbol)
        return cleaned

    def _raise_if_lock_error(self, exc: Exception) -> None:
        msg = str(exc).lower()
        lock_hint = (
            ("lock" in msg or "locked" in msg)
            and ("file" in msg or "database" in msg or "duckdb" in msg or "resource" in msg)
        ) or "resource temporarily unavailable" in msg
        if lock_hint:
            raise OptionBarsStoreError(
                "DuckDB database is locked. Avoid concurrent ingestion and retry."
            ) from exc

    def _normalize_bars_payload(
        self,
        df: pd.DataFrame | None,
        *,
        interval_name: str,
        provider_name: str,
        updated_at: datetime | None = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        table = df.copy()
        if table.empty:
            return pd.DataFrame()
        table.columns = [str(c) for c in table.columns]

        def _get_column(*names: str) -> pd.Series:
            for name in names:
                if name in table.columns:
                    return table[name]
            return pd.Series([None] * len(table), index=table.index)

        contract_symbol = _get_column("contractSymbol", "contract_symbol", "symbol", "contract")
        ts = _get_column("ts", "timestamp", "time")

        contract_symbol = contract_symbol.map(self._clean_symbol)
        ts = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert(None)

        open_px = pd.to_numeric(_get_column("open", "Open"), errors="coerce")
        high_px = pd.to_numeric(_get_column("high", "High"), errors="coerce")
        low_px = pd.to_numeric(_get_column("low", "Low"), errors="coerce")
        close_px = pd.to_numeric(_get_column("close", "Close"), errors="coerce")
        volume = pd.to_numeric(_get_column("volume", "Volume"), errors="coerce")
        vwap = pd.to_numeric(_get_column("vwap", "VWAP"), errors="coerce")
        trade_count = pd.to_numeric(
            _get_column("trade_count", "tradeCount", "Trade Count", "trade_count"),
            errors="coerce",
        ).astype("Int64")

        normalized = pd.DataFrame(
            {
                "contract_symbol": contract_symbol,
                "ts": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": volume,
                "vwap": vwap,
                "trade_count": trade_count,
            }
        )
        normalized = normalized.dropna(subset=["contract_symbol", "ts"])
        normalized = normalized[normalized["contract_symbol"].map(lambda v: bool(str(v).strip()))]
        if normalized.empty:
            return pd.DataFrame()
        normalized = normalized.drop_duplicates(subset=["contract_symbol", "ts"], keep="last")
        if normalized.empty:
            return pd.DataFrame()

        now = updated_at or datetime.now(timezone.utc)
        normalized["interval"] = interval_name
        normalized["provider"] = provider_name
        normalized["updated_at"] = now
        return normalized

    def _upsert_bars_tx(self, tx: Any, normalized: pd.DataFrame) -> None:
        if normalized is None or normalized.empty:
            return
        tx.register("tmp_option_bars", normalized)
        try:
            tx.execute(
                """
                INSERT INTO option_bars(
                  contract_symbol, interval, ts,
                  open, high, low, close, volume, vwap, trade_count,
                  provider, updated_at
                )
                SELECT contract_symbol, interval, ts,
                       open, high, low, close, volume, vwap, trade_count,
                       provider, updated_at
                FROM tmp_option_bars
                ON CONFLICT(contract_symbol, interval, ts, provider)
                DO UPDATE SET
                  open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low = EXCLUDED.low,
                  close = EXCLUDED.close,
                  volume = EXCLUDED.volume,
                  vwap = EXCLUDED.vwap,
                  trade_count = EXCLUDED.trade_count,
                  updated_at = EXCLUDED.updated_at
                """
            )
        finally:
            tx.unregister("tmp_option_bars")

    def _normalize_symbol_map(self, value: object | dict[str, object] | None) -> dict[str, object] | None:
        if not isinstance(value, dict):
            return None
        out: dict[str, object] = {}
        for key, item in value.items():
            sym = self._clean_symbol(key)
            if sym:
                out[sym] = item
        return out

    def _build_meta_success_frame(
        self,
        symbols: list[str],
        *,
        interval_name: str,
        provider_name: str,
        status: str = "ok",
        rows: int | dict[str, int] | None = None,
        start_ts: object | dict[str, object] | None = None,
        end_ts: object | dict[str, object] | None = None,
        last_success_at: datetime | None = None,
        last_attempt_at: datetime | None = None,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        status_value = str(status or "ok").strip().lower() or "ok"
        rows_map = self._normalize_symbol_map(rows)
        start_map = self._normalize_symbol_map(start_ts)
        end_map = self._normalize_symbol_map(end_ts)

        now = datetime.now(timezone.utc)
        success_at = last_success_at or now
        attempt_at = last_attempt_at or success_at

        meta_rows: list[dict[str, object]] = []
        for sym in symbols:
            row_value = rows_map.get(sym) if rows_map is not None else rows
            start_value = start_map.get(sym) if start_map is not None else start_ts
            end_value = end_map.get(sym) if end_map is not None else end_ts
            meta_rows.append(
                {
                    "contract_symbol": sym,
                    "interval": interval_name,
                    "provider": provider_name,
                    "status": status_value,
                    "rows": int(row_value) if row_value is not None else 0,
                    "start_ts": _to_utc_naive(start_value),
                    "end_ts": _to_utc_naive(end_value),
                    "last_success_at": success_at,
                    "last_attempt_at": attempt_at,
                    "last_error": None,
                    "error_count": 0,
                }
            )
        return pd.DataFrame(meta_rows)

    def _upsert_meta_success_tx(self, tx: Any, meta_df: pd.DataFrame) -> None:
        if meta_df is None or meta_df.empty:
            return
        tx.register("tmp_option_bars_meta", meta_df)
        try:
            tx.execute(
                """
                UPDATE option_bars_meta AS meta
                SET status = tmp.status,
                    rows = greatest(coalesce(meta.rows, 0), coalesce(tmp.rows, 0)),
                    start_ts = CASE
                      WHEN meta.start_ts IS NULL THEN tmp.start_ts
                      WHEN tmp.start_ts IS NULL THEN meta.start_ts
                      ELSE least(meta.start_ts, tmp.start_ts)
                    END,
                    end_ts = CASE
                      WHEN meta.end_ts IS NULL THEN tmp.end_ts
                      WHEN tmp.end_ts IS NULL THEN meta.end_ts
                      ELSE greatest(meta.end_ts, tmp.end_ts)
                    END,
                    last_success_at = tmp.last_success_at,
                    last_attempt_at = tmp.last_attempt_at,
                    last_error = NULL,
                    error_count = 0
                FROM tmp_option_bars_meta AS tmp
                WHERE meta.contract_symbol = tmp.contract_symbol
                  AND meta.interval = tmp.interval
                  AND meta.provider = tmp.provider
                """
            )
            tx.execute(
                """
                INSERT INTO option_bars_meta(
                  contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                  last_success_at, last_attempt_at, last_error, error_count
                )
                SELECT tmp.contract_symbol, tmp.interval, tmp.provider, tmp.status, tmp.rows,
                       tmp.start_ts, tmp.end_ts, tmp.last_success_at, tmp.last_attempt_at,
                       tmp.last_error, tmp.error_count
                FROM tmp_option_bars_meta AS tmp
                LEFT JOIN option_bars_meta AS meta
                  ON meta.contract_symbol = tmp.contract_symbol
                 AND meta.interval = tmp.interval
                 AND meta.provider = tmp.provider
                WHERE meta.contract_symbol IS NULL
                """
            )
        finally:
            tx.unregister("tmp_option_bars_meta")

    def _build_meta_error_frame(
        self,
        symbols: list[str],
        *,
        interval_name: str,
        provider_name: str,
        error: str | None = None,
        status: str = "error",
        last_attempt_at: datetime | None = None,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        status_value = str(status or "error").strip().lower() or "error"
        attempt_at = last_attempt_at or datetime.now(timezone.utc)
        error_text = str(error) if error else None

        return pd.DataFrame(
            {
                "contract_symbol": symbols,
                "interval": interval_name,
                "provider": provider_name,
                "status": status_value,
                "rows": 0,
                "start_ts": None,
                "end_ts": None,
                "last_success_at": None,
                "last_attempt_at": attempt_at,
                "last_error": error_text,
                "error_count": 1,
            }
        )

    def _upsert_meta_error_tx(self, tx: Any, meta_df: pd.DataFrame) -> None:
        if meta_df is None or meta_df.empty:
            return
        tx.register("tmp_option_bars_meta", meta_df)
        try:
            tx.execute(
                """
                UPDATE option_bars_meta
                SET status = tmp.status,
                    last_attempt_at = tmp.last_attempt_at,
                    last_error = tmp.last_error,
                    error_count = option_bars_meta.error_count + tmp.error_count
                FROM tmp_option_bars_meta AS tmp
                WHERE option_bars_meta.contract_symbol = tmp.contract_symbol
                  AND option_bars_meta.interval = tmp.interval
                  AND option_bars_meta.provider = tmp.provider
                """
            )
            tx.execute(
                """
                INSERT INTO option_bars_meta(
                  contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                  last_success_at, last_attempt_at, last_error, error_count
                )
                SELECT tmp.contract_symbol, tmp.interval, tmp.provider, tmp.status, tmp.rows,
                       tmp.start_ts, tmp.end_ts, tmp.last_success_at, tmp.last_attempt_at,
                       tmp.last_error, tmp.error_count
                FROM tmp_option_bars_meta AS tmp
                LEFT JOIN option_bars_meta AS meta
                  ON meta.contract_symbol = tmp.contract_symbol
                 AND meta.interval = tmp.interval
                 AND meta.provider = tmp.provider
                WHERE meta.contract_symbol IS NULL
                """
            )
        finally:
            tx.unregister("tmp_option_bars_meta")

    def upsert_bars(
        self,
        df: pd.DataFrame,
        *,
        interval: str = "1d",
        provider: str,
        updated_at: datetime | None = None,
    ) -> None:
        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)
        normalized = self._normalize_bars_payload(
            df,
            interval_name=interval_name,
            provider_name=provider_name,
            updated_at=updated_at,
        )
        if normalized.empty:
            return

        try:
            with self.warehouse.transaction() as tx:
                self._upsert_bars_tx(tx, normalized)
        except Exception as exc:  # noqa: BLE001
            self._raise_if_lock_error(exc)
            raise

    def mark_meta_success(
        self,
        contract_symbols: Iterable[str] | str,
        interval: str,
        provider: str,
        *,
        status: str = "ok",
        rows: int | dict[str, int] | None = None,
        start_ts: object | dict[str, object] | None = None,
        end_ts: object | dict[str, object] | None = None,
        last_success_at: datetime | None = None,
        last_attempt_at: datetime | None = None,
    ) -> None:
        symbols = self._normalize_contract_symbols(contract_symbols)
        if not symbols:
            return

        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)
        meta_df = self._build_meta_success_frame(
            symbols,
            interval_name=interval_name,
            provider_name=provider_name,
            status=status,
            rows=rows,
            start_ts=start_ts,
            end_ts=end_ts,
            last_success_at=last_success_at,
            last_attempt_at=last_attempt_at,
        )
        if meta_df.empty:
            return

        try:
            with self.warehouse.transaction() as tx:
                self._upsert_meta_success_tx(tx, meta_df)
        except Exception as exc:  # noqa: BLE001
            self._raise_if_lock_error(exc)
            raise

    def mark_meta_error(
        self,
        contract_symbols: Iterable[str] | str,
        interval: str,
        provider: str,
        *,
        error: str | None = None,
        status: str = "error",
    ) -> None:
        symbols = self._normalize_contract_symbols(contract_symbols)
        if not symbols:
            return

        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)
        meta_df = self._build_meta_error_frame(
            symbols,
            interval_name=interval_name,
            provider_name=provider_name,
            error=error,
            status=status,
        )
        if meta_df.empty:
            return

        try:
            with self.warehouse.transaction() as tx:
                self._upsert_meta_error_tx(tx, meta_df)
        except Exception as exc:  # noqa: BLE001
            self._raise_if_lock_error(exc)
            raise

    def apply_write_batch(
        self,
        *,
        bars_df: pd.DataFrame | None,
        interval: str = "1d",
        provider: str,
        success_symbols: Iterable[str] | str | None = None,
        success_rows: int | dict[str, int] | None = None,
        success_start_ts: object | dict[str, object] | None = None,
        success_end_ts: object | dict[str, object] | None = None,
        error_groups: Iterable[tuple[Iterable[str] | str, str, str]] | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)

        normalized_bars = self._normalize_bars_payload(
            bars_df,
            interval_name=interval_name,
            provider_name=provider_name,
            updated_at=updated_at,
        )

        success_symbol_list = self._normalize_contract_symbols(success_symbols or [])
        success_meta_df = self._build_meta_success_frame(
            success_symbol_list,
            interval_name=interval_name,
            provider_name=provider_name,
            status="ok",
            rows=success_rows,
            start_ts=success_start_ts,
            end_ts=success_end_ts,
        )

        error_frames: list[pd.DataFrame] = []
        for symbols, status, error_text in error_groups or []:
            error_symbols = self._normalize_contract_symbols(symbols)
            if not error_symbols:
                continue
            frame = self._build_meta_error_frame(
                error_symbols,
                interval_name=interval_name,
                provider_name=provider_name,
                error=error_text,
                status=status,
            )
            if frame is not None and not frame.empty:
                error_frames.append(frame)

        error_meta_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
        if not error_meta_df.empty:
            error_meta_df = error_meta_df.drop_duplicates(
                subset=["contract_symbol", "interval", "provider"],
                keep="last",
            )

        if normalized_bars.empty and success_meta_df.empty and error_meta_df.empty:
            return

        try:
            with self.warehouse.transaction() as tx:
                self._upsert_bars_tx(tx, normalized_bars)
                self._upsert_meta_success_tx(tx, success_meta_df)
                self._upsert_meta_error_tx(tx, error_meta_df)
        except Exception as exc:  # noqa: BLE001
            self._raise_if_lock_error(exc)
            raise

    def coverage(
        self,
        contract_symbol: str,
        *,
        interval: str = "1d",
        provider: str,
    ) -> dict[str, Any]:
        symbol = self._clean_symbol(contract_symbol)
        if not symbol:
            return {}
        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)

        df = self.warehouse.fetch_df(
            """
            SELECT contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                   last_success_at, last_attempt_at, last_error, error_count
            FROM option_bars_meta
            WHERE contract_symbol = ? AND interval = ? AND provider = ?
            """,
            [symbol, interval_name, provider_name],
        )
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def coverage_bulk(
        self,
        contract_symbols: Iterable[str],
        *,
        interval: str = "1d",
        provider: str,
    ) -> dict[str, dict[str, Any]]:
        symbols = self._normalize_contract_symbols(contract_symbols)
        if not symbols:
            return {}
        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)
        chunk_size = 2000
        out: dict[str, dict[str, Any]] = {}

        conn = self.warehouse.connect(read_only=True)
        try:
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i : i + chunk_size]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                sql = f"""
                    SELECT contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                           last_success_at, last_attempt_at, last_error, error_count
                    FROM option_bars_meta
                    WHERE interval = ? AND provider = ? AND contract_symbol IN ({placeholders})
                """
                df = conn.execute(sql, [interval_name, provider_name, *chunk]).df()
                if df is None or df.empty:
                    continue
                for row in df.to_dict("records"):
                    symbol = self._clean_symbol(row.get("contract_symbol"))
                    if symbol:
                        out[symbol] = row
        finally:
            conn.close()
        return out

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


_IV_SURFACE_TENOR_LOAD_COLUMNS = tuple(IV_SURFACE_TENOR_FIELDS) + ("provider", "updated_at")
_IV_SURFACE_DELTA_BUCKET_LOAD_COLUMNS = tuple(IV_SURFACE_DELTA_BUCKET_FIELDS) + ("provider", "updated_at")
_DEALER_EXPOSURE_STRIKES_LOAD_COLUMNS = tuple(EXPOSURE_STRIKE_FIELDS) + ("provider", "updated_at")
_INTRADAY_OPTION_FLOW_LOAD_COLUMNS = tuple(INTRADAY_FLOW_CONTRACT_FIELDS) + ("provider", "updated_at")


@dataclass(frozen=True)
class DuckDBResearchMetricsStore:
    """DuckDB-backed persistence for research metrics tables."""

    root_dir: Path
    warehouse: DuckDBWarehouse

    def _normalize_provider(self, provider: str | None) -> str:
        default_provider = get_default_provider_name()
        raw = provider if provider is not None else default_provider
        cleaned = str(raw or "").strip().lower()
        if not cleaned:
            cleaned = str(default_provider).strip().lower()
        if not cleaned:
            raise ValueError("provider required")
        return cleaned

    def _clean_symbol(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        cleaned = str(value).strip().upper()
        return cleaned or None

    def _clean_text(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        cleaned = str(value).strip()
        return cleaned or None

    def _clean_option_type(self, value: Any) -> str | None:
        cleaned = self._clean_text(value)
        return cleaned.lower() if cleaned else None

    def _warnings_to_json(self, value: Any) -> str:
        values: list[str]
        if value is None:
            values = []
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                values = []
            else:
                try:
                    parsed = json.loads(text)
                except Exception:  # noqa: BLE001
                    values = [text]
                else:
                    if isinstance(parsed, list):
                        values = [str(item).strip() for item in parsed if str(item).strip()]
                    elif parsed is None:
                        values = []
                    else:
                        parsed_text = str(parsed).strip()
                        values = [parsed_text] if parsed_text else []
        elif isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            values = [text] if text else []
        return json.dumps(values, ensure_ascii=True)

    def _warnings_from_json(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001
            return [text]
        if not isinstance(parsed, list):
            parsed_text = str(parsed).strip() if parsed is not None else ""
            return [parsed_text] if parsed_text else []
        return [str(item).strip() for item in parsed if str(item).strip()]

    def _coerce_numeric(self, column: pd.Series) -> pd.Series:
        return pd.to_numeric(column, errors="coerce")

    def _coerce_int(self, column: pd.Series) -> pd.Series:
        return pd.to_numeric(column, errors="coerce").astype("Int64")

    def _coerce_iso_date(self, column: pd.Series) -> pd.Series:
        return pd.to_datetime(column, errors="coerce").dt.date

    def _build_input_frame(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        table = df.copy()
        table.columns = [str(c) for c in table.columns]
        return table

    def _column_or_null(self, table: pd.DataFrame, *names: str) -> pd.Series:
        for name in names:
            if name in table.columns:
                return table[name]
        return pd.Series([None] * len(table), index=table.index)

    def _resolve_latest_date(
        self,
        *,
        table: str,
        date_column: str,
        symbol_column: str,
        symbol: str,
        provider: str,
    ) -> date | None:
        df = self.warehouse.fetch_df(
            f"""
            SELECT max({date_column}) AS max_date
            FROM {table}
            WHERE {symbol_column} = ? AND provider = ?
            """,
            [symbol, provider],
        )
        if df is None or df.empty:
            return None
        return _coerce_date(df.iloc[0].get("max_date"))

    def _normalize_iv_surface_tenor(self, df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
        table = self._build_input_frame(df)
        if table.empty:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "symbol": self._column_or_null(table, "symbol").map(self._clean_symbol),
                "as_of": self._coerce_iso_date(self._column_or_null(table, "as_of")),
                "tenor_target_dte": self._coerce_int(
                    self._column_or_null(table, "tenor_target_dte")
                ),
                "expiry": self._coerce_iso_date(self._column_or_null(table, "expiry")),
                "dte": self._coerce_int(self._column_or_null(table, "dte")),
                "tenor_gap_dte": self._coerce_int(self._column_or_null(table, "tenor_gap_dte")),
                "atm_strike": self._coerce_numeric(self._column_or_null(table, "atm_strike")),
                "atm_iv": self._coerce_numeric(self._column_or_null(table, "atm_iv")),
                "atm_mark": self._coerce_numeric(self._column_or_null(table, "atm_mark")),
                "straddle_mark": self._coerce_numeric(self._column_or_null(table, "straddle_mark")),
                "expected_move_pct": self._coerce_numeric(
                    self._column_or_null(table, "expected_move_pct")
                ),
                "skew_25d_pp": self._coerce_numeric(self._column_or_null(table, "skew_25d_pp")),
                "skew_10d_pp": self._coerce_numeric(self._column_or_null(table, "skew_10d_pp")),
                "contracts_used": self._coerce_int(self._column_or_null(table, "contracts_used")),
                "warnings_json": self._column_or_null(table, "warnings_json", "warnings").map(
                    self._warnings_to_json
                ),
                "provider": provider,
            }
        )
        out = out.dropna(subset=["symbol", "as_of", "tenor_target_dte"]).copy()
        if out.empty:
            return pd.DataFrame()
        out["tenor_target_dte"] = out["tenor_target_dte"].astype(int)
        out = out.drop_duplicates(subset=["symbol", "as_of", "tenor_target_dte", "provider"], keep="last")
        return out

    def upsert_iv_surface_tenor(self, df: pd.DataFrame, *, provider: str | None = None) -> int:
        provider_name = self._normalize_provider(provider)
        normalized = self._normalize_iv_surface_tenor(df, provider=provider_name)
        if normalized.empty:
            return 0
        normalized = normalized.copy()
        normalized["updated_at"] = datetime.now(timezone.utc)

        with self.warehouse.transaction() as tx:
            tx.register("tmp_iv_surface_tenor", normalized)
            tx.execute(
                """
                INSERT INTO iv_surface_tenor(
                  symbol, as_of, tenor_target_dte, expiry, dte, tenor_gap_dte,
                  atm_strike, atm_iv, atm_mark, straddle_mark, expected_move_pct,
                  skew_25d_pp, skew_10d_pp, contracts_used, warnings_json, provider, updated_at
                )
                SELECT symbol, as_of, tenor_target_dte, expiry, dte, tenor_gap_dte,
                       atm_strike, atm_iv, atm_mark, straddle_mark, expected_move_pct,
                       skew_25d_pp, skew_10d_pp, contracts_used, warnings_json, provider, updated_at
                FROM tmp_iv_surface_tenor
                ON CONFLICT(symbol, as_of, tenor_target_dte, provider)
                DO UPDATE SET
                  expiry = EXCLUDED.expiry,
                  dte = EXCLUDED.dte,
                  tenor_gap_dte = EXCLUDED.tenor_gap_dte,
                  atm_strike = EXCLUDED.atm_strike,
                  atm_iv = EXCLUDED.atm_iv,
                  atm_mark = EXCLUDED.atm_mark,
                  straddle_mark = EXCLUDED.straddle_mark,
                  expected_move_pct = EXCLUDED.expected_move_pct,
                  skew_25d_pp = EXCLUDED.skew_25d_pp,
                  skew_10d_pp = EXCLUDED.skew_10d_pp,
                  contracts_used = EXCLUDED.contracts_used,
                  warnings_json = EXCLUDED.warnings_json,
                  updated_at = EXCLUDED.updated_at
                """
            )
            tx.unregister("tmp_iv_surface_tenor")
        return int(len(normalized))

    def load_iv_surface_tenor(
        self,
        *,
        symbol: str,
        as_of: date | str | None = None,
        provider: str | None = None,
    ) -> pd.DataFrame:
        symbol_norm = self._clean_symbol(symbol)
        if symbol_norm is None:
            return pd.DataFrame(columns=_IV_SURFACE_TENOR_LOAD_COLUMNS)
        provider_name = self._normalize_provider(provider)
        as_of_date = _coerce_date(as_of) if as_of is not None else self._resolve_latest_date(
            table="iv_surface_tenor",
            date_column="as_of",
            symbol_column="symbol",
            symbol=symbol_norm,
            provider=provider_name,
        )
        if as_of_date is None:
            return pd.DataFrame(columns=_IV_SURFACE_TENOR_LOAD_COLUMNS)

        df = self.warehouse.fetch_df(
            """
            SELECT symbol, as_of, tenor_target_dte, expiry, dte, tenor_gap_dte,
                   atm_strike, atm_iv, atm_mark, straddle_mark, expected_move_pct,
                   skew_25d_pp, skew_10d_pp, contracts_used, warnings_json, provider, updated_at
            FROM iv_surface_tenor
            WHERE symbol = ? AND as_of = ? AND provider = ?
            ORDER BY tenor_target_dte ASC
            """,
            [symbol_norm, as_of_date, provider_name],
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=_IV_SURFACE_TENOR_LOAD_COLUMNS)
        out = df.copy()
        out["as_of"] = out["as_of"].astype(str)
        out["expiry"] = out["expiry"].map(lambda value: value.isoformat() if isinstance(value, date) else None)
        out["warnings"] = out["warnings_json"].map(self._warnings_from_json)
        out = out.drop(columns=["warnings_json"])
        return out[list(_IV_SURFACE_TENOR_LOAD_COLUMNS)].copy()

    def _normalize_iv_surface_delta_buckets(self, df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
        table = self._build_input_frame(df)
        if table.empty:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "symbol": self._column_or_null(table, "symbol").map(self._clean_symbol),
                "as_of": self._coerce_iso_date(self._column_or_null(table, "as_of")),
                "tenor_target_dte": self._coerce_int(
                    self._column_or_null(table, "tenor_target_dte")
                ),
                "expiry": self._coerce_iso_date(self._column_or_null(table, "expiry")),
                "option_type": self._column_or_null(table, "option_type").map(self._clean_option_type),
                "delta_bucket": self._column_or_null(table, "delta_bucket").map(self._clean_text),
                "avg_iv": self._coerce_numeric(self._column_or_null(table, "avg_iv")),
                "median_iv": self._coerce_numeric(self._column_or_null(table, "median_iv")),
                "n_contracts": self._coerce_int(self._column_or_null(table, "n_contracts")),
                "warnings_json": self._column_or_null(table, "warnings_json", "warnings").map(
                    self._warnings_to_json
                ),
                "provider": provider,
            }
        )
        out = out.dropna(
            subset=["symbol", "as_of", "tenor_target_dte", "option_type", "delta_bucket"]
        ).copy()
        if out.empty:
            return pd.DataFrame()
        out["tenor_target_dte"] = out["tenor_target_dte"].astype(int)
        out = out.drop_duplicates(
            subset=["symbol", "as_of", "tenor_target_dte", "option_type", "delta_bucket", "provider"],
            keep="last",
        )
        return out

    def upsert_iv_surface_delta_buckets(self, df: pd.DataFrame, *, provider: str | None = None) -> int:
        provider_name = self._normalize_provider(provider)
        normalized = self._normalize_iv_surface_delta_buckets(df, provider=provider_name)
        if normalized.empty:
            return 0
        normalized = normalized.copy()
        normalized["updated_at"] = datetime.now(timezone.utc)

        with self.warehouse.transaction() as tx:
            tx.register("tmp_iv_surface_delta", normalized)
            tx.execute(
                """
                INSERT INTO iv_surface_delta_buckets(
                  symbol, as_of, tenor_target_dte, expiry, option_type, delta_bucket,
                  avg_iv, median_iv, n_contracts, warnings_json, provider, updated_at
                )
                SELECT symbol, as_of, tenor_target_dte, expiry, option_type, delta_bucket,
                       avg_iv, median_iv, n_contracts, warnings_json, provider, updated_at
                FROM tmp_iv_surface_delta
                ON CONFLICT(symbol, as_of, tenor_target_dte, option_type, delta_bucket, provider)
                DO UPDATE SET
                  expiry = EXCLUDED.expiry,
                  avg_iv = EXCLUDED.avg_iv,
                  median_iv = EXCLUDED.median_iv,
                  n_contracts = EXCLUDED.n_contracts,
                  warnings_json = EXCLUDED.warnings_json,
                  updated_at = EXCLUDED.updated_at
                """
            )
            tx.unregister("tmp_iv_surface_delta")
        return int(len(normalized))

    def load_iv_surface_delta_buckets(
        self,
        *,
        symbol: str,
        as_of: date | str | None = None,
        provider: str | None = None,
    ) -> pd.DataFrame:
        symbol_norm = self._clean_symbol(symbol)
        if symbol_norm is None:
            return pd.DataFrame(columns=_IV_SURFACE_DELTA_BUCKET_LOAD_COLUMNS)
        provider_name = self._normalize_provider(provider)
        as_of_date = _coerce_date(as_of) if as_of is not None else self._resolve_latest_date(
            table="iv_surface_delta_buckets",
            date_column="as_of",
            symbol_column="symbol",
            symbol=symbol_norm,
            provider=provider_name,
        )
        if as_of_date is None:
            return pd.DataFrame(columns=_IV_SURFACE_DELTA_BUCKET_LOAD_COLUMNS)

        df = self.warehouse.fetch_df(
            """
            SELECT symbol, as_of, tenor_target_dte, expiry, option_type, delta_bucket,
                   avg_iv, median_iv, n_contracts, warnings_json, provider, updated_at
            FROM iv_surface_delta_buckets
            WHERE symbol = ? AND as_of = ? AND provider = ?
            ORDER BY tenor_target_dte ASC, option_type ASC, delta_bucket ASC
            """,
            [symbol_norm, as_of_date, provider_name],
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=_IV_SURFACE_DELTA_BUCKET_LOAD_COLUMNS)
        out = df.copy()
        out["as_of"] = out["as_of"].astype(str)
        out["expiry"] = out["expiry"].map(lambda value: value.isoformat() if isinstance(value, date) else None)
        out["warnings"] = out["warnings_json"].map(self._warnings_from_json)
        out = out.drop(columns=["warnings_json"])
        return out[list(_IV_SURFACE_DELTA_BUCKET_LOAD_COLUMNS)].copy()

    def _normalize_dealer_exposure_strikes(self, df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
        table = self._build_input_frame(df)
        if table.empty:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "symbol": self._column_or_null(table, "symbol").map(self._clean_symbol),
                "as_of": self._coerce_iso_date(self._column_or_null(table, "as_of")),
                "expiry": self._coerce_iso_date(self._column_or_null(table, "expiry")),
                "strike": self._coerce_numeric(self._column_or_null(table, "strike")),
                "call_oi": self._coerce_numeric(self._column_or_null(table, "call_oi")),
                "put_oi": self._coerce_numeric(self._column_or_null(table, "put_oi")),
                "call_gex": self._coerce_numeric(self._column_or_null(table, "call_gex")),
                "put_gex": self._coerce_numeric(self._column_or_null(table, "put_gex")),
                "net_gex": self._coerce_numeric(self._column_or_null(table, "net_gex")),
                "provider": provider,
            }
        )
        out = out.dropna(subset=["symbol", "as_of", "expiry", "strike"]).copy()
        if out.empty:
            return pd.DataFrame()
        out = out.drop_duplicates(subset=["symbol", "as_of", "expiry", "strike", "provider"], keep="last")
        return out

    def upsert_dealer_exposure_strikes(self, df: pd.DataFrame, *, provider: str | None = None) -> int:
        provider_name = self._normalize_provider(provider)
        normalized = self._normalize_dealer_exposure_strikes(df, provider=provider_name)
        if normalized.empty:
            return 0
        normalized = normalized.copy()
        normalized["updated_at"] = datetime.now(timezone.utc)

        with self.warehouse.transaction() as tx:
            tx.register("tmp_dealer_exposure_strikes", normalized)
            tx.execute(
                """
                INSERT INTO dealer_exposure_strikes(
                  symbol, as_of, expiry, strike, call_oi, put_oi, call_gex,
                  put_gex, net_gex, provider, updated_at
                )
                SELECT symbol, as_of, expiry, strike, call_oi, put_oi, call_gex,
                       put_gex, net_gex, provider, updated_at
                FROM tmp_dealer_exposure_strikes
                ON CONFLICT(symbol, as_of, expiry, strike, provider)
                DO UPDATE SET
                  call_oi = EXCLUDED.call_oi,
                  put_oi = EXCLUDED.put_oi,
                  call_gex = EXCLUDED.call_gex,
                  put_gex = EXCLUDED.put_gex,
                  net_gex = EXCLUDED.net_gex,
                  updated_at = EXCLUDED.updated_at
                """
            )
            tx.unregister("tmp_dealer_exposure_strikes")
        return int(len(normalized))

    def load_dealer_exposure_strikes(
        self,
        *,
        symbol: str,
        as_of: date | str | None = None,
        provider: str | None = None,
    ) -> pd.DataFrame:
        symbol_norm = self._clean_symbol(symbol)
        if symbol_norm is None:
            return pd.DataFrame(columns=_DEALER_EXPOSURE_STRIKES_LOAD_COLUMNS)
        provider_name = self._normalize_provider(provider)
        as_of_date = _coerce_date(as_of) if as_of is not None else self._resolve_latest_date(
            table="dealer_exposure_strikes",
            date_column="as_of",
            symbol_column="symbol",
            symbol=symbol_norm,
            provider=provider_name,
        )
        if as_of_date is None:
            return pd.DataFrame(columns=_DEALER_EXPOSURE_STRIKES_LOAD_COLUMNS)

        df = self.warehouse.fetch_df(
            """
            SELECT symbol, as_of, expiry, strike, call_oi, put_oi, call_gex,
                   put_gex, net_gex, provider, updated_at
            FROM dealer_exposure_strikes
            WHERE symbol = ? AND as_of = ? AND provider = ?
            ORDER BY expiry ASC, strike ASC
            """,
            [symbol_norm, as_of_date, provider_name],
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=_DEALER_EXPOSURE_STRIKES_LOAD_COLUMNS)
        out = df.copy()
        out["as_of"] = out["as_of"].astype(str)
        out["expiry"] = out["expiry"].map(lambda value: value.isoformat() if isinstance(value, date) else None)
        return out[list(_DEALER_EXPOSURE_STRIKES_LOAD_COLUMNS)].copy()

    def _normalize_intraday_option_flow(self, df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
        table = self._build_input_frame(df)
        if table.empty:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "symbol": self._column_or_null(table, "symbol").map(self._clean_symbol),
                "market_date": self._coerce_iso_date(self._column_or_null(table, "market_date")),
                "source": self._column_or_null(table, "source").map(self._clean_text),
                "contract_symbol": self._column_or_null(table, "contract_symbol").map(self._clean_symbol),
                "expiry": self._coerce_iso_date(self._column_or_null(table, "expiry")),
                "option_type": self._column_or_null(table, "option_type").map(self._clean_option_type),
                "strike": self._coerce_numeric(self._column_or_null(table, "strike")),
                "delta_bucket": self._column_or_null(table, "delta_bucket").map(self._clean_text),
                "buy_volume": self._coerce_numeric(self._column_or_null(table, "buy_volume")),
                "sell_volume": self._coerce_numeric(self._column_or_null(table, "sell_volume")),
                "unknown_volume": self._coerce_numeric(self._column_or_null(table, "unknown_volume")),
                "buy_notional": self._coerce_numeric(self._column_or_null(table, "buy_notional")),
                "sell_notional": self._coerce_numeric(self._column_or_null(table, "sell_notional")),
                "net_notional": self._coerce_numeric(self._column_or_null(table, "net_notional")),
                "trade_count": self._coerce_int(self._column_or_null(table, "trade_count")),
                "unknown_trade_share": self._coerce_numeric(
                    self._column_or_null(table, "unknown_trade_share")
                ),
                "quote_coverage_pct": self._coerce_numeric(
                    self._column_or_null(table, "quote_coverage_pct")
                ),
                "warnings_json": self._column_or_null(table, "warnings_json", "warnings").map(
                    self._warnings_to_json
                ),
                "provider": provider,
            }
        )
        out = out.dropna(subset=["symbol", "market_date", "source", "contract_symbol"]).copy()
        if out.empty:
            return pd.DataFrame()
        out = out.drop_duplicates(
            subset=["symbol", "market_date", "source", "contract_symbol", "provider"],
            keep="last",
        )
        return out

    def upsert_intraday_option_flow(self, df: pd.DataFrame, *, provider: str | None = None) -> int:
        provider_name = self._normalize_provider(provider)
        normalized = self._normalize_intraday_option_flow(df, provider=provider_name)
        if normalized.empty:
            return 0
        normalized = normalized.copy()
        normalized["updated_at"] = datetime.now(timezone.utc)

        with self.warehouse.transaction() as tx:
            tx.register("tmp_intraday_option_flow", normalized)
            tx.execute(
                """
                INSERT INTO intraday_option_flow(
                  symbol, market_date, source, contract_symbol, expiry, option_type, strike, delta_bucket,
                  buy_volume, sell_volume, unknown_volume, buy_notional, sell_notional, net_notional,
                  trade_count, unknown_trade_share, quote_coverage_pct, warnings_json, provider, updated_at
                )
                SELECT symbol, market_date, source, contract_symbol, expiry, option_type, strike, delta_bucket,
                       buy_volume, sell_volume, unknown_volume, buy_notional, sell_notional, net_notional,
                       trade_count, unknown_trade_share, quote_coverage_pct, warnings_json, provider, updated_at
                FROM tmp_intraday_option_flow
                ON CONFLICT(symbol, market_date, source, contract_symbol, provider)
                DO UPDATE SET
                  expiry = EXCLUDED.expiry,
                  option_type = EXCLUDED.option_type,
                  strike = EXCLUDED.strike,
                  delta_bucket = EXCLUDED.delta_bucket,
                  buy_volume = EXCLUDED.buy_volume,
                  sell_volume = EXCLUDED.sell_volume,
                  unknown_volume = EXCLUDED.unknown_volume,
                  buy_notional = EXCLUDED.buy_notional,
                  sell_notional = EXCLUDED.sell_notional,
                  net_notional = EXCLUDED.net_notional,
                  trade_count = EXCLUDED.trade_count,
                  unknown_trade_share = EXCLUDED.unknown_trade_share,
                  quote_coverage_pct = EXCLUDED.quote_coverage_pct,
                  warnings_json = EXCLUDED.warnings_json,
                  updated_at = EXCLUDED.updated_at
                """
            )
            tx.unregister("tmp_intraday_option_flow")
        return int(len(normalized))

    def load_intraday_option_flow(
        self,
        *,
        symbol: str,
        market_date: date | str | None = None,
        provider: str | None = None,
        contract_symbol: str | None = None,
    ) -> pd.DataFrame:
        symbol_norm = self._clean_symbol(symbol)
        if symbol_norm is None:
            return pd.DataFrame(columns=_INTRADAY_OPTION_FLOW_LOAD_COLUMNS)
        provider_name = self._normalize_provider(provider)
        resolved_market_date = _coerce_date(market_date) if market_date is not None else self._resolve_latest_date(
            table="intraday_option_flow",
            date_column="market_date",
            symbol_column="symbol",
            symbol=symbol_norm,
            provider=provider_name,
        )
        if resolved_market_date is None:
            return pd.DataFrame(columns=_INTRADAY_OPTION_FLOW_LOAD_COLUMNS)

        where_sql = "symbol = ? AND market_date = ? AND provider = ?"
        params: list[Any] = [symbol_norm, resolved_market_date, provider_name]

        contract_symbol_norm = self._clean_symbol(contract_symbol)
        if contract_symbol_norm is not None:
            where_sql += " AND contract_symbol = ?"
            params.append(contract_symbol_norm)

        df = self.warehouse.fetch_df(
            f"""
            SELECT symbol, market_date, source, contract_symbol, expiry, option_type, strike, delta_bucket,
                   buy_volume, sell_volume, unknown_volume, buy_notional, sell_notional, net_notional,
                   trade_count, unknown_trade_share, quote_coverage_pct, warnings_json, provider, updated_at
            FROM intraday_option_flow
            WHERE {where_sql}
            ORDER BY ABS(COALESCE(net_notional, 0.0)) DESC, contract_symbol ASC
            """,
            params,
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=_INTRADAY_OPTION_FLOW_LOAD_COLUMNS)
        out = df.copy()
        out["market_date"] = out["market_date"].astype(str)
        out["expiry"] = out["expiry"].map(lambda value: value.isoformat() if isinstance(value, date) else None)
        out["warnings"] = out["warnings_json"].map(self._warnings_from_json)
        out = out.drop(columns=["warnings_json"])
        return out[list(_INTRADAY_OPTION_FLOW_LOAD_COLUMNS)].copy()


def _sql_quote(path: str) -> str:
    return str(path).replace("'", "''")
