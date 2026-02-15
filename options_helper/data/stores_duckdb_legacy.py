from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

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
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.db.warehouse import DuckDBWarehouse

if TYPE_CHECKING:
    from options_helper.data.stores_duckdb.bars import DuckDBOptionBarsStore
    from options_helper.data.stores_duckdb.research_metrics import DuckDBResearchMetricsStore
    from options_helper.data.stores_duckdb.snapshots import DuckDBOptionsSnapshotStore


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
__all__ = [
    "DuckDBCandleStore",
    "DuckDBDerivedStore",
    "DuckDBJournalStore",
    "DuckDBOptionContractsStore",
    "DuckDBOptionBarsStore",
    "DuckDBOptionsSnapshotStore",
    "DuckDBResearchMetricsStore",
]


def __getattr__(name: str) -> Any:
    # Compatibility exports for callers that still import these classes from
    # the legacy module path.
    if name == "DuckDBOptionBarsStore":
        from options_helper.data.stores_duckdb.bars import DuckDBOptionBarsStore

        return DuckDBOptionBarsStore
    if name == "DuckDBOptionsSnapshotStore":
        from options_helper.data.stores_duckdb.snapshots import DuckDBOptionsSnapshotStore

        return DuckDBOptionsSnapshotStore
    if name == "DuckDBResearchMetricsStore":
        from options_helper.data.stores_duckdb.research_metrics import DuckDBResearchMetricsStore

        return DuckDBResearchMetricsStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
