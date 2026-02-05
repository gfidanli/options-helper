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
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
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
                  open, high, low, close, volume, vwap, trade_count, dividends, splits, capital_gains
                )
                SELECT symbol, interval, auto_adjust, back_adjust, ts,
                       open, high, low, close, volume, vwap, trade_count, dividends, splits, capital_gains
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
                DELETE FROM option_contracts
                WHERE contract_symbol IN (SELECT contract_symbol FROM tmp_contracts)
                """
            )
            tx.execute(
                """
                INSERT INTO option_contracts(
                  contract_symbol, underlying, expiry, option_type, strike, multiplier, provider, updated_at
                )
                SELECT contract_symbol, underlying, expiry, option_type, strike, multiplier, provider, updated_at
                FROM tmp_contracts
                """
            )
            tx.unregister("tmp_contracts")

            if not snapshots_df.empty:
                tx.register("tmp_contract_snapshots", snapshots_df)
                tx.execute(
                    """
                    DELETE FROM option_contract_snapshots
                    WHERE provider = ? AND as_of_date = ?
                      AND contract_symbol IN (SELECT contract_symbol FROM tmp_contract_snapshots)
                    """,
                    [provider_name, as_of],
                )
                tx.execute(
                    """
                    INSERT INTO option_contract_snapshots(
                      contract_symbol, as_of_date, open_interest, open_interest_date,
                      close_price, close_price_date, provider, updated_at, raw_json
                    )
                    SELECT contract_symbol, as_of_date, open_interest, open_interest_date,
                           close_price, close_price_date, provider, updated_at, raw_json
                    FROM tmp_contract_snapshots
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

    def upsert_bars(
        self,
        df: pd.DataFrame,
        *,
        interval: str = "1d",
        provider: str,
        updated_at: datetime | None = None,
    ) -> None:
        if df is None or df.empty:
            return

        provider_name = self._normalize_provider(provider)
        interval_name = self._normalize_interval(interval)

        table = df.copy()
        if table.empty:
            return
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
            return
        normalized = normalized.drop_duplicates(subset=["contract_symbol", "ts"], keep="last")
        if normalized.empty:
            return

        now = updated_at or datetime.now(timezone.utc)
        normalized["interval"] = interval_name
        normalized["provider"] = provider_name
        normalized["updated_at"] = now

        try:
            with self.warehouse.transaction() as tx:
                tx.register("tmp_option_bars", normalized)
                tx.execute(
                    """
                    DELETE FROM option_bars
                    WHERE (contract_symbol, interval, ts, provider) IN (
                      SELECT contract_symbol, interval, ts, provider FROM tmp_option_bars
                    )
                    """
                )
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
                    """
                )
                tx.unregister("tmp_option_bars")
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
        status_value = str(status or "ok").strip().lower() or "ok"

        def _normalize_map(value: object | dict[str, object] | None) -> dict[str, object] | None:
            if not isinstance(value, dict):
                return None
            out: dict[str, object] = {}
            for key, item in value.items():
                sym = self._clean_symbol(key)
                if sym:
                    out[sym] = item
            return out

        rows_map = _normalize_map(rows)
        start_map = _normalize_map(start_ts)
        end_map = _normalize_map(end_ts)

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

        meta_df = pd.DataFrame(meta_rows)

        try:
            with self.warehouse.transaction() as tx:
                tx.register("tmp_option_bars_meta", meta_df)
                tx.execute(
                    """
                    DELETE FROM option_bars_meta
                    WHERE (contract_symbol, interval, provider) IN (
                      SELECT contract_symbol, interval, provider FROM tmp_option_bars_meta
                    )
                    """
                )
                tx.execute(
                    """
                    INSERT INTO option_bars_meta(
                      contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                      last_success_at, last_attempt_at, last_error, error_count
                    )
                    SELECT contract_symbol, interval, provider, status, rows, start_ts, end_ts,
                           last_success_at, last_attempt_at, last_error, error_count
                    FROM tmp_option_bars_meta
                    """
                )
                tx.unregister("tmp_option_bars_meta")
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
        status_value = str(status or "error").strip().lower() or "error"

        attempt_at = datetime.now(timezone.utc)
        error_text = str(error) if error else None

        meta_df = pd.DataFrame(
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

        try:
            with self.warehouse.transaction() as tx:
                tx.register("tmp_option_bars_meta", meta_df)
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
                tx.unregister("tmp_option_bars_meta")
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
            coerced = _coerce_date(v)
            if coerced is not None:
                out.append(coerced)
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
