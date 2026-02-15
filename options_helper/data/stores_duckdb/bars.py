from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.stores_duckdb.common import _to_utc_naive
from options_helper.db.warehouse import DuckDBWarehouse


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

__all__ = ["DuckDBOptionBarsStore"]
