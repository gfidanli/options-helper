from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.stores_duckdb.common import _coerce_date
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.research_metrics_contracts import (
    EXPOSURE_STRIKE_FIELDS,
    INTRADAY_FLOW_CONTRACT_FIELDS,
    IV_SURFACE_DELTA_BUCKET_FIELDS,
    IV_SURFACE_TENOR_FIELDS,
)

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

__all__ = ["DuckDBResearchMetricsStore"]
