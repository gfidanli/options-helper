from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.analysis.osi import parse_contract_symbol
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df, option_chain_to_rows
from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol
from options_helper.data.candles import _parse_period_to_start
from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData
from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.providers.base import MarketDataProvider, normalize_option_chain

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    return val if val > 0 else None


def _normalize_option_type_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in {"call", "put"}:
        return raw
    if raw in {"c", "p"}:
        return "call" if raw == "c" else "put"
    if raw.startswith("call"):
        return "call"
    if raw.startswith("put"):
        return "put"
    return None


def _needs_fill(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _merge_contract_metadata(chain_df: pd.DataFrame, contracts_df: pd.DataFrame | None) -> pd.DataFrame:
    if chain_df is None or chain_df.empty:
        return chain_df
    if contracts_df is None or contracts_df.empty:
        return chain_df
    if "contractSymbol" not in chain_df.columns or "contractSymbol" not in contracts_df.columns:
        return chain_df

    merged = chain_df.merge(contracts_df, on="contractSymbol", how="left", suffixes=("", "_contract"))
    for field in (
        "strike",
        "optionType",
        "openInterest",
        "openInterestDate",
        "closePrice",
        "closePriceDate",
        "expiry",
    ):
        contract_field = f"{field}_contract"
        if contract_field not in merged.columns:
            continue
        if field not in merged.columns:
            merged[field] = merged[contract_field]
        else:
            mask = merged[field].map(_needs_fill)
            if mask.any():
                merged.loc[mask, field] = merged.loc[mask, contract_field]
    drop_cols = [col for col in merged.columns if col.endswith("_contract")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def _fill_missing_from_osi(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "contractSymbol" not in df.columns:
        return df

    parsed = df["contractSymbol"].map(lambda raw: parse_contract_symbol(str(raw)) if raw else None)

    if "optionType" in df.columns:
        df["optionType"] = df["optionType"].map(_normalize_option_type_value)
        mask = df["optionType"].map(_needs_fill)
        if mask.any():
            df.loc[mask, "optionType"] = parsed.map(lambda item: item.option_type if item else None)
    else:
        df["optionType"] = parsed.map(lambda item: item.option_type if item else None)

    if "strike" in df.columns:
        mask = df["strike"].map(_needs_fill)
        if mask.any():
            df.loc[mask, "strike"] = parsed.map(lambda item: item.strike if item else None)
    else:
        df["strike"] = parsed.map(lambda item: item.strike if item else None)

    if "expiry" in df.columns:
        mask = df["expiry"].map(_needs_fill)
        if mask.any():
            df.loc[mask, "expiry"] = parsed.map(lambda item: item.expiry.isoformat() if item else None)
    else:
        df["expiry"] = parsed.map(lambda item: item.expiry.isoformat() if item else None)

    return df


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_price_from(obj: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = _get_field(obj, key)
        price = _as_float(val)
        if price is not None:
            return price
    return None


def _extract_snapshot_price(snapshot: Any, symbol: str) -> float | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        maybe = snapshot.get(symbol) or snapshot.get(symbol.upper())
        if maybe is not None:
            snapshot = maybe
    trade = (
        _get_field(snapshot, "latest_trade")
        or _get_field(snapshot, "latestTrade")
        or _get_field(snapshot, "last_trade")
        or _get_field(snapshot, "trade")
    )
    price = _extract_price_from(trade, ("price", "p", "last", "last_price", "lastPrice"))
    if price is not None:
        return price
    price = _extract_price_from(snapshot, ("price", "last", "last_price", "lastPrice", "regularMarketPrice"))
    if price is not None:
        return price
    bar = (
        _get_field(snapshot, "daily_bar")
        or _get_field(snapshot, "dailyBar")
        or _get_field(snapshot, "prev_daily_bar")
        or _get_field(snapshot, "prevDailyBar")
    )
    return _extract_price_from(bar, ("close", "c", "close_price", "closePrice"))


def _last_close(history: pd.DataFrame) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _coerce_expiry_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return None


def _expiries_from_contracts(df: pd.DataFrame) -> list[date]:
    if df is None or df.empty:
        return []
    expiries: set[date] = set()
    if "expiry" in df.columns:
        for val in df["expiry"].dropna().tolist():
            exp = _coerce_expiry_date(val)
            if exp is not None:
                expiries.add(exp)
    if not expiries and "contractSymbol" in df.columns:
        for raw in df["contractSymbol"].dropna().tolist():
            parsed = parse_contract_symbol(str(raw))
            if parsed is not None:
                expiries.add(parsed.expiry)
    return sorted(expiries)


class AlpacaProvider(MarketDataProvider):
    name = "alpaca"

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        contracts_store: OptionContractsStore | None = None,
        contracts_cache_dir: Path | None = None,
    ) -> None:
        self._client = client or AlpacaClient()
        self._warned_back_adjust = False
        cache_dir = contracts_cache_dir or Path("data/option_contracts")
        self._contracts_store = contracts_store or OptionContractsStore(cache_dir)

    def get_history(
        self,
        symbol: str,
        *,
        start: date | None,
        end: date | None,
        interval: str,
        auto_adjust: bool,
        back_adjust: bool,
    ) -> pd.DataFrame:
        if start is None and end is None:
            start = date(1970, 1, 1)
        adjustment = "all" if auto_adjust or back_adjust else "raw"
        if back_adjust and not self._warned_back_adjust:
            logger.warning("Alpaca does not support back_adjust; using adjustment='all'.")
            self._warned_back_adjust = True
        return self._client.get_stock_bars(
            symbol,
            start=start,
            end=end,
            interval=interval,
            adjustment=adjustment,
        )

    @property
    def provider_params(self) -> dict[str, Any]:
        return {
            "options_feed": getattr(self._client, "options_feed", None),
            "stock_feed": getattr(self._client, "stock_feed", None),
            "recent_bars_buffer_minutes": getattr(self._client, "recent_bars_buffer_minutes", None),
        }

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        try:
            start = _parse_period_to_start(period, today=date.today())
        except ValueError as exc:
            raise DataFetchError(f"Unsupported period format: {period}") from exc
        history = self.get_history(
            symbol,
            start=start,
            end=None,
            interval=interval,
            auto_adjust=True,
            back_adjust=False,
        )
        last_price = _last_close(history)
        if last_price is None:
            try:
                last_price = self.get_quote(symbol)
            except DataFetchError:
                last_price = None
        return UnderlyingData(symbol=to_repo_symbol(symbol), last_price=last_price, history=history)

    def get_quote(self, symbol: str) -> float | None:
        alpaca_symbol = to_alpaca_symbol(symbol)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid symbol: {symbol}")
        client = self._client.stock_client

        snapshot_error: Exception | None = None
        trade_error: Exception | None = None

        snapshot = None
        try:
            if hasattr(client, "get_stock_snapshot"):
                snapshot = client.get_stock_snapshot(alpaca_symbol)
            elif hasattr(client, "get_stock_snapshots"):
                snapshots = client.get_stock_snapshots([alpaca_symbol])
                snapshot = snapshots.get(alpaca_symbol) if isinstance(snapshots, dict) else snapshots
        except Exception as exc:  # noqa: BLE001
            snapshot_error = exc

        price = _extract_snapshot_price(snapshot, alpaca_symbol)
        if price is not None:
            return price

        try:
            if hasattr(client, "get_stock_latest_trade"):
                trade = client.get_stock_latest_trade(alpaca_symbol)
                price = _extract_price_from(trade, ("price", "p", "last", "last_price", "lastPrice"))
                if price is not None:
                    return price
        except Exception as exc:  # noqa: BLE001
            trade_error = exc

        try:
            history = self._client.get_stock_bars(
                symbol,
                start=date.today() - timedelta(days=5),
                end=None,
                interval="1d",
                adjustment="raw",
            )
            return _last_close(history)
        except DataFetchError as exc:
            if snapshot_error is not None:
                raise DataFetchError(f"Failed to fetch Alpaca snapshot for {alpaca_symbol}.") from snapshot_error
            if trade_error is not None:
                raise DataFetchError(f"Failed to fetch Alpaca latest trade for {alpaca_symbol}.") from trade_error
            raise exc

    def list_option_expiries(self, symbol: str) -> list[date]:
        sym = to_repo_symbol(symbol)
        if not sym:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        as_of = date.today()
        cached = self._contracts_store.load(sym, as_of)
        if cached is not None:
            return _expiries_from_contracts(cached)

        try:
            raw_contracts = self._client.list_option_contracts(
                sym,
                exp_gte=as_of,
                exp_lte=None,
                limit=1000,
                page_limit=50,
            )
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch Alpaca option contracts for {sym}.") from exc

        df = contracts_to_df(raw_contracts)
        meta = {
            "provider": self.name,
            "provider_version": self._client.provider_version,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "exp_gte": as_of.isoformat(),
            "exp_lte": None,
            "contracts": int(len(raw_contracts)),
        }
        self._contracts_store.save(sym, as_of, df, raw={"contracts": raw_contracts}, meta=meta)
        return _expiries_from_contracts(df)

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        sym = to_repo_symbol(symbol)
        if not sym:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        raw = self.get_options_chain_raw(sym, expiry)
        calls = pd.DataFrame(raw.get("calls", []))
        puts = pd.DataFrame(raw.get("puts", []))

        calls = normalize_option_chain(calls, option_type="call", expiry=expiry)
        puts = normalize_option_chain(puts, option_type="put", expiry=expiry)

        return OptionsChain(symbol=sym, expiry=expiry, calls=calls, puts=puts)

    def get_options_chain_raw(self, symbol: str, expiry: date, *, snapshot_date: date | None = None) -> dict:
        sym = to_repo_symbol(symbol)
        if not sym:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        try:
            payload = self._client.get_option_chain_snapshots(
                sym,
                expiry=expiry,
                feed=self._client.options_feed,
            )
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch Alpaca option chain for {sym} {expiry.isoformat()}.") from exc

        rows = option_chain_to_rows(payload)
        chain_df = pd.DataFrame(rows)

        contracts_df: pd.DataFrame | None = None
        as_of = date.today()
        cached = self._contracts_store.load(sym, as_of)
        if cached is not None:
            contracts_df = cached
        else:
            try:
                raw_contracts = self._client.list_option_contracts(
                    sym,
                    exp_gte=expiry,
                    exp_lte=expiry,
                    limit=1000,
                    page_limit=50,
                )
            except DataFetchError:
                raw_contracts = []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch Alpaca contracts for %s: %s", sym, exc)
                raw_contracts = []

            if raw_contracts:
                contracts_df = contracts_to_df(raw_contracts)
                meta = {
                    "provider": self.name,
                    "provider_version": self._client.provider_version,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "exp_gte": expiry.isoformat(),
                    "exp_lte": expiry.isoformat(),
                    "contracts": int(len(raw_contracts)),
                }
                self._contracts_store.save(sym, as_of, contracts_df, raw={"contracts": raw_contracts}, meta=meta)

        if contracts_df is not None and "expiry" in contracts_df.columns:
            contracts_df = contracts_df[contracts_df["expiry"] == expiry.isoformat()]

        chain_df = _merge_contract_metadata(chain_df, contracts_df)
        chain_df = _fill_missing_from_osi(chain_df)

        if "contractSymbol" in chain_df.columns and not chain_df.empty:
            snapshot_day = snapshot_date or date.today()
            if snapshot_day <= date.today():
                start_dt = datetime.combine(snapshot_day, datetime.min.time()).replace(tzinfo=timezone.utc)
                end_dt = None
                if snapshot_day < date.today():
                    end_dt = datetime.combine(snapshot_day, datetime.max.time()).replace(tzinfo=timezone.utc)

                symbols = [
                    str(raw).strip()
                    for raw in chain_df["contractSymbol"].dropna().unique().tolist()
                    if str(raw).strip()
                ]
                if symbols:
                    try:
                        bars_df = self._client.get_option_bars(
                            symbols,
                            start=start_dt,
                            end=end_dt,
                            interval="1d",
                        )
                    except DataFetchError as exc:
                        logger.warning("Failed to fetch Alpaca option bars for %s %s: %s", sym, expiry, exc)
                        bars_df = pd.DataFrame()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to fetch Alpaca option bars for %s %s: %s", sym, expiry, exc)
                        bars_df = pd.DataFrame()

                    if not bars_df.empty:
                        bars_df = bars_df.rename(
                            columns={
                                "volume": "volume_bar",
                                "vwap": "vwap_bar",
                                "trade_count": "trade_count_bar",
                            }
                        )
                        merge_cols = ["contractSymbol", "volume_bar", "vwap_bar", "trade_count_bar"]
                        chain_df = chain_df.merge(
                            bars_df[merge_cols],
                            on="contractSymbol",
                            how="left",
                        )

                        if "volume" not in chain_df.columns:
                            chain_df["volume"] = pd.NA
                        volume_numeric = pd.to_numeric(chain_df["volume"], errors="coerce")
                        fill_volume = volume_numeric.isna() | (volume_numeric <= 0)
                        chain_df.loc[fill_volume, "volume"] = chain_df.loc[fill_volume, "volume_bar"]

                        if "vwap" not in chain_df.columns:
                            chain_df["vwap"] = chain_df["vwap_bar"]
                        else:
                            mask = chain_df["vwap"].map(_needs_fill)
                            chain_df.loc[mask, "vwap"] = chain_df.loc[mask, "vwap_bar"]

                        if "trade_count" not in chain_df.columns:
                            chain_df["trade_count"] = chain_df["trade_count_bar"]
                        else:
                            mask = chain_df["trade_count"].map(_needs_fill)
                            chain_df.loc[mask, "trade_count"] = chain_df.loc[mask, "trade_count_bar"]

                        chain_df = chain_df.drop(
                            columns=["volume_bar", "vwap_bar", "trade_count_bar"],
                            errors="ignore",
                        )

        if "optionType" not in chain_df.columns:
            chain_df["optionType"] = pd.NA

        calls_df = chain_df[chain_df.get("optionType") == "call"].copy()
        puts_df = chain_df[chain_df.get("optionType") == "put"].copy()

        underlying_payload = None
        if isinstance(payload, dict):
            underlying_payload = payload.get("underlying") or payload.get("underlying_snapshot")
        else:
            underlying_payload = getattr(payload, "underlying", None)

        underlying: dict[str, Any] = {"symbol": sym}
        if isinstance(underlying_payload, dict):
            underlying.update(underlying_payload)
            spot = _extract_snapshot_price(underlying_payload, sym)
            if spot is not None:
                underlying["spot"] = spot
        elif underlying_payload is not None:
            underlying["raw"] = str(underlying_payload)

        return {
            "underlying": underlying,
            "calls": calls_df.to_dict(orient="records"),
            "puts": puts_df.to_dict(orient="records"),
            "provider_raw": payload,
        }

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: earnings event not implemented yet (see IMP-022).")
