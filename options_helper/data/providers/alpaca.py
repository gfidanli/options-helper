from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.analysis.osi import parse_contract_symbol
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol
from options_helper.data.candles import _parse_period_to_start
from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData
from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    return val if val > 0 else None


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
            auto_adjust=False,
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
        _ = self._client.option_client
        raise DataFetchError("Alpaca provider scaffold: options chain not implemented yet (see IMP-024).")

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict:
        _ = self._client.option_client
        raise DataFetchError("Alpaca provider scaffold: raw chain not implemented yet (see IMP-024).")

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: earnings event not implemented yet (see IMP-022).")
