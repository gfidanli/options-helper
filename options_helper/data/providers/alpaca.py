from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol
from options_helper.data.candles import _parse_period_to_start
from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData
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


class AlpacaProvider(MarketDataProvider):
    name = "alpaca"

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self._client = client or AlpacaClient()
        self._warned_back_adjust = False

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
        _ = self._client.option_client
        raise DataFetchError("Alpaca provider scaffold: expiries not implemented yet (see IMP-023).")

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        _ = self._client.option_client
        raise DataFetchError("Alpaca provider scaffold: options chain not implemented yet (see IMP-024).")

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict:
        _ = self._client.option_client
        raise DataFetchError("Alpaca provider scaffold: raw chain not implemented yet (see IMP-024).")

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: earnings event not implemented yet (see IMP-022).")
