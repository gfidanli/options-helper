from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.providers.base import MarketDataProvider, normalize_option_chain
from options_helper.data.yf_client import EarningsEvent, OptionsChain, UnderlyingData, YFinanceClient


class YahooProvider(MarketDataProvider):
    name = "yahoo"

    def __init__(self, client: YFinanceClient | None = None) -> None:
        self._client = client or YFinanceClient()

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
        return self._client.get_history(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust,
        )

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        return self._client.get_underlying(symbol, period=period, interval=interval)

    def get_quote(self, symbol: str) -> float | None:
        return self._client.get_quote(symbol)

    def list_option_expiries(self, symbol: str) -> list[date]:
        return self._client.list_option_expiries(symbol)

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        chain = self._client.get_options_chain(symbol, expiry)
        calls = normalize_option_chain(chain.calls, option_type="call", expiry=expiry)
        puts = normalize_option_chain(chain.puts, option_type="put", expiry=expiry)
        return OptionsChain(symbol=chain.symbol, expiry=chain.expiry, calls=calls, puts=puts)

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict:
        return self._client.get_options_chain_raw(symbol, expiry)

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        return self._client.get_next_earnings_event(symbol, today=today)
