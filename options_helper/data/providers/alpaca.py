from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData
from options_helper.data.providers.base import MarketDataProvider


class AlpacaProvider(MarketDataProvider):
    name = "alpaca"

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self._client = client or AlpacaClient()

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
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: history not implemented yet (see IMP-022).")

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: underlying not implemented yet (see IMP-022).")

    def get_quote(self, symbol: str) -> float | None:
        _ = self._client.stock_client
        raise DataFetchError("Alpaca provider scaffold: quote not implemented yet (see IMP-022).")

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
