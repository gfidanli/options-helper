from __future__ import annotations

from typing import Callable

from options_helper.data.providers.base import (
    OPTION_CHAIN_REQUIRED_COLUMNS,
    MarketDataProvider,
    normalize_option_chain,
)
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.providers.mock import MockProvider
from options_helper.data.providers.yahoo import YahooProvider
from options_helper.data.yf_client import YFinanceClient


def _alpaca_provider_factory() -> MarketDataProvider:
    from options_helper.data.providers.alpaca import AlpacaProvider

    return AlpacaProvider()


_PROVIDERS: dict[str, Callable[[], MarketDataProvider]] = {
    "yahoo": YahooProvider,
    "alpaca": _alpaca_provider_factory,
}

_ALIASES: dict[str, str] = {
    "yf": "yahoo",
    "yfinance": "yahoo",
    "apca": "alpaca",
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS.keys())


def get_provider(
    name: str | None = None,
    *,
    client: YFinanceClient | None = None,
) -> MarketDataProvider:
    provider_name = (name if name is not None else get_default_provider_name()).strip().lower()
    provider_name = _ALIASES.get(provider_name, provider_name)
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {', '.join(available_providers())}")
    if provider_name == "yahoo":
        return YahooProvider(client=client)
    return _PROVIDERS[provider_name]()


__all__ = [
    "OPTION_CHAIN_REQUIRED_COLUMNS",
    "MarketDataProvider",
    "MockProvider",
    "available_providers",
    "get_provider",
    "normalize_option_chain",
]
