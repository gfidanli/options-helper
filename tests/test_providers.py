from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from options_helper.data.providers import get_provider
from options_helper.data.providers.base import OPTION_CHAIN_REQUIRED_COLUMNS, normalize_option_chain
from options_helper.data.yf_client import EarningsEvent, OptionsChain, UnderlyingData


@dataclass(frozen=True)
class StubClient:
    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        history = pd.DataFrame({"Close": [100.0]})
        return UnderlyingData(symbol=symbol.upper(), last_price=100.0, history=history)

    def get_quote(self, symbol: str) -> float | None:
        return 101.25

    def list_option_expiries(self, symbol: str) -> list[date]:
        return [date(2026, 1, 17)]

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        calls = pd.DataFrame({"strike": [100.0]})
        puts = pd.DataFrame({"strike": [95.0]})
        return OptionsChain(symbol=symbol.upper(), expiry=expiry, calls=calls, puts=puts)

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict:
        return {"calls": [], "puts": [], "underlying": {}}

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        return EarningsEvent(symbol=symbol.upper(), next_date=None, source="stub")


def test_normalize_option_chain_adds_required_columns() -> None:
    df = pd.DataFrame({"strike": [100.0], "bid": [1.2]})
    out = normalize_option_chain(df, option_type="call", expiry=date(2026, 1, 17))
    for col in OPTION_CHAIN_REQUIRED_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "optionType"] == "call"
    assert out.loc[0, "expiry"] == "2026-01-17"


def test_get_provider_yahoo_normalizes_chain() -> None:
    provider = get_provider("yahoo", client=StubClient())
    chain = provider.get_options_chain("AAPL", date(2026, 1, 17))
    for col in OPTION_CHAIN_REQUIRED_COLUMNS:
        assert col in chain.calls.columns
        assert col in chain.puts.columns
    assert chain.calls.loc[0, "optionType"] == "call"
    assert chain.puts.loc[0, "optionType"] == "put"
    assert chain.calls.loc[0, "expiry"] == "2026-01-17"
