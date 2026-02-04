from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.providers import alpaca as alpaca_provider
from options_helper.data.providers.alpaca import AlpacaProvider


class _FixedDate(date):
    @classmethod
    def today(cls):  # noqa: N805
        return date(2026, 2, 3)


class _StubClient:
    def __init__(self, contracts: list[dict]) -> None:
        self.contracts = contracts
        self.calls = 0
        self.provider_version = "0.0.0-test"

    def list_option_contracts(self, *args, **kwargs):  # noqa: D401, ARG002
        self.calls += 1
        return self.contracts


def test_list_option_expiries_uses_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpaca_provider, "date", _FixedDate)
    store = OptionContractsStore(tmp_path)
    as_of = _FixedDate.today()
    df = pd.DataFrame(
        {
            "contractSymbol": ["SPY260320C00450000"],
            "expiry": ["2026-03-20"],
        }
    )
    store.save("SPY", as_of, df, raw=None, meta={"provider": "alpaca"})

    stub = _StubClient([])
    provider = AlpacaProvider(client=stub, contracts_store=store)

    expiries = provider.list_option_expiries("SPY")

    assert expiries == [date(2026, 3, 20)]
    assert stub.calls == 0


def test_list_option_expiries_fetches_and_caches(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpaca_provider, "date", _FixedDate)
    store = OptionContractsStore(tmp_path)
    contracts = [
        {"symbol": "SPY260320C00450000", "expiration_date": "2026-03-20"},
        {"symbol": "SPY260214P00400000", "expiration_date": "2026-02-14"},
    ]
    stub = _StubClient(contracts)
    provider = AlpacaProvider(client=stub, contracts_store=store)

    expiries = provider.list_option_expiries("SPY")

    assert expiries == [date(2026, 2, 14), date(2026, 3, 20)]
    assert stub.calls == 1
    cached = store.load("SPY", _FixedDate.today())
    assert cached is not None
