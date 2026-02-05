from __future__ import annotations

import pandas as pd

from options_helper.data.yf_client import YFinanceClient


class _StubTicker:
    def __init__(self) -> None:
        self.last_history_kwargs: dict[str, object] = {}

    def history(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_history_kwargs = dict(kwargs)
        return pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-01-03"]))


def test_yfinance_underlying_requests_adjusted_history() -> None:
    stub = _StubTicker()
    client = YFinanceClient()
    client.ticker = lambda symbol: stub  # type: ignore[method-assign]

    out = client.get_underlying("AAPL", period="5d", interval="1d")

    assert out.symbol == "AAPL"
    assert out.last_price == 100.0
    assert stub.last_history_kwargs["auto_adjust"] is True
    assert stub.last_history_kwargs["back_adjust"] is False

