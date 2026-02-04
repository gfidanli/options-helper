from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from options_helper.data import alpaca_client
from options_helper.data.alpaca_client import AlpacaClient


class _StubTimeFrame:
    Day = "1Day"


class _StubBars:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class _StubOptionClient:
    def __init__(self, payload: _StubBars) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def get_option_bars(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_effective_end_defaults_to_buffer() -> None:
    client = AlpacaClient(api_key_id="key", api_secret_key="secret", recent_bars_buffer_minutes=20)
    before = datetime.now(timezone.utc)
    end = client.effective_end(None)
    after = datetime.now(timezone.utc)

    assert end is not None
    lower = before - timedelta(minutes=22)
    upper = after - timedelta(minutes=18)
    assert lower <= end <= upper


def test_effective_end_respects_explicit_value() -> None:
    client = AlpacaClient(api_key_id="key", api_secret_key="secret", recent_bars_buffer_minutes=20)
    explicit = datetime(2026, 2, 3, 12, 0, tzinfo=timezone.utc)
    assert client.effective_end(explicit) == explicit


def test_get_option_bars_normalizes_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.MultiIndex.from_product(
        [
            ["SPY260621C00100000", "SPY260621P00100000"],
            pd.to_datetime(["2026-02-02T14:30:00Z", "2026-02-02T15:30:00Z"], utc=True),
        ],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {"v": [10, 11, 20, 21], "vw": [1.0, 1.1, 2.0, 2.1], "n": [1, 2, 3, 4]},
        index=idx,
    )
    stub = _StubOptionClient(_StubBars(df))

    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._option_client = stub

    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "_load_option_bars_request", lambda: None)

    out = client.get_option_bars(
        ["SPY260621C00100000", "SPY260621P00100000"],
        start=date(2026, 2, 2),
        end=None,
        interval="1d",
    )

    assert len(stub.calls) == 1
    assert out.shape[0] == 2
    assert set(out["contractSymbol"]) == {"SPY260621C00100000", "SPY260621P00100000"}
    call_row = out[out["contractSymbol"] == "SPY260621C00100000"].iloc[0]
    put_row = out[out["contractSymbol"] == "SPY260621P00100000"].iloc[0]
    assert call_row["volume"] == 11
    assert put_row["volume"] == 21
    assert call_row["vwap"] == 1.1
    assert put_row["trade_count"] == 4
