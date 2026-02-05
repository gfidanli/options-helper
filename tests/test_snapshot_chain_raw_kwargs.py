from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from options_helper.data.options_snapshotter import snapshot_full_chain_for_symbols


class _StubProvider:
    name = "stub"

    def __init__(self) -> None:
        self.called_snapshot_date: date | None = None

    def list_option_expiries(self, symbol: str):  # noqa: ARG002
        return [date(2026, 2, 21)]

    def get_options_chain_raw(self, symbol: str, expiry: date, *, snapshot_date: date | None = None):  # noqa: ARG002
        self.called_snapshot_date = snapshot_date
        return {
            "underlying": {"symbol": symbol},
            "calls": [
                {
                    "contractSymbol": "SPY260221C00450000",
                    "bid": 1.0,
                    "ask": 1.2,
                    "lastPrice": 1.1,
                    "volume": 10,
                    "openInterest": 100,
                }
            ],
            "puts": [
                {
                    "contractSymbol": "SPY260221P00450000",
                    "bid": 0.9,
                    "ask": 1.1,
                    "lastPrice": 1.0,
                    "volume": 8,
                    "openInterest": 80,
                }
            ],
        }

    def get_underlying(self, symbol: str, *, period: str = "10d", interval: str = "1d"):  # noqa: ARG002
        raise AssertionError("Should not call get_underlying when candles are available")


def test_snapshotter_passes_snapshot_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candle_day = date(2026, 2, 3)

    def _stub_history(self, symbol: str, *, period: str = "10d"):  # noqa: ARG001
        idx = pd.to_datetime([candle_day.replace(day=candle_day.day - 1), candle_day])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)

    provider = _StubProvider()
    snapshot_dir = tmp_path / "snapshots"
    candle_dir = tmp_path / "candles"

    results = snapshot_full_chain_for_symbols(
        ["SPY"],
        cache_dir=snapshot_dir,
        candle_cache_dir=candle_dir,
        provider=provider,
    )

    assert results and results[0].status == "ok"
    assert provider.called_snapshot_date == candle_day
