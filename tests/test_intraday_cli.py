from __future__ import annotations

from datetime import date

import pandas as pd
from typer.testing import CliRunner

from options_helper.commands import intraday
from options_helper.cli import app
from options_helper.data.option_contracts import OptionContractsStore


class _StubProvider:
    name = "alpaca"


def test_intraday_pull_stocks_bars(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    runner = CliRunner()
    instances: list[object] = []

    class _StubClient:
        provider_version = "test"
        stock_feed = "sip"

        def __init__(self) -> None:
            instances.append(self)
            self.calls: list[dict[str, object]] = []

        def get_stock_bars_intraday(self, symbol: str, *, day: date, timeframe: str, feed: str | None):
            self.calls.append(
                {"symbol": symbol, "day": day, "timeframe": timeframe, "feed": feed}
            )
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-02-03T14:30:00Z", "2026-02-03T14:31:00Z"], utc=True
                    ),
                    "open": [10.0, 10.5],
                    "high": [10.6, 10.8],
                    "low": [9.9, 10.2],
                    "close": [10.2, 10.7],
                    "volume": [100, 150],
                }
            )

    monkeypatch.setattr(intraday, "build_provider", lambda: _StubProvider())
    monkeypatch.setattr(intraday, "AlpacaClient", _StubClient)

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "intraday",
            "pull-stocks-bars",
            "--symbol",
            "AAPL",
            "--day",
            "2026-02-03",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert instances
    assert instances[0].calls

    out_path = tmp_path / "stocks" / "bars" / "1Min" / "AAPL" / "2026-02-03.csv.gz"
    meta_path = tmp_path / "stocks" / "bars" / "1Min" / "AAPL" / "2026-02-03.meta.json"
    assert out_path.exists()
    assert meta_path.exists()


def test_intraday_pull_options_bars(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    runner = CliRunner()
    instances: list[object] = []

    contracts_dir = tmp_path / "contracts"
    contracts_store = OptionContractsStore(contracts_dir)
    as_of = date(2026, 2, 2)
    contracts_df = pd.DataFrame(
        {
            "contractSymbol": ["SPY260621C00100000", "SPY260621P00100000"],
            "underlying": ["SPY", "SPY"],
            "expiry": ["2026-06-21", "2026-06-21"],
            "optionType": ["call", "put"],
            "strike": [100.0, 100.0],
            "multiplier": [100, 100],
            "openInterest": [10, 12],
            "openInterestDate": ["2026-02-01", "2026-02-01"],
            "closePrice": [1.1, 1.2],
            "closePriceDate": ["2026-02-01", "2026-02-01"],
        }
    )
    contracts_store.save("SPY", as_of, contracts_df, raw=None, meta={})

    class _StubClient:
        provider_version = "test"
        options_feed = "opra"

        def __init__(self) -> None:
            instances.append(self)
            self.calls: list[dict[str, object]] = []

        def get_option_bars_intraday(
            self,
            symbols: list[str],
            *,
            day: date,
            timeframe: str,
            feed: str | None,
        ):
            self.calls.append(
                {"symbols": symbols, "day": day, "timeframe": timeframe, "feed": feed}
            )
            return pd.DataFrame(
                {
                    "contractSymbol": [
                        "SPY260621C00100000",
                        "SPY260621P00100000",
                    ],
                    "timestamp": pd.to_datetime(
                        ["2026-02-03T14:30:00Z", "2026-02-03T14:31:00Z"], utc=True
                    ),
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.05, 2.05],
                    "volume": [10, 20],
                }
            )

    monkeypatch.setattr(intraday, "build_provider", lambda: _StubProvider())
    monkeypatch.setattr(intraday, "AlpacaClient", _StubClient)

    out_dir = tmp_path / "intraday"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "intraday",
            "pull-options-bars",
            "--underlying",
            "SPY",
            "--contracts-dir",
            str(contracts_dir),
            "--day",
            "2026-02-03",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert instances
    assert instances[0].calls

    call_path = (
        out_dir
        / "options"
        / "bars"
        / "1Min"
        / "SPY260621C00100000"
        / "2026-02-03.csv.gz"
    )
    put_path = (
        out_dir
        / "options"
        / "bars"
        / "1Min"
        / "SPY260621P00100000"
        / "2026-02-03.csv.gz"
    )
    assert call_path.exists()
    assert put_path.exists()
