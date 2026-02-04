from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.option_contracts import OptionContractsStore


def test_stream_capture_parses_symbols(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def run(self, *, duration_seconds=None):  # type: ignore[no-untyped-def]
            captured["duration"] = duration_seconds
            return []

    import options_helper.commands.stream as stream_cmd

    monkeypatch.setattr(stream_cmd, "StreamRunner", FakeRunner)
    monkeypatch.setattr(stream_cmd, "_require_alpaca_provider", lambda: None)

    runner = CliRunner()
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "intraday"
    result = runner.invoke(
        app,
        [
            "--log-dir",
            str(log_dir),
            "stream",
            "capture",
            "--stocks",
            "brk-b,AAPL",
            "--duration",
            "0",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stocks"] == ["AAPL", "BRK-B"]
    assert kwargs["option_contracts"] == []
    assert kwargs["out_dir"] == out_dir
    assert captured["duration"] == 0.0


def test_stream_capture_expands_underlyings(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def run(self, *, duration_seconds=None):  # type: ignore[no-untyped-def]
            return []

    import options_helper.commands.stream as stream_cmd

    monkeypatch.setattr(stream_cmd, "StreamRunner", FakeRunner)
    monkeypatch.setattr(stream_cmd, "_require_alpaca_provider", lambda: None)

    contracts_dir = tmp_path / "contracts"
    store = OptionContractsStore(contracts_dir)
    as_of = date(2026, 2, 3)
    df = pd.DataFrame(
        [
            {"contractSymbol": "SPY260320C00500000", "expiry": "2026-03-20"},
            {"contractSymbol": "SPY260327P00480000", "expiry": "2026-03-27"},
        ]
    )
    store.save("SPY", as_of, df, raw=None, meta=None)

    runner = CliRunner()
    out_dir = tmp_path / "intraday"
    log_dir = tmp_path / "logs"
    result = runner.invoke(
        app,
        [
            "--log-dir",
            str(log_dir),
            "stream",
            "capture",
            "--options-underlyings",
            "SPY",
            "--contracts-dir",
            str(contracts_dir),
            "--contracts-as-of",
            as_of.isoformat(),
            "--expiry",
            "2026-03-20",
            "--duration",
            "0",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stocks"] == []
    assert kwargs["option_contracts"] == ["SPY260320C00500000"]

