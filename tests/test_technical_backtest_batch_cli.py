from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from options_helper.cli import app


def _stub_config(tmp_path: Path) -> dict[str, object]:
    return {
        "logging": {"level": "INFO", "log_dir": str(tmp_path / "logs")},
        "backtest": {"commission": 0.001, "slippage_bps": 1.0},
        "strategies": {
            "MeanReversionIBS": {
                "enabled": True,
                "defaults": {"lookback_high": 10},
            }
        },
    }


def _patch_command_dependencies(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "options_helper.commands.technicals.backtest_batch.load_technical_backtesting_config",
        lambda _path: _stub_config(tmp_path),
    )
    monkeypatch.setattr(
        "options_helper.commands.technicals.backtest_batch.setup_technicals_logging",
        lambda _cfg: None,
    )


def test_backtest_batch_command_is_registered_under_technicals() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technicals", "--help"])
    assert result.exit_code == 0, result.output
    assert "backtest-batch" in result.output


def test_backtest_batch_help_lists_overlay_and_cost_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technicals", "backtest-batch", "--help"])
    assert result.exit_code == 0, result.output
    assert "Commission override" in result.output
    assert "Slippage override in" in result.output
    assert "SMA trend overlay" in result.output
    assert "weekly trend overlay" in result.output
    assert "MA direction overlay" in result.output


def test_backtest_batch_invocation_supports_single_symbol_path(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    _patch_command_dependencies(tmp_path, monkeypatch)

    def _fake_runtime(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return SimpleNamespace(
            symbols=tuple(kwargs["symbols"]),
            success_count=1,
            failure_count=0,
            outcomes=(SimpleNamespace(symbol="SPY", ok=True),),
        )

    monkeypatch.setattr(
        "options_helper.commands.technicals.backtest_batch.run_technicals_backtest_batch_runtime",
        _fake_runtime,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "backtest-batch",
            "--symbol",
            "spy",
            "--cache-dir",
            str(tmp_path / "candles"),
            "--config-path",
            str(tmp_path / "technical_backtesting.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ("SPY",)
    assert "Batch backtest complete: symbols=1 success=1 failed=0" in result.output
    assert "Failed symbols:" not in result.output
    assert "Informational output only; not financial advice." in result.output


def test_backtest_batch_multi_ticker_path_continues_after_partial_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    _patch_command_dependencies(tmp_path, monkeypatch)

    def _fake_runtime(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return SimpleNamespace(
            symbols=tuple(kwargs["symbols"]),
            success_count=2,
            failure_count=1,
            outcomes=(
                SimpleNamespace(symbol="QQQ", ok=True),
                SimpleNamespace(symbol="SPY", ok=True),
                SimpleNamespace(symbol="IWM", ok=False),
            ),
        )

    monkeypatch.setattr(
        "options_helper.commands.technicals.backtest_batch.run_technicals_backtest_batch_runtime",
        _fake_runtime,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "backtest-batch",
            "--tickers",
            "qqq, spy, iwm",
            "--cache-dir",
            str(tmp_path / "candles"),
            "--config-path",
            str(tmp_path / "technical_backtesting.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ("QQQ", "SPY", "IWM")
    assert "Batch backtest complete: symbols=3 success=2 failed=1" in result.output
    assert "Failed symbols: IWM" in result.output
    assert "Informational output only; not financial advice." in result.output


def test_backtest_batch_invocation_delegates_to_runtime_with_overlay_and_cost_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    _patch_command_dependencies(tmp_path, monkeypatch)

    def _fake_runtime(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return SimpleNamespace(
            symbols=tuple(kwargs["symbols"]),
            success_count=2,
            failure_count=1,
            outcomes=(
                SimpleNamespace(symbol="SPY", ok=True),
                SimpleNamespace(symbol="QQQ", ok=True),
                SimpleNamespace(symbol="IWM", ok=False),
            ),
        )

    monkeypatch.setattr(
        "options_helper.commands.technicals.backtest_batch.run_technicals_backtest_batch_runtime",
        _fake_runtime,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "backtest-batch",
            "--symbol",
            "spy",
            "--tickers",
            "qqq, iwm",
            "--cache-dir",
            str(tmp_path / "candles"),
            "--config-path",
            str(tmp_path / "technical_backtesting.yaml"),
            "--commission",
            "0.004",
            "--slippage-bps",
            "7.5",
            "--require-sma-trend",
            "--sma-trend-window",
            "150",
            "--require-weekly-trend",
            "--no-require-ma-direction",
            "--ma-direction-window",
            "30",
            "--ma-direction-lookback",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ("SPY", "QQQ", "IWM")
    assert captured["strategy"] == "MeanReversionIBS"
    assert captured["cache_dir"] == tmp_path / "candles"
    assert captured["cli_commission"] == 0.004
    assert captured["cli_slippage_bps"] == 7.5

    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["strategies"]["MeanReversionIBS"]["defaults"] == {
        "lookback_high": 10,
        "require_sma_trend": True,
        "sma_trend_window": 150,
        "require_weekly_trend": True,
        "require_ma_direction": False,
        "ma_direction_window": 30,
        "ma_direction_lookback": 3,
    }
    assert "Batch backtest complete: symbols=3 success=2 failed=1" in result.output
    assert "Failed symbols: IWM" in result.output
    assert "Informational output only; not financial advice." in result.output
