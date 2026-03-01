from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.technical_backtesting_intraday import (
    IntradayCandleLoadResult,
    IntradayCoverage,
)


def _ohlc_frame(*, rows: int = 12, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2025-01-02 14:30:00+00:00", periods=rows, freq=freq)
    values = pd.Series(range(rows), dtype=float).to_numpy()
    return pd.DataFrame(
        {
            "Open": 100.0 + values,
            "High": 101.0 + values,
            "Low": 99.0 + values,
            "Close": 100.5 + values,
            "Volume": 1_000.0 + values,
        },
        index=idx,
    )


def _patch_common_optimize_dependencies(monkeypatch, *, tmp_path: Path, captured: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.compute_features",
        lambda frame, _cfg: frame.copy(),
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.warmup_bars",
        lambda _cfg: 0,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.feature_selection.required_feature_columns_for_strategy",
        lambda _strategy, _strat_cfg: [],
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.strategies.registry.get_strategy",
        lambda _strategy: object,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.backtest.optimizer.optimize_params",
        lambda *_args, **_kwargs: ({"atr_window": 14}, {"ret": 1.0}, None),
    )

    def _fake_write_artifacts(_cfg, **kwargs):  # noqa: ANN001
        captured["write_kwargs"] = kwargs
        return SimpleNamespace(
            params_path=tmp_path / "params.json",
            report_path=tmp_path / "summary.md",
        )

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_artifacts.write_artifacts",
        _fake_write_artifacts,
    )


def test_technicals_optimize_cli_prefers_ohlc_path_and_threads_interval(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {
        "path_calls": 0,
        "intraday_calls": 0,
        "cache_calls": 0,
    }

    def _fake_load_ohlc_from_path(_path):  # noqa: ANN001
        captured["path_calls"] += 1
        return _ohlc_frame()

    def _forbid_intraday(**_kwargs):  # noqa: ANN003
        captured["intraday_calls"] += 1
        raise AssertionError("intraday loader should not be used when --ohlc-path is provided")

    def _forbid_cache(*_args, **_kwargs):  # noqa: ANN002,ANN003
        captured["cache_calls"] += 1
        raise AssertionError("cache loader should not be used when --ohlc-path is provided")

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_io.load_ohlc_from_path",
        _fake_load_ohlc_from_path,
    )
    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_intraday.load_intraday_candles",
        _forbid_intraday,
    )
    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_io.load_ohlc_from_cache",
        _forbid_cache,
    )
    _patch_common_optimize_dependencies(monkeypatch, tmp_path=tmp_path, captured=captured)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "optimize",
            "--strategy",
            "TrendPullbackATR",
            "--ohlc-path",
            str(tmp_path / "input.csv"),
            "--symbol",
            "SPY",
            "--cache-dir",
            str(tmp_path / "candles"),
            "--intraday-dir",
            str(tmp_path / "intraday"),
            "--intraday-timeframe",
            "1Min",
            "--intraday-start",
            "2025-01-02",
            "--intraday-end",
            "2025-01-03",
            "--interval",
            "30m",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["path_calls"] == 1
    assert captured["intraday_calls"] == 0
    assert captured["cache_calls"] == 0
    assert captured["write_kwargs"]["interval"] == "30m"
    assert "intraday_coverage" not in captured["write_kwargs"]["data_meta"]


def test_technicals_walk_forward_cli_intraday_threads_coverage_into_data_meta(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {"intraday_calls": 0}

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_io.load_ohlc_from_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("path loader must not be used in intraday mode")),
    )
    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_io.load_ohlc_from_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache loader must not be used when --intraday-dir is provided")
        ),
    )

    def _fake_load_intraday_candles(**kwargs):  # noqa: ANN003
        captured["intraday_calls"] += 1
        captured["intraday_kwargs"] = kwargs
        coverage = IntradayCoverage(
            symbol="SPY",
            base_timeframe="1Min",
            target_interval="15Min",
            requested_days=(date(2025, 1, 2), date(2025, 1, 3)),
            loaded_days=(date(2025, 1, 2),),
            missing_days=(date(2025, 1, 3),),
            empty_days=(),
            loaded_row_count=12,
            output_row_count=4,
        )
        return IntradayCandleLoadResult(candles=_ohlc_frame(rows=8, freq="15min"), coverage=coverage)

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_intraday.load_intraday_candles",
        _fake_load_intraday_candles,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.compute_features",
        lambda frame, _cfg: frame.copy(),
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.warmup_bars",
        lambda _cfg: 0,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.feature_selection.required_feature_columns_for_strategy",
        lambda _strategy, _strat_cfg: [],
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.strategies.registry.get_strategy",
        lambda _strategy: object,
    )

    def _fake_walk_forward(*_args, **_kwargs):  # noqa: ANN002,ANN003
        return SimpleNamespace(
            params={"atr_window": 14},
            folds=[
                {
                    "train_start": pd.Timestamp("2025-01-02T14:30:00Z"),
                    "train_end": pd.Timestamp("2025-01-02T15:30:00Z"),
                    "validate_start": pd.Timestamp("2025-01-02T15:45:00Z"),
                    "validate_end": pd.Timestamp("2025-01-02T16:15:00Z"),
                    "best_params": {"atr_window": 14},
                    "train_stats": {"ret": 1.0},
                    "validate_stats": {"ret": 0.5},
                    "validate_score": 0.5,
                    "heatmap": None,
                }
            ],
            stability={"stable": True},
            used_defaults=False,
            reason=None,
        )

    monkeypatch.setattr(
        "options_helper.technicals_backtesting.backtest.walk_forward.walk_forward_optimize",
        _fake_walk_forward,
    )

    def _fake_write_artifacts(_cfg, **kwargs):  # noqa: ANN001
        captured["write_kwargs"] = kwargs
        return SimpleNamespace(
            params_path=tmp_path / "params.json",
            report_path=tmp_path / "summary.md",
        )

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_artifacts.write_artifacts",
        _fake_write_artifacts,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "walk-forward",
            "--strategy",
            "TrendPullbackATR",
            "--symbol",
            "SPY",
            "--cache-dir",
            str(tmp_path / "candles"),
            "--intraday-dir",
            str(tmp_path / "intraday"),
            "--intraday-timeframe",
            "1Min",
            "--intraday-start",
            "2025-01-02",
            "--intraday-end",
            "2025-01-03",
            "--interval",
            "15m",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["intraday_calls"] == 1
    assert captured["intraday_kwargs"]["target_interval"] == "15m"
    assert captured["write_kwargs"]["interval"] == "15Min"
    intraday_meta = captured["write_kwargs"]["data_meta"]["intraday_coverage"]
    assert intraday_meta["base_timeframe"] == "1Min"
    assert intraday_meta["target_interval"] == "15Min"
    assert intraday_meta["requested_day_count"] == 2
    assert intraday_meta["loaded_day_count"] == 1
    assert intraday_meta["missing_day_count"] == 1
    assert intraday_meta["requested_days"] == ["2025-01-02", "2025-01-03"]
    assert intraday_meta["missing_days"] == ["2025-01-03"]
