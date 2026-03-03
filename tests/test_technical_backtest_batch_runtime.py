from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from options_helper.commands.technicals.backtest_batch_runtime import (
    run_technicals_backtest_batch_runtime,
)


def _feature_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.2, 10.1, 10.3],
            "High": [10.2, 10.3, 10.4, 10.2, 10.5],
            "Low": [9.8, 9.9, 10.0, 9.9, 10.1],
            "Close": [10.1, 10.2, 10.1, 10.2, 10.4],
            "Volume": [1000, 1200, 1300, 1250, 1400],
            "weekly_trend_up": [True, True, False, True, True],
            "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ignored_col": [9, 9, 9, 9, 9],
        },
        index=index,
    )


def test_run_technicals_backtest_batch_runtime_wires_cost_precedence_and_feature_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "backtest": {"commission": 0.001, "slippage_bps": 3.0},
        "strategies": {
            "MeanReversionIBS": {
                "enabled": True,
                "defaults": {"lookback_high": 10, "ibs_threshold": 0.3},
                "cost_overrides": {"slippage_bps": 6.5},
            }
        },
    }
    loaded_symbols: list[str] = []
    run_calls: list[dict[str, object]] = []

    def _load_from_cache(symbol: str, _cache_dir: Path, **_kwargs: object) -> pd.DataFrame:
        loaded_symbols.append(symbol)
        return _feature_frame()

    monkeypatch.setattr(
        "options_helper.data.technical_backtesting_io.load_ohlc_from_cache",
        _load_from_cache,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.compute_features",
        lambda frame, _cfg: frame.copy(),
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.pipeline.warmup_bars",
        lambda _cfg: 7,
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.feature_selection.required_feature_columns_for_strategy",
        lambda _strategy, _strat_cfg: ["feature_x", "weekly_trend_up"],
    )
    monkeypatch.setattr(
        "options_helper.technicals_backtesting.strategies.registry.get_strategy",
        lambda _strategy: object,
    )

    def _run_backtest_stub(
        strategy_features: pd.DataFrame,
        strategy_class: object,
        bt_cfg: dict,
        strat_params: dict,
        *,
        warmup_bars: int,
        indicator_cols: tuple[str, ...],
    ) -> pd.Series:
        run_calls.append(
            {
                "columns": tuple(strategy_features.columns),
                "strategy_class": strategy_class,
                "bt_cfg": dict(bt_cfg),
                "strat_params": dict(strat_params),
                "warmup_bars": warmup_bars,
                "indicator_cols": tuple(indicator_cols),
            }
        )
        return pd.Series(
            {
                "Return [%]": 0.5,
                "_equity_curve": pd.DataFrame({"Equity": [100_000.0, 100_500.0]}),
                "_trades": pd.DataFrame({"Size": [1]}),
            }
        )

    monkeypatch.setattr(
        "options_helper.technicals_backtesting.backtest.runner.run_backtest",
        _run_backtest_stub,
    )

    result = run_technicals_backtest_batch_runtime(
        symbols=["spy", "qqq"],
        strategy="MeanReversionIBS",
        cfg=cfg,
        cache_dir=tmp_path / "candles",
        cli_commission=0.004,
    )

    assert result.success_count == 2
    assert result.failure_count == 0
    assert loaded_symbols == ["SPY", "QQQ"]
    assert len(run_calls) == 2
    for call in run_calls:
        assert call["columns"] == (
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "feature_x",
            "weekly_trend_up",
        )
        assert call["strategy_class"] is object
        assert call["bt_cfg"]["commission"] == 0.004
        assert call["bt_cfg"]["slippage_bps"] == 6.5
        assert call["strat_params"] == {"lookback_high": 10, "ibs_threshold": 0.3}
        assert call["warmup_bars"] == 7
        assert call["indicator_cols"] == ("feature_x", "weekly_trend_up")


def test_run_technicals_backtest_batch_runtime_rejects_disabled_strategy(tmp_path: Path) -> None:
    cfg = {
        "backtest": {"commission": 0.001, "slippage_bps": 3.0},
        "strategies": {
            "MeanReversionIBS": {
                "enabled": False,
                "defaults": {"lookback_high": 10},
            }
        },
    }

    with pytest.raises(ValueError, match="disabled"):
        run_technicals_backtest_batch_runtime(
            symbols="SPY",
            strategy="MeanReversionIBS",
            cfg=cfg,
            cache_dir=tmp_path / "candles",
        )
