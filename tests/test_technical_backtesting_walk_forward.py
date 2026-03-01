from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.technicals_backtesting.backtest import walk_forward as walk_forward_module
from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
from options_helper.technicals_backtesting.strategies.trend_pullback_atr import TrendPullbackATR
from tests.technical_backtesting_helpers import make_synthetic_ohlc


def _stub_optimize_params(*args: Any, **kwargs: Any) -> tuple[dict[str, int], dict[str, float], None]:
    return {"atr_window": 14}, {"ret": 1.0, "dd": -1.0, "trades": 5.0}, None


def _stub_run_backtest(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {"ret": 1.0, "dd": -0.5, "trades": 5.0}


def _intraday_ohlc(rows: int) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02 14:30:00+00:00", periods=rows, freq="30min")
    base = pd.Series(range(rows), dtype=float).to_numpy()
    return pd.DataFrame(
        {
            "Open": 100.0 + base,
            "High": 101.0 + base,
            "Low": 99.0 + base,
            "Close": 100.5 + base,
        },
        index=idx,
    )


def _custom_score_cfg() -> dict:
    return {
        "return_key": "ret",
        "max_drawdown_key": "dd",
        "trades_key": "trades",
        "weights": {"drawdown_lambda": 0.0, "turnover_mu": 0.0},
        "min_trades": 1,
    }


def test_walk_forward_smoke() -> None:
    cfg = load_technical_backtesting_config()
    cfg["data"]["warmup_bars"] = 30
    cfg["optimization"]["method"] = "grid"
    cfg["optimization"]["min_train_bars"] = 120
    cfg["optimization"]["custom_score"]["min_trades"] = 1
    cfg["walk_forward"]["train_years"] = 1
    cfg["walk_forward"]["validate_months"] = 3
    cfg["walk_forward"]["step_months"] = 3
    cfg["walk_forward"]["min_history_years"] = 2
    cfg["walk_forward"]["selection"]["stability"]["max_validate_score_cv"] = 2.0

    strat_cfg = cfg["strategies"]["TrendPullbackATR"]
    strat_cfg["search_space"] = {
        "atr_window": [10, 14],
        "sma_window": [20],
        "z_window": [20],
        "add_z": [-1.0],
        "trim_ext_atr": [1.5],
        "stop_mult_atr": [2.0],
        "use_weekly_filter": [True],
    }

    df = make_synthetic_ohlc(rows=600, seed=11)
    features = compute_features(df, cfg)

    result = walk_forward_optimize(
        features,
        TrendPullbackATR,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        cfg["optimization"]["maximize"],
        cfg["optimization"]["method"],
        cfg["optimization"].get("sambo", {}),
        cfg["optimization"]["custom_score"],
        cfg["walk_forward"],
        strat_cfg["defaults"],
        warmup_bars=warmup_bars(cfg),
        min_train_bars=cfg["optimization"].get("min_train_bars", 0),
    )

    assert result.params
    assert "stable" in result.stability


def test_walk_forward_intraday_bar_splits_are_gap_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(walk_forward_module, "optimize_params", _stub_optimize_params)
    monkeypatch.setattr(walk_forward_module, "run_backtest", _stub_run_backtest)
    features = _intraday_ohlc(rows=40)
    walk_cfg = {
        "train_years": 10,
        "validate_months": 6,
        "step_months": 6,
        "min_history_years": 20,
        "min_history_bars": 12,
        "train_bars": 10,
        "validate_bars": 4,
        "step_bars": 3,
        "selection": {"stability": {"max_validate_score_cv": 2.0}},
    }
    result = walk_forward_optimize(
        features,
        TrendPullbackATR,
        {},
        {"atr_window": [14]},
        [],
        "custom_score",
        "grid",
        {},
        _custom_score_cfg(),
        walk_cfg,
        {"atr_window": 10},
    )

    assert not result.used_defaults
    assert len(result.folds) == 9
    for fold in result.folds:
        train_start_pos = features.index.get_loc(fold["train_start"])
        train_end_pos = features.index.get_loc(fold["train_end"])
        validate_start_pos = features.index.get_loc(fold["validate_start"])
        validate_end_pos = features.index.get_loc(fold["validate_end"])
        assert train_end_pos - train_start_pos + 1 == 10
        assert validate_end_pos - validate_start_pos + 1 == 4
        assert validate_start_pos == train_end_pos + 1


def test_walk_forward_min_history_bars_checked_after_warmup() -> None:
    features = _intraday_ohlc(rows=12)
    walk_cfg = {
        "train_years": 1,
        "validate_months": 1,
        "step_months": 1,
        "min_history_years": 0,
        "min_history_bars": 10,
        "train_bars": 6,
        "validate_bars": 3,
        "step_bars": 3,
        "selection": {"stability": {"max_validate_score_cv": 2.0}},
    }
    result = walk_forward_optimize(
        features,
        TrendPullbackATR,
        {},
        {"atr_window": [14]},
        [],
        "custom_score",
        "grid",
        {},
        _custom_score_cfg(),
        walk_cfg,
        {"atr_window": 10},
        warmup_bars=4,
    )

    assert result.used_defaults
    assert result.reason == "insufficient_history"
