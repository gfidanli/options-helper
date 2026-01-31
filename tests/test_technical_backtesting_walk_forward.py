from __future__ import annotations

from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
from options_helper.technicals_backtesting.strategies.trend_pullback_atr import TrendPullbackATR
from tests.technical_backtesting_helpers import make_synthetic_ohlc


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
