from __future__ import annotations

import numpy as np
import pandas as pd

from options_helper.analysis.tail_risk import TailRiskConfig, compute_tail_risk


def _synthetic_close(days: int = 400, *, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.02, size=days)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.Series(prices, index=idx, dtype="float64")


def test_tail_risk_is_deterministic_for_same_seed() -> None:
    close = _synthetic_close()
    cfg = TailRiskConfig(
        lookback_days=252,
        horizon_days=30,
        num_simulations=4000,
        seed=42,
        var_confidence=0.95,
    )
    result1 = compute_tail_risk(close, config=cfg)
    result2 = compute_tail_risk(close, config=cfg)

    assert result1.as_of == "2025-02-03"
    assert result1.daily_price_bands.shape[0] == 31
    assert result1.sample_price_paths.shape[0] == 31
    assert result1.end_returns.shape[0] == 4000
    assert np.allclose(result1.end_returns, result2.end_returns)
    assert result1.end_price_percentiles == result2.end_price_percentiles


def test_tail_risk_percentile_bands_and_var_cvar_sanity() -> None:
    close = _synthetic_close(days=500, seed=17)
    cfg = TailRiskConfig(
        lookback_days=300,
        horizon_days=40,
        num_simulations=5000,
        seed=5,
        var_confidence=0.95,
    )
    result = compute_tail_risk(close, config=cfg)

    assert (result.daily_price_bands["p05"] <= result.daily_price_bands["p50"]).all()
    assert (result.daily_price_bands["p50"] <= result.daily_price_bands["p95"]).all()
    assert result.var_return < 0.0
    assert result.cvar_return is not None
    assert result.cvar_return <= result.var_return


def test_tail_risk_warnings_for_short_history_and_degenerate_vol() -> None:
    short_close = _synthetic_close(days=20)
    short_cfg = TailRiskConfig(lookback_days=252, horizon_days=20, num_simulations=2000, seed=1)
    short_result = compute_tail_risk(short_close, config=short_cfg)
    assert "insufficient_history" in short_result.warnings

    flat_close = pd.Series(
        [100.0] * 120,
        index=pd.date_range("2025-01-01", periods=120, freq="D"),
        dtype="float64",
    )
    flat_cfg = TailRiskConfig(lookback_days=100, horizon_days=20, num_simulations=2000, seed=1)
    flat_result = compute_tail_risk(flat_close, config=flat_cfg)
    assert "degenerate_vol" in flat_result.warnings

