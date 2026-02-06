from __future__ import annotations

import warnings

import pandas as pd

from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.technicals_backtesting.pipeline import compute_features
from tests.technical_backtesting_helpers import make_synthetic_ohlc


def test_indicator_columns_and_no_lookahead() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=260, seed=7)

    features = compute_features(df, cfg)

    expected_cols = [
        "atr_14",
        "atrp_14",
        "bb_mavg_20",
        "bb_hband_20_2",
        "bb_lband_20_2",
        "bb_pband_20_2",
        "bb_wband_20_2",
        "rsi_14",
        "sma_20",
        "zscore_20",
        "extension_atr_20_14",
        "weekly_trend_up",
    ]
    for col in expected_cols:
        assert col in features.columns

    df_shifted = df.copy()
    df_shifted.iloc[-5:, df_shifted.columns.get_loc("Close")] += 5.0
    df_shifted.iloc[-5:, df_shifted.columns.get_loc("Open")] += 5.0
    df_shifted.iloc[-5:, df_shifted.columns.get_loc("High")] += 5.0
    df_shifted.iloc[-5:, df_shifted.columns.get_loc("Low")] += 5.0

    features_shifted = compute_features(df_shifted, cfg)

    cutoff = features.index[-30]
    for col in ["atr_14", "sma_20", "zscore_20"]:
        pd.testing.assert_series_equal(
            features.loc[:cutoff, col],
            features_shifted.loc[:cutoff, col],
            check_names=False,
        )


def test_compute_features_does_not_emit_pandas_fragmentation_or_downcast_warnings() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=260, seed=7)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = compute_features(df, cfg)

    messages = [str(w.message) for w in captured]
    assert not any("DataFrame is highly fragmented" in m for m in messages)
    assert not any("Downcasting object dtype arrays on .fillna" in m for m in messages)


def test_weekly_regime_logic_close_above_fast_is_more_permissive() -> None:
    cfg = load_technical_backtesting_config()
    cfg = dict(cfg)
    cfg["weekly_regime"] = dict(cfg["weekly_regime"])
    cfg["weekly_regime"]["enabled"] = True
    cfg["weekly_regime"]["resample_rule"] = "W-FRI"
    cfg["weekly_regime"]["ma_type"] = "sma"
    cfg["weekly_regime"]["fast_ma"] = 3
    cfg["weekly_regime"]["slow_ma"] = 6

    # Construct weekly closes where the last week closes above fast MA,
    # but fast MA is still below slow MA (so strict logic is False).
    weekly_closes = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 105]
    start = pd.Timestamp("2024-01-01")  # Monday
    dates = pd.bdate_range(start=start, periods=len(weekly_closes) * 5)
    close = []
    for w in weekly_closes:
        close.extend([float(w)] * 5)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1.0 for c in close],
            "Low": [c - 1.0 for c in close],
            "Close": close,
            "Volume": [1_000_000] * len(close),
        },
        index=dates,
    )

    cfg_strict = dict(cfg)
    cfg_strict["weekly_regime"] = dict(cfg["weekly_regime"])
    cfg_strict["weekly_regime"]["logic"] = "close_above_fast_and_fast_above_slow"
    feat_strict = compute_features(df, cfg_strict)

    cfg_relaxed = dict(cfg)
    cfg_relaxed["weekly_regime"] = dict(cfg["weekly_regime"])
    cfg_relaxed["weekly_regime"]["logic"] = "close_above_fast"
    feat_relaxed = compute_features(df, cfg_relaxed)

    last_day = df.index.max()
    assert bool(feat_relaxed.loc[last_day, "weekly_trend_up"]) is True
    assert bool(feat_strict.loc[last_day, "weekly_trend_up"]) is False


def test_compute_features_handles_short_history() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=7, seed=11)

    features = compute_features(df, cfg)

    assert not features.empty
    assert "atr_14" in features.columns
    assert features["atr_14"].isna().all()
