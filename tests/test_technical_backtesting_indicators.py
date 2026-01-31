from __future__ import annotations

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

