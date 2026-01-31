from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlc(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0.0, 0.002, size=rows))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.002, size=rows)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.002, size=rows)))
    volume = rng.integers(100_000, 1_000_000, size=rows)
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )

