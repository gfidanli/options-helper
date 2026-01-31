from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_helper.data.candles import CandleStore


def load_ohlc_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OHLC file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raise ValueError("Unsupported OHLC file format (use .csv or .parquet)")


def load_ohlc_from_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    store = CandleStore(cache_dir)
    return store.load(symbol)

