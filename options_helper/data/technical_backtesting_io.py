from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from options_helper.data.candles import CandleCacheError
from options_helper.data.store_factory import get_candle_store

logger = logging.getLogger(__name__)


def load_ohlc_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OHLC file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raise ValueError("Unsupported OHLC file format (use .csv or .parquet)")


def load_ohlc_from_cache(
    symbol: str,
    cache_dir: Path,
    *,
    backfill_if_missing: bool = False,
    period: str = "max",
    raise_on_backfill_error: bool = False,
) -> pd.DataFrame:
    store = get_candle_store(cache_dir)
    history = store.load(symbol)
    if history.empty and backfill_if_missing:
        try:
            history = store.get_daily_history(symbol, period=period)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to backfill OHLC for %s: %s", symbol, exc)
            if raise_on_backfill_error:
                raise CandleCacheError(str(exc)) from exc
            return pd.DataFrame()
    return history
