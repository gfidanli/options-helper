from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, ClassVar

import pandas as pd
import yfinance as yf

from options_helper.data.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)


class CandleCacheError(RuntimeError):
    pass


HistoryFetcher = Callable[[str, date | None, date | None], pd.DataFrame]


def _parse_period_to_start(period: str, *, today: date) -> date | None:
    period = period.strip().lower()
    if period == "max":
        return None
    if period == "ytd":
        return date(today.year, 1, 1)

    units = {"d": 1, "wk": 7, "mo": 30, "y": 365}

    # yfinance periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    if period.endswith("wk"):
        n = int(period[:-2])
        days = n * units["wk"]
        return today - timedelta(days=days)
    for suffix in ("d", "mo", "y"):
        if period.endswith(suffix):
            n = int(period[: -len(suffix)])
            days = n * units[suffix]
            return today - timedelta(days=days)

    raise ValueError(f"Unsupported period format: {period}")


def _default_fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
    # NOTE: Kept for backward compatibility with older tests / call sites that pass a
    # custom fetcher. New code should prefer CandleStore(auto_adjust=..., back_adjust=...).
    ticker = yf.Ticker(symbol)
    if start is None and end is None:
        return ticker.history(period="max", interval="1d", auto_adjust=False, back_adjust=False)

    kwargs = {"interval": "1d", "auto_adjust": False, "back_adjust": False}
    if start is not None:
        kwargs["start"] = start.isoformat()
    if end is not None:
        kwargs["end"] = end.isoformat()
    return ticker.history(**kwargs)


def _yfinance_fetch(
    symbol: str,
    start: date | None,
    end: date | None,
    *,
    interval: str,
    auto_adjust: bool,
    back_adjust: bool,
) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if start is None and end is None:
        return ticker.history(period="max", interval=interval, auto_adjust=auto_adjust, back_adjust=back_adjust)

    kwargs: dict[str, object] = {"interval": interval, "auto_adjust": auto_adjust, "back_adjust": back_adjust}
    if start is not None:
        kwargs["start"] = start.isoformat()
    if end is not None:
        kwargs["end"] = end.isoformat()
    return ticker.history(**kwargs)


def _auto_adjust_from_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply yfinance-style auto_adjust locally using Adj Close / Close ratio.

    This mirrors yfinance.utils.auto_adjust:
    - Open/High/Low/Close are multiplied by (Adj Close / Close)
    - Close is replaced with Adj Close
    - "Adj Close" column is dropped
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if "Adj Close" not in df.columns or "Close" not in df.columns:
        raise CandleCacheError("Unable to auto-adjust candles: missing 'Adj Close' and/or 'Close' column")

    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    adj_close = pd.to_numeric(out["Adj Close"], errors="coerce")

    ratio = adj_close / close
    # Avoid blasting the series with NaNs due to a single bad row; treat invalid ratios as 1.0.
    ratio = ratio.where(ratio.notna() & (ratio != 0.0), other=1.0)

    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * ratio

    out = out.drop(columns=["Adj Close"])
    return out


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Normalize index to a timezone-naive DatetimeIndex to avoid mixed-tz concat issues.
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    if isinstance(idx, pd.DatetimeIndex):
        if idx.isna().all():
            raise CandleCacheError("Unable to normalize candle index to datetime")
        out = out.loc[~idx.isna()].copy()
        out.index = idx[~idx.isna()].tz_localize(None)
    else:
        raise CandleCacheError("Unable to normalize candle index to datetime")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


@dataclass(frozen=True)
class CandleStore:
    root_dir: Path
    # If provided, overrides yfinance fetching entirely (used for tests).
    fetcher: HistoryFetcher | None = None
    # If provided, uses a market data provider abstraction for history fetches.
    provider: MarketDataProvider | None = None
    backfill_days: int = 5
    interval: str = "1d"
    max_fetch_attempts: int = 3
    retry_backoff_seconds: float = 2.0

    # yfinance adjustments materially impact indicators/backtests.
    # Defaults follow the repo's technical_backtesting.yaml recommendation.
    auto_adjust: bool = True
    back_adjust: bool = False

    # yfinance can error on very old start dates (negative epoch). 1970-01-01 is a
    # pragmatic "max" backfill floor for modern equities/options workflows.
    _MAX_BACKFILL_START: ClassVar[date] = date(1970, 1, 1)
    _META_SCHEMA_VERSION: ClassVar[int] = 1

    def _path_for_symbol(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_")
        return self.root_dir / f"{safe}.csv"

    def _meta_path_for_symbol(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_")
        return self.root_dir / f"{safe}.meta.json"

    def load_meta(self, symbol: str) -> dict | None:
        """
        Best-effort read of cache metadata.

        Notes:
        - Legacy caches (pre-meta) are assumed to be auto_adjust=False/back_adjust=False.
        - If the meta file is unreadable, returns None.
        """
        path = self._meta_path_for_symbol(symbol)
        if not path.exists():
            return {
                "schema_version": 0,
                "symbol": symbol.upper(),
                "interval": "1d",
                "auto_adjust": False,
                "back_adjust": False,
                "source": "legacy",
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _write_meta(self, symbol: str, history: pd.DataFrame) -> None:
        path = self._meta_path_for_symbol(symbol)
        payload = {
            "schema_version": self._META_SCHEMA_VERSION,
            "symbol": symbol.upper(),
            "interval": self.interval,
            "auto_adjust": bool(self.auto_adjust),
            "back_adjust": bool(self.back_adjust),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(history)) if history is not None else 0,
            "start": None if history is None or history.empty else history.index.min().isoformat(),
            "end": None if history is None or history.empty else history.index.max().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _meta_matches_settings(self, meta: dict | None) -> bool:
        if not meta:
            return False
        try:
            return bool(meta.get("auto_adjust")) == bool(self.auto_adjust) and bool(meta.get("back_adjust")) == bool(
                self.back_adjust
            )
        except Exception:  # noqa: BLE001
            return False

    def _fetch(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        if self.fetcher is not None:
            return self.fetcher(symbol, start, end)
        if self.provider is not None:
            return self.provider.get_history(
                symbol,
                start=start,
                end=end,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
                back_adjust=self.back_adjust,
            )
        return _yfinance_fetch(
            symbol,
            start,
            end,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            back_adjust=self.back_adjust,
        )

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        current: Exception | None = exc
        for _ in range(4):
            if current is None:
                break
            name = current.__class__.__name__
            msg = str(current).lower()
            if (
                name == "YFRateLimitError"
                or "rate limit" in msg
                or "too many requests" in msg
                or "429" in msg
            ):
                return True
            next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
            current = next_exc if isinstance(next_exc, Exception) else None
        return False

    def _fetch_with_retry(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        attempt = 1
        while True:
            try:
                return self._fetch(symbol, start, end)
            except Exception as exc:  # noqa: BLE001
                if not self._is_rate_limit_error(exc) or attempt >= self.max_fetch_attempts:
                    raise
                delay = float(self.retry_backoff_seconds) * (2 ** (attempt - 1))
                if delay < 0:
                    delay = 0.0
                logger.warning(
                    "Rate limited fetching %s; retrying in %.1fs (%d/%d)",
                    symbol,
                    delay,
                    attempt,
                    self.max_fetch_attempts,
                )
                time.sleep(delay)
                attempt += 1

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path_for_symbol(symbol)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_history(df)

    def save(self, symbol: str, history: pd.DataFrame) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_symbol(symbol)
        history.to_csv(path)
        self._write_meta(symbol, history)

    def get_daily_history(self, symbol: str, *, period: str = "2y", today: date | None = None) -> pd.DataFrame:
        """
        Returns a locally cached daily OHLCV DataFrame, updating it incrementally from the provider
        (default: Yahoo via yfinance).

        Behavior:
        - Ensures the cache covers at least the requested `period` (best-effort).
        - Always refreshes the most recent candles by re-fetching a small tail window.
        """
        today = today or date.today()
        period_norm = period.strip().lower()
        desired_start = _parse_period_to_start(period_norm, today=today)
        wants_max = period_norm == "max"

        cached = self.load(symbol)
        cached_meta = self.load_meta(symbol) if not cached.empty else None

        # If a legacy/unversioned cache exists but we now want adjusted candles, upgrade it
        # locally (no network) using Adj Close ratio.
        if (
            not cached.empty
            and not self._meta_matches_settings(cached_meta)
            and bool(self.auto_adjust)
            and not bool(self.back_adjust)
            and cached_meta is not None
            and not bool(cached_meta.get("auto_adjust"))
            and not bool(cached_meta.get("back_adjust"))
            and "Adj Close" in cached.columns
        ):
            try:
                cached = _normalize_history(_auto_adjust_from_adj_close(cached))
                self.save(symbol, cached)
                cached_meta = self.load_meta(symbol)
            except Exception:  # noqa: BLE001
                # Fall back to a re-fetch below.
                cached = pd.DataFrame()
                cached_meta = None

        # If cache settings don't match, ignore cached data to avoid mixing adjusted/unadjusted series.
        if not cached.empty and not self._meta_matches_settings(cached_meta):
            cached = pd.DataFrame()

        merged = cached

        did_full_fetch = False

        # Backfill earlier history if required.
        if merged.empty:
            if wants_max:
                merged = _normalize_history(self._fetch_with_retry(symbol, None, None))
                did_full_fetch = True
            elif desired_start is None:
                merged = _normalize_history(self._fetch_with_retry(symbol, None, None))
                did_full_fetch = True
            else:
                merged = _normalize_history(self._fetch_with_retry(symbol, desired_start, None))
        elif wants_max:
            first_cached = merged.index.min().date()
            fetched = self._fetch_with_retry(symbol, self._MAX_BACKFILL_START, first_cached)
            fetched = _normalize_history(fetched)
            if not fetched.empty:
                merged = pd.concat([fetched, merged])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        elif desired_start is not None:
            first_cached = merged.index.min().date()
            if desired_start < first_cached:
                fetched = self._fetch_with_retry(symbol, desired_start, first_cached)
                fetched = _normalize_history(fetched)
                merged = pd.concat([fetched, merged])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        # Refresh the recent tail (captures late revisions / newly available data).
        if not did_full_fetch and not merged.empty and self.backfill_days > 0:
            last_cached_dt = merged.index.max()
            refresh_start = (last_cached_dt.date() - timedelta(days=self.backfill_days))
            fetched = self._fetch_with_retry(symbol, refresh_start, None)
            fetched = _normalize_history(fetched)
            if not fetched.empty:
                # If we are pulling adjusted candles and Yahoo reports a corporate action in the
                # refresh window, adjustment factors can change for *all* earlier rows. To avoid
                # stitching incompatible adjustment regimes, do a full refresh for the requested
                # period when we detect an action event.
                action_cols = [c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in fetched.columns]
                has_action = False
                for col in action_cols:
                    ser = pd.to_numeric(fetched[col], errors="coerce").fillna(0.0)
                    if (ser != 0.0).any():
                        has_action = True
                        break

                if has_action and (bool(self.auto_adjust) or bool(self.back_adjust)):
                    if wants_max or desired_start is None:
                        merged = _normalize_history(self._fetch_with_retry(symbol, None, None))
                    else:
                        merged = _normalize_history(self._fetch_with_retry(symbol, desired_start, None))
                    did_full_fetch = True
                else:
                    merged = pd.concat([merged, fetched])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        if merged is None:
            merged = pd.DataFrame()

        # Persist only if we have something; avoid writing empty files for invalid symbols.
        if not merged.empty:
            self.save(symbol, merged)

        return merged


def last_close(history: pd.DataFrame) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    val = history["Close"].dropna()
    if val.empty:
        return None
    return float(val.iloc[-1])


def close_asof(history: pd.DataFrame, as_of: date) -> float | None:
    """
    Return the most recent close at or before `as_of` (date).

    Intended for offline/deterministic workflows where an "as-of" snapshot date
    should align to the candle cache.
    """
    if history is None or history.empty or "Close" not in history.columns:
        return None

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    if isinstance(close.index, pd.DatetimeIndex):
        cutoff = pd.Timestamp(as_of)
        close = close.loc[close.index <= cutoff]
        if close.empty:
            return None

    return float(close.iloc[-1])
