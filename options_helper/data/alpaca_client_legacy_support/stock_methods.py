from __future__ import annotations

import os
import random
import time
from datetime import date, datetime
from typing import Any

import pandas as pd

from options_helper.data.market_types import DataFetchError


class _StockBarsDeps(dict[str, Any]):
    pass


def _call_with_backoff(make_call, *, deps: _StockBarsDeps, max_retries: int):
    for attempt in range(max_retries):
        try:
            return make_call()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries - 1:
                raise
            status_code = deps["extract_status_code"](exc)
            retry_after = deps["extract_retry_after_seconds"](exc)
            is_rate_limit = status_code == 429 or deps["is_rate_limit_error"](exc)
            if status_code is not None:
                if 400 <= status_code < 500 and status_code not in (408, 429):
                    raise
                should_retry = status_code >= 500 or status_code in (408, 429)
            else:
                should_retry = is_rate_limit or deps["is_timeout_error"](exc)
            if not should_retry:
                raise
            base_delay = 0.5 * (2**attempt)
            delay = base_delay
            if is_rate_limit:
                delay += random.uniform(0, base_delay * 0.25)
            if retry_after is not None:
                delay = max(delay, retry_after)
            time.sleep(delay)
    return None


def get_stock_bars_impl(
    client_obj: Any,
    symbol: str,
    *,
    start: date | datetime | None,
    end: date | datetime | None,
    interval: str,
    adjustment: str,
    deps: _StockBarsDeps,
) -> pd.DataFrame:
    alpaca_symbol = deps["to_alpaca_symbol"](symbol)
    if not alpaca_symbol:
        raise DataFetchError(f"Invalid symbol: {symbol}")

    timeframe = deps["resolve_timeframe"](interval)
    start_dt = deps["coerce_datetime"](start, end_of_day=False)
    end_dt = client_obj.effective_end(end, end_of_day=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise DataFetchError("End date is before start date for Alpaca bars request.")

    request_cls = deps["load_stock_bars_request"]()
    max_retries = max(1, deps["coerce_int"](os.getenv("OH_ALPACA_MAX_RETRIES"), default=3))

    try:
        if request_cls is not None:
            request = request_cls(
                symbol_or_symbols=alpaca_symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                adjustment=adjustment,
            )
            payload = _call_with_backoff(
                lambda: client_obj.stock_client.get_stock_bars(request),
                deps=deps,
                max_retries=max_retries,
            )
        else:
            payload = _call_with_backoff(
                lambda: client_obj.stock_client.get_stock_bars(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                    adjustment=adjustment,
                ),
                deps=deps,
                max_retries=max_retries,
            )
    except DataFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError(f"Failed to fetch Alpaca stock bars for {alpaca_symbol} ({interval}).") from exc

    df = deps["bars_to_dataframe"](payload, alpaca_symbol)
    return deps["normalize_stock_bars"](df, symbol=alpaca_symbol)


def get_stock_bars_intraday_impl(
    client_obj: Any,
    symbol: str,
    *,
    day: date,
    timeframe: str,
    feed: str | None,
    adjustment: str,
    deps: _StockBarsDeps,
) -> pd.DataFrame:
    alpaca_symbol = deps["to_alpaca_symbol"](symbol)
    if not alpaca_symbol:
        raise DataFetchError(f"Invalid symbol: {symbol}")

    market_tz = deps["load_market_tz"]()
    market_today = datetime.now(market_tz).date()
    if day > market_today:
        raise DataFetchError(f"Requested intraday day is in the future: {day.isoformat()}")

    timeframe_val = deps["resolve_timeframe"](timeframe)
    start_dt, day_end_dt = deps["market_day_bounds"](day, market_tz)
    end_dt = _intraday_end(client_obj, day=day, market_today=market_today, start_dt=start_dt, day_end_dt=day_end_dt)

    request_cls = deps["load_stock_bars_request"]()
    kwargs = {
        "symbol_or_symbols": alpaca_symbol,
        "timeframe": timeframe_val,
        "start": start_dt,
        "end": end_dt,
        "adjustment": adjustment,
        "feed": feed or client_obj._stock_feed,
    }

    try:
        if request_cls is not None:
            request_kwargs = deps["filter_kwargs"](request_cls, kwargs)
            request = request_cls(**request_kwargs)
            payload = client_obj.stock_client.get_stock_bars(request)
        else:
            payload = client_obj.stock_client.get_stock_bars(
                **deps["filter_kwargs"](client_obj.stock_client.get_stock_bars, kwargs)
            )
    except DataFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError(f"Failed to fetch Alpaca intraday stock bars for {alpaca_symbol} ({timeframe}).") from exc

    df = deps["bars_to_dataframe"](payload, alpaca_symbol)
    return deps["normalize_intraday_stock_bars"](df, symbol=alpaca_symbol)


def _intraday_end(
    client_obj: Any,
    *,
    day: date,
    market_today: date,
    start_dt: datetime,
    day_end_dt: datetime,
) -> datetime:
    if day != market_today:
        return day_end_dt
    end_dt = client_obj.effective_end(None, end_of_day=True)
    if end_dt is None:
        end_dt = day_end_dt
    end_dt = min(end_dt, day_end_dt)
    if end_dt < start_dt:
        end_dt = start_dt
    return end_dt
