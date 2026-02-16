from __future__ import annotations

import random
import time
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from options_helper.data.market_types import DataFetchError


class _OptionBarsDeps(dict[str, Any]):
    pass


def get_option_bars_daily_full_impl(
    client_obj: Any,
    symbols: list[str],
    *,
    start: date | datetime | None,
    end: date | datetime | None,
    interval: str,
    feed: str | None,
    chunk_size: int,
    max_retries: int,
    page_limit: int | None,
    deps: _OptionBarsDeps,
) -> pd.DataFrame:
    unique_symbols = _unique_symbols(symbols)
    columns = ["contractSymbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    if not unique_symbols:
        return pd.DataFrame(columns=columns)

    timeframe = deps["resolve_timeframe"](interval)
    start_dt = deps["coerce_datetime"](start, end_of_day=False)
    end_dt = client_obj.effective_end(end, end_of_day=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise DataFetchError("End date is before start date for Alpaca option bars request.")

    method = _option_bars_method(client_obj)
    request_cls = deps["load_option_bars_request"]()
    feed_val = feed or client_obj._options_feed
    all_chunks = _collect_daily_chunks(
        method=method,
        request_cls=request_cls,
        unique_symbols=unique_symbols,
        timeframe=timeframe,
        start_dt=start_dt,
        end_dt=end_dt,
        feed_val=feed_val,
        chunk_size=_bounded_chunk_size(chunk_size),
        max_retries=_bounded_retries(max_retries),
        page_limit=page_limit,
        deps=deps,
    )
    if not all_chunks:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["contractSymbol", "ts"], keep="last")
    combined = combined.sort_values(["contractSymbol", "ts"], na_position="last")
    for col in columns:
        if col not in combined.columns:
            combined[col] = pd.NA
    return combined[columns].reset_index(drop=True)


def get_option_bars_intraday_impl(
    client_obj: Any,
    symbols: list[str],
    *,
    day: date,
    timeframe: str,
    feed: str | None,
    max_chunk_size: int,
    max_retries: int,
    deps: _OptionBarsDeps,
) -> pd.DataFrame:
    unique_symbols = _unique_symbols(symbols)
    columns = ["contractSymbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    if not unique_symbols:
        return pd.DataFrame(columns=columns)

    market_tz = deps["load_market_tz"]()
    market_today = datetime.now(market_tz).date()
    if day > market_today:
        raise DataFetchError(f"Requested intraday day is in the future: {day.isoformat()}")

    timeframe_val = deps["resolve_timeframe"](timeframe)
    start_dt, day_end_dt = deps["market_day_bounds"](day, market_tz)
    end_dt = _resolve_intraday_end(client_obj, day=day, market_today=market_today, start_dt=start_dt, day_end_dt=day_end_dt)

    method = _option_bars_method(client_obj)
    request_cls = deps["load_option_bars_request"]()
    feed_val = feed or client_obj._options_feed
    all_chunks = _collect_intraday_chunks(
        method=method,
        request_cls=request_cls,
        unique_symbols=unique_symbols,
        timeframe=timeframe_val,
        start_dt=start_dt,
        end_dt=end_dt,
        feed_val=feed_val,
        max_chunk_size=max_chunk_size,
        max_retries=max_retries,
        deps=deps,
    )
    if not all_chunks:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["contractSymbol", "timestamp"], keep="last")
    return combined.reset_index(drop=True)


def _collect_daily_chunks(
    *,
    method,
    request_cls,
    unique_symbols: list[str],
    timeframe,
    start_dt,
    end_dt,
    feed_val,
    chunk_size: int,
    max_retries: int,
    page_limit: int | None,
    deps: _OptionBarsDeps,
) -> list[pd.DataFrame]:
    all_chunks: list[pd.DataFrame] = []
    for chunk in _chunked(unique_symbols, chunk_size):
        payload = _request_chunk(
            method=method,
            request_cls=request_cls,
            chunk=chunk,
            timeframe=timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            feed_val=feed_val,
            page_token=None,
            max_retries=max_retries,
            deps=deps,
        )
        if payload is None:
            continue
        all_chunks.extend(_collect_daily_pages(payload, method, request_cls, chunk, timeframe, start_dt, end_dt, feed_val, max_retries, page_limit, deps))
    return all_chunks


def _collect_daily_pages(
    payload,
    method,
    request_cls,
    chunk,
    timeframe,
    start_dt,
    end_dt,
    feed_val,
    max_retries,
    page_limit,
    deps: _OptionBarsDeps,
) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    page_count = 1
    while payload is not None:
        df = deps["option_bars_to_dataframe"](payload)
        norm = deps["normalize_option_bars_daily_full"](df)
        if not norm.empty:
            out.append(norm)

        page_token = deps["extract_option_bars_page_token"](payload)
        if not page_token:
            break
        page_count += 1
        if page_limit is not None and page_count > page_limit:
            raise DataFetchError("Exceeded Alpaca option bars page limit.")
        payload = _request_chunk(
            method=method,
            request_cls=request_cls,
            chunk=chunk,
            timeframe=timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            feed_val=feed_val,
            page_token=page_token,
            max_retries=max_retries,
            deps=deps,
        )
    return out


def _collect_intraday_chunks(
    *,
    method,
    request_cls,
    unique_symbols: list[str],
    timeframe,
    start_dt,
    end_dt,
    feed_val,
    max_chunk_size: int,
    max_retries: int,
    deps: _OptionBarsDeps,
) -> list[pd.DataFrame]:
    all_chunks: list[pd.DataFrame] = []
    for chunk in _chunked(unique_symbols, max_chunk_size):
        payload = _request_chunk(
            method=method,
            request_cls=request_cls,
            chunk=chunk,
            timeframe=timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
            feed_val=feed_val,
            page_token=None,
            max_retries=max_retries,
            deps=deps,
        )
        if payload is None:
            continue
        df = deps["option_bars_to_dataframe"](payload)
        norm = deps["normalize_intraday_option_bars"](df)
        if not norm.empty:
            all_chunks.append(norm)
    return all_chunks


def _request_chunk(
    *,
    method,
    request_cls,
    chunk,
    timeframe,
    start_dt,
    end_dt,
    feed_val,
    page_token,
    max_retries,
    deps: _OptionBarsDeps,
):
    kwargs = {
        "symbol_or_symbols": chunk,
        "symbols": chunk,
        "timeframe": timeframe,
        "start": start_dt,
        "end": end_dt,
        "feed": feed_val,
        "page_token": page_token,
    }
    payload = None
    if request_cls is not None:
        try:
            request_kwargs = deps["filter_kwargs"](request_cls, kwargs)
            request = request_cls(**request_kwargs)
            payload = _call_with_backoff(lambda: method(request), max_retries=max_retries, deps=deps)
        except TypeError:
            payload = None
    if payload is None:
        payload = _call_with_backoff(
            lambda: method(**deps["filter_kwargs"](method, kwargs)),
            max_retries=max_retries,
            deps=deps,
        )
    return payload


def _call_with_backoff(make_call, *, max_retries: int, deps: _OptionBarsDeps):
    max_retries = _bounded_retries(max_retries)
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
            delay = 0.5 * (2**attempt)
            if is_rate_limit:
                delay += random.uniform(0, delay * 0.25)
            if retry_after is not None:
                delay = max(delay, retry_after)
            time.sleep(delay)
    return None


def _option_bars_method(client_obj: Any):
    method = getattr(client_obj.option_client, "get_option_bars", None)
    if method is None:
        raise DataFetchError("Alpaca data client missing get_option_bars.")
    return method


def _unique_symbols(symbols: list[str]) -> list[str]:
    raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
    return sorted({sym for sym in raw_symbols if sym})


def _chunked(values: list[str], chunk_size: int) -> Iterable[list[str]]:
    chunk_size = _bounded_chunk_size(chunk_size)
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def _bounded_chunk_size(chunk_size: int) -> int:
    size = int(chunk_size) if chunk_size else 200
    return size if size > 0 else 200


def _bounded_retries(max_retries: int) -> int:
    retries = int(max_retries) if max_retries else 1
    return retries if retries > 0 else 1


def _resolve_intraday_end(
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
