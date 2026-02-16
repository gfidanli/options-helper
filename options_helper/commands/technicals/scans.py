from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console

from options_helper.commands.technicals.common import (
    _compute_extension_percentile_series,
    _date_only_label,
    _load_ohlc_df,
    _round2,
    _rsi_series,
)


def _validate_scan_args(*, swing_left_bars: int, swing_right_bars: int, min_swing_distance_bars: int) -> None:
    if swing_left_bars < 1:
        raise typer.BadParameter("--swing-left-bars must be >= 1")
    if swing_right_bars < 1:
        raise typer.BadParameter("--swing-right-bars must be >= 1")
    if min_swing_distance_bars < 1:
        raise typer.BadParameter("--min-swing-distance-bars must be >= 1")


def _validate_sfp_scan_args(
    *,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    divergence_window_bars: int,
) -> None:
    _validate_scan_args(
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
    )
    if divergence_window_bars < 2:
        raise typer.BadParameter("--divergence-window-bars must be >= 2")


def _asof_label(index: object) -> str:
    import pandas as pd

    if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
        return index.max().date().isoformat()
    return str(index[-1]) if len(index) else "-"


def _parse_event_timestamp(value: object) -> object | None:
    import pandas as pd

    try:
        return pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None


def _build_forward_return_lookup(signals) -> tuple[Callable[[object, int], float | None], object, dict[object, int]]:
    import pandas as pd

    open_series = signals["Open"].astype("float64")
    close_series = signals["Close"].astype("float64")
    index_to_pos = {ts: i for i, ts in enumerate(signals.index)}

    def _forward_return_pct(start_ts: object, horizon: int) -> float | None:
        i = index_to_pos.get(start_ts)
        if i is None:
            return None
        entry_i = i + 1
        j = i + int(horizon)
        if entry_i >= len(open_series) or j >= len(close_series) or j < entry_i:
            return None
        c0 = float(open_series.iloc[entry_i])
        c1 = float(close_series.iloc[j])
        if c0 == 0.0 or pd.isna(c0) or pd.isna(c1):
            return None
        return round((c1 / c0 - 1.0) * 100.0, 2)

    return _forward_return_pct, open_series, index_to_pos


def _entry_anchor(
    *,
    event_ts: object | None,
    signals,
    open_series,
    index_to_pos: dict[object, int],
) -> tuple[object | None, float | None]:
    import pandas as pd

    i = index_to_pos.get(event_ts) if event_ts is not None else None
    if i is None:
        return None, None
    entry_i = i + 1
    if entry_i >= len(open_series):
        return None, None
    candidate = open_series.iloc[entry_i]
    if pd.isna(candidate):
        return None, None
    return signals.index[entry_i], float(candidate)


def _attach_rsi_fields(
    event: dict[str, object],
    *,
    event_ts: object | None,
    rsi_series,
    rsi_regime_tag: Callable[..., str],
    rsi_overbought: float,
    rsi_oversold: float,
) -> None:
    import pandas as pd

    if event_ts is None or rsi_series is None or event_ts not in rsi_series.index:
        event["rsi"] = None
        event["rsi_regime"] = None
        return
    rsi_value = rsi_series.loc[event_ts]
    if pd.isna(rsi_value):
        event["rsi"] = None
        event["rsi_regime"] = None
        return
    event["rsi"] = float(rsi_value)
    event["rsi_regime"] = rsi_regime_tag(
        rsi_value=float(rsi_value),
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
    )


def _attach_divergence_field(event: dict[str, object], *, event_ts: object | None, divergence_flags) -> None:
    if event_ts is None or divergence_flags is None or event_ts not in divergence_flags.index:
        event["rsi_divergence_same_bar"] = None
        return
    divergence = divergence_flags.loc[event_ts]
    event["rsi_divergence_same_bar"] = divergence.get("divergence")


def _attach_common_event_fields(
    event: dict[str, object],
    *,
    event_ts: object | None,
    signals,
    open_series,
    index_to_pos: dict[object, int],
    extension_series,
    extension_percentile_series,
    forward_return_pct: Callable[[object, int], float | None],
    swing_key: str,
    level_key: str,
) -> None:
    entry_anchor_ts, entry_anchor_price = _entry_anchor(
        event_ts=event_ts,
        signals=signals,
        open_series=open_series,
        index_to_pos=index_to_pos,
    )
    event["timestamp"] = _date_only_label(event.get("timestamp"))
    event[swing_key] = _date_only_label(event.get(swing_key))
    event["entry_anchor_timestamp"] = _date_only_label(entry_anchor_ts) if entry_anchor_ts is not None else None
    event["entry_anchor_price"] = _round2(entry_anchor_price)
    event["candle_open"] = _round2(event.get("candle_open"))
    event["candle_high"] = _round2(event.get("candle_high"))
    event["candle_low"] = _round2(event.get("candle_low"))
    event["candle_close"] = _round2(event.get("candle_close"))
    event[level_key] = _round2(event.get(level_key))
    event["rsi"] = _round2(event.get("rsi"))

    if event_ts is not None and event_ts in extension_series.index:
        event["extension_atr"] = _round2(extension_series.loc[event_ts])
    else:
        event["extension_atr"] = None
    if event_ts is not None and event_ts in extension_percentile_series.index:
        event["extension_percentile"] = _round2(extension_percentile_series.loc[event_ts])
    else:
        event["extension_percentile"] = None

    if event_ts is None:
        event["forward_returns_pct"] = {"1d": None, "5d": None, "10d": None}
        return
    event["forward_returns_pct"] = {
        "1d": forward_return_pct(event_ts, 1),
        "5d": forward_return_pct(event_ts, 5),
        "10d": forward_return_pct(event_ts, 10),
    }


def _enrich_scan_events(
    events: list[dict[str, object]],
    *,
    signals,
    rsi_series,
    rsi_regime_tag: Callable[..., str],
    rsi_overbought: float,
    rsi_oversold: float,
    divergence_flags,
    include_divergence: bool,
    extension_series,
    extension_percentile_series,
    forward_return_pct: Callable[[object, int], float | None],
    open_series,
    index_to_pos: dict[object, int],
    swing_key: str,
    level_key: str,
) -> None:
    for event in events:
        event_ts = _parse_event_timestamp(event.get("timestamp"))
        _attach_rsi_fields(
            event,
            event_ts=event_ts,
            rsi_series=rsi_series,
            rsi_regime_tag=rsi_regime_tag,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
        )
        if include_divergence:
            _attach_divergence_field(event, event_ts=event_ts, divergence_flags=divergence_flags)
        _attach_common_event_fields(
            event,
            event_ts=event_ts,
            signals=signals,
            open_series=open_series,
            index_to_pos=index_to_pos,
            extension_series=extension_series,
            extension_percentile_series=extension_percentile_series,
            forward_return_pct=forward_return_pct,
            swing_key=swing_key,
            level_key=level_key,
        )


def _build_recent_events_lines(
    *,
    events: list[dict[str, object]],
    pattern_label: str,
    level_key: str,
    swing_key: str,
    include_divergence: bool,
) -> list[str]:
    lines: list[str] = []
    recent_events = events[-20:]
    if not recent_events:
        lines.append(f"No {pattern_label} events found.")
        return lines

    def _fmt_price(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2f}"

    def _fmt_pct(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2f}%"

    for event in reversed(recent_events):
        regime = event.get("rsi_regime")
        regime_text = f", rsi_regime={regime}" if regime else ""
        divergence = event.get("rsi_divergence_same_bar")
        divergence_text = f", rsi_div={divergence}" if include_divergence and divergence else ""
        fwd = event.get("forward_returns_pct", {}) or {}
        ext_pct = event.get("extension_percentile")
        ext_pct_text = f"{float(ext_pct):.2f}" if ext_pct is not None else "n/a"
        lines.append(
            (
                f"- `{event['timestamp']}` {event['direction']} {pattern_label.upper()} | "
                f"{level_key.replace('_level', '')}={_fmt_price(event.get(level_key))}, "
                f"close={_fmt_price(event.get('candle_close'))}, entry={_fmt_price(event.get('entry_anchor_price'))}, "
                f"swing={event[swing_key]}, age={int(event['bars_since_swing'])} bars, ext_pct={ext_pct_text}, "
                f"fwd(1/5/10d)={_fmt_pct(fwd.get('1d'))}/{_fmt_pct(fwd.get('5d'))}/{_fmt_pct(fwd.get('10d'))}"
                f"{regime_text}{divergence_text}"
            )
        )
    return lines


def _build_sfp_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# SFP Scan ({payload['symbol']})",
        "",
        "Informational output only; not financial advice.",
        "",
        f"- As of: `{payload['asof']}`",
        f"- Timeframe: `{payload['timeframe']}`",
        f"- Swings: highs `{payload['counts']['swing_highs']}`, lows `{payload['counts']['swing_lows']}`",
        f"- SFP events: bearish `{payload['counts']['bearish_sfp']}`, bullish `{payload['counts']['bullish_sfp']}`",
        "",
        "## Recent Events",
        "",
    ]
    lines.extend(
        _build_recent_events_lines(
            events=payload["events"],
            pattern_label="sfp",
            level_key="sweep_level",
            swing_key="swept_swing_timestamp",
            include_divergence=True,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_msb_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Market Structure Break Scan ({payload['symbol']})",
        "",
        "Informational output only; not financial advice.",
        "",
        f"- As of: `{payload['asof']}`",
        f"- Timeframe: `{payload['timeframe']}`",
        f"- Swings: highs `{payload['counts']['swing_highs']}`, lows `{payload['counts']['swing_lows']}`",
        f"- MSB events: bullish `{payload['counts']['bullish_msb']}`, bearish `{payload['counts']['bearish_msb']}`",
        "",
        "## Recent Events",
        "",
    ]
    lines.extend(
        _build_recent_events_lines(
            events=payload["events"],
            pattern_label="msb",
            level_key="break_level",
            swing_key="broken_swing_timestamp",
            include_divergence=False,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_scan_artifacts(
    *,
    payload: dict[str, Any],
    markdown: str,
    out: Path,
    write_json: bool,
    write_md: bool,
    print_to_console: bool,
    console: Console,
) -> tuple[Path, Path]:
    symbol_label = str(payload["symbol"])
    base = out / symbol_label
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"{payload['asof']}.json"
    md_path = base / f"{payload['asof']}.md"
    if write_json:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if write_md:
        md_path.write_text(markdown, encoding="utf-8")
    if print_to_console:
        console.print(markdown)
    return json_path, md_path


def _print_scan_summary(
    *,
    console: Console,
    payload: dict[str, Any],
    scan_name: str,
    count_keys: tuple[str, str],
    count_labels: tuple[str, str],
    json_path: Path,
    md_path: Path,
    write_json: bool,
    write_md: bool,
) -> None:
    left_key, right_key = count_keys
    left_label, right_label = count_labels
    console.print(
        (
            f"{scan_name} scan complete: bars={payload['counts']['bars']} "
            f"swings(high/low)={payload['counts']['swing_highs']}/{payload['counts']['swing_lows']} "
            f"events({left_label}/{right_label})={payload['counts'][left_key]}/{payload['counts'][right_key]}"
        )
    )
    if write_json:
        console.print(f"Wrote JSON: {json_path}")
    if write_md:
        console.print(f"Wrote Markdown: {md_path}")


def _build_sfp_payload(
    *,
    symbol: str | None,
    timeframe: str,
    signals,
    event_rows: list[dict[str, object]],
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    include_rsi_divergence: bool,
    divergence_window_bars: int,
) -> dict[str, Any]:
    bearish_events = [event for event in event_rows if event.get("direction") == "bearish"]
    bullish_events = [event for event in event_rows if event.get("direction") == "bullish"]
    return {
        "schema_version": 3,
        "symbol": symbol.upper() if symbol else "UNKNOWN",
        "asof": _asof_label(signals.index),
        "timeframe": timeframe,
        "config": {
            "swing_left_bars": int(swing_left_bars),
            "swing_right_bars": int(swing_right_bars),
            "min_swing_distance_bars": int(min_swing_distance_bars),
            "rsi_window": int(rsi_window),
            "rsi_overbought": float(rsi_overbought),
            "rsi_oversold": float(rsi_oversold),
            "include_rsi_divergence": bool(include_rsi_divergence),
            "divergence_window_bars": int(divergence_window_bars),
            "extension_sma_window": 20,
            "extension_atr_window": 14,
            "forward_returns_entry_anchor": "next_bar_open",
        },
        "counts": {
            "bars": int(len(signals)),
            "swing_highs": int(signals["swing_high"].sum()),
            "swing_lows": int(signals["swing_low"].sum()),
            "bearish_sfp": int(signals["bearish_sfp"].sum()),
            "bullish_sfp": int(signals["bullish_sfp"].sum()),
            "total_sfp_events": int(len(event_rows)),
        },
        "latest": {
            "bearish": bearish_events[-1] if bearish_events else None,
            "bullish": bullish_events[-1] if bullish_events else None,
        },
        "events": event_rows,
    }


def _build_msb_payload(
    *,
    symbol: str | None,
    timeframe: str,
    signals,
    event_rows: list[dict[str, object]],
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
) -> dict[str, Any]:
    bullish_events = [event for event in event_rows if event.get("direction") == "bullish"]
    bearish_events = [event for event in event_rows if event.get("direction") == "bearish"]
    return {
        "schema_version": 2,
        "symbol": symbol.upper() if symbol else "UNKNOWN",
        "asof": _asof_label(signals.index),
        "timeframe": timeframe,
        "config": {
            "swing_left_bars": int(swing_left_bars),
            "swing_right_bars": int(swing_right_bars),
            "min_swing_distance_bars": int(min_swing_distance_bars),
            "rsi_window": int(rsi_window),
            "rsi_overbought": float(rsi_overbought),
            "rsi_oversold": float(rsi_oversold),
            "extension_sma_window": 20,
            "extension_atr_window": 14,
            "forward_returns_entry_anchor": "next_bar_open",
        },
        "counts": {
            "bars": int(len(signals)),
            "swing_highs": int(signals["swing_high"].sum()),
            "swing_lows": int(signals["swing_low"].sum()),
            "bullish_msb": int(signals["bullish_msb"].sum()),
            "bearish_msb": int(signals["bearish_msb"].sum()),
            "total_msb_events": int(len(event_rows)),
        },
        "latest": {
            "bullish": bullish_events[-1] if bullish_events else None,
            "bearish": bearish_events[-1] if bearish_events else None,
        },
        "events": event_rows,
    }


def _prepare_sfp_scan_data(
    *,
    symbol: str | None,
    ohlc_path: Path | None,
    cache_dir: Path,
    timeframe: str,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    include_rsi_divergence: bool,
    divergence_window_bars: int,
    rsi_overbought: float,
    rsi_oversold: float,
) -> tuple[object, list[dict[str, object]], object, object, object]:
    from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
    from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for SFP scan.")
    try:
        signals = compute_sfp_signals(
            df,
            swing_left_bars=int(swing_left_bars),
            swing_right_bars=int(swing_right_bars),
            min_swing_distance_bars=int(min_swing_distance_bars),
            timeframe=timeframe,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if signals.empty:
        raise typer.BadParameter("No bars available after timeframe normalization.")
    event_rows: list[dict[str, object]] = [event.to_dict() for event in extract_sfp_events(signals)]
    extension_series, extension_percentile_series = _compute_extension_percentile_series(signals)
    rsi_series = _rsi_series(signals["Close"], window=int(rsi_window)) if rsi_window > 0 else None
    divergence_flags = None
    if include_rsi_divergence and rsi_series is not None:
        divergence_flags = compute_rsi_divergence_flags(
            close_series=signals["Close"],
            rsi_series=rsi_series,
            extension_percentile_series=None,
            window_days=int(divergence_window_bars),
            min_extension_days=0,
            min_extension_percentile=100.0,
            max_extension_percentile=0.0,
            min_price_delta_pct=0.0,
            min_rsi_delta=0.0,
            rsi_overbought=float(rsi_overbought),
            rsi_oversold=float(rsi_oversold),
            require_rsi_extreme=False,
        )
    return signals, event_rows, extension_series, extension_percentile_series, divergence_flags


def _prepare_msb_scan_data(
    *,
    symbol: str | None,
    ohlc_path: Path | None,
    cache_dir: Path,
    timeframe: str,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
) -> tuple[object, list[dict[str, object]], object, object]:
    from options_helper.analysis.msb import compute_msb_signals, extract_msb_events

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for MSB scan.")
    try:
        signals = compute_msb_signals(
            df,
            swing_left_bars=int(swing_left_bars),
            swing_right_bars=int(swing_right_bars),
            min_swing_distance_bars=int(min_swing_distance_bars),
            timeframe=timeframe,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if signals.empty:
        raise typer.BadParameter("No bars available after timeframe normalization.")
    event_rows: list[dict[str, object]] = [event.to_dict() for event in extract_msb_events(signals)]
    extension_series, extension_percentile_series = _compute_extension_percentile_series(signals)
    return signals, event_rows, extension_series, extension_percentile_series


def _build_enriched_sfp_payload(
    *,
    symbol: str | None,
    timeframe: str,
    signals,
    event_rows: list[dict[str, object]],
    extension_series,
    extension_percentile_series,
    divergence_flags,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    include_rsi_divergence: bool,
    divergence_window_bars: int,
    rsi_regime_tag: Callable[..., str],
) -> dict[str, Any]:
    rsi_series = _rsi_series(signals["Close"], window=int(rsi_window)) if rsi_window > 0 else None
    forward_return_pct, open_series, index_to_pos = _build_forward_return_lookup(signals)
    _enrich_scan_events(
        event_rows,
        signals=signals,
        rsi_series=rsi_series,
        rsi_regime_tag=rsi_regime_tag,
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        divergence_flags=divergence_flags,
        include_divergence=True,
        extension_series=extension_series,
        extension_percentile_series=extension_percentile_series,
        forward_return_pct=forward_return_pct,
        open_series=open_series,
        index_to_pos=index_to_pos,
        swing_key="swept_swing_timestamp",
        level_key="sweep_level",
    )
    return _build_sfp_payload(
        symbol=symbol,
        timeframe=timeframe,
        signals=signals,
        event_rows=event_rows,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        include_rsi_divergence=include_rsi_divergence,
        divergence_window_bars=divergence_window_bars,
    )


def _build_enriched_msb_payload(
    *,
    symbol: str | None,
    timeframe: str,
    signals,
    event_rows: list[dict[str, object]],
    extension_series,
    extension_percentile_series,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    rsi_regime_tag: Callable[..., str],
) -> dict[str, Any]:
    rsi_series = _rsi_series(signals["Close"], window=int(rsi_window)) if rsi_window > 0 else None
    forward_return_pct, open_series, index_to_pos = _build_forward_return_lookup(signals)
    _enrich_scan_events(
        event_rows,
        signals=signals,
        rsi_series=rsi_series,
        rsi_regime_tag=rsi_regime_tag,
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        divergence_flags=None,
        include_divergence=False,
        extension_series=extension_series,
        extension_percentile_series=extension_percentile_series,
        forward_return_pct=forward_return_pct,
        open_series=open_series,
        index_to_pos=index_to_pos,
        swing_key="broken_swing_timestamp",
        level_key="break_level",
    )
    return _build_msb_payload(
        symbol=symbol,
        timeframe=timeframe,
        signals=signals,
        event_rows=event_rows,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )


def _run_sfp_scan_impl(
    *,
    symbol: str | None,
    ohlc_path: Path | None,
    cache_dir: Path,
    timeframe: str,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    divergence_window_bars: int,
    include_rsi_divergence: bool,
    out: Path,
    write_json: bool,
    write_md: bool,
    print_to_console: bool,
) -> None:
    from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag
    console = Console(width=200)
    _validate_sfp_scan_args(swing_left_bars=swing_left_bars, swing_right_bars=swing_right_bars, min_swing_distance_bars=min_swing_distance_bars, divergence_window_bars=divergence_window_bars)
    signals, event_rows, extension_series, extension_percentile_series, divergence_flags = _prepare_sfp_scan_data(
        symbol=symbol,
        ohlc_path=ohlc_path,
        cache_dir=cache_dir,
        timeframe=timeframe,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        include_rsi_divergence=include_rsi_divergence,
        divergence_window_bars=divergence_window_bars,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    payload = _build_enriched_sfp_payload(
        symbol=symbol,
        timeframe=timeframe,
        signals=signals,
        event_rows=event_rows,
        extension_series=extension_series,
        extension_percentile_series=extension_percentile_series,
        divergence_flags=divergence_flags,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        include_rsi_divergence=include_rsi_divergence,
        divergence_window_bars=divergence_window_bars,
        rsi_regime_tag=rsi_regime_tag,
    )
    markdown = _build_sfp_markdown(payload)
    json_path, md_path = _write_scan_artifacts(
        payload=payload,
        markdown=markdown,
        out=out,
        write_json=write_json,
        write_md=write_md,
        print_to_console=print_to_console,
        console=console,
    )
    _print_scan_summary(
        console=console,
        payload=payload,
        scan_name="SFP",
        count_keys=("bearish_sfp", "bullish_sfp"),
        count_labels=("bearish", "bullish"),
        json_path=json_path,
        md_path=md_path,
        write_json=write_json, write_md=write_md,
    )


def _run_msb_scan_impl(
    *,
    symbol: str | None,
    ohlc_path: Path | None,
    cache_dir: Path,
    timeframe: str,
    swing_left_bars: int,
    swing_right_bars: int,
    min_swing_distance_bars: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    out: Path,
    write_json: bool,
    write_md: bool,
    print_to_console: bool,
) -> None:
    from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag

    console = Console(width=200)
    _validate_scan_args(
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
    )
    signals, event_rows, extension_series, extension_percentile_series = _prepare_msb_scan_data(
        symbol=symbol,
        ohlc_path=ohlc_path,
        cache_dir=cache_dir,
        timeframe=timeframe,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
    )
    payload = _build_enriched_msb_payload(
        symbol=symbol,
        timeframe=timeframe,
        signals=signals,
        event_rows=event_rows,
        extension_series=extension_series,
        extension_percentile_series=extension_percentile_series,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        rsi_regime_tag=rsi_regime_tag,
    )
    markdown = _build_msb_markdown(payload)
    json_path, md_path = _write_scan_artifacts(
        payload=payload,
        markdown=markdown,
        out=out,
        write_json=write_json,
        write_md=write_md,
        print_to_console=print_to_console,
        console=console,
    )
    _print_scan_summary(
        console=console,
        payload=payload,
        scan_name="MSB",
        count_keys=("bullish_msb", "bearish_msb"),
        count_labels=("bullish", "bearish"),
        json_path=json_path,
        md_path=md_path,
        write_json=write_json,
        write_md=write_md,
    )


def technicals_sfp_scan(
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    timeframe: str = typer.Option(
        "native",
        "--timeframe",
        help=(
            "Resample rule for OHLC bars (for example: native, W-FRI, 2W-FRI, 4H). "
            "Use native to keep input bars unchanged. Weekly rules are labeled to week-start Monday."
        ),
    ),
    swing_left_bars: int = typer.Option(2, "--swing-left-bars", help="Bars to the left of a swing point."),
    swing_right_bars: int = typer.Option(2, "--swing-right-bars", help="Bars to the right of a swing point."),
    min_swing_distance_bars: int = typer.Option(
        1,
        "--min-swing-distance-bars",
        help="Minimum bars between the swept swing point and the SFP candle.",
    ),
    rsi_window: int = typer.Option(
        14,
        "--rsi-window",
        help="RSI window for SFP event tagging. Set <= 0 to disable RSI fields.",
    ),
    rsi_overbought: float = typer.Option(70.0, "--rsi-overbought", help="RSI overbought threshold."),
    rsi_oversold: float = typer.Option(30.0, "--rsi-oversold", help="RSI oversold threshold."),
    divergence_window_bars: int = typer.Option(
        14,
        "--divergence-window-bars",
        help="Window for optional same-bar RSI divergence tagging.",
    ),
    include_rsi_divergence: bool = typer.Option(
        True,
        "--include-rsi-divergence/--no-rsi-divergence",
        help="Attach same-bar RSI divergence tags to SFP events.",
    ),
    out: Path = typer.Option(
        Path("data/reports/technicals/sfp"),
        "--out",
        help="Output root for SFP artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write Markdown artifact."),
    print_to_console: bool = typer.Option(False, "--print/--no-print", help="Print markdown to console."),
) -> None:
    """Detect swing highs/lows and bullish/bearish swing failure pattern (SFP) candles."""
    _run_sfp_scan_impl(
        symbol=symbol,
        ohlc_path=ohlc_path,
        cache_dir=cache_dir,
        timeframe=timeframe,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        divergence_window_bars=divergence_window_bars,
        include_rsi_divergence=include_rsi_divergence,
        out=out,
        write_json=write_json,
        write_md=write_md,
        print_to_console=print_to_console,
    )


def technicals_msb_scan(
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    timeframe: str = typer.Option(
        "native",
        "--timeframe",
        help=(
            "Resample rule for OHLC bars (for example: native, W-FRI, 2W-FRI, 4H). "
            "Use native to keep input bars unchanged. Weekly rules are labeled to week-start Monday."
        ),
    ),
    swing_left_bars: int = typer.Option(2, "--swing-left-bars", help="Bars to the left of a swing point."),
    swing_right_bars: int = typer.Option(2, "--swing-right-bars", help="Bars to the right of a swing point."),
    min_swing_distance_bars: int = typer.Option(
        1,
        "--min-swing-distance-bars",
        help="Minimum bars between the broken swing point and the MSB candle.",
    ),
    rsi_window: int = typer.Option(
        14,
        "--rsi-window",
        help="RSI window for MSB event tagging. Set <= 0 to disable RSI fields.",
    ),
    rsi_overbought: float = typer.Option(70.0, "--rsi-overbought", help="RSI overbought threshold."),
    rsi_oversold: float = typer.Option(30.0, "--rsi-oversold", help="RSI oversold threshold."),
    out: Path = typer.Option(
        Path("data/reports/technicals/msb"),
        "--out",
        help="Output root for MSB artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write Markdown artifact."),
    print_to_console: bool = typer.Option(False, "--print/--no-print", help="Print markdown to console."),
) -> None:
    """Detect bullish/bearish market structure breaks (MSB) from prior swing highs/lows."""
    _run_msb_scan_impl(
        symbol=symbol,
        ohlc_path=ohlc_path,
        cache_dir=cache_dir,
        timeframe=timeframe,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        out=out,
        write_json=write_json,
        write_md=write_md,
        print_to_console=print_to_console,
    )


def register(app: typer.Typer) -> None:
    app.command("sfp-scan")(technicals_sfp_scan)
    app.command("msb-scan")(technicals_msb_scan)


__all__ = ["register", "technicals_sfp_scan", "technicals_msb_scan"]
