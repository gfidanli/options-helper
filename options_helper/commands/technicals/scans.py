from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from options_helper.commands.technicals.common import (
    _compute_extension_percentile_series,
    _date_only_label,
    _load_ohlc_df,
    _round2,
    _rsi_series,
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
    import pandas as pd

    from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
    from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags, rsi_regime_tag

    console = Console(width=200)

    if swing_left_bars < 1:
        raise typer.BadParameter("--swing-left-bars must be >= 1")
    if swing_right_bars < 1:
        raise typer.BadParameter("--swing-right-bars must be >= 1")
    if min_swing_distance_bars < 1:
        raise typer.BadParameter("--min-swing-distance-bars must be >= 1")
    if divergence_window_bars < 2:
        raise typer.BadParameter("--divergence-window-bars must be >= 2")

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

    events = extract_sfp_events(signals)
    event_rows: list[dict[str, object]] = [ev.to_dict() for ev in events]
    extension_series, extension_percentile_series = _compute_extension_percentile_series(signals)

    idx = signals.index
    asof_label = (
        idx.max().date().isoformat()
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0
        else (str(idx[-1]) if len(idx) else "-")
    )

    rsi_series = None
    if rsi_window > 0:
        rsi_series = _rsi_series(signals["Close"], window=int(rsi_window))

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

    open_series = signals["Open"].astype("float64")
    close_series = signals["Close"].astype("float64")
    index_to_pos = {ts: i for i, ts in enumerate(signals.index)}

    def _forward_return_pct(start_ts: pd.Timestamp, horizon: int) -> float | None:
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

    for ev in event_rows:
        ts = ev["timestamp"]
        try:
            event_ts = pd.Timestamp(ts)
        except Exception:  # noqa: BLE001
            event_ts = None

        if event_ts is not None and rsi_series is not None and event_ts in rsi_series.index:
            rsi_value = rsi_series.loc[event_ts]
            if pd.notna(rsi_value):
                ev["rsi"] = float(rsi_value)
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_value),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            else:
                ev["rsi"] = None
                ev["rsi_regime"] = None
        else:
            ev["rsi"] = None
            ev["rsi_regime"] = None

        if event_ts is not None and divergence_flags is not None and event_ts in divergence_flags.index:
            div = divergence_flags.loc[event_ts]
            ev["rsi_divergence_same_bar"] = div.get("divergence")
        else:
            ev["rsi_divergence_same_bar"] = None

        entry_anchor_ts = None
        entry_anchor_price = None
        i = index_to_pos.get(event_ts) if event_ts is not None else None
        if i is not None:
            entry_i = i + 1
            if entry_i < len(open_series):
                entry_candidate = open_series.iloc[entry_i]
                if pd.notna(entry_candidate):
                    entry_anchor_ts = signals.index[entry_i]
                    entry_anchor_price = float(entry_candidate)

        ev["timestamp"] = _date_only_label(ev.get("timestamp"))
        ev["swept_swing_timestamp"] = _date_only_label(ev.get("swept_swing_timestamp"))
        ev["entry_anchor_timestamp"] = (
            _date_only_label(entry_anchor_ts) if entry_anchor_ts is not None else None
        )
        ev["entry_anchor_price"] = _round2(entry_anchor_price)
        ev["candle_open"] = _round2(ev.get("candle_open"))
        ev["candle_high"] = _round2(ev.get("candle_high"))
        ev["candle_low"] = _round2(ev.get("candle_low"))
        ev["candle_close"] = _round2(ev.get("candle_close"))
        ev["sweep_level"] = _round2(ev.get("sweep_level"))
        ev["rsi"] = _round2(ev.get("rsi"))

        if event_ts is not None and event_ts in extension_series.index:
            ev["extension_atr"] = _round2(extension_series.loc[event_ts])
        else:
            ev["extension_atr"] = None
        if event_ts is not None and event_ts in extension_percentile_series.index:
            ev["extension_percentile"] = _round2(extension_percentile_series.loc[event_ts])
        else:
            ev["extension_percentile"] = None

        if event_ts is not None:
            ev["forward_returns_pct"] = {
                "1d": _forward_return_pct(event_ts, 1),
                "5d": _forward_return_pct(event_ts, 5),
                "10d": _forward_return_pct(event_ts, 10),
            }
        else:
            ev["forward_returns_pct"] = {"1d": None, "5d": None, "10d": None}

    bearish_events = [ev for ev in event_rows if ev.get("direction") == "bearish"]
    bullish_events = [ev for ev in event_rows if ev.get("direction") == "bullish"]

    payload = {
        "schema_version": 3,
        "symbol": symbol.upper() if symbol else "UNKNOWN",
        "asof": asof_label,
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

    md_lines: list[str] = [
        f"# SFP Scan ({payload['symbol']})",
        "",
        "Informational output only; not financial advice.",
        "",
        f"- As of: `{payload['asof']}`",
        f"- Timeframe: `{timeframe}`",
        (
            f"- Swings: highs `{payload['counts']['swing_highs']}`, "
            f"lows `{payload['counts']['swing_lows']}`"
        ),
        (
            f"- SFP events: bearish `{payload['counts']['bearish_sfp']}`, "
            f"bullish `{payload['counts']['bullish_sfp']}`"
        ),
        "",
        "## Recent Events",
        "",
    ]

    recent_events = event_rows[-20:]
    if not recent_events:
        md_lines.append("No SFP events found.")
    else:

        def _fmt_price(value: object) -> str:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}"

        def _fmt_pct(value: object) -> str:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}%"

        for ev in reversed(recent_events):
            regime = ev.get("rsi_regime")
            regime_text = f", rsi_regime={regime}" if regime else ""
            div_text = f", rsi_div={ev.get('rsi_divergence_same_bar')}" if ev.get("rsi_divergence_same_bar") else ""
            fwd = ev.get("forward_returns_pct", {}) or {}
            ext_pct = ev.get("extension_percentile")
            ext_pct_text = f"{float(ext_pct):.2f}" if ext_pct is not None else "n/a"
            md_lines.append(
                (
                    f"- `{ev['timestamp']}` {ev['direction']} SFP | "
                    f"sweep={_fmt_price(ev.get('sweep_level'))}, close={_fmt_price(ev.get('candle_close'))}, "
                    f"entry={_fmt_price(ev.get('entry_anchor_price'))}, "
                    f"swing={ev['swept_swing_timestamp']}, age={int(ev['bars_since_swing'])} bars"
                    f", ext_pct={ext_pct_text}, fwd(1/5/10d)="
                    f"{_fmt_pct(fwd.get('1d'))}/{_fmt_pct(fwd.get('5d'))}/{_fmt_pct(fwd.get('10d'))}"
                    f"{regime_text}{div_text}"
                )
            )

    markdown = "\n".join(md_lines).rstrip() + "\n"

    symbol_label = payload["symbol"]
    base = out / str(symbol_label)
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"{payload['asof']}.json"
    md_path = base / f"{payload['asof']}.md"

    if write_json:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if write_md:
        md_path.write_text(markdown, encoding="utf-8")
    if print_to_console:
        console.print(markdown)

    console.print(
        (
            f"SFP scan complete: bars={payload['counts']['bars']} "
            f"swings(high/low)={payload['counts']['swing_highs']}/{payload['counts']['swing_lows']} "
            f"events(bearish/bullish)={payload['counts']['bearish_sfp']}/{payload['counts']['bullish_sfp']}"
        )
    )
    if write_json:
        console.print(f"Wrote JSON: {json_path}")
    if write_md:
        console.print(f"Wrote Markdown: {md_path}")


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
    import pandas as pd

    from options_helper.analysis.msb import compute_msb_signals, extract_msb_events
    from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag

    console = Console(width=200)

    if swing_left_bars < 1:
        raise typer.BadParameter("--swing-left-bars must be >= 1")
    if swing_right_bars < 1:
        raise typer.BadParameter("--swing-right-bars must be >= 1")
    if min_swing_distance_bars < 1:
        raise typer.BadParameter("--min-swing-distance-bars must be >= 1")

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

    events = extract_msb_events(signals)
    event_rows: list[dict[str, object]] = [ev.to_dict() for ev in events]
    extension_series, extension_percentile_series = _compute_extension_percentile_series(signals)

    idx = signals.index
    asof_label = (
        idx.max().date().isoformat()
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0
        else (str(idx[-1]) if len(idx) else "-")
    )

    rsi_series = None
    if rsi_window > 0:
        rsi_series = _rsi_series(signals["Close"], window=int(rsi_window))

    open_series = signals["Open"].astype("float64")
    close_series = signals["Close"].astype("float64")
    index_to_pos = {ts: i for i, ts in enumerate(signals.index)}

    def _forward_return_pct(start_ts: pd.Timestamp, horizon: int) -> float | None:
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

    for ev in event_rows:
        ts = ev["timestamp"]
        try:
            event_ts = pd.Timestamp(ts)
        except Exception:  # noqa: BLE001
            event_ts = None

        if event_ts is not None and rsi_series is not None and event_ts in rsi_series.index:
            rsi_value = rsi_series.loc[event_ts]
            if pd.notna(rsi_value):
                ev["rsi"] = float(rsi_value)
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_value),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            else:
                ev["rsi"] = None
                ev["rsi_regime"] = None
        else:
            ev["rsi"] = None
            ev["rsi_regime"] = None

        entry_anchor_ts = None
        entry_anchor_price = None
        i = index_to_pos.get(event_ts) if event_ts is not None else None
        if i is not None:
            entry_i = i + 1
            if entry_i < len(open_series):
                entry_candidate = open_series.iloc[entry_i]
                if pd.notna(entry_candidate):
                    entry_anchor_ts = signals.index[entry_i]
                    entry_anchor_price = float(entry_candidate)

        ev["timestamp"] = _date_only_label(ev.get("timestamp"))
        ev["broken_swing_timestamp"] = _date_only_label(ev.get("broken_swing_timestamp"))
        ev["entry_anchor_timestamp"] = (
            _date_only_label(entry_anchor_ts) if entry_anchor_ts is not None else None
        )
        ev["entry_anchor_price"] = _round2(entry_anchor_price)
        ev["candle_open"] = _round2(ev.get("candle_open"))
        ev["candle_high"] = _round2(ev.get("candle_high"))
        ev["candle_low"] = _round2(ev.get("candle_low"))
        ev["candle_close"] = _round2(ev.get("candle_close"))
        ev["break_level"] = _round2(ev.get("break_level"))
        ev["rsi"] = _round2(ev.get("rsi"))

        if event_ts is not None and event_ts in extension_series.index:
            ev["extension_atr"] = _round2(extension_series.loc[event_ts])
        else:
            ev["extension_atr"] = None
        if event_ts is not None and event_ts in extension_percentile_series.index:
            ev["extension_percentile"] = _round2(extension_percentile_series.loc[event_ts])
        else:
            ev["extension_percentile"] = None

        if event_ts is not None:
            ev["forward_returns_pct"] = {
                "1d": _forward_return_pct(event_ts, 1),
                "5d": _forward_return_pct(event_ts, 5),
                "10d": _forward_return_pct(event_ts, 10),
            }
        else:
            ev["forward_returns_pct"] = {"1d": None, "5d": None, "10d": None}

    bullish_events = [ev for ev in event_rows if ev.get("direction") == "bullish"]
    bearish_events = [ev for ev in event_rows if ev.get("direction") == "bearish"]

    payload = {
        "schema_version": 2,
        "symbol": symbol.upper() if symbol else "UNKNOWN",
        "asof": asof_label,
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

    md_lines: list[str] = [
        f"# Market Structure Break Scan ({payload['symbol']})",
        "",
        "Informational output only; not financial advice.",
        "",
        f"- As of: `{payload['asof']}`",
        f"- Timeframe: `{timeframe}`",
        (
            f"- Swings: highs `{payload['counts']['swing_highs']}`, "
            f"lows `{payload['counts']['swing_lows']}`"
        ),
        (
            f"- MSB events: bullish `{payload['counts']['bullish_msb']}`, "
            f"bearish `{payload['counts']['bearish_msb']}`"
        ),
        "",
        "## Recent Events",
        "",
    ]

    recent_events = event_rows[-20:]
    if not recent_events:
        md_lines.append("No MSB events found.")
    else:

        def _fmt_price(value: object) -> str:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}"

        def _fmt_pct(value: object) -> str:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}%"

        for ev in reversed(recent_events):
            regime = ev.get("rsi_regime")
            regime_text = f", rsi_regime={regime}" if regime else ""
            fwd = ev.get("forward_returns_pct", {}) or {}
            ext_pct = ev.get("extension_percentile")
            ext_pct_text = f"{float(ext_pct):.2f}" if ext_pct is not None else "n/a"
            md_lines.append(
                (
                    f"- `{ev['timestamp']}` {ev['direction']} MSB | "
                    f"break={_fmt_price(ev.get('break_level'))}, close={_fmt_price(ev.get('candle_close'))}, "
                    f"entry={_fmt_price(ev.get('entry_anchor_price'))}, "
                    f"swing={ev['broken_swing_timestamp']}, age={int(ev['bars_since_swing'])} bars"
                    f", ext_pct={ext_pct_text}, fwd(1/5/10d)="
                    f"{_fmt_pct(fwd.get('1d'))}/{_fmt_pct(fwd.get('5d'))}/{_fmt_pct(fwd.get('10d'))}"
                    f"{regime_text}"
                )
            )

    markdown = "\n".join(md_lines).rstrip() + "\n"

    symbol_label = payload["symbol"]
    base = out / str(symbol_label)
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"{payload['asof']}.json"
    md_path = base / f"{payload['asof']}.md"

    if write_json:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if write_md:
        md_path.write_text(markdown, encoding="utf-8")
    if print_to_console:
        console.print(markdown)

    console.print(
        (
            f"MSB scan complete: bars={payload['counts']['bars']} "
            f"swings(high/low)={payload['counts']['swing_highs']}/{payload['counts']['swing_lows']} "
            f"events(bullish/bearish)={payload['counts']['bullish_msb']}/{payload['counts']['bearish_msb']}"
        )
    )
    if write_json:
        console.print(f"Wrote JSON: {json_path}")
    if write_md:
        console.print(f"Wrote Markdown: {md_path}")


def register(app: typer.Typer) -> None:
    app.command("sfp-scan")(technicals_sfp_scan)
    app.command("msb-scan")(technicals_msb_scan)


__all__ = ["register", "technicals_sfp_scan", "technicals_msb_scan"]
