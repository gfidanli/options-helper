from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config

if TYPE_CHECKING:
    import pandas as pd
    from options_helper.analysis.strategy_simulator import StrategyRTarget

app = typer.Typer(help="Technical indicators + backtesting/optimization.")


def _load_ohlc_df(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
) -> pd.DataFrame:
    from options_helper.data.candles import CandleCacheError
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path

    if ohlc_path:
        return load_ohlc_from_path(ohlc_path)
    if symbol:
        try:
            return load_ohlc_from_cache(
                symbol,
                cache_dir,
                backfill_if_missing=True,
                period="max",
                raise_on_backfill_error=True,
            )
        except CandleCacheError as exc:
            raise typer.BadParameter(f"Failed to backfill OHLC for {symbol}: {exc}") from exc
    raise typer.BadParameter("Provide --ohlc-path or --symbol/--cache-dir")


def _stats_to_dict(stats: object | None) -> dict | None:
    import pandas as pd

    if stats is None:
        return None
    if isinstance(stats, pd.Series):
        return {k: v for k, v in stats.to_dict().items() if not str(k).startswith("_")}
    if isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": stats}


def _rsi_series(close: pd.Series, *, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _date_only_label(value: object) -> str:
    import pandas as pd

    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:  # noqa: BLE001
        return str(value).split("T")[0]


def _round2(value: object) -> float | None:
    import pandas as pd

    try:
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(val):
        return None
    return round(val, 2)


def _compute_extension_percentile_series(signals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    import pandas as pd

    close = signals["Close"].astype("float64")
    high = signals["High"].astype("float64")
    low = signals["Low"].astype("float64")

    # Use the same defaults used in technicals extension features.
    sma_window = 20
    atr_window = 14

    sma = close.rolling(window=sma_window, min_periods=sma_window).mean()
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    extension = (close - sma) / atr
    extension = extension.replace([float("inf"), float("-inf")], float("nan"))

    valid_ext = extension.dropna()
    percentile = pd.Series(index=signals.index, dtype="float64")
    if len(valid_ext) >= 2:
        # Percentile rank of each candle against all extension history available up to that candle.
        def _pct(arr: pd.Series) -> float:
            s = pd.Series(arr).dropna().astype("float64")
            if len(s) < 2:
                return float("nan")
            return float(s.rank(pct=True, method="average").iloc[-1] * 100.0)

        expanding_pct = valid_ext.expanding(min_periods=2).apply(_pct, raw=False)
        percentile.loc[expanding_pct.index] = expanding_pct

    return extension, percentile


@app.command("compute-indicators")
def technicals_compute_indicators(
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    output: Path | None = typer.Option(None, "--output", help="Output CSV/parquet path."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Compute indicators from OHLC data and optionally persist to disk."""
    from options_helper.technicals_backtesting.pipeline import compute_features

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for indicator computation.")

    features = compute_features(df, cfg)
    console.print(f"Computed features: {len(features)} rows, {len(features.columns)} columns")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".parquet":
            features.to_parquet(output)
        else:
            features.to_csv(output)
        console.print(f"Wrote features to {output}")


@app.command("sfp-scan")
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


@app.command("msb-scan")
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


@app.command("extension-stats")
def technicals_extension_stats(
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Symmetric tail threshold percentile (e.g. 5 => low<=5, high>=95). Overrides config tail_high_pct/tail_low_pct.",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Rolling window (years) for extension percentiles + tail events. Default: auto (1y if <5y history, else 3y).",
    ),
    out: Path | None = typer.Option(
        Path("data/reports/technicals/extension"),
        "--out",
        help="Output root for extension stats artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write Markdown artifact."),
    print_to_console: bool = typer.Option(False, "--print/--no-print", help="Print Markdown to console."),
    divergence_window_days: int = typer.Option(
        14, "--divergence-window-days", help="Lookback window (trading bars) for RSI divergence detection."
    ),
    divergence_min_extension_days: int = typer.Option(
        5,
        "--divergence-min-extension-days",
        help="Minimum days in the window where extension percentile is elevated/depressed to qualify.",
    ),
    divergence_min_extension_percentile: float | None = typer.Option(
        None,
        "--divergence-min-extension-percentile",
        help="High extension percentile threshold for bearish divergence gating (default: tail_high_pct).",
    ),
    divergence_max_extension_percentile: float | None = typer.Option(
        None,
        "--divergence-max-extension-percentile",
        help="Low extension percentile threshold for bullish divergence gating (default: tail_low_pct).",
    ),
    divergence_min_price_delta_pct: float = typer.Option(
        0.0,
        "--divergence-min-price-delta-pct",
        help="Minimum % move between swing points (in the divergence direction).",
    ),
    divergence_min_rsi_delta: float = typer.Option(
        0.0,
        "--divergence-min-rsi-delta",
        help="Minimum RSI difference between swing points.",
    ),
    rsi_overbought: float = typer.Option(70.0, "--rsi-overbought", help="RSI threshold for overbought tagging."),
    rsi_oversold: float = typer.Option(30.0, "--rsi-oversold", help="RSI threshold for oversold tagging."),
    require_rsi_extreme: bool = typer.Option(
        False,
        "--require-rsi-extreme/--allow-rsi-neutral",
        help="If set, only keep bearish divergences at overbought RSI and bullish divergences at oversold RSI.",
    ),
) -> None:
    """Compute extension percentile stats (tail events + rolling windows) from cached candles."""
    import numpy as np
    import pandas as pd

    from options_helper.technicals_backtesting.extension_percentiles import (
        build_weekly_extension_series,
        compute_extension_percentiles,
        rolling_percentile_rank,
    )
    from options_helper.technicals_backtesting.max_forward_returns import (
        forward_max_down_move,
        forward_max_up_move,
    )
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags, rsi_regime_tag

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for extension stats.")

    features = compute_features(df, cfg)
    w = warmup_bars(cfg)
    if w > 0 and len(features) > w:
        features = features.iloc[w:]
    elif w > 0 and len(features) <= w:
        console.print("[yellow]Warning:[/yellow] insufficient history for warmup; using full history.")
    if features.empty:
        raise typer.BadParameter("No features after warmup; check candle history.")

    atr_window = int(cfg["indicators"]["atr"]["window_default"])
    sma_window = int(cfg["indicators"]["sma"]["window_default"])
    ext_col = f"extension_atr_{sma_window}_{atr_window}"
    if ext_col not in features.columns:
        raise typer.BadParameter(f"Missing extension column: {ext_col}")

    ext_cfg = cfg.get("extension_percentiles", {})
    days_per_year = int(ext_cfg.get("days_per_year", 252))

    # Tail thresholds:
    # - Used to select tail events
    # - Used as default extension gating for RSI divergence (unless explicitly overridden)
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_high_pct = tail_high_cfg
        tail_low_pct = tail_low_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp

    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")

    # Rolling window selection for extension percentiles:
    # - If the ticker has <5 years of history, use a 1-year rolling window.
    # - Otherwise, use a 3-year rolling window.
    # Rationale: if window bars >= history bars, percentiles are only defined at the last bar (min_periods=window),
    # which yields very few tail events and weak divergence gating.
    available_bars = int(features[ext_col].dropna().shape[0])
    if percentile_window_years is None:
        history_years = (float(available_bars) / float(days_per_year)) if days_per_year > 0 else 0.0
        window_years = 1 if history_years < 5.0 else 3
    else:
        window_years = int(percentile_window_years)

    if window_years <= 0:
        raise typer.BadParameter("--percentile-window-years must be >= 1")

    windows_years = [window_years]

    forward_days_base = [int(d) for d in (ext_cfg.get("forward_days", [1, 3, 5, 10]) or [])]
    forward_days_daily = [
        int(d)
        for d in (
            ext_cfg.get("forward_days_daily", None)
            or sorted({*forward_days_base, 15})  # include +15D (2 trading weeks) by default
        )
    ]
    forward_days_weekly = [
        int(d) for d in (ext_cfg.get("forward_days_weekly", None) or forward_days_base or [1, 3, 5, 10])
    ]

    report_daily = compute_extension_percentiles(
        extension_series=features[ext_col],
        open_series=features["Open"],
        close_series=features["Close"],
        windows_years=windows_years,
        days_per_year=days_per_year,
        tail_high_pct=float(tail_high_pct),
        tail_low_pct=float(tail_low_pct),
        forward_days=forward_days_daily,
        include_tail_events=True,
    )
    weekly_rule = cfg["weekly_regime"].get("resample_rule", "W-FRI")
    weekly_candles = (
        df[["Open", "High", "Low", "Close"]]
        .resample(weekly_rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    weekly_ext, weekly_close = build_weekly_extension_series(
        df[["Open", "High", "Low", "Close"]],
        sma_window=sma_window,
        atr_window=atr_window,
        resample_rule=weekly_rule,
    )
    report_weekly = compute_extension_percentiles(
        extension_series=weekly_ext,
        open_series=weekly_candles["Open"],
        close_series=weekly_close,
        windows_years=windows_years,
        days_per_year=int(days_per_year / 5),
        tail_high_pct=float(tail_high_pct),
        tail_low_pct=float(tail_low_pct),
        forward_days=forward_days_weekly,
        include_tail_events=True,
    )

    if report_daily.asof == "-":
        fallback_daily = None
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            try:
                fallback_daily = df.index.max().date().isoformat()
            except Exception:  # noqa: BLE001
                fallback_daily = None
        if fallback_daily:
            report_daily = replace(report_daily, asof=fallback_daily)

    if report_weekly.asof == "-":
        fallback_weekly = None
        if weekly_close is not None and not weekly_close.empty:
            try:
                fallback_weekly = weekly_close.index.max().date().isoformat()
            except Exception:  # noqa: BLE001
                fallback_weekly = None
        if fallback_weekly:
            report_weekly = replace(report_weekly, asof=fallback_weekly)

    sym_label = symbol.upper() if symbol else "UNKNOWN"

    def _none_if_nan(val: object) -> object | None:
        try:
            if val is None:
                return None
            if isinstance(val, (float, int)) and pd.isna(val):
                return None
            if pd.isna(val):
                return None
            return val
        except Exception:  # noqa: BLE001
            return None

    # RSI config (used for event tagging + divergence enrichment).
    rsi_cfg = (cfg.get("indicators", {}) or {}).get("rsi", {}) or {}
    rsi_enabled = bool(rsi_cfg.get("enabled", False))
    rsi_window = int(rsi_cfg.get("window_default", 14)) if rsi_enabled else None
    rsi_col = f"rsi_{rsi_window}" if rsi_window is not None else None

    # Optional enrichment: RSI divergence (daily + weekly, anchored on swing points).
    rsi_divergence_cfg: dict | None = None
    rsi_divergence_daily: dict | None = None
    rsi_divergence_weekly: dict | None = None

    # Pre-align daily series for deterministic iloc-based lookups.
    ext_series_daily = features[ext_col].dropna()
    open_series_daily = features["Open"].reindex(ext_series_daily.index)
    close_series_daily = features["Close"].reindex(ext_series_daily.index)
    high_series_daily = features["High"].reindex(ext_series_daily.index) if "High" in features.columns else None
    low_series_daily = features["Low"].reindex(ext_series_daily.index) if "Low" in features.columns else None
    rsi_series_daily = (
        features[rsi_col].reindex(ext_series_daily.index) if rsi_col and rsi_col in features.columns else None
    )

    weekly_open_series = weekly_candles["Open"]
    weekly_close_series = weekly_candles["Close"]
    weekly_high_series = weekly_candles["High"]
    weekly_low_series = weekly_candles["Low"]

    weekly_rsi_series = None
    if rsi_window is not None and not weekly_close_series.empty:
        try:
            from ta.momentum import RSIIndicator

            weekly_rsi_series = RSIIndicator(close=weekly_close_series, window=int(rsi_window)).rsi()
        except Exception:  # noqa: BLE001
            weekly_rsi_series = None

    # Divergence config (shared between daily/weekly; window is interpreted as bars for each timeframe).
    min_ext_pct = (
        float(divergence_min_extension_percentile)
        if divergence_min_extension_percentile is not None
        else float(tail_high_pct)
    )
    max_ext_pct = (
        float(divergence_max_extension_percentile)
        if divergence_max_extension_percentile is not None
        else float(tail_low_pct)
    )

    if rsi_window is not None:
        rsi_divergence_cfg = {
            "window_bars": int(divergence_window_days),
            "min_extension_bars": int(divergence_min_extension_days),
            "min_extension_percentile": min_ext_pct,
            "max_extension_percentile": max_ext_pct,
            "min_price_delta_pct": float(divergence_min_price_delta_pct),
            "min_rsi_delta": float(divergence_min_rsi_delta),
            "rsi_overbought": float(rsi_overbought),
            "rsi_oversold": float(rsi_oversold),
            "require_rsi_extreme": bool(require_rsi_extreme),
            "rsi_window": int(rsi_window),
        }

    # Daily RSI divergence.
    try:
        tail_years = report_daily.tail_window_years or (
            max(report_daily.current_percentiles.keys()) if report_daily.current_percentiles else None
        )
        bars = int(tail_years * int(ext_cfg.get("days_per_year", 252))) if tail_years else None
        if rsi_series_daily is not None and bars and bars > 1:
            aligned = pd.concat(
                [
                    ext_series_daily.rename("ext"),
                    close_series_daily.rename("close"),
                    rsi_series_daily.rename("rsi"),
                ],
                axis=1,
            ).dropna()
            if not aligned.empty:
                ext_series = aligned["ext"]
                close_series = aligned["close"]
                rsi_series = aligned["rsi"]

                bars = bars if len(ext_series) >= bars else len(ext_series)
                ext_pct = rolling_percentile_rank(ext_series, bars)

                flags = compute_rsi_divergence_flags(
                    close_series=close_series,
                    rsi_series=rsi_series,
                    extension_percentile_series=ext_pct,
                    window_days=divergence_window_days,
                    min_extension_days=divergence_min_extension_days,
                    min_extension_percentile=min_ext_pct,
                    max_extension_percentile=max_ext_pct,
                    min_price_delta_pct=divergence_min_price_delta_pct,
                    min_rsi_delta=divergence_min_rsi_delta,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    require_rsi_extreme=require_rsi_extreme,
                )

                events = flags[(flags["bearish_divergence"]) | (flags["bullish_divergence"])]
                events_by_date: dict[str, dict] = {}
                for idx, row in events.iterrows():
                    d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                    events_by_date[d] = {
                        "date": d,
                        "divergence": _none_if_nan(row.get("divergence")),
                        "rsi_regime": _none_if_nan(row.get("rsi_regime")),
                        "swing1_date": _none_if_nan(row.get("swing1_date")),
                        "swing2_date": _none_if_nan(row.get("swing2_date")),
                        "close1": _none_if_nan(row.get("close1")),
                        "close2": _none_if_nan(row.get("close2")),
                        "rsi1": _none_if_nan(row.get("rsi1")),
                        "rsi2": _none_if_nan(row.get("rsi2")),
                        "price_delta_pct": _none_if_nan(row.get("price_delta_pct")),
                        "rsi_delta": _none_if_nan(row.get("rsi_delta")),
                    }

                recent = flags.tail(max(1, int(divergence_window_days) + 2))
                last_bearish = None
                last_bullish = None
                for idx, row in reversed(list(recent.iterrows())):
                    if last_bearish is None and bool(row.get("bearish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bearish = events_by_date.get(d)
                    if last_bullish is None and bool(row.get("bullish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bullish = events_by_date.get(d)
                    if last_bearish is not None and last_bullish is not None:
                        break

                rsi_divergence_daily = {
                    "asof": report_daily.asof,
                    "current": {
                        "bearish": last_bearish,
                        "bullish": last_bullish,
                    },
                    "events_by_date": events_by_date,
                }
    except Exception:  # noqa: BLE001
        rsi_divergence_daily = None

    # Weekly RSI divergence (weekly bars).
    try:
        if weekly_rsi_series is not None and not weekly_ext.empty:
            tail_years_w = report_weekly.tail_window_years or (
                max(report_weekly.current_percentiles.keys()) if report_weekly.current_percentiles else None
            )
            bars_w = int(tail_years_w * int(ext_cfg.get("days_per_year", 252) / 5)) if tail_years_w else None
            ext_w = weekly_ext.dropna()
            aligned_w = pd.concat(
                [
                    ext_w.rename("ext"),
                    weekly_close_series.reindex(ext_w.index).rename("close"),
                    weekly_rsi_series.reindex(ext_w.index).rename("rsi"),
                ],
                axis=1,
            ).dropna()
            if not aligned_w.empty and bars_w and bars_w > 1:
                ext_w = aligned_w["ext"]
                close_w = aligned_w["close"]
                rsi_w = aligned_w["rsi"]

                bars_w = bars_w if len(ext_w) >= bars_w else len(ext_w)
                ext_pct_w = rolling_percentile_rank(ext_w, bars_w)

                flags_w = compute_rsi_divergence_flags(
                    close_series=close_w,
                    rsi_series=rsi_w,
                    extension_percentile_series=ext_pct_w,
                    window_days=divergence_window_days,
                    min_extension_days=divergence_min_extension_days,
                    min_extension_percentile=min_ext_pct,
                    max_extension_percentile=max_ext_pct,
                    min_price_delta_pct=divergence_min_price_delta_pct,
                    min_rsi_delta=divergence_min_rsi_delta,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    require_rsi_extreme=require_rsi_extreme,
                )

                events_w = flags_w[(flags_w["bearish_divergence"]) | (flags_w["bullish_divergence"])]
                events_by_date_w: dict[str, dict] = {}
                for idx, row in events_w.iterrows():
                    d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                    events_by_date_w[d] = {
                        "date": d,
                        "divergence": _none_if_nan(row.get("divergence")),
                        "rsi_regime": _none_if_nan(row.get("rsi_regime")),
                        "swing1_date": _none_if_nan(row.get("swing1_date")),
                        "swing2_date": _none_if_nan(row.get("swing2_date")),
                        "close1": _none_if_nan(row.get("close1")),
                        "close2": _none_if_nan(row.get("close2")),
                        "rsi1": _none_if_nan(row.get("rsi1")),
                        "rsi2": _none_if_nan(row.get("rsi2")),
                        "price_delta_pct": _none_if_nan(row.get("price_delta_pct")),
                        "rsi_delta": _none_if_nan(row.get("rsi_delta")),
                    }

                recent_w = flags_w.tail(max(1, int(divergence_window_days) + 2))
                last_bearish_w = None
                last_bullish_w = None
                for idx, row in reversed(list(recent_w.iterrows())):
                    if last_bearish_w is None and bool(row.get("bearish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bearish_w = events_by_date_w.get(d)
                    if last_bullish_w is None and bool(row.get("bullish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bullish_w = events_by_date_w.get(d)
                    if last_bearish_w is not None and last_bullish_w is not None:
                        break

                rsi_divergence_weekly = {
                    "asof": report_weekly.asof,
                    "current": {
                        "bearish": last_bearish_w,
                        "bullish": last_bullish_w,
                    },
                    "events_by_date": events_by_date_w,
                }
    except Exception:  # noqa: BLE001
        rsi_divergence_weekly = None

    ext_cfg_effective = dict(ext_cfg or {})
    ext_cfg_effective["windows_years"] = windows_years
    ext_cfg_effective["tail_high_pct"] = float(tail_high_pct)
    ext_cfg_effective["tail_low_pct"] = float(tail_low_pct)
    ext_cfg_effective["forward_days_daily"] = forward_days_daily
    ext_cfg_effective["forward_days_weekly"] = forward_days_weekly

    max_return_horizons_days = {"1w": 5, "4w": 20, "3m": 63, "6m": 126, "9m": 189, "1y": 252}

    payload = {
        "schema_version": 5,
        "symbol": sym_label,
        "asof": report_daily.asof,
        "config": {
            "extension_percentiles": ext_cfg_effective,
            "atr_window": atr_window,
            "sma_window": sma_window,
            "extension_column": ext_col,
            "rsi_divergence": rsi_divergence_cfg,
            "max_forward_returns": {
                "method": "directional_mfe",  # low-tail uses High (up move), high-tail uses Low (down move)
                "entry_anchor": "next_bar_open",
                "horizons_days": max_return_horizons_days,
            },
        },
        "report_daily": asdict(report_daily),
        "report_weekly": asdict(report_weekly),
        "rsi_divergence_daily": rsi_divergence_daily,
        "rsi_divergence_weekly": rsi_divergence_weekly,
    }

    # Enrich tail events (daily + weekly) with:
    # - RSI-at-event regime
    # - max-upside forward returns (High-based, MFE-style)
    # - divergence details (if available)
    # - weekly context attached to daily tail events

    daily = payload.get("report_daily", {}) or {}
    daily_tail_events = daily.get("tail_events", []) or []

    weekly = payload.get("report_weekly", {}) or {}
    weekly_tail_events = weekly.get("tail_events", []) or []

    # Deterministic iloc lookup: date string -> position in the aligned daily extension series.
    daily_date_to_iloc: dict[str, int] = {}
    for i, idx in enumerate(ext_series_daily.index):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        daily_date_to_iloc[d] = i

    # Weekly extension percentile series (for daily context) using the same window as the weekly report.
    weekly_ext_pct = None
    try:
        tail_years_w = report_weekly.tail_window_years or (
            max(report_weekly.current_percentiles.keys()) if report_weekly.current_percentiles else None
        )
        bars_w = int(tail_years_w * int(ext_cfg.get("days_per_year", 252) / 5)) if tail_years_w else None
        ext_w = weekly_ext.dropna()
        if bars_w and bars_w > 1 and not ext_w.empty:
            bars_w = bars_w if len(ext_w) >= bars_w else len(ext_w)
            if bars_w > 1:
                weekly_ext_pct = rolling_percentile_rank(ext_w, bars_w)
    except Exception:  # noqa: BLE001
        weekly_ext_pct = None

    weekly_pct_on_daily = (
        weekly_ext_pct.reindex(ext_series_daily.index, method="ffill") if weekly_ext_pct is not None else None
    )

    weekly_rsi_regime_on_daily = None
    if weekly_rsi_series is not None:
        try:
            weekly_rsi_regime = weekly_rsi_series.dropna().apply(
                lambda v: rsi_regime_tag(
                    rsi_value=float(v), rsi_overbought=float(rsi_overbought), rsi_oversold=float(rsi_oversold)
                )
            )
            weekly_rsi_regime_on_daily = weekly_rsi_regime.reindex(ext_series_daily.index, method="ffill")
        except Exception:  # noqa: BLE001
            weekly_rsi_regime_on_daily = None

    weekly_div_type_on_daily = None
    weekly_div_rsi_on_daily = None
    by_weekly_date = (
        (rsi_divergence_weekly or {}).get("events_by_date", {}) if isinstance(rsi_divergence_weekly, dict) else {}
    )
    if isinstance(by_weekly_date, dict) and by_weekly_date:
        try:
            s_div = pd.Series(index=weekly_close_series.index, dtype="object")
            s_tag = pd.Series(index=weekly_close_series.index, dtype="object")
            for d, ev in by_weekly_date.items():
                try:
                    ts = pd.Timestamp(d)
                except Exception:  # noqa: BLE001
                    continue
                if ts in s_div.index:
                    s_div.loc[ts] = (ev or {}).get("divergence")
                    s_tag.loc[ts] = (ev or {}).get("rsi_regime")
            weekly_div_type_on_daily = s_div.reindex(ext_series_daily.index, method="ffill")
            weekly_div_rsi_on_daily = s_tag.reindex(ext_series_daily.index, method="ffill")
        except Exception:  # noqa: BLE001
            weekly_div_type_on_daily = None
            weekly_div_rsi_on_daily = None

    by_daily_date = (
        (rsi_divergence_daily or {}).get("events_by_date", {}) if isinstance(rsi_divergence_daily, dict) else {}
    )

    # Daily tail events enrichment.
    for ev in daily_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_daily_date.get(d) if isinstance(by_daily_date, dict) else None

        i = daily_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = None
        if i is not None and rsi_series_daily is not None:
            rsi_val = _none_if_nan(rsi_series_daily.iloc[i])
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = None
        if rsi_val is not None:
            try:
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_val),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            except Exception:  # noqa: BLE001
                ev["rsi_regime"] = None

        # Max favorable move + drawdown (directional):
        # - low tail: favorable is bounce using High (>= 0), drawdown is pullback using Low (<= 0)
        # - high tail: favorable is pullback using Low (<= 0), drawdown is squeeze using High (<= 0)
        max_up_short: dict[int, float | None] = {int(h): None for h in forward_days_daily}
        max_down_short: dict[int, float | None] = {int(h): None for h in forward_days_daily}
        max_up_long: dict[str, float | None] = {k: None for k in max_return_horizons_days.keys()}
        max_down_long: dict[str, float | None] = {k: None for k in max_return_horizons_days.keys()}

        if i is not None:
            if high_series_daily is not None:
                for h in forward_days_daily:
                    r = forward_max_up_move(
                        open_series=open_series_daily,
                        high_series=high_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_up_short[int(h)] = None if r is None else float(r)
                for label, h in max_return_horizons_days.items():
                    r = forward_max_up_move(
                        open_series=open_series_daily,
                        high_series=high_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_up_long[str(label)] = None if r is None else float(r)

            if low_series_daily is not None:
                for h in forward_days_daily:
                    r = forward_max_down_move(
                        open_series=open_series_daily,
                        low_series=low_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_down_short[int(h)] = None if r is None else float(r)
                for label, h in max_return_horizons_days.items():
                    r = forward_max_down_move(
                        open_series=open_series_daily,
                        low_series=low_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_down_long[str(label)] = None if r is None else float(r)

        def _neg0_to_0(val: float) -> float:
            return 0.0 if float(val) == 0.0 else float(val)

        direction = ev.get("direction")
        if direction == "low":
            max_fav_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            max_fav_long = {k: _none_if_nan(v) for k, v in max_up_long.items()}
            dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            dd_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
        elif direction == "high":
            max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            max_fav_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
            # Adverse move (drawdown) for high-tail mean reversion is a further squeeze up.
            dd_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            dd_long = {k: _none_if_nan(v) for k, v in max_up_long.items()}
        else:
            max_fav_short = {int(h): None for h in forward_days_daily}
            max_fav_long = {k: None for k in max_return_horizons_days.keys()}
            dd_short = {int(h): None for h in forward_days_daily}
            dd_long = {k: None for k in max_return_horizons_days.keys()}

        # Retain up/down maps for debugging, but prefer *favorable* maps in displays/summaries.
        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["max_up_returns"] = max_up_long
        ev["max_down_returns"] = max_down_long
        ev["max_fav_returns"] = max_fav_long
        ev["forward_drawdown_returns"] = dd_short
        ev["drawdown_returns"] = dd_long

        # Weekly context (ffill weekly values onto daily dates).
        wctx: dict[str, object] = {
            "extension_percentile": None,
            "rsi_regime": None,
            "divergence": None,
            "divergence_rsi_regime": None,
        }
        if i is not None:
            if weekly_pct_on_daily is not None:
                wctx["extension_percentile"] = _none_if_nan(weekly_pct_on_daily.iloc[i])
            if weekly_rsi_regime_on_daily is not None:
                wctx["rsi_regime"] = _none_if_nan(weekly_rsi_regime_on_daily.iloc[i])
            if weekly_div_type_on_daily is not None:
                wctx["divergence"] = _none_if_nan(weekly_div_type_on_daily.iloc[i])
            if weekly_div_rsi_on_daily is not None:
                wctx["divergence_rsi_regime"] = _none_if_nan(weekly_div_rsi_on_daily.iloc[i])
        ev["weekly_context"] = wctx

    # Weekly tail events enrichment (RSI-at-event + divergence + max-up returns).
    weekly_date_to_iloc: dict[str, int] = {}
    for i, idx in enumerate(weekly_close_series.index):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        weekly_date_to_iloc[d] = i

    by_weekly_date = (
        (rsi_divergence_weekly or {}).get("events_by_date", {}) if isinstance(rsi_divergence_weekly, dict) else {}
    )
    for ev in weekly_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_weekly_date.get(d) if isinstance(by_weekly_date, dict) else None

        i = weekly_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = None
        if i is not None and weekly_rsi_series is not None:
            rsi_val = _none_if_nan(weekly_rsi_series.iloc[i])
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = None
        if rsi_val is not None:
            try:
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_val),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            except Exception:  # noqa: BLE001
                ev["rsi_regime"] = None

        max_up_short: dict[int, float | None] = {int(h): None for h in forward_days_weekly}
        max_down_short: dict[int, float | None] = {int(h): None for h in forward_days_weekly}
        if i is not None:
            for h in forward_days_weekly:
                r_up = forward_max_up_move(
                    open_series=weekly_open_series,
                    high_series=weekly_high_series,
                    start_iloc=i,
                    horizon_bars=int(h),
                )
                r_dn = forward_max_down_move(
                    open_series=weekly_open_series,
                    low_series=weekly_low_series,
                    start_iloc=i,
                    horizon_bars=int(h),
                )
                max_up_short[int(h)] = None if r_up is None else float(r_up)
                max_down_short[int(h)] = None if r_dn is None else float(r_dn)

        def _neg0_to_0(val: float) -> float:
            return 0.0 if float(val) == 0.0 else float(val)

        direction = ev.get("direction")
        if direction == "low":
            max_fav_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
        elif direction == "high":
            max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            dd_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
        else:
            max_fav_short = {int(h): None for h in forward_days_weekly}
            dd_short = {int(h): None for h in forward_days_weekly}

        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["forward_drawdown_returns"] = dd_short

    # Store max favorable-move summaries (JSON) for both tails:
    # - low tail: bounce (max-up move using High)
    # - high tail: pullback (max-down move using Low)
    max_move_summary_daily: dict[str, object] = {"horizons_days": max_return_horizons_days, "buckets": []}
    try:
        def _quantile(values: list[float], q: float) -> float | None:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return None
            return float(np.percentile(vals, q * 100.0))

        buckets = [
            ("low_tail_all", "Low tail (all)", lambda ev: ev.get("direction") == "low"),
            (
                "low_tail_rsi_oversold",
                "Low tail + RSI oversold (event)",
                lambda ev: ev.get("direction") == "low" and ev.get("rsi_regime") == "oversold",
            ),
            (
                "low_tail_bull_div",
                "Low tail + bullish divergence",
                lambda ev: ev.get("direction") == "low"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish",
            ),
            (
                "low_tail_bull_div_rsi_oversold",
                "Low tail + bullish divergence + RSI oversold (event)",
                lambda ev: ev.get("direction") == "low"
                and ev.get("rsi_regime") == "oversold"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish",
            ),
            ("high_tail_all", "High tail (all)", lambda ev: ev.get("direction") == "high"),
            (
                "high_tail_rsi_overbought",
                "High tail + RSI overbought (event)",
                lambda ev: ev.get("direction") == "high" and ev.get("rsi_regime") == "overbought",
            ),
            (
                "high_tail_bear_div",
                "High tail + bearish divergence",
                lambda ev: ev.get("direction") == "high"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish",
            ),
            (
                "high_tail_bear_div_rsi_overbought",
                "High tail + bearish divergence + RSI overbought (event)",
                lambda ev: ev.get("direction") == "high"
                and ev.get("rsi_regime") == "overbought"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish",
            ),
        ]
        for key, label, fn in buckets:
            rows = [ev for ev in daily_tail_events if fn(ev)]
            bucket_out: dict[str, object] = {"key": key, "label": label, "n": len(rows), "stats": {}}
            for h_label in max_return_horizons_days.keys():
                is_high_tail_bucket = str(key).startswith("high_tail")
                fav_sign = -1.0 if is_high_tail_bucket else 1.0
                dd_sign = 1.0 if is_high_tail_bucket else -1.0

                fav_mags = []
                dd_mags = []
                for ev in rows:
                    v = (ev.get("max_fav_returns") or {}).get(h_label)
                    if v is not None and not pd.isna(v):
                        fav_mags.append(abs(float(v)) * 100.0)
                    dd = (ev.get("drawdown_returns") or {}).get(h_label)
                    if dd is not None and not pd.isna(dd):
                        dd_mags.append(abs(float(dd)) * 100.0)

                fav_med = _quantile(fav_mags, 0.50)
                fav_p75 = _quantile(fav_mags, 0.75)
                dd_med = _quantile(dd_mags, 0.50)
                dd_p75 = _quantile(dd_mags, 0.75)

                bucket_out["stats"][h_label] = {
                    "fav_median": None if fav_med is None else float(fav_sign) * float(fav_med),
                    "fav_p75": None if fav_p75 is None else float(fav_sign) * float(fav_p75),
                    "dd_median": None if dd_med is None else float(dd_sign) * float(dd_med),
                    "dd_p75": None if dd_p75 is None else float(dd_sign) * float(dd_p75),
                }
            max_move_summary_daily["buckets"].append(bucket_out)
    except Exception:  # noqa: BLE001
        max_move_summary_daily = {"horizons_days": max_return_horizons_days, "buckets": []}

    payload["max_move_summary_daily"] = max_move_summary_daily
    # Backward-compatible alias (schema v3 called this "max_upside_summary_daily").
    payload["max_upside_summary_daily"] = max_move_summary_daily

    # Divergence-conditioned summaries (daily): tail events with divergence vs without, using max-up returns.
    if rsi_divergence_daily is not None:
        # Conditional summaries: tail events with divergence vs without.
        def _median(values: list[float]) -> float | None:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return None
            vals.sort()
            m = len(vals) // 2
            if len(vals) % 2:
                return float(vals[m])
            return float((vals[m - 1] + vals[m]) / 2.0)

        def _get_forward(dct: dict, day: int) -> object | None:
            if not dct:
                return None
            if day in dct:
                return dct.get(day)
            return dct.get(str(day))

        fwd_days_int = [int(d) for d in forward_days_daily]
        summary: dict[str, dict[str, dict]] = {"high": {}, "low": {}}
        for tail in ("high", "low"):
            want = "bearish" if tail == "high" else "bullish"

            def _bucket(has_div: bool) -> dict[str, object]:
                rows = []
                for ev in daily_tail_events:
                    if ev.get("direction") != tail:
                        continue
                    div = ev.get("rsi_divergence") or None
                    match = bool(div) and div.get("divergence") == want
                    if match != has_div:
                        continue
                    rows.append(ev)

                out: dict[str, object] = {"n": len(rows)}
                for d in fwd_days_int:
                    maxrets = []
                    pcts = []
                    for ev in rows:
                        r = _get_forward(ev.get("forward_max_fav_returns") or {}, d)
                        p = _get_forward(ev.get("forward_extension_percentiles") or {}, d)
                        if r is not None and not pd.isna(r):
                            maxrets.append(float(r) * 100.0)
                        if p is not None and not pd.isna(p):
                            pcts.append(float(p))
                    out[f"median_fwd_max_fav_move_pct_{d}d"] = _median(maxrets)
                    out[f"median_fwd_extension_percentile_{d}d"] = _median(pcts)
                return out

            summary[tail]["with_divergence"] = _bucket(True)
            summary[tail]["without_divergence"] = _bucket(False)

        rsi_divergence_daily["tail_event_summary"] = summary

    daily_tail_events = sorted(
        daily_tail_events,
        key=lambda ev: (ev.get("date") or ""),
        reverse=True,
    )
    weekly_tail_events = sorted(
        weekly_tail_events,
        key=lambda ev: (ev.get("date") or ""),
        reverse=True,
    )
    daily["tail_events"] = daily_tail_events
    weekly["tail_events"] = weekly_tail_events

    payload["report_daily"] = daily
    payload["report_weekly"] = weekly

    md_lines: list[str] = [
        f"# {sym_label} — Extension Percentile Stats",
        "",
        f"- As-of (daily): `{report_daily.asof}`",
        f"- Extension (daily, ATR units): `{'-' if report_daily.extension_atr is None else f'{report_daily.extension_atr:+.2f}'}`",
        "",
        "## Current Percentiles",
    ]
    if report_daily.current_percentiles:
        for years, pct in sorted(report_daily.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Daily p5 / p50 / p95)")
    if report_daily.quantiles_by_window:
        for years, q in sorted(report_daily.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Daily)")
    if rsi_divergence_daily is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_daily.get("current") or {}) if isinstance(rsi_divergence_daily, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
        if cur_bear is None and cur_bull is None:
            md_lines.append("- No divergences detected in the most recent window.")
        if cur_bear is not None:
            try:
                drsi = "-" if cur_bear.get("rsi_delta") is None else f"{float(cur_bear.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = (
                    "-"
                    if cur_bear.get("price_delta_pct") is None
                    else f"{float(cur_bear.get('price_delta_pct')):+.2f}%"
                )
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
                f"(RSI tag `{cur_bear.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )
        if cur_bull is not None:
            try:
                drsi = "-" if cur_bull.get("rsi_delta") is None else f"{float(cur_bull.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = (
                    "-"
                    if cur_bull.get("price_delta_pct") is None
                    else f"{float(cur_bull.get('price_delta_pct')):+.2f}%"
                )
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
                f"(RSI tag `{cur_bull.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )

        # Compact conditional summary table (focus on the most commonly used horizons).
        summ = rsi_divergence_daily.get("tail_event_summary") if isinstance(rsi_divergence_daily, dict) else None
        if isinstance(summ, dict):
            md_lines.append("")
            md_lines.append("### Tail Outcomes With vs Without Divergence (Daily)")
            md_lines.append("| Tail | Divergence | N | Med max move (5d/15d) | Med fwd pctl (5d/15d) |")
            md_lines.append("|---|---|---:|---|---|")
            for tail, want in (("high", "bearish"), ("low", "bullish")):
                for bucket_name, label in (
                    ("with_divergence", f"with {want}"),
                    ("without_divergence", f"without {want}"),
                ):
                    b = (summ.get(tail) or {}).get(bucket_name) if isinstance(summ.get(tail), dict) else None
                    if not isinstance(b, dict):
                        continue
                    n = b.get("n", 0)
                    r5 = b.get("median_fwd_max_fav_move_pct_5d")
                    r15 = b.get("median_fwd_max_fav_move_pct_15d")
                    p5 = b.get("median_fwd_extension_percentile_5d")
                    p15 = b.get("median_fwd_extension_percentile_15d")

                    def _fmt_ret(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):+.1f}%"
                        except Exception:  # noqa: BLE001
                            return "-"

                    def _fmt_pct(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):.1f}"
                        except Exception:  # noqa: BLE001
                            return "-"

                    r_str = f"{_fmt_ret(r5)} / {_fmt_ret(r15)}"
                    p_str = f"{_fmt_pct(p5)} / {_fmt_pct(p15)}"
                    md_lines.append(f"| {tail} | {label} | {n} | {r_str} | {p_str} |")

    md_lines.append("")
    md_lines.append("## Max Favorable Move (Daily)")
    md_lines.append(
        "Directional metrics: fav is in the mean-reversion direction; dd is max adverse move against it (both use High/Low vs entry Close)."
    )
    md_lines.append("Cells: fav (med/p75); dd (med/p75).")
    md_lines.append("Descriptive (not financial advice).")
    max_up = payload.get("max_move_summary_daily", {}) if isinstance(payload, dict) else {}
    buckets = max_up.get("buckets", []) if isinstance(max_up, dict) else []
    if buckets:
        md_lines.append("| Bucket | N | 1w | 4w | 3m | 6m | 9m | 1y |")
        md_lines.append("|---|---:|---|---|---|---|---|---|")

        def _fmt_pair(med: object, p75: object) -> str:
            try:
                if med is None or p75 is None:
                    return "-"
                return f"{float(med):+.1f}% / {float(p75):+.1f}%"
            except Exception:  # noqa: BLE001
                return "-"

        def _fmt_cell(fav_med: object, fav_p75: object, dd_med: object, dd_p75: object) -> str:
            fav = _fmt_pair(fav_med, fav_p75)
            dd = _fmt_pair(dd_med, dd_p75)
            if fav == "-" and dd == "-":
                return "-"
            if dd == "-":
                return fav
            if fav == "-":
                return dd
            return f"{fav}; {dd}"

        for b in buckets:
            if not isinstance(b, dict):
                continue
            n = int(b.get("n", 0) or 0)
            stats = b.get("stats", {}) if isinstance(b.get("stats"), dict) else {}

            def _get(label: str) -> str:
                s = stats.get(label, {}) if isinstance(stats.get(label), dict) else {}
                return _fmt_cell(s.get("fav_median"), s.get("fav_p75"), s.get("dd_median"), s.get("dd_p75"))

            md_lines.append(
                f"| {b.get('label', '-')} | {n} | {_get('1w')} | {_get('4w')} | {_get('3m')} | {_get('6m')} | {_get('9m')} | {_get('1y')} |"
            )
    else:
        md_lines.append("- Not available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Tail Events (Daily, all)")
    if daily_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_daily)
        max_horiz = "/".join(max_return_horizons_days.keys())
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | W pctl | W RSI | W div | Fwd pctl ({horiz}) | Max ret% ({max_horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---:|---|---|---|---|")
        for ev in daily_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_daily]
            maxrets = [(ev.get("max_fav_returns") or {}).get(str(lbl)) for lbl in max_return_horizons_days.keys()]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            wctx = ev.get("weekly_context") if isinstance(ev.get("weekly_context"), dict) else {}
            w_pct = wctx.get("extension_percentile")
            w_pct_str = "-" if w_pct is None else f"{float(w_pct):.1f}"
            w_rsi = wctx.get("rsi_regime") or "-"
            w_div = wctx.get("divergence") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {w_pct_str} | {w_rsi} | {w_div} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    md_lines.append("")
    md_lines.append("## Weekly Context")
    md_lines.append(f"- As-of (weekly): `{report_weekly.asof}`")
    md_lines.append(
        f"- Extension (weekly, ATR units): `{'-' if report_weekly.extension_atr is None else f'{report_weekly.extension_atr:+.2f}'}`"
    )
    weekly_rsi_val = None
    weekly_rsi_tag = None
    if weekly_rsi_series is not None:
        try:
            v = weekly_rsi_series.dropna()
            if not v.empty:
                weekly_rsi_val = float(v.iloc[-1])
                weekly_rsi_tag = rsi_regime_tag(
                    rsi_value=weekly_rsi_val,
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
        except Exception:  # noqa: BLE001
            weekly_rsi_val = None
            weekly_rsi_tag = None
    if weekly_rsi_val is not None and weekly_rsi_tag is not None:
        md_lines.append(f"- RSI (weekly): `{weekly_rsi_val:.1f}` (tag `{weekly_rsi_tag}`)")

    md_lines.append("")
    md_lines.append("## Current Percentiles (Weekly)")
    if report_weekly.current_percentiles:
        for years, pct in sorted(report_weekly.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Weekly p5 / p50 / p95)")
    if report_weekly.quantiles_by_window:
        for years, q in sorted(report_weekly.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Weekly)")
    if rsi_divergence_weekly is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_weekly.get("current") or {}) if isinstance(rsi_divergence_weekly, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
        if cur_bear is None and cur_bull is None:
            md_lines.append("- No divergences detected in the most recent window.")
        if cur_bear is not None:
            md_lines.append(
                f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
                f"(RSI tag `{cur_bear.get('rsi_regime')}`)"
            )
        if cur_bull is not None:
            md_lines.append(
                f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
                f"(RSI tag `{cur_bull.get('rsi_regime')}`)"
            )

    md_lines.append("")
    md_lines.append("## Tail Events (Weekly, all)")
    if weekly_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_weekly)
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | Fwd pctl ({horiz}) | Max move% ({horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---|---|")
        for ev in weekly_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_weekly]
            maxrets = [(ev.get("forward_max_fav_returns") or {}).get(int(d)) for d in forward_days_weekly]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    md = "\n".join(md_lines).rstrip() + "\n"

    if out:
        base = out / sym_label
        base.mkdir(parents=True, exist_ok=True)
        json_path = base / f"{report_daily.asof}.json"
        md_path = base / f"{report_daily.asof}.md"
        if write_json:
            json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            console.print(f"Wrote JSON: {json_path}")
        if write_md:
            md_path.write_text(md, encoding="utf-8")
            console.print(f"Wrote Markdown: {md_path}")

    if print_to_console:
        try:
            from rich.markdown import Markdown

            console.print(Markdown(md))
        except Exception:  # noqa: BLE001
            console.print(md)


@app.command("optimize")
def technicals_optimize(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Optimize strategy parameters for a single dataset."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for optimization.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [c for c in needed if c in features.columns]
    features = features.loc[:, [c for c in cols if c in features.columns]]

    warmup = warmup_bars(cfg)
    best_params, best_stats, heatmap = optimize_params(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        warmup_bars=warmup,
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    console.print(f"Best params: {best_params}")
    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=best_params,
        train_stats=best_stats,
        walk_forward_result=None,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


@app.command("walk-forward")
def technicals_walk_forward(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run walk-forward optimization and write artifacts."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for walk-forward.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    walk_cfg = cfg["walk_forward"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [c for c in needed if c in features.columns]
    features = features.loc[:, [c for c in cols if c in features.columns]]

    warmup = warmup_bars(cfg)
    result = walk_forward_optimize(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        walk_cfg,
        strat_cfg["defaults"],
        warmup_bars=warmup,
        min_train_bars=opt_cfg.get("min_train_bars", 0),
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    folds_out = []
    for fold in result.folds:
        folds_out.append(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validate_start": fold["validate_start"],
                "validate_end": fold["validate_end"],
                "best_params": fold["best_params"],
                "train_stats": _stats_to_dict(fold["train_stats"]),
                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                "validate_score": fold["validate_score"],
            }
        )
    wf_dict = {
        "params": result.params,
        "folds": folds_out,
        "stability": result.stability,
        "used_defaults": result.used_defaults,
        "reason": result.reason,
    }
    heatmap = None
    if result.folds:
        best_fold = max(result.folds, key=lambda f: f.get("validate_score", float("-inf")))
        heatmap = best_fold.get("heatmap")

    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=result.params,
        train_stats=None,
        walk_forward_result=wf_dict,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


@app.command("run-all")
def technicals_run_all(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run both strategies for a list of tickers."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache
    from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
    from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise typer.BadParameter("Provide at least one ticker.")

    for symbol in symbols:
        try:
            df = load_ohlc_from_cache(symbol, cache_dir)
            if df.empty:
                console.print(f"[yellow]No data for {symbol} in cache.[/yellow]")
                continue
            features = compute_features(df, cfg)
            warmup = warmup_bars(cfg)
            for strategy, strat_cfg in cfg["strategies"].items():
                if not strat_cfg.get("enabled", False):
                    continue
                needed = required_feature_columns_for_strategy(strategy, strat_cfg)
                cols = ["Open", "High", "Low", "Close"]
                if "Volume" in features.columns:
                    cols.append("Volume")
                cols += [c for c in needed if c in features.columns]
                strat_features = features.loc[:, [c for c in cols if c in features.columns]]
                StrategyClass = get_strategy(strategy)
                opt_cfg = cfg["optimization"]
                if cfg["walk_forward"]["enabled"]:
                    result = walk_forward_optimize(
                        strat_features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        cfg["walk_forward"],
                        strat_cfg["defaults"],
                        warmup_bars=warmup,
                        min_train_bars=opt_cfg.get("min_train_bars", 0),
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    folds_out = []
                    for fold in result.folds:
                        folds_out.append(
                            {
                                "train_start": fold["train_start"],
                                "train_end": fold["train_end"],
                                "validate_start": fold["validate_start"],
                                "validate_end": fold["validate_end"],
                                "best_params": fold["best_params"],
                                "train_stats": _stats_to_dict(fold["train_stats"]),
                                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                                "validate_score": fold["validate_score"],
                            }
                        )
                    wf_dict = {
                        "params": result.params,
                        "folds": folds_out,
                        "stability": result.stability,
                        "used_defaults": result.used_defaults,
                        "reason": result.reason,
                    }
                    heatmap = None
                    if result.folds:
                        best_fold = max(
                            result.folds, key=lambda f: f.get("validate_score", float("-inf"))
                        )
                        heatmap = best_fold.get("heatmap")
                    train_stats = None
                    params = result.params
                else:
                    best_params, train_stats, heatmap = optimize_params(
                        strat_features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        warmup_bars=warmup,
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    wf_dict = None
                    params = best_params

                data_meta = {
                    "start": features.index.min(),
                    "end": features.index.max(),
                    "bars": len(features),
                    "warmup_bars": warmup,
                }
                optimize_meta = {
                    "method": opt_cfg["method"],
                    "maximize": opt_cfg["maximize"],
                    "constraints": strat_cfg["constraints"],
                }
                write_artifacts(
                    cfg,
                    ticker=symbol,
                    strategy=strategy,
                    params=params,
                    train_stats=train_stats,
                    walk_forward_result=wf_dict,
                    optimize_meta=optimize_meta,
                    data_meta=data_meta,
                    heatmap=heatmap,
                )
            console.print(f"[green]Completed[/green] {symbol}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{symbol} failed:[/red] {exc}")


def _parse_iso_date(value: str | None, *, option_name: str) -> date | None:
    if value is None:
        return None
    token = value.strip()
    if not token:
        return None
    try:
        return date.fromisoformat(token)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be in YYYY-MM-DD format.") from exc


def _split_csv_option(value: str | None, *, uppercase: bool = True) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if uppercase:
        return [item.upper() for item in items]
    return items


def _normalize_hhmm_et_cutoff(value: str, *, option_name: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise typer.BadParameter(f"{option_name} must be HH:MM in 24-hour ET time.")
    try:
        datetime.strptime(token, "%H:%M")
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be HH:MM in 24-hour ET time.") from exc
    return token


def _normalize_orb_stop_policy(value: str, *, option_name: str) -> str:
    token = str(value or "").strip().lower()
    allowed = ("base", "orb_range", "tighten")
    if token not in allowed:
        raise typer.BadParameter(f"{option_name} must be one of: {', '.join(allowed)}")
    return token


def _normalize_ma_type(value: str, *, option_name: str) -> str:
    token = str(value or "").strip().lower()
    allowed = ("sma", "ema")
    if token not in allowed:
        raise typer.BadParameter(f"{option_name} must be one of: {', '.join(allowed)}")
    return token


def _parse_allowed_volatility_regimes(value: str, *, option_name: str) -> tuple[str, ...]:
    allowed = ("low", "normal", "high")
    parsed = tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())
    if not parsed:
        raise typer.BadParameter(
            f"{option_name} must include at least one regime from: {', '.join(allowed)}"
        )
    if len(set(parsed)) != len(parsed):
        raise typer.BadParameter(f"{option_name} must not include duplicate regimes.")
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise typer.BadParameter(
            f"{option_name} contains invalid regime(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(allowed)}"
        )
    return parsed


def _build_strategy_signal_kwargs(
    *,
    strategy: str,
    ma_fast_window: int,
    ma_slow_window: int,
    ma_trend_window: int,
    ma_fast_type: str,
    ma_slow_type: str,
    ma_trend_type: str,
    trend_slope_lookback_bars: int,
    atr_window: int,
    atr_stop_multiple: float,
) -> dict[str, object]:
    if ma_fast_window < 1:
        raise typer.BadParameter("--ma-fast-window must be >= 1")
    if ma_slow_window < 1:
        raise typer.BadParameter("--ma-slow-window must be >= 1")
    if ma_trend_window < 1:
        raise typer.BadParameter("--ma-trend-window must be >= 1")
    if trend_slope_lookback_bars < 1:
        raise typer.BadParameter("--trend-slope-lookback-bars must be >= 1")
    if atr_window < 1:
        raise typer.BadParameter("--atr-window must be >= 1")
    if atr_stop_multiple <= 0.0:
        raise typer.BadParameter("--atr-stop-multiple must be > 0")

    normalized_fast_type = _normalize_ma_type(ma_fast_type, option_name="--ma-fast-type")
    normalized_slow_type = _normalize_ma_type(ma_slow_type, option_name="--ma-slow-type")
    normalized_trend_type = _normalize_ma_type(ma_trend_type, option_name="--ma-trend-type")

    if strategy == "ma_crossover":
        if ma_fast_window >= ma_slow_window:
            raise typer.BadParameter("--ma-fast-window must be < --ma-slow-window for ma_crossover.")
        return {
            "fast_window": int(ma_fast_window),
            "slow_window": int(ma_slow_window),
            "fast_type": normalized_fast_type,
            "slow_type": normalized_slow_type,
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    if strategy == "trend_following":
        return {
            "trend_window": int(ma_trend_window),
            "trend_type": normalized_trend_type,
            "fast_window": int(ma_fast_window),
            "fast_type": normalized_fast_type,
            "slope_lookback_bars": int(trend_slope_lookback_bars),
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    return {}


def _build_strategy_filter_config(
    *,
    allow_shorts: bool,
    enable_orb_confirmation: bool,
    orb_range_minutes: int,
    orb_confirmation_cutoff_et: str,
    orb_stop_policy: str,
    enable_atr_stop_floor: bool,
    atr_stop_floor_multiple: float,
    enable_rsi_extremes: bool,
    enable_ema9_regime: bool,
    ema9_slope_lookback_bars: int,
    enable_volatility_regime: bool,
    allowed_volatility_regimes: str,
):
    from pydantic import ValidationError

    from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig

    if orb_range_minutes < 1 or orb_range_minutes > 120:
        raise typer.BadParameter("--orb-range-minutes must be between 1 and 120.")
    if atr_stop_floor_multiple <= 0.0:
        raise typer.BadParameter("--atr-stop-floor-multiple must be > 0.")
    if ema9_slope_lookback_bars < 1:
        raise typer.BadParameter("--ema9-slope-lookback-bars must be >= 1.")

    cutoff = _normalize_hhmm_et_cutoff(
        orb_confirmation_cutoff_et,
        option_name="--orb-confirmation-cutoff-et",
    )
    normalized_stop_policy = _normalize_orb_stop_policy(
        orb_stop_policy,
        option_name="--orb-stop-policy",
    )
    parsed_regimes = _parse_allowed_volatility_regimes(
        allowed_volatility_regimes,
        option_name="--allowed-volatility-regimes",
    )

    payload = {
        "allow_shorts": bool(allow_shorts),
        "enable_orb_confirmation": bool(enable_orb_confirmation),
        "orb_range_minutes": int(orb_range_minutes),
        "orb_confirmation_cutoff_et": cutoff,
        "orb_stop_policy": normalized_stop_policy,
        "enable_atr_stop_floor": bool(enable_atr_stop_floor),
        "atr_stop_floor_multiple": float(atr_stop_floor_multiple),
        "enable_rsi_extremes": bool(enable_rsi_extremes),
        "enable_ema9_regime": bool(enable_ema9_regime),
        "ema9_slope_lookback_bars": int(ema9_slope_lookback_bars),
        "enable_volatility_regime": bool(enable_volatility_regime),
        "allowed_volatility_regimes": parsed_regimes,
    }
    try:
        return StrategyEntryFilterConfig.model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
        raise typer.BadParameter(f"Invalid strategy filter configuration: {detail}") from exc


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mapping_view(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except TypeError:
            return {}
    return {}


def _extract_directional_headline(bucket: object) -> tuple[int | None, float | None]:
    payload = _mapping_view(bucket)
    if not payload:
        return None, None

    trade_count = _coerce_int(payload.get("trade_count"))
    if trade_count is None:
        trade_count = _coerce_int(payload.get("closed_trade_count"))

    total_return = _coerce_float(payload.get("total_return_pct"))
    if total_return is None:
        total_return = _coerce_float(_mapping_view(payload.get("portfolio_metrics")).get("total_return_pct"))
    return trade_count, total_return


def _build_target_ladder(
    *,
    min_tenths: int,
    max_tenths: int,
    step_tenths: int,
) -> tuple[StrategyRTarget, ...]:
    from options_helper.analysis.strategy_simulator import build_r_target_ladder

    if min_tenths < 1:
        raise typer.BadParameter("--r-ladder-min-tenths must be >= 1")
    if max_tenths < min_tenths:
        raise typer.BadParameter("--r-ladder-max-tenths must be >= --r-ladder-min-tenths")
    if step_tenths < 1:
        raise typer.BadParameter("--r-ladder-step-tenths must be >= 1")

    return build_r_target_ladder(
        min_target_tenths=min_tenths,
        max_target_tenths=max_tenths,
        step_tenths=step_tenths,
    )


_OUTPUT_TIMEZONE_ALIASES: dict[str, str] = {
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
}


def _normalize_output_timezone(value: str, *, option_name: str = "--output-timezone") -> str:
    token = str(value or "").strip()
    if not token:
        raise typer.BadParameter(f"{option_name} must be a non-empty IANA timezone.")

    canonical = _OUTPUT_TIMEZONE_ALIASES.get(token.upper(), token)
    try:
        ZoneInfo(canonical)
    except ZoneInfoNotFoundError as exc:
        raise typer.BadParameter(
            f"{option_name}={value!r} is invalid. Use an IANA timezone (for example America/Chicago)."
        ) from exc
    return canonical


def _resolve_strategy_symbols(
    *,
    service: object,
    requested_symbols: list[str],
    excluded_symbols: set[str],
    universe_limit: int | None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if requested_symbols:
        candidates = requested_symbols
    else:
        loader = getattr(service, "list_universe_loader", None)
        if not callable(loader):
            raise typer.BadParameter(
                "No --symbols provided and strategy-modeling service does not expose universe loader."
            )
        universe = loader(database_path=None)
        symbols = getattr(universe, "symbols", ()) or ()
        candidates = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in candidates:
        token = str(symbol).strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)

    filtered = [symbol for symbol in deduped if symbol not in excluded_symbols]
    if universe_limit is not None:
        filtered = filtered[: int(universe_limit)]

    return tuple(filtered)


def _intraday_coverage_block_message(preflight: object | None) -> str | None:
    if preflight is None:
        return None

    blocked_symbols = tuple(getattr(preflight, "blocked_symbols", ()) or ())
    if not blocked_symbols:
        return None

    coverage_by_symbol = getattr(preflight, "coverage_by_symbol", {}) or {}
    details: list[str] = []
    for symbol in blocked_symbols:
        coverage = coverage_by_symbol.get(symbol)
        required_days = getattr(coverage, "required_days", ()) or ()
        missing_days = getattr(coverage, "missing_days", ()) or ()
        required = len(tuple(required_days))
        missing = len(tuple(missing_days))
        details.append(f"{symbol}({missing}/{required} missing)")

    detail_text = ", ".join(details) if details else ", ".join(blocked_symbols)
    message = f"Missing required intraday coverage for requested scope: {detail_text}"
    notes = tuple(getattr(preflight, "notes", ()) or ())
    if notes:
        message = f"{message} | notes: {'; '.join(str(note) for note in notes)}"
    return message


def _option_was_set_on_command_line(ctx: typer.Context, option_name: str) -> bool:
    from click.core import ParameterSource

    source = ctx.get_parameter_source(option_name)
    return source == ParameterSource.COMMANDLINE


def _merge_strategy_modeling_profile_values(
    *,
    ctx: typer.Context,
    loaded_profile,
    cli_values: dict[str, object],
):
    from options_helper.schemas.strategy_modeling_profile import StrategyModelingProfile

    payload = loaded_profile.model_dump() if loaded_profile is not None else {}
    for field_name, field_value in cli_values.items():
        if loaded_profile is None or _option_was_set_on_command_line(ctx, field_name):
            payload[field_name] = field_value
        else:
            payload.setdefault(field_name, field_value)
    return StrategyModelingProfile.model_validate(payload)


@app.command("strategy-model")
def technicals_strategy_model(
    ctx: typer.Context,
    strategy: str = typer.Option(
        "sfp",
        "--strategy",
        help="Strategy to model: sfp, msb, orb, ma_crossover, or trend_following.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Load a named strategy-modeling profile from --profile-path.",
    ),
    save_profile: str | None = typer.Option(
        None,
        "--save-profile",
        help="Save effective strategy-modeling inputs under this profile name.",
    ),
    overwrite_profile: bool = typer.Option(
        False,
        "--overwrite-profile/--no-overwrite-profile",
        help="Allow --save-profile to overwrite an existing profile name.",
    ),
    profile_path: Path = typer.Option(
        Path("config/strategy_modeling_profiles.json"),
        "--profile-path",
        help="Strategy-modeling profile store path (JSON).",
    ),
    allow_shorts: bool = typer.Option(
        True,
        "--allow-shorts/--no-allow-shorts",
        help="Allow short-direction events to pass entry filters.",
    ),
    enable_orb_confirmation: bool = typer.Option(
        False,
        "--enable-orb-confirmation/--no-enable-orb-confirmation",
        help="Require ORB breakout confirmation by cutoff for SFP/MSB events.",
    ),
    orb_range_minutes: int = typer.Option(
        15,
        "--orb-range-minutes",
        help="Opening-range window in minutes for ORB confirmation.",
    ),
    orb_confirmation_cutoff_et: str = typer.Option(
        "10:30",
        "--orb-confirmation-cutoff-et",
        help="Cutoff for ORB confirmation in ET (HH:MM, 24-hour).",
    ),
    orb_stop_policy: str = typer.Option(
        "base",
        "--orb-stop-policy",
        help="ORB stop policy: base, orb_range, or tighten.",
    ),
    enable_atr_stop_floor: bool = typer.Option(
        False,
        "--enable-atr-stop-floor/--no-enable-atr-stop-floor",
        help="Apply ATR stop floor gate to entries.",
    ),
    atr_stop_floor_multiple: float = typer.Option(
        0.5,
        "--atr-stop-floor-multiple",
        help="ATR multiple required for ATR stop floor gate.",
    ),
    enable_rsi_extremes: bool = typer.Option(
        False,
        "--enable-rsi-extremes/--no-enable-rsi-extremes",
        help="Require RSI extreme context for entries.",
    ),
    enable_ema9_regime: bool = typer.Option(
        False,
        "--enable-ema9-regime/--no-enable-ema9-regime",
        help="Require EMA9 regime alignment for entries.",
    ),
    ema9_slope_lookback_bars: int = typer.Option(
        3,
        "--ema9-slope-lookback-bars",
        help="Lookback bars for EMA9 slope regime computation.",
    ),
    enable_volatility_regime: bool = typer.Option(
        False,
        "--enable-volatility-regime/--no-enable-volatility-regime",
        help="Enable volatility-regime allowlist filtering.",
    ),
    allowed_volatility_regimes: str = typer.Option(
        "low,normal,high",
        "--allowed-volatility-regimes",
        help="Comma-separated volatility regimes to allow: low,normal,high.",
    ),
    ma_fast_window: int = typer.Option(
        20,
        "--ma-fast-window",
        help="Fast MA window for ma_crossover/trend_following signal generation.",
    ),
    ma_slow_window: int = typer.Option(
        50,
        "--ma-slow-window",
        help="Slow MA window for ma_crossover signal generation.",
    ),
    ma_trend_window: int = typer.Option(
        200,
        "--ma-trend-window",
        help="Trend MA window for trend_following signal generation.",
    ),
    ma_fast_type: str = typer.Option(
        "sma",
        "--ma-fast-type",
        help="Fast MA type for strategy signal generation: sma or ema.",
    ),
    ma_slow_type: str = typer.Option(
        "sma",
        "--ma-slow-type",
        help="Slow MA type for strategy signal generation: sma or ema.",
    ),
    ma_trend_type: str = typer.Option(
        "sma",
        "--ma-trend-type",
        help="Trend MA type for strategy signal generation: sma or ema.",
    ),
    trend_slope_lookback_bars: int = typer.Option(
        3,
        "--trend-slope-lookback-bars",
        help="Lookback bars used for trend MA slope checks in trend_following.",
    ),
    atr_window: int = typer.Option(
        14,
        "--atr-window",
        help="ATR window used by ma_crossover/trend_following signal stops.",
    ),
    atr_stop_multiple: float = typer.Option(
        2.0,
        "--atr-stop-multiple",
        help="ATR stop multiple used by ma_crossover/trend_following signal stops.",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated symbols (if omitted, uses universe loader).",
    ),
    exclude_symbols: str | None = typer.Option(
        None,
        "--exclude-symbols",
        help="Comma-separated symbols to exclude from modeled universe.",
    ),
    universe_limit: int | None = typer.Option(
        None,
        "--universe-limit",
        help="Optional max number of resolved universe symbols.",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Optional ISO start date (YYYY-MM-DD).",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="Optional ISO end date (YYYY-MM-DD).",
    ),
    intraday_timeframe: str = typer.Option(
        "5Min",
        "--intraday-timeframe",
        help="Required intraday bar timeframe for simulation.",
    ),
    intraday_source: str = typer.Option(
        "alpaca",
        "--intraday-source",
        help="Intraday source label for service request metadata.",
    ),
    r_ladder_min_tenths: int = typer.Option(
        10,
        "--r-ladder-min-tenths",
        help="Minimum target R in tenths (10 => 1.0R).",
    ),
    r_ladder_max_tenths: int = typer.Option(
        20,
        "--r-ladder-max-tenths",
        help="Maximum target R in tenths (20 => 2.0R).",
    ),
    r_ladder_step_tenths: int = typer.Option(
        1,
        "--r-ladder-step-tenths",
        help="Step size for target ladder in tenths.",
    ),
    starting_capital: float = typer.Option(
        100000.0,
        "--starting-capital",
        help="Starting portfolio capital for strategy modeling.",
    ),
    risk_per_trade_pct: float = typer.Option(
        1.0,
        "--risk-per-trade-pct",
        help="Per-trade risk percent used with risk_pct_of_equity sizing.",
    ),
    gap_fill_policy: str = typer.Option(
        "fill_at_open",
        "--gap-fill-policy",
        help="Gap fill policy for stop/target-through-open fills.",
    ),
    max_hold_bars: int | None = typer.Option(
        None,
        "--max-hold-bars",
        help="Optional max hold duration in bars for simulated trades.",
    ),
    one_open_per_symbol: bool = typer.Option(
        True,
        "--one-open-per-symbol/--no-one-open-per-symbol",
        help="Allow at most one open trade per symbol at a time.",
    ),
    signal_confirmation_lag_bars: int | None = typer.Option(
        None,
        "--signal-confirmation-lag-bars",
        help="Optional right-side confirmation lag metadata for close-confirmed signals.",
    ),
    segment_dimensions: str | None = typer.Option(
        None,
        "--segment-dimensions",
        help="Comma-separated segment dimensions to display first.",
    ),
    segment_values: str | None = typer.Option(
        None,
        "--segment-values",
        help="Comma-separated segment values to prioritize in output.",
    ),
    segment_min_trades: int = typer.Option(
        1,
        "--segment-min-trades",
        help="Minimum trades required for segment display filters.",
    ),
    segment_limit: int = typer.Option(
        10,
        "--segment-limit",
        help="Maximum segments to summarize in CLI output.",
    ),
    output_timezone: str = typer.Option(
        "America/Chicago",
        "--output-timezone",
        help="Timezone used for serialized report timestamps (IANA name, e.g. America/Chicago).",
    ),
    out: Path = typer.Option(
        Path("data/reports/technicals/strategy_modeling"),
        "--out",
        help="Output root for strategy-modeling artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write summary JSON."),
    write_csv: bool = typer.Option(True, "--write-csv/--no-write-csv", help="Write CSV artifacts."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write markdown summary."),
    show_progress: bool = typer.Option(
        True,
        "--show-progress/--no-show-progress",
        help="Print stage status updates while strategy modeling runs.",
    ),
) -> None:
    """Run strategy-modeling service and write JSON/CSV/Markdown artifacts."""
    from pydantic import ValidationError

    from options_helper.analysis.strategy_modeling import StrategyModelingRequest
    from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts
    from options_helper.data.strategy_modeling_profiles import (
        StrategyModelingProfileStoreError,
        load_strategy_modeling_profile,
        save_strategy_modeling_profile,
    )

    console = Console(width=200)

    normalized_cli_strategy = strategy.strip().lower()
    if normalized_cli_strategy not in {"sfp", "msb", "orb", "ma_crossover", "trend_following"}:
        raise typer.BadParameter(
            "--strategy must be one of: sfp, msb, orb, ma_crossover, trend_following"
        )
    if starting_capital <= 0.0:
        raise typer.BadParameter("--starting-capital must be > 0")
    if risk_per_trade_pct <= 0.0 or risk_per_trade_pct > 100.0:
        raise typer.BadParameter("--risk-per-trade-pct must be > 0 and <= 100")
    if gap_fill_policy != "fill_at_open":
        raise typer.BadParameter("--gap-fill-policy currently supports only fill_at_open")
    if max_hold_bars is not None and int(max_hold_bars) < 1:
        raise typer.BadParameter("--max-hold-bars must be >= 1")
    if segment_min_trades < 1:
        raise typer.BadParameter("--segment-min-trades must be >= 1")
    if segment_limit < 1:
        raise typer.BadParameter("--segment-limit must be >= 1")
    if universe_limit is not None and universe_limit < 1:
        raise typer.BadParameter("--universe-limit must be >= 1")
    normalized_output_timezone = _normalize_output_timezone(output_timezone)

    parsed_cli_start_date = _parse_iso_date(start_date, option_name="--start-date")
    parsed_cli_end_date = _parse_iso_date(end_date, option_name="--end-date")
    if parsed_cli_start_date and parsed_cli_end_date and parsed_cli_start_date > parsed_cli_end_date:
        raise typer.BadParameter("--start-date must be <= --end-date")
    # Preserve option-specific error messages for explicit CLI validation before profile merging.
    _build_strategy_signal_kwargs(
        strategy=normalized_cli_strategy,
        ma_fast_window=int(ma_fast_window),
        ma_slow_window=int(ma_slow_window),
        ma_trend_window=int(ma_trend_window),
        ma_fast_type=ma_fast_type,
        ma_slow_type=ma_slow_type,
        ma_trend_type=ma_trend_type,
        trend_slope_lookback_bars=int(trend_slope_lookback_bars),
        atr_window=int(atr_window),
        atr_stop_multiple=float(atr_stop_multiple),
    )
    _build_strategy_filter_config(
        allow_shorts=allow_shorts,
        enable_orb_confirmation=enable_orb_confirmation,
        orb_range_minutes=int(orb_range_minutes),
        orb_confirmation_cutoff_et=orb_confirmation_cutoff_et,
        orb_stop_policy=orb_stop_policy,
        enable_atr_stop_floor=enable_atr_stop_floor,
        atr_stop_floor_multiple=float(atr_stop_floor_multiple),
        enable_rsi_extremes=enable_rsi_extremes,
        enable_ema9_regime=enable_ema9_regime,
        ema9_slope_lookback_bars=int(ema9_slope_lookback_bars),
        enable_volatility_regime=enable_volatility_regime,
        allowed_volatility_regimes=allowed_volatility_regimes,
    )
    parsed_cli_allowed_volatility_regimes = _parse_allowed_volatility_regimes(
        allowed_volatility_regimes,
        option_name="--allowed-volatility-regimes",
    )

    loaded_profile = None
    if profile is not None:
        try:
            loaded_profile = load_strategy_modeling_profile(profile_path, profile)
        except StrategyModelingProfileStoreError as exc:
            raise typer.BadParameter(str(exc)) from exc

    cli_profile_values: dict[str, object] = {
        "strategy": normalized_cli_strategy,
        "symbols": tuple(_split_csv_option(symbols)),
        "start_date": parsed_cli_start_date,
        "end_date": parsed_cli_end_date,
        "intraday_timeframe": str(intraday_timeframe),
        "intraday_source": str(intraday_source),
        "starting_capital": float(starting_capital),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "gap_fill_policy": str(gap_fill_policy),
        "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        "one_open_per_symbol": bool(one_open_per_symbol),
        "r_ladder_min_tenths": int(r_ladder_min_tenths),
        "r_ladder_max_tenths": int(r_ladder_max_tenths),
        "r_ladder_step_tenths": int(r_ladder_step_tenths),
        "allow_shorts": bool(allow_shorts),
        "enable_orb_confirmation": bool(enable_orb_confirmation),
        "orb_range_minutes": int(orb_range_minutes),
        "orb_confirmation_cutoff_et": str(orb_confirmation_cutoff_et),
        "orb_stop_policy": str(orb_stop_policy),
        "enable_atr_stop_floor": bool(enable_atr_stop_floor),
        "atr_stop_floor_multiple": float(atr_stop_floor_multiple),
        "enable_rsi_extremes": bool(enable_rsi_extremes),
        "enable_ema9_regime": bool(enable_ema9_regime),
        "ema9_slope_lookback_bars": int(ema9_slope_lookback_bars),
        "enable_volatility_regime": bool(enable_volatility_regime),
        "allowed_volatility_regimes": parsed_cli_allowed_volatility_regimes,
        "ma_fast_window": int(ma_fast_window),
        "ma_slow_window": int(ma_slow_window),
        "ma_trend_window": int(ma_trend_window),
        "ma_fast_type": str(ma_fast_type),
        "ma_slow_type": str(ma_slow_type),
        "ma_trend_type": str(ma_trend_type),
        "trend_slope_lookback_bars": int(trend_slope_lookback_bars),
        "atr_window": int(atr_window),
        "atr_stop_multiple": float(atr_stop_multiple),
    }
    try:
        effective_profile = _merge_strategy_modeling_profile_values(
            ctx=ctx,
            loaded_profile=loaded_profile,
            cli_values=cli_profile_values,
        )
    except ValidationError as exc:
        detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
        raise typer.BadParameter(f"Invalid strategy-modeling profile values: {detail}") from exc

    normalized_strategy = effective_profile.strategy
    signal_kwargs = _build_strategy_signal_kwargs(
        strategy=normalized_strategy,
        ma_fast_window=int(effective_profile.ma_fast_window),
        ma_slow_window=int(effective_profile.ma_slow_window),
        ma_trend_window=int(effective_profile.ma_trend_window),
        ma_fast_type=effective_profile.ma_fast_type,
        ma_slow_type=effective_profile.ma_slow_type,
        ma_trend_type=effective_profile.ma_trend_type,
        trend_slope_lookback_bars=int(effective_profile.trend_slope_lookback_bars),
        atr_window=int(effective_profile.atr_window),
        atr_stop_multiple=float(effective_profile.atr_stop_multiple),
    )
    filter_config = _build_strategy_filter_config(
        allow_shorts=effective_profile.allow_shorts,
        enable_orb_confirmation=effective_profile.enable_orb_confirmation,
        orb_range_minutes=int(effective_profile.orb_range_minutes),
        orb_confirmation_cutoff_et=effective_profile.orb_confirmation_cutoff_et,
        orb_stop_policy=effective_profile.orb_stop_policy,
        enable_atr_stop_floor=effective_profile.enable_atr_stop_floor,
        atr_stop_floor_multiple=float(effective_profile.atr_stop_floor_multiple),
        enable_rsi_extremes=effective_profile.enable_rsi_extremes,
        enable_ema9_regime=effective_profile.enable_ema9_regime,
        ema9_slope_lookback_bars=int(effective_profile.ema9_slope_lookback_bars),
        enable_volatility_regime=effective_profile.enable_volatility_regime,
        allowed_volatility_regimes=",".join(effective_profile.allowed_volatility_regimes),
    )

    target_ladder = _build_target_ladder(
        min_tenths=int(effective_profile.r_ladder_min_tenths),
        max_tenths=int(effective_profile.r_ladder_max_tenths),
        step_tenths=int(effective_profile.r_ladder_step_tenths),
    )

    if save_profile is not None:
        try:
            save_strategy_modeling_profile(
                profile_path,
                save_profile,
                effective_profile,
                overwrite=overwrite_profile,
            )
        except StrategyModelingProfileStoreError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"Saved strategy-modeling profile '{save_profile}' -> {profile_path}")

    service = cli_deps.build_strategy_modeling_service()
    stage_timings: dict[str, float] = {}

    def _format_seconds(value: float) -> str:
        return f"{value:.2f}s"

    def _accumulate_timing(name: str, elapsed: float) -> None:
        stage_timings[name] = stage_timings.get(name, 0.0) + elapsed

    def _safe_detail(detail_builder, output, args, kwargs) -> str:  # noqa: ANN001
        if detail_builder is None:
            return ""
        try:
            detail = detail_builder(output, args, kwargs)
        except Exception:  # noqa: BLE001
            return ""
        if not detail:
            return ""
        return f" | {detail}"

    def _timed_stage(name: str, fn, detail_builder=None):  # noqa: ANN001
        def _wrapped(*args, **kwargs):  # noqa: ANN002,ANN003
            if show_progress:
                console.print(f"[cyan]... {name}[/cyan]")
            started = time.perf_counter()
            output = fn(*args, **kwargs)
            elapsed = time.perf_counter() - started
            _accumulate_timing(name, elapsed)
            if show_progress:
                detail = _safe_detail(detail_builder, output, args, kwargs)
                console.print(f"[green]OK  {name} ({_format_seconds(elapsed)}){detail}[/green]")
            return output

        return _wrapped

    service_stage_attrs = (
        "list_universe_loader",
        "daily_loader",
        "required_sessions_builder",
        "intraday_loader",
        "feature_computer",
        "signal_builder",
        "trade_simulator",
        "portfolio_builder",
        "metrics_computer",
        "segmentation_aggregator",
    )
    if show_progress and all(hasattr(service, attr) for attr in service_stage_attrs):
        from options_helper.analysis.strategy_modeling import StrategyModelingService

        service = StrategyModelingService(
            list_universe_loader=_timed_stage(
                "Loading universe",
                service.list_universe_loader,
                lambda out, _a, _k: f"symbols={len(tuple(getattr(out, 'symbols', ()) or ())) or 0}",
            ),
            daily_loader=_timed_stage(
                "Loading daily candles",
                service.daily_loader,
                lambda out, _a, _k: (
                    f"loaded={len(getattr(out, 'candles_by_symbol', {}) or {})} "
                    f"skipped={len(tuple(getattr(out, 'skipped_symbols', ()) or ())) or 0} "
                    f"missing={len(tuple(getattr(out, 'missing_symbols', ()) or ())) or 0}"
                ),
            ),
            required_sessions_builder=_timed_stage(
                "Building required sessions",
                service.required_sessions_builder,
                lambda out, _a, _k: (
                    f"symbols={len(out or {})} "
                    f"sessions={sum(len(tuple(v or ())) for v in (out or {}).values())}"
                ),
            ),
            intraday_loader=_timed_stage(
                "Loading intraday bars",
                service.intraday_loader,
                lambda out, _a, _k: (
                    f"symbols={len(getattr(out, 'bars_by_symbol', {}) or {})} "
                    f"blocked={len(tuple(getattr(getattr(out, 'preflight', None), 'blocked_symbols', ()) or ())) or 0}"
                ),
            ),
            feature_computer=_timed_stage("Computing features", service.feature_computer),
            signal_builder=_timed_stage(
                "Building signals",
                service.signal_builder,
                lambda out, _a, kwargs: (
                    f"symbol={str(kwargs.get('symbol', '')).upper()} events={len(tuple(out or ())) or 0}"
                ),
            ),
            trade_simulator=_timed_stage(
                "Simulating trades",
                service.trade_simulator,
                lambda out, args, kwargs: (
                    f"events={len(tuple(args[0] or ())) if args else 0} "
                    f"targets={len(tuple(kwargs.get('target_ladder') or ())) or 0} "
                    f"trades={len(tuple(out or ())) or 0}"
                ),
            ),
            portfolio_builder=_timed_stage(
                "Building portfolio ledger",
                service.portfolio_builder,
                lambda out, _a, _k: (
                    f"accepted={len(tuple(getattr(out, 'accepted_trade_ids', ()) or ())) or 0} "
                    f"skipped={len(tuple(getattr(out, 'skipped_trade_ids', ()) or ())) or 0}"
                ),
            ),
            metrics_computer=_timed_stage("Computing metrics", service.metrics_computer),
            segmentation_aggregator=_timed_stage(
                "Building segmentation",
                service.segmentation_aggregator,
                lambda out, _a, _k: f"segments={len(tuple(getattr(out, 'segments', ()) or ())) or 0}",
            ),
        )

    resolved_symbols = _resolve_strategy_symbols(
        service=service,
        requested_symbols=list(effective_profile.symbols),
        excluded_symbols=set(_split_csv_option(exclude_symbols)),
        universe_limit=universe_limit,
    )
    if not resolved_symbols:
        raise typer.BadParameter("No symbols remain after applying include/exclude/universe filters.")

    base_request = StrategyModelingRequest(
        strategy=normalized_strategy,
        symbols=resolved_symbols,
        start_date=effective_profile.start_date,
        end_date=effective_profile.end_date,
        intraday_timeframe=effective_profile.intraday_timeframe,
        target_ladder=target_ladder,
        starting_capital=float(effective_profile.starting_capital),
        max_hold_bars=effective_profile.max_hold_bars,
        filter_config=filter_config,
        signal_kwargs=signal_kwargs,
        policy={
            "require_intraday_bars": True,
            "sizing_rule": "risk_pct_of_equity",
            "risk_per_trade_pct": float(effective_profile.risk_per_trade_pct),
            "gap_fill_policy": effective_profile.gap_fill_policy,
            "max_hold_bars": effective_profile.max_hold_bars,
            "one_open_per_symbol": bool(effective_profile.one_open_per_symbol),
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
    )

    request = SimpleNamespace(
        **vars(base_request),
        intraday_source=effective_profile.intraday_source,
        gap_fill_policy=effective_profile.gap_fill_policy,
        intra_bar_tie_break_rule="stop_first",
        signal_confirmation_lag_bars=signal_confirmation_lag_bars,
        output_timezone=normalized_output_timezone,
        segment_dimensions=tuple(_split_csv_option(segment_dimensions, uppercase=False)),
        segment_values=tuple(_split_csv_option(segment_values, uppercase=False)),
        segment_min_trades=int(segment_min_trades),
        segment_limit=int(segment_limit),
    )

    if show_progress:
        console.print(
            (
                f"[cyan]Starting strategy-model run: strategy={normalized_strategy} "
                f"symbols={len(resolved_symbols)} timeframe={effective_profile.intraday_timeframe}[/cyan]"
            )
        )
    run_started = time.perf_counter()
    run_result = service.run(request)
    run_elapsed = time.perf_counter() - run_started
    if show_progress:
        console.print(f"[green]Run complete in {_format_seconds(run_elapsed)}[/green]")
    block_message = _intraday_coverage_block_message(getattr(run_result, "intraday_preflight", None))
    if block_message is not None:
        raise typer.BadParameter(block_message)

    paths = write_strategy_modeling_artifacts(
        out_dir=out,
        strategy=normalized_strategy,
        request=request,
        run_result=run_result,
        write_json=write_json,
        write_csv=write_csv,
        write_md=write_md,
    )

    segment_count = len(tuple(getattr(run_result, "segment_records", ()) or ()))
    segments_shown = min(segment_count, int(segment_limit))
    trade_count = len(tuple(getattr(run_result, "trade_simulations", ()) or ()))
    console.print(
        (
            f"strategy={normalized_strategy} symbols={len(resolved_symbols)} "
            f"trades={trade_count} segments_shown={segments_shown}"
        )
    )

    filter_summary = _mapping_view(getattr(run_result, "filter_summary", None))
    if filter_summary:
        base_events = _coerce_int(filter_summary.get("base_event_count"))
        kept_events = _coerce_int(filter_summary.get("kept_event_count"))
        rejected_events = _coerce_int(filter_summary.get("rejected_event_count"))
        if base_events is not None and kept_events is not None and rejected_events is not None:
            console.print(f"filters base={base_events} kept={kept_events} rejected={rejected_events}")

        reject_counts = _mapping_view(filter_summary.get("reject_counts"))
        reject_parts: list[str] = []
        for reason, count in reject_counts.items():
            parsed = _coerce_int(count)
            if parsed is None or parsed <= 0:
                continue
            reject_parts.append(f"{reason}={parsed}")
        if reject_parts:
            console.print(f"filter_rejects {', '.join(reject_parts)}")

    directional_metrics = _mapping_view(getattr(run_result, "directional_metrics", None))
    if directional_metrics:
        directional_parts: list[str] = []
        for label in ("combined", "long_only", "short_only"):
            trade_count_value, total_return_value = _extract_directional_headline(
                directional_metrics.get(label)
            )
            if trade_count_value is None and total_return_value is None:
                continue

            part = label
            if trade_count_value is not None:
                part = f"{part} trades={trade_count_value}"
            if total_return_value is not None:
                part = f"{part} return={total_return_value:.2f}%"
            directional_parts.append(part)
        if directional_parts:
            console.print(f"directional {' | '.join(directional_parts)}")

    if write_json:
        console.print(f"Wrote summary JSON: {paths.summary_json}")
    if write_csv:
        console.print(f"Wrote trades CSV: {paths.trade_log_csv}")
        console.print(f"Wrote R ladder CSV: {paths.r_ladder_csv}")
        console.print(f"Wrote segments CSV: {paths.segments_csv}")
    if write_md:
        console.print(f"Wrote summary Markdown: {paths.summary_md}")
        console.print(f"Wrote LLM analysis prompt: {paths.llm_analysis_prompt_md}")
    if show_progress and stage_timings:
        console.print("Stage timings:")
        for name in sorted(stage_timings, key=stage_timings.get, reverse=True):
            console.print(f"  - {name}: {_format_seconds(stage_timings[name])}")
