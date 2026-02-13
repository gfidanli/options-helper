from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands.technicals.extension_stats_legacy import (
    technicals_extension_stats as _technicals_extension_stats_impl,
)
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


technicals_extension_stats = _technicals_extension_stats_impl
app.command("extension-stats")(technicals_extension_stats)


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
    fib_retracement_pct: float,
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
    if strategy == "fib_retracement":
        from options_helper.analysis.fib_retracement import normalize_fib_retracement_pct

        try:
            normalized_fib_retracement_pct = normalize_fib_retracement_pct(fib_retracement_pct)
        except ValueError as exc:
            raise typer.BadParameter(f"--fib-retracement-pct {exc}") from exc
        return {"fib_retracement_pct": float(normalized_fib_retracement_pct)}
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
        help="Strategy to model: sfp, msb, orb, ma_crossover, trend_following, or fib_retracement.",
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
    fib_retracement_pct: float = typer.Option(
        61.8,
        "--fib-retracement-pct",
        help="Fib retracement percent (e.g., 61.8; accepts 0.618 as ratio).",
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
    stop_move_rules: list[str] = typer.Option(
        [],
        "--stop-move",
        help=(
            "Repeatable stop adjustment rule expressed as 'trigger_r:stop_r' in R multiples "
            "(evaluated on bar close, applied starting next bar). Example: --stop-move 1.0:0.0 "
            "to move to breakeven after a +1.0R close."
        ),
    ),
    disable_stop_moves: bool = typer.Option(
        False,
        "--disable-stop-moves",
        help="Disable stop-move rules even if a loaded profile includes them.",
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
    max_hold_timeframe: str = typer.Option(
        "entry",
        "--max-hold-timeframe",
        help="Timeframe unit for max hold bars (entry, 10Min, 1H, 1D, 1W).",
    ),
    disable_max_hold: bool = typer.Option(
        False,
        "--disable-max-hold",
        help="Disable max-hold time stop even if profile values set one.",
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
    from options_helper.schemas.strategy_modeling_policy import normalize_max_hold_timeframe

    console = Console(width=200)

    normalized_cli_strategy = strategy.strip().lower()
    if normalized_cli_strategy not in {
        "sfp",
        "msb",
        "orb",
        "ma_crossover",
        "trend_following",
        "fib_retracement",
    }:
        raise typer.BadParameter(
            "--strategy must be one of: sfp, msb, orb, ma_crossover, trend_following, fib_retracement"
        )
    if starting_capital <= 0.0:
        raise typer.BadParameter("--starting-capital must be > 0")
    if risk_per_trade_pct <= 0.0 or risk_per_trade_pct > 100.0:
        raise typer.BadParameter("--risk-per-trade-pct must be > 0 and <= 100")
    if gap_fill_policy != "fill_at_open":
        raise typer.BadParameter("--gap-fill-policy currently supports only fill_at_open")
    if max_hold_bars is not None and int(max_hold_bars) < 1:
        raise typer.BadParameter("--max-hold-bars must be >= 1")
    try:
        normalized_max_hold_timeframe = normalize_max_hold_timeframe(max_hold_timeframe)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if disable_max_hold and _option_was_set_on_command_line(ctx, "max_hold_bars"):
        raise typer.BadParameter("--disable-max-hold cannot be combined with --max-hold-bars")
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
        fib_retracement_pct=float(fib_retracement_pct),
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
        "fib_retracement_pct": float(fib_retracement_pct),
        "gap_fill_policy": str(gap_fill_policy),
        "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        "max_hold_timeframe": normalized_max_hold_timeframe,
        "one_open_per_symbol": bool(one_open_per_symbol),
        "stop_move_rules": tuple(stop_move_rules),
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
    if disable_max_hold:
        effective_profile = effective_profile.model_copy(update={"max_hold_bars": None})
    if disable_stop_moves:
        effective_profile = effective_profile.model_copy(update={"stop_move_rules": ()})

    normalized_strategy = effective_profile.strategy
    signal_kwargs = _build_strategy_signal_kwargs(
        strategy=normalized_strategy,
        fib_retracement_pct=float(effective_profile.fib_retracement_pct),
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
            "max_hold_timeframe": effective_profile.max_hold_timeframe,
            "one_open_per_symbol": bool(effective_profile.one_open_per_symbol),
            "stop_move_rules": effective_profile.stop_move_rules,
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
    )

    request = SimpleNamespace(
        **vars(base_request),
        intraday_source=effective_profile.intraday_source,
        gap_fill_policy=effective_profile.gap_fill_policy,
        max_hold_timeframe=effective_profile.max_hold_timeframe,
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
