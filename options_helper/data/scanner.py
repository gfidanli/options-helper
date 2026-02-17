from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.extension_scan import ExtensionScanResult, compute_current_extension_percentile
from options_helper.analysis.options_liquidity import LiquidityResult, evaluate_liquidity
from options_helper.analysis.research import Direction, analyze_underlying, build_confluence_inputs
from options_helper.analysis.scanner_rank import ScannerRankInputs, ScannerRankResult, rank_scanner_candidate
from options_helper.data.candles import CandleStore
from options_helper.data.derived import DerivedStore
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.models import RiskProfile


@dataclass(frozen=True)
class ScannerScanRow:
    symbol: str
    asof: str | None
    extension_atr: float | None
    percentile: float | None
    window_years: int | None
    window_bars: int | None
    tail: bool
    status: str
    error: str | None


@dataclass(frozen=True)
class ScannerLiquidityRow:
    symbol: str
    snapshot_date: str | None
    eligible_contracts: int
    eligible_expiries: str
    is_liquid: bool
    status: str
    error: str | None


@dataclass(frozen=True)
class ScannerShortlistRow:
    symbol: str
    score: float | None
    coverage: float | None
    top_reasons: str


_HYPHEN_SUFFIXES_DEFAULT = {
    "W",
    "WS",
    "WT",
    "WTA",
    "WTS",
    "WSA",
    "WSB",
    "WSR",
    "U",
    "UN",
    "UNIT",
    "R",
    "RT",
    "RIGHT",
}

_AGGRESSIVE_ENDINGS = _HYPHEN_SUFFIXES_DEFAULT


def prefilter_symbols(
    symbols: list[str],
    *,
    mode: str = "default",
    exclude: set[str] | None = None,
    scanned: set[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    mode_norm = (mode or "default").strip().lower()
    counts: dict[str, int] = {}
    filtered: list[str] = []

    def _bump(reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1

    exclude = exclude or set()
    scanned = scanned or set()
    for raw in symbols:
        sym = (raw or "").strip().upper()
        if not sym:
            _bump("empty")
            continue
        if sym in scanned:
            _bump("scanned")
            continue
        if sym in exclude:
            _bump("excluded")
            continue
        if mode_norm == "none":
            filtered.append(sym)
            continue
        if any(ch in sym for ch in ("$", "^", "/", " ")):
            _bump("special_char")
            continue

        # Drop hyphenated warrants/units/rights.
        if "-" in sym:
            suffix = sym.split("-", 1)[1].strip().upper()
            if suffix in _HYPHEN_SUFFIXES_DEFAULT:
                _bump("hyphen_suffix")
                continue

        if mode_norm == "aggressive":
            if len(sym) >= 5:
                for suf in _AGGRESSIVE_ENDINGS:
                    if sym.endswith(suf):
                        _bump("aggressive_suffix")
                        break
                else:
                    filtered.append(sym)
                continue

        if mode_norm != "none":
            filtered.append(sym)
        else:
            filtered.append(sym)

    return sorted(set(filtered)), counts


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_symbol(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def _safe_emit_row(row_callback: Callable[[ScannerScanRow], None] | None, row: ScannerScanRow) -> None:
    if row_callback is None:
        return
    try:
        row_callback(row)
    except Exception:  # noqa: BLE001
        pass


def _scan_symbol_row(
    symbol: str,
    *,
    candle_store: CandleStore,
    cfg: dict,
    scan_period: str,
    tail_low_pct: float,
    tail_high_pct: float,
    percentile_window_years: int | None,
) -> tuple[ScannerScanRow, bool]:
    sym = symbol.strip().upper()
    try:
        history = candle_store.get_daily_history(sym, period=scan_period)
        if history.empty:
            return (
                ScannerScanRow(
                    symbol=sym,
                    asof=None,
                    extension_atr=None,
                    percentile=None,
                    window_years=None,
                    window_bars=None,
                    tail=False,
                    status="no_candles",
                    error=None,
                ),
                False,
            )
        result: ExtensionScanResult = compute_current_extension_percentile(
            history,
            cfg,
            percentile_window_years=percentile_window_years,
        )
        pct = result.percentile
        tail = bool(pct is not None and (pct <= tail_low_pct or pct >= tail_high_pct))
        row = ScannerScanRow(
            symbol=sym,
            asof=result.asof,
            extension_atr=result.extension_atr,
            percentile=pct,
            window_years=result.window_years,
            window_bars=result.window_bars,
            tail=tail,
            status="ok" if pct is not None else "no_percentile",
            error=None,
        )
        return row, tail
    except Exception as exc:  # noqa: BLE001
        return (
            ScannerScanRow(
                symbol=sym,
                asof=None,
                extension_atr=None,
                percentile=None,
                window_years=None,
                window_bars=None,
                tail=False,
                status="error",
                error=str(exc),
            ),
            False,
        )


def _collect_scan_result(
    *,
    row: ScannerScanRow,
    tail: bool,
    rows: list[ScannerScanRow],
    tail_symbols: set[str],
    row_callback: Callable[[ScannerScanRow], None] | None,
) -> None:
    rows.append(row)
    _safe_emit_row(row_callback, row)
    if tail:
        tail_symbols.add(row.symbol)


def _scan_symbols_sequential(
    symbols: list[str],
    *,
    candle_store: CandleStore,
    cfg: dict,
    scan_period: str,
    tail_low_pct: float,
    tail_high_pct: float,
    percentile_window_years: int | None,
    row_callback: Callable[[ScannerScanRow], None] | None,
) -> tuple[list[ScannerScanRow], set[str]]:
    rows: list[ScannerScanRow] = []
    tail_symbols: set[str] = set()
    for symbol in symbols:
        if not symbol:
            continue
        row, tail = _scan_symbol_row(
            symbol,
            candle_store=candle_store,
            cfg=cfg,
            scan_period=scan_period,
            tail_low_pct=tail_low_pct,
            tail_high_pct=tail_high_pct,
            percentile_window_years=percentile_window_years,
        )
        _collect_scan_result(
            row=row,
            tail=tail,
            rows=rows,
            tail_symbols=tail_symbols,
            row_callback=row_callback,
        )
    return rows, tail_symbols


def _scan_symbols_parallel(
    symbols: list[str],
    *,
    workers: int,
    batch_size: int,
    batch_sleep_seconds: float,
    candle_store: CandleStore,
    cfg: dict,
    scan_period: str,
    tail_low_pct: float,
    tail_high_pct: float,
    percentile_window_years: int | None,
    row_callback: Callable[[ScannerScanRow], None] | None,
) -> tuple[list[ScannerScanRow], set[str]]:
    rows: list[ScannerScanRow] = []
    tail_symbols: set[str] = set()
    for batch in _chunked(symbols, batch_size):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _scan_symbol_row,
                    sym,
                    candle_store=candle_store,
                    cfg=cfg,
                    scan_period=scan_period,
                    tail_low_pct=tail_low_pct,
                    tail_high_pct=tail_high_pct,
                    percentile_window_years=percentile_window_years,
                ): sym
                for sym in batch
                if sym
            }
            for fut in as_completed(futures):
                row, tail = fut.result()
                _collect_scan_result(
                    row=row,
                    tail=tail,
                    rows=rows,
                    tail_symbols=tail_symbols,
                    row_callback=row_callback,
                )
        if batch_sleep_seconds > 0:
            time.sleep(batch_sleep_seconds)
    return rows, tail_symbols


def _evaluate_liquidity_row(
    symbol: str,
    *,
    store: OptionsSnapshotStore,
    min_dte: int,
    min_volume: int,
    min_open_interest: int,
) -> tuple[ScannerLiquidityRow, bool]:
    try:
        dates = store.latest_dates(symbol, n=1)
        if not dates:
            return (
                ScannerLiquidityRow(
                    symbol=symbol,
                    snapshot_date=None,
                    eligible_contracts=0,
                    eligible_expiries="",
                    is_liquid=False,
                    status="no_snapshot",
                    error=None,
                ),
                False,
            )
        snapshot_date = dates[-1]
        df = store.load_day(symbol, snapshot_date)
        if df.empty:
            return (
                ScannerLiquidityRow(
                    symbol=symbol,
                    snapshot_date=snapshot_date.isoformat(),
                    eligible_contracts=0,
                    eligible_expiries="",
                    is_liquid=False,
                    status="empty_snapshot",
                    error=None,
                ),
                False,
            )
        result: LiquidityResult = evaluate_liquidity(
            df,
            snapshot_date=snapshot_date,
            min_dte=min_dte,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )
        return (
            ScannerLiquidityRow(
                symbol=symbol,
                snapshot_date=snapshot_date.isoformat(),
                eligible_contracts=result.eligible_contracts,
                eligible_expiries=",".join(result.eligible_expiries),
                is_liquid=result.is_liquid,
                status="ok",
                error=None,
            ),
            result.is_liquid,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            ScannerLiquidityRow(
                symbol=symbol,
                snapshot_date=None,
                eligible_contracts=0,
                eligible_expiries="",
                is_liquid=False,
                status="error",
                error=str(exc),
            ),
            False,
        )


def read_symbol_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    out: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(_normalize_symbol(line))
    return out


def write_symbol_set(path: Path, symbols: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({_normalize_symbol(s) for s in symbols if s and s.strip()})
    payload = "\n".join(ordered).rstrip() + "\n"
    path.write_text(payload, encoding="utf-8")


def read_exclude_symbols(path: Path) -> set[str]:
    return read_symbol_set(path)


def write_exclude_symbols(path: Path, symbols: set[str]) -> None:
    write_symbol_set(path, symbols)


def read_scanned_symbols(path: Path) -> set[str]:
    return read_symbol_set(path)


def write_scanned_symbols(path: Path, symbols: set[str]) -> None:
    write_symbol_set(path, symbols)


def scan_symbols(
    symbols: list[str],
    *,
    candle_store: CandleStore,
    cfg: dict,
    scan_period: str,
    tail_low_pct: float,
    tail_high_pct: float,
    percentile_window_years: int | None = None,
    workers: int | None = None,
    batch_size: int = 50,
    batch_sleep_seconds: float = 0.0,
    row_callback: Callable[[ScannerScanRow], None] | None = None,
) -> tuple[list[ScannerScanRow], list[str]]:
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 4)))
    if workers <= 1:
        rows, tail_symbols = _scan_symbols_sequential(
            symbols,
            candle_store=candle_store,
            cfg=cfg,
            scan_period=scan_period,
            tail_low_pct=tail_low_pct,
            tail_high_pct=tail_high_pct,
            percentile_window_years=percentile_window_years,
            row_callback=row_callback,
        )
        return rows, sorted(tail_symbols)
    rows, tail_symbols = _scan_symbols_parallel(
        symbols,
        workers=workers,
        batch_size=batch_size,
        batch_sleep_seconds=batch_sleep_seconds,
        candle_store=candle_store,
        cfg=cfg,
        scan_period=scan_period,
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
        percentile_window_years=percentile_window_years,
        row_callback=row_callback,
    )
    return rows, sorted(tail_symbols)


def write_scan_csv(rows: list[ScannerScanRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "asof",
                "extension_atr",
                "percentile",
                "window_years",
                "window_bars",
                "tail",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def evaluate_liquidity_for_symbols(
    symbols: list[str],
    *,
    store: OptionsSnapshotStore,
    min_dte: int,
    min_volume: int,
    min_open_interest: int,
) -> tuple[list[ScannerLiquidityRow], list[str]]:
    rows: list[ScannerLiquidityRow] = []
    liquid_symbols: list[str] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if not sym:
            continue
        row, is_liquid = _evaluate_liquidity_row(
            sym,
            store=store,
            min_dte=min_dte,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )
        rows.append(row)
        if is_liquid:
            liquid_symbols.append(sym)
    return rows, sorted(set(liquid_symbols))


def score_shortlist_confluence(
    symbols: list[str],
    *,
    candle_store: CandleStore,
    confluence_cfg: dict | None,
    extension_percentiles: dict[str, float | None] | None = None,
    period: str = "5y",
    risk_profile: RiskProfile | None = None,
) -> dict[str, ConfluenceScore]:
    results: dict[str, ConfluenceScore] = {}
    rp = risk_profile or RiskProfile()
    for symbol in symbols:
        sym = (symbol or "").strip().upper()
        if not sym:
            continue
        try:
            history = candle_store.get_daily_history(sym, period=period)
            setup = analyze_underlying(sym, history=history, risk_profile=rp)
            ext_pct = None
            if extension_percentiles is not None:
                ext_pct = extension_percentiles.get(sym)
            inputs = build_confluence_inputs(
                setup,
                extension_percentile=ext_pct,
                vol_context=None,
            )
            results[sym] = score_confluence(inputs, cfg=confluence_cfg)
        except Exception:  # noqa: BLE001
            continue
    return results


def rank_shortlist_candidates(
    symbols: list[str],
    *,
    candle_store: CandleStore,
    rank_cfg: dict | None,
    scan_rows: list[ScannerScanRow] | None = None,
    liquidity_rows: list[ScannerLiquidityRow] | None = None,
    derived_store: DerivedStore | None = None,
    period: str = "5y",
    risk_profile: RiskProfile | None = None,
) -> dict[str, ScannerRankResult]:
    results: dict[str, ScannerRankResult] = {}
    rp = risk_profile or RiskProfile()
    extension_percentiles = _extension_percentile_map(scan_rows)
    liquidity_scores = _liquidity_score_map(liquidity_rows)
    iv_rv = _iv_rv_map(symbols, derived_store)
    for symbol in symbols:
        sym = (symbol or "").strip().upper()
        if not sym:
            continue
        try:
            history = candle_store.get_daily_history(sym, period=period)
            setup = analyze_underlying(sym, history=history, risk_profile=rp)
            trend = _map_trend(setup.direction)
            inputs = ScannerRankInputs(
                extension_percentile=extension_percentiles.get(sym),
                weekly_trend=trend,
                rsi_divergence=None,
                iv_rv_20d=iv_rv.get(sym),
                liquidity_score=liquidity_scores.get(sym),
                flow_delta_oi_notional=None,
            )
            results[sym] = rank_scanner_candidate(inputs, cfg=rank_cfg)
        except Exception:  # noqa: BLE001
            continue
    return results


def write_liquidity_csv(rows: list[ScannerLiquidityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "snapshot_date",
                "eligible_contracts",
                "eligible_expiries",
                "is_liquid",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_shortlist_csv(rows: list[ScannerShortlistRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "score",
                "coverage",
                "top_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _extension_percentile_map(rows: list[ScannerScanRow] | None) -> dict[str, float | None]:
    if not rows:
        return {}
    return {row.symbol.upper(): row.percentile for row in rows}


def _liquidity_score_map(rows: list[ScannerLiquidityRow] | None) -> dict[str, float]:
    if not rows:
        return {}
    max_contracts = max((int(row.eligible_contracts) for row in rows), default=0)
    max_expiries = max((_count_expiries(row.eligible_expiries) for row in rows), default=0)
    scores: dict[str, float] = {}
    for row in rows:
        sym = (row.symbol or "").strip().upper()
        if not sym:
            continue
        parts: list[float] = []
        if max_contracts > 0:
            parts.append(max(0.0, float(row.eligible_contracts)) / float(max_contracts))
        if max_expiries > 0:
            parts.append(float(_count_expiries(row.eligible_expiries)) / float(max_expiries))
        if not parts:
            continue
        score = sum(parts) / float(len(parts))
        scores[sym] = max(0.0, min(1.0, score))
    return scores


def _count_expiries(value: str) -> int:
    if not value:
        return 0
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return len(parts)


def _iv_rv_map(symbols: list[str], store: DerivedStore | None) -> dict[str, float | None]:
    if store is None:
        return {}
    results: dict[str, float | None] = {}
    for symbol in symbols:
        sym = (symbol or "").strip().upper()
        if not sym:
            continue
        try:
            df = store.load(sym)
            if df.empty or "iv_rv_20d" not in df.columns:
                results[sym] = None
                continue
            series = pd.to_numeric(df["iv_rv_20d"], errors="coerce").dropna()
            results[sym] = float(series.iloc[-1]) if not series.empty else None
        except Exception:  # noqa: BLE001
            results[sym] = None
    return results


def _map_trend(direction: Direction | None) -> str | None:
    if direction == Direction.BULLISH:
        return "up"
    if direction == Direction.BEARISH:
        return "down"
    if direction == Direction.NEUTRAL:
        return "flat"
    return None
