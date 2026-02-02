from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import logging

import pandas as pd

from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.data.candles import CandleStore, last_close
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.yf_client import DataFetchError, YFinanceClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotSymbolResult:
    symbol: str
    snapshot_date: date | None
    expiries: int
    status: str
    error: str | None


def snapshot_full_chain_for_symbols(
    symbols: list[str],
    *,
    cache_dir: Path,
    candle_cache_dir: Path,
    spot_period: str = "10d",
    max_expiries: int | None = None,
    risk_free_rate: float = 0.0,
    symbol_source: str = "scanner",
    watchlists: list[str] | None = None,
) -> list[SnapshotSymbolResult]:
    client = YFinanceClient()
    store = OptionsSnapshotStore(cache_dir)
    candle_store = CandleStore(candle_cache_dir)

    results: list[SnapshotSymbolResult] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if not sym:
            continue
        try:
            history = candle_store.get_daily_history(sym, period=spot_period)
            spot = last_close(history)
            data_date: date | None = history.index.max().date() if not history.empty else None

            if spot is None or spot <= 0:
                try:
                    underlying = client.get_underlying(sym, period=spot_period, interval="1d")
                    spot = underlying.last_price
                    if data_date is None and underlying.history is not None and not underlying.history.empty:
                        try:
                            data_date = underlying.history.index.max().date()
                        except Exception:  # noqa: BLE001
                            pass
                except DataFetchError as exc:
                    results.append(
                        SnapshotSymbolResult(
                            symbol=sym,
                            snapshot_date=data_date,
                            expiries=0,
                            status="no_spot",
                            error=str(exc),
                        )
                    )
                    continue

            if spot is None or spot <= 0:
                results.append(
                    SnapshotSymbolResult(
                        symbol=sym,
                        snapshot_date=data_date,
                        expiries=0,
                        status="no_spot",
                        error="Missing spot price",
                    )
                )
                continue

            snapshot_date = data_date or date.today()
            expiry_strs = list(client.ticker(sym).options or [])
            if not expiry_strs:
                results.append(
                    SnapshotSymbolResult(
                        symbol=sym,
                        snapshot_date=snapshot_date,
                        expiries=0,
                        status="no_expiries",
                        error="No listed option expiries",
                    )
                )
                continue
            if max_expiries is not None:
                expiry_strs = expiry_strs[:max_expiries]

            meta = {
                "spot": spot,
                "spot_period": spot_period,
                "full_chain": True,
                "all_expiries": True,
                "risk_free_rate": risk_free_rate,
                "window_pct": None,
                "strike_min": None,
                "strike_max": None,
                "snapshot_date": snapshot_date.isoformat(),
                "symbol_source": symbol_source,
                "watchlists": watchlists or [],
            }

            ok_expiries = 0
            for exp_str in expiry_strs:
                try:
                    exp = date.fromisoformat(exp_str)
                except ValueError:
                    logger.warning("Invalid expiry format for %s: %s", sym, exp_str)
                    continue
                try:
                    raw = client.get_options_chain_raw(sym, exp)
                except DataFetchError as exc:
                    logger.warning("%s %s: %s", sym, exp.isoformat(), exc)
                    continue

                meta_with_underlying = dict(meta)
                meta_with_underlying["underlying"] = raw.get("underlying", {})

                calls = pd.DataFrame(raw.get("calls", []))
                puts = pd.DataFrame(raw.get("puts", []))
                calls["optionType"] = "call"
                puts["optionType"] = "put"
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=snapshot_date,
                    r=risk_free_rate,
                )

                store.save_expiry_snapshot(
                    sym,
                    snapshot_date,
                    expiry=exp,
                    snapshot=df,
                    meta=meta_with_underlying,
                )
                store.save_expiry_snapshot_raw(
                    sym,
                    snapshot_date,
                    expiry=exp,
                    raw=raw,
                    meta=meta_with_underlying,
                )
                ok_expiries += 1

            results.append(
                SnapshotSymbolResult(
                    symbol=sym,
                    snapshot_date=snapshot_date,
                    expiries=ok_expiries,
                    status="ok",
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                SnapshotSymbolResult(
                    symbol=sym,
                    snapshot_date=None,
                    expiries=0,
                    status="error",
                    error=str(exc),
                )
            )

    return results
