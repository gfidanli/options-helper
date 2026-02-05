from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from options_helper.data.candles import CandleStore


@dataclass(frozen=True)
class CandleIngestResult:
    symbol: str
    status: str
    last_date: date | None
    error: str | None


def ingest_candles(
    store: CandleStore,
    symbols: list[str],
    *,
    period: str = "max",
    best_effort: bool = True,
) -> list[CandleIngestResult]:
    results: list[CandleIngestResult] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if not sym:
            continue
        try:
            history = store.get_daily_history(sym, period=period)
            if history is None or history.empty:
                results.append(
                    CandleIngestResult(symbol=sym, status="empty", last_date=None, error=None)
                )
            else:
                last_dt = history.index.max()
                results.append(
                    CandleIngestResult(symbol=sym, status="ok", last_date=last_dt.date(), error=None)
                )
        except Exception as exc:  # noqa: BLE001
            results.append(
                CandleIngestResult(symbol=sym, status="error", last_date=None, error=str(exc))
            )
            if not best_effort:
                raise
    return results
