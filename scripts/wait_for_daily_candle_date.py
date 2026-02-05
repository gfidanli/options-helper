from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import sys
import time
from datetime import timedelta
from zoneinfo import ZoneInfo

from options_helper.data.market_types import DataFetchError
from options_helper.data.providers import get_provider


@dataclass(frozen=True)
class CandleCheckResult:
    symbol: str
    last_date: date | None
    error: str | None = None


def _parse_period_to_start(period: str, *, today: date) -> date | None:
    period = period.strip().lower()
    if period == "max":
        return None
    if period == "ytd":
        return date(today.year, 1, 1)

    units = {"d": 1, "wk": 7, "mo": 30, "y": 365}

    if period.endswith("wk"):
        n = int(period[:-2])
        return today - timedelta(days=n * units["wk"])

    for suffix in ("d", "mo", "y"):
        if period.endswith(suffix):
            n = int(period[: -len(suffix)])
            return today - timedelta(days=n * units[suffix])

    raise ValueError(f"Unsupported period format: {period}")


def _fetch_last_daily_candle_date(provider, symbol: str, *, start: date | None, end: date | None) -> CandleCheckResult:  # noqa: ANN001
    sym = (symbol or "").strip().upper()
    if not sym:
        return CandleCheckResult(symbol=sym, last_date=None, error="empty symbol")
    try:
        df = provider.get_history(sym, start=start, end=end, interval="1d", auto_adjust=True, back_adjust=False)
        if df is None or df.empty:
            return CandleCheckResult(symbol=sym, last_date=None, error="empty history")
        ts = df.index.max()
        try:
            return CandleCheckResult(symbol=sym, last_date=ts.date())
        except Exception:  # noqa: BLE001
            # Last resort: parse YYYY-MM-DD from string form.
            s = str(ts)[:10]
            return CandleCheckResult(symbol=sym, last_date=date.fromisoformat(s))
    except DataFetchError as exc:
        return CandleCheckResult(symbol=sym, last_date=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return CandleCheckResult(symbol=sym, last_date=None, error=str(exc))


def _resolve_expected_date(spec: str, *, tz: str) -> date:
    spec_norm = (spec or "").strip().lower()
    if spec_norm in {"today", "now", ""}:
        return datetime.now(ZoneInfo(tz)).date()
    return date.fromisoformat(spec_norm)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Wait until daily candles are published for an expected date.")
    p.add_argument(
        "--symbols",
        default="SPY",
        help="Comma-separated list of canary symbols (default: SPY).",
    )
    p.add_argument(
        "--provider",
        default="alpaca",
        help="Market data provider for candles (default: alpaca).",
    )
    p.add_argument(
        "--period",
        default="10d",
        help="Period window to request (yfinance-style: 10d/1mo/3mo/1y/ytd/max; default: 10d).",
    )
    p.add_argument(
        "--expected-date",
        default="today",
        help="Expected data date: YYYY-MM-DD or 'today' (default: today).",
    )
    p.add_argument(
        "--tz",
        default="America/Chicago",
        help="Timezone used to interpret 'today' (default: America/Chicago).",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=10800,
        help="Max seconds to wait (default: 10800 = 3h). Use 0 to check once.",
    )
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between checks (default: 300).",
    )
    args = p.parse_args(argv)

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s and s.strip()]
    if not symbols:
        print("Error: no symbols provided", file=sys.stderr)
        return 2

    try:
        expected = _resolve_expected_date(args.expected_date, tz=args.tz)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: invalid --expected-date: {exc}", file=sys.stderr)
        return 2

    provider_name = str(args.provider or "").strip().lower()
    if not provider_name:
        print("Error: empty --provider", file=sys.stderr)
        return 2
    try:
        provider = get_provider(provider_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: failed to initialize provider '{provider_name}': {exc}", file=sys.stderr)
        return 2

    try:
        start_date = _parse_period_to_start(args.period, today=expected)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: invalid --period: {exc}", file=sys.stderr)
        return 2

    timeout = max(0, int(args.timeout_seconds))
    poll = max(1, int(args.poll_seconds))

    start_ts = time.time()
    attempt = 0
    while True:
        attempt += 1
        results = [_fetch_last_daily_candle_date(provider, sym, start=start_date, end=None) for sym in symbols]
        ready = [r for r in results if r.last_date is not None and r.last_date >= expected]

        parts: list[str] = []
        for r in results:
            if r.last_date is not None:
                parts.append(f"{r.symbol}:{r.last_date.isoformat()}")
            elif r.error:
                parts.append(f"{r.symbol}:err")
            else:
                parts.append(f"{r.symbol}:-")
        status = ", ".join(parts)

        if ready:
            print(
                f"ready provider={provider_name} expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
                flush=True,
            )
            return 0

        elapsed = time.time() - start_ts
        if elapsed >= timeout:
            print(
                f"timeout provider={provider_name} expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
                file=sys.stderr,
                flush=True,
            )
            return 1

        print(
            f"waiting provider={provider_name} expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
            flush=True,
        )
        time.sleep(min(poll, max(1, timeout - int(elapsed))))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
