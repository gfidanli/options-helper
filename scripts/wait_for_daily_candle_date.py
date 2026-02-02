from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import sys
import time
from zoneinfo import ZoneInfo

import yfinance as yf


@dataclass(frozen=True)
class CandleCheckResult:
    symbol: str
    last_date: date | None
    error: str | None = None


def _fetch_last_daily_candle_date(symbol: str, *, period: str) -> CandleCheckResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return CandleCheckResult(symbol=sym, last_date=None, error="empty symbol")
    try:
        df = yf.Ticker(sym).history(
            period=period,
            interval="1d",
            auto_adjust=True,
            back_adjust=False,
        )
        if df is None or df.empty:
            return CandleCheckResult(symbol=sym, last_date=None, error="empty history")
        ts = df.index.max()
        try:
            return CandleCheckResult(symbol=sym, last_date=ts.date())
        except Exception:  # noqa: BLE001
            # Last resort: parse YYYY-MM-DD from string form.
            s = str(ts)[:10]
            return CandleCheckResult(symbol=sym, last_date=date.fromisoformat(s))
    except Exception as exc:  # noqa: BLE001
        return CandleCheckResult(symbol=sym, last_date=None, error=str(exc))


def _resolve_expected_date(spec: str, *, tz: str) -> date:
    spec_norm = (spec or "").strip().lower()
    if spec_norm in {"today", "now", ""}:
        return datetime.now(ZoneInfo(tz)).date()
    return date.fromisoformat(spec_norm)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Wait until Yahoo daily candles are published for an expected date.")
    p.add_argument(
        "--symbols",
        default="SPY",
        help="Comma-separated list of canary symbols (default: SPY).",
    )
    p.add_argument(
        "--period",
        default="10d",
        help="yfinance period to request (default: 10d).",
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

    timeout = max(0, int(args.timeout_seconds))
    poll = max(1, int(args.poll_seconds))

    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        results = [_fetch_last_daily_candle_date(sym, period=args.period) for sym in symbols]
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
                f"ready expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
                flush=True,
            )
            return 0

        elapsed = time.time() - start
        if elapsed >= timeout:
            print(
                f"timeout expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
                file=sys.stderr,
                flush=True,
            )
            return 1

        print(
            f"waiting expected={expected.isoformat()} tz={args.tz} attempts={attempt} last=[{status}]",
            flush=True,
        )
        time.sleep(min(poll, max(1, timeout - int(elapsed))))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

