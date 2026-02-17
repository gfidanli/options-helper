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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait until daily candles are published for an expected date.")
    parser.add_argument(
        "--symbols",
        default="SPY",
        help="Comma-separated list of canary symbols (default: SPY).",
    )
    parser.add_argument(
        "--provider",
        default="alpaca",
        help="Market data provider for candles (default: alpaca).",
    )
    parser.add_argument(
        "--period",
        default="10d",
        help="Period window to request (yfinance-style: 10d/1mo/3mo/1y/ytd/max; default: 10d).",
    )
    parser.add_argument(
        "--expected-date",
        default="today",
        help="Expected data date: YYYY-MM-DD or 'today' (default: today).",
    )
    parser.add_argument(
        "--tz",
        default="America/Chicago",
        help="Timezone used to interpret 'today' (default: America/Chicago).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=10800,
        help="Max seconds to wait (default: 10800 = 3h). Use 0 to check once.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between checks (default: 300).",
    )
    return parser


def _parse_symbols(raw: str) -> list[str]:
    symbols = [value.strip().upper() for value in (raw or "").split(",") if value and value.strip()]
    if not symbols:
        raise ValueError("no symbols provided")
    return symbols


def _resolve_provider(provider_name: str):  # noqa: ANN201
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        raise ValueError("empty --provider")
    try:
        provider = get_provider(normalized)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to initialize provider '{normalized}': {exc}") from exc
    return normalized, provider


def _format_status(results: list[CandleCheckResult]) -> str:
    parts: list[str] = []
    for result in results:
        if result.last_date is not None:
            parts.append(f"{result.symbol}:{result.last_date.isoformat()}")
        elif result.error:
            parts.append(f"{result.symbol}:err")
        else:
            parts.append(f"{result.symbol}:-")
    return ", ".join(parts)


def _run_wait_loop(
    *,
    provider,
    provider_name: str,  # noqa: ANN001
    symbols: list[str],
    expected: date,
    tz: str,
    start_date: date | None,
    timeout_seconds: int,
    poll_seconds: int,
) -> int:
    start_ts = time.time()
    attempt = 0
    while True:
        attempt += 1
        results = [_fetch_last_daily_candle_date(provider, symbol, start=start_date, end=None) for symbol in symbols]
        status = _format_status(results)
        if any(result.last_date is not None and result.last_date >= expected for result in results):
            print(
                f"ready provider={provider_name} expected={expected.isoformat()} tz={tz} attempts={attempt} last=[{status}]",
                flush=True,
            )
            return 0
        elapsed = time.time() - start_ts
        if elapsed >= timeout_seconds:
            print(
                f"timeout provider={provider_name} expected={expected.isoformat()} tz={tz} attempts={attempt} last=[{status}]",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            f"waiting provider={provider_name} expected={expected.isoformat()} tz={tz} attempts={attempt} last=[{status}]",
            flush=True,
        )
        time.sleep(min(poll_seconds, max(1, timeout_seconds - int(elapsed))))


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    try:
        symbols = _parse_symbols(args.symbols)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    try:
        expected = _resolve_expected_date(args.expected_date, tz=args.tz)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: invalid --expected-date: {exc}", file=sys.stderr)
        return 2
    try:
        provider_name, provider = _resolve_provider(args.provider)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    try:
        start_date = _parse_period_to_start(args.period, today=expected)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: invalid --period: {exc}", file=sys.stderr)
        return 2
    return _run_wait_loop(
        provider=provider,
        provider_name=provider_name,
        symbols=symbols,
        expected=expected,
        tz=args.tz,
        start_date=start_date,
        timeout_seconds=max(0, int(args.timeout_seconds)),
        poll_seconds=max(1, int(args.poll_seconds)),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
