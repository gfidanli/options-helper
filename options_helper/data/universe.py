from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import json
import logging
import urllib.request


logger = logging.getLogger(__name__)


class UniverseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ListedSymbol:
    symbol: str
    is_etf: bool


NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
DEFAULT_USER_AGENT = "options-helper/0.1 (scanner)"


def _normalize_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if not sym:
        return ""
    # Yahoo uses '-' for class shares (e.g. BRK-B instead of BRK.B).
    return sym.replace(".", "-")


def _fetch_text(url: str, *, timeout: int = 30, user_agent: str | None = None) -> str:
    try:
        headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise UniverseError(f"Failed to fetch universe file: {url}") from exc


def _parse_symbol_rows(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    reader = csv.reader(text.splitlines(), delimiter="|")
    try:
        header = next(reader)
    except StopIteration:
        return []
    header = [h.strip() for h in header]
    rows: list[dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        if row[0].startswith("File Creation Time"):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        data = {header[i]: row[i].strip() for i in range(len(header))}
        rows.append(data)
    return rows


def _listed_symbols_from_text(text: str) -> list[ListedSymbol]:
    rows = _parse_symbol_rows(text)
    out: list[ListedSymbol] = []
    for row in rows:
        symbol = row.get("Symbol") or row.get("ACT Symbol") or ""
        if not symbol:
            continue
        test_issue = (row.get("Test Issue") or "").strip().upper()
        if test_issue == "Y":
            continue
        etf_flag = (row.get("ETF") or "").strip().upper()
        out.append(ListedSymbol(symbol=_normalize_symbol(symbol), is_etf=etf_flag == "Y"))
    return out


def _merge_listed_symbols(*lists: list[ListedSymbol]) -> dict[str, ListedSymbol]:
    merged: dict[str, ListedSymbol] = {}
    for symbols in lists:
        for item in symbols:
            if not item.symbol:
                continue
            existing = merged.get(item.symbol)
            if existing is None:
                merged[item.symbol] = item
            else:
                merged[item.symbol] = ListedSymbol(symbol=item.symbol, is_etf=existing.is_etf or item.is_etf)
    return merged


def _load_sec_company_tickers_exchange() -> dict[str, ListedSymbol]:
    text = _fetch_text(SEC_COMPANY_TICKERS_EXCHANGE_URL)
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise UniverseError("Invalid SEC company_tickers_exchange.json payload") from exc
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    if not isinstance(fields, list) or not isinstance(data, list):
        raise UniverseError("Unexpected SEC company_tickers_exchange.json structure")

    try:
        ticker_idx = fields.index("ticker")
        exchange_idx = fields.index("exchange")
        name_idx = fields.index("name")
    except ValueError as exc:
        raise UniverseError("Missing expected fields in SEC company_tickers_exchange.json") from exc

    out: dict[str, ListedSymbol] = {}
    for row in data:
        if not isinstance(row, list) or len(row) <= max(ticker_idx, exchange_idx, name_idx):
            continue
        ticker = _normalize_symbol(str(row[ticker_idx]))
        if not ticker:
            continue
        exchange = str(row[exchange_idx]) if row[exchange_idx] is not None else ""
        if exchange.strip().upper().startswith("OTC") or not exchange.strip():
            continue
        # SEC list does not label ETFs; treat as unknown/non-ETF.
        out[ticker] = ListedSymbol(symbol=ticker, is_etf=False)
    return out


def _load_nasdaq_trader_symbols() -> dict[str, ListedSymbol]:
    other_text = _fetch_text(OTHER_LISTED_URL)
    nasdaq_text = _fetch_text(NASDAQ_LISTED_URL)
    other_symbols = _listed_symbols_from_text(other_text)
    nasdaq_symbols = _listed_symbols_from_text(nasdaq_text)
    return _merge_listed_symbols(other_symbols, nasdaq_symbols)


def _load_us_listed_symbols(*, prefer: str = "sec") -> tuple[dict[str, ListedSymbol], str]:
    prefer_norm = (prefer or "").strip().lower()
    if prefer_norm == "nasdaq":
        try:
            return _load_nasdaq_trader_symbols(), "nasdaq"
        except UniverseError as exc:
            logger.warning("Nasdaq Trader universe failed; falling back to SEC: %s", exc)
            return _load_sec_company_tickers_exchange(), "sec"

    try:
        return _load_sec_company_tickers_exchange(), "sec"
    except UniverseError as exc:
        logger.warning("SEC universe failed; falling back to Nasdaq Trader: %s", exc)
        return _load_nasdaq_trader_symbols(), "nasdaq"


def _symbols_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise UniverseError(f"Universe file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            raise UniverseError(f"Invalid JSON universe file: {path}") from exc
        symbols: list[str] = []
        if isinstance(payload, dict):
            if "symbols" in payload:
                symbols = payload.get("symbols", [])
            elif "fields" in payload and "data" in payload:
                fields = payload.get("fields", [])
                data = payload.get("data", [])
                if isinstance(fields, list) and isinstance(data, list) and "ticker" in fields:
                    ticker_idx = fields.index("ticker")
                    exchange_idx = fields.index("exchange") if "exchange" in fields else None
                    for row in data:
                        if not isinstance(row, list) or len(row) <= ticker_idx:
                            continue
                        if exchange_idx is not None and len(row) > exchange_idx:
                            exchange = str(row[exchange_idx]) if row[exchange_idx] is not None else ""
                            if exchange.strip().upper().startswith("OTC") or not exchange.strip():
                                continue
                        symbols.append(str(row[ticker_idx]))
            else:
                # SEC company_tickers.json style: dict of numeric keys -> {ticker, title, cik_str}
                for val in payload.values():
                    if isinstance(val, dict) and "ticker" in val:
                        symbols.append(str(val.get("ticker")))
        elif isinstance(payload, list):
            symbols = payload
        if not isinstance(symbols, list):
            raise UniverseError(f"Universe JSON must be a list or {{symbols: []}}: {path}")
        return sorted({_normalize_symbol(str(s)) for s in symbols if s})

    if path.suffix.lower() == ".csv":
        reader = csv.reader(text.splitlines())
        try:
            header = next(reader)
        except StopIteration:
            return []
        header_lower = [str(h).strip().lower() for h in header]
        sym_idx = header_lower.index("symbol") if "symbol" in header_lower else 0
        syms = []
        for row in reader:
            if not row:
                continue
            if sym_idx >= len(row):
                continue
            syms.append(_normalize_symbol(row[sym_idx]))
        return sorted({s for s in syms if s})

    # Default: newline-delimited text.
    syms = [_normalize_symbol(line) for line in text.splitlines()]
    return sorted({s for s in syms if s})


def _cache_fresh(path: Path, *, refresh_days: int) -> bool:
    if not path.exists():
        return False
    if refresh_days <= 0:
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime < timedelta(days=refresh_days)


def load_universe_symbols(
    source: str,
    *,
    cache_dir: Path,
    refresh_days: int = 1,
) -> list[str]:
    source_raw = (source or "").strip()
    source_norm = source_raw.lower()
    if source_norm.startswith("file:"):
        path = Path(source_raw[len("file:") :]).expanduser()
        return _symbols_from_file(path)

    if source_norm in {"us-equities", "us-equity", "us_stocks", "us-stocks"}:
        key = "us-equities"
        want_etf = False
        want_equity = True
        prefer = "sec"
    elif source_norm in {"us-etfs", "us-etf"}:
        key = "us-etfs"
        want_etf = True
        want_equity = False
        prefer = "nasdaq"
    elif source_norm in {"us-all", "us", "us-equities-etfs", "us-stock-etf"}:
        key = "us-all"
        want_etf = True
        want_equity = True
        prefer = "sec"
    else:
        raise UniverseError(
            "Unknown universe source. Use us-all/us-equities/us-etfs or file:/path/to/list.txt"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if _cache_fresh(cache_path, refresh_days=refresh_days):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            symbols = payload.get("symbols", []) if isinstance(payload, dict) else payload
            if isinstance(symbols, list) and symbols:
                return sorted({_normalize_symbol(str(s)) for s in symbols if s})
        except Exception:  # noqa: BLE001
            logger.warning("Invalid universe cache; rebuilding: %s", cache_path)

    listing, source_used = _load_us_listed_symbols(prefer=prefer)
    if key == "us-etfs" and source_used != "nasdaq":
        raise UniverseError(
            "ETF-only universe requires Nasdaq Trader symbol directory access; SEC list has no ETF flag."
        )
    symbols: list[str] = []
    for sym, info in listing.items():
        if info.is_etf and not want_etf:
            continue
        if not info.is_etf and not want_equity:
            continue
        symbols.append(sym)
    symbols = sorted({s for s in symbols if s})

    payload = {"asof": datetime.now(timezone.utc).isoformat(), "source": key, "symbols": symbols}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return symbols
