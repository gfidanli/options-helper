from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from options_helper.models import OptionType

_UNDERLYING_ALIASES = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BF.B": "BF-B",
    "BF.A": "BF-A",
}

_PM_SETTLED_ROOTS: set[str] = {
    "SPY",
    "SPXW",
    "XSP",
    "QQQ",
    "IWM",
    "DIA",
}

_AM_SETTLED_ROOTS: set[str] = {
    "SPX",
    "RUT",
    "NDX",
    "OEX",
}


def normalize_underlying(symbol: str | None) -> str:
    if symbol is None:
        return ""
    s = str(symbol).strip().upper()
    if not s:
        return ""
    if s in _UNDERLYING_ALIASES:
        return _UNDERLYING_ALIASES[s]
    if "." in s and "-" not in s:
        s = s.replace(".", "-")
    return s


@dataclass(frozen=True)
class ParsedContract:
    underlying: str
    underlying_norm: str
    expiry: date
    option_type: OptionType
    strike: float
    raw: str | None = None


def _parse_osi_date(value: str) -> date:
    if len(value) != 6 or not value.isdigit():
        raise ValueError("invalid OSI date")
    yy = int(value[0:2])
    mm = int(value[2:4])
    dd = int(value[4:6])
    year = 1900 + yy if yy >= 70 else 2000 + yy
    return date(year, mm, dd)


def parse_contract_symbol(raw: str | None) -> ParsedContract | None:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or len(s) < 15:
        return None
    tail = s[-15:]
    root = s[:-15].strip()
    if not root:
        return None
    date_part = tail[:6]
    option_part = tail[6]
    strike_part = tail[7:]
    if not date_part.isdigit() or option_part not in {"C", "P"} or not strike_part.isdigit():
        return None
    try:
        expiry = _parse_osi_date(date_part)
    except ValueError:
        return None
    strike = int(strike_part) / 1000.0
    option_type: OptionType = "call" if option_part == "C" else "put"
    underlying_norm = normalize_underlying(root)
    return ParsedContract(
        underlying=root,
        underlying_norm=underlying_norm,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
        raw=s,
    )


def format_osi(parsed: ParsedContract) -> str:
    root = normalize_underlying(parsed.underlying_norm or parsed.underlying)
    if not root:
        raise ValueError("missing underlying")
    if not isinstance(parsed.expiry, date):
        raise ValueError("missing expiry")
    if parsed.option_type not in {"call", "put"}:
        raise ValueError("invalid option_type")
    if parsed.strike is None:
        raise ValueError("missing strike")
    strike_int = int(round(float(parsed.strike) * 1000))
    if strike_int < 0 or strike_int > 99999999:
        raise ValueError("strike out of range for OSI encoding")
    root_part = root.ljust(6)
    date_part = parsed.expiry.strftime("%y%m%d")
    option_part = "C" if parsed.option_type == "call" else "P"
    return f"{root_part}{date_part}{option_part}{strike_int:08d}"


def infer_settlement_style(value: ParsedContract | str | None) -> str | None:
    """Best-effort inference for option settlement style.

    Returns:
      - ``"pm"`` for known PM-settled roots (for example SPXW/SPY).
      - ``"am"`` for known AM-settled index monthlies (for example SPX).
      - ``None`` when the root cannot be inferred.
    """

    parsed: ParsedContract | None
    if isinstance(value, ParsedContract):
        parsed = value
    else:
        parsed = parse_contract_symbol(value)
    if parsed is None:
        return None

    root = normalize_underlying(parsed.underlying_norm or parsed.underlying)
    if not root:
        return None
    if root in _PM_SETTLED_ROOTS or root.endswith("W"):
        return "pm"
    if root in _AM_SETTLED_ROOTS:
        return "am"
    return None


__all__ = [
    "ParsedContract",
    "format_osi",
    "infer_settlement_style",
    "normalize_underlying",
    "parse_contract_symbol",
]
