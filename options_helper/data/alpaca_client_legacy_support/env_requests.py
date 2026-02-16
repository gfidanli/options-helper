from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_DEFAULT_MARKET_TZ = "America/New_York"

StockBarsRequest = None
OptionContractsRequest = None
OptionChainRequest = None
OptionBarsRequest = None
CorporateActionsClient = None
CorporateActionsRequest = None
NewsClient = None
NewsRequest = None


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


def _env_truthy(name: str) -> bool:
    raw = _clean_env(os.getenv(name)).lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _load_market_tz_name() -> str:
    return _clean_env(os.getenv("OH_ALPACA_MARKET_TZ")) or _DEFAULT_MARKET_TZ


def _load_market_tz() -> ZoneInfo:
    name = _load_market_tz_name()
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001
        return ZoneInfo(_DEFAULT_MARKET_TZ)


def _market_day_bounds(day: date, market_tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(day, datetime.min.time()).replace(tzinfo=market_tz)
    end_local = datetime.combine(day, datetime.max.time()).replace(tzinfo=market_tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _coerce_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _coerce_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export ") :].strip()
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_quotes(value)


def _load_env_file(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except Exception:  # noqa: BLE001
        return

    for line in content.splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not (key.startswith("APCA_") or key.startswith("OH_ALPACA_")):
            continue
        if key in os.environ:
            continue
        os.environ[key] = value


def _maybe_load_alpaca_env() -> None:
    override = _clean_env(os.getenv("OH_ALPACA_ENV_FILE"))
    if override:
        _load_env_file(Path(override))
        return
    _load_env_file(Path("config/alpaca.env"))
    _load_env_file(Path(".env"))


def _load_stock_bars_request():
    global StockBarsRequest  # noqa: PLW0603 - intentional lazy import
    if StockBarsRequest is not None:
        return StockBarsRequest
    try:
        from alpaca.data.requests import StockBarsRequest as AlpacaStockBarsRequest

        StockBarsRequest = AlpacaStockBarsRequest
    except Exception:  # noqa: BLE001
        return None
    return StockBarsRequest


def _load_option_contracts_request():
    global OptionContractsRequest  # noqa: PLW0603 - intentional lazy import
    if OptionContractsRequest is not None:
        return OptionContractsRequest
    try:
        from alpaca.trading.requests import GetOptionContractsRequest as AlpacaGetOptionContractsRequest

        OptionContractsRequest = AlpacaGetOptionContractsRequest
    except Exception:  # noqa: BLE001
        try:
            from alpaca.trading.requests import OptionContractsRequest as AlpacaOptionContractsRequest

            OptionContractsRequest = AlpacaOptionContractsRequest
        except Exception:  # noqa: BLE001
            return None
    return OptionContractsRequest


def _load_option_chain_request():
    global OptionChainRequest  # noqa: PLW0603 - intentional lazy import
    if OptionChainRequest is not None:
        return OptionChainRequest
    try:
        from alpaca.data.requests import OptionChainRequest as AlpacaOptionChainRequest

        OptionChainRequest = AlpacaOptionChainRequest
    except Exception:  # noqa: BLE001
        return None
    return OptionChainRequest


def _load_option_bars_request():
    global OptionBarsRequest  # noqa: PLW0603 - intentional lazy import
    if OptionBarsRequest is not None:
        return OptionBarsRequest
    try:
        from alpaca.data.requests import OptionBarsRequest as AlpacaOptionBarsRequest

        OptionBarsRequest = AlpacaOptionBarsRequest
    except Exception:  # noqa: BLE001
        return None
    return OptionBarsRequest


def _load_corporate_actions_client():
    global CorporateActionsClient  # noqa: PLW0603 - intentional lazy import
    if CorporateActionsClient is not None:
        return CorporateActionsClient
    try:
        from alpaca.data.historical.corporate_actions import (
            CorporateActionsClient as AlpacaCorporateActionsClient,
        )

        CorporateActionsClient = AlpacaCorporateActionsClient
    except Exception:  # noqa: BLE001
        return None
    return CorporateActionsClient


def _load_corporate_actions_request():
    global CorporateActionsRequest  # noqa: PLW0603 - intentional lazy import
    if CorporateActionsRequest is not None:
        return CorporateActionsRequest
    try:
        from alpaca.data.requests import CorporateActionsRequest as AlpacaCorporateActionsRequest

        CorporateActionsRequest = AlpacaCorporateActionsRequest
    except Exception:  # noqa: BLE001
        return None
    return CorporateActionsRequest


def _load_news_client():
    global NewsClient  # noqa: PLW0603 - intentional lazy import
    if NewsClient is not None:
        return NewsClient
    try:
        from alpaca.data.historical.news import NewsClient as AlpacaNewsClient

        NewsClient = AlpacaNewsClient
    except Exception:  # noqa: BLE001
        return None
    return NewsClient


def _load_news_request():
    global NewsRequest  # noqa: PLW0603 - intentional lazy import
    if NewsRequest is not None:
        return NewsRequest
    try:
        from alpaca.data.requests import NewsRequest as AlpacaNewsRequest

        NewsRequest = AlpacaNewsRequest
    except Exception:  # noqa: BLE001
        return None
    return NewsRequest


def _coerce_datetime(value: date | datetime | None, *, end_of_day: bool) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.combine(value, datetime.max.time() if end_of_day else datetime.min.time())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
