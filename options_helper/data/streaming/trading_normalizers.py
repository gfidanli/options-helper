from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _get_field(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj.get(name)
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


def _coerce_lower_str(value: Any) -> str | None:
    raw = _coerce_str(value)
    if raw is None:
        return None
    return raw.lower()


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "to_pydatetime"):
        try:
            dt = value.to_pydatetime()
        except Exception:  # noqa: BLE001
            dt = None
    elif isinstance(value, (int, float)):
        val = float(value)
        if val > 1e15:
            dt = datetime.fromtimestamp(val / 1e9, tz=timezone.utc)
        elif val > 1e12:
            dt = datetime.fromtimestamp(val / 1e3, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_trade_update(update: Any) -> dict[str, Any] | None:
    if update is None:
        return None
    order = _get_field(update, ("order",))

    timestamp = _coerce_timestamp(
        _get_field(update, ("timestamp", "at", "time", "updated_at", "ts"))
    )
    if timestamp is None:
        timestamp = _coerce_timestamp(
            _get_field(order, ("updated_at", "filled_at", "submitted_at", "created_at"))
        )

    event = _coerce_lower_str(_get_field(update, ("event", "event_type")))
    order_id = _coerce_str(_get_field(update, ("order_id", "id")))
    if order_id is None:
        order_id = _coerce_str(_get_field(order, ("id", "order_id", "client_order_id")))

    symbol = _coerce_str(_get_field(update, ("symbol", "asset_symbol")))
    if symbol is None:
        symbol = _coerce_str(_get_field(order, ("symbol", "asset_symbol")))
    if symbol is not None:
        symbol = symbol.upper()

    side = _coerce_lower_str(_get_field(update, ("side",)))
    if side is None:
        side = _coerce_lower_str(_get_field(order, ("side",)))

    qty = _coerce_float(_get_field(update, ("qty", "quantity")))
    if qty is None:
        qty = _coerce_float(_get_field(order, ("qty", "quantity", "order_qty")))

    filled_qty = _coerce_float(_get_field(update, ("filled_qty", "filled_quantity")))
    if filled_qty is None:
        filled_qty = _coerce_float(
            _get_field(order, ("filled_qty", "filled_quantity", "filled"))
        )

    filled_avg_price = _coerce_float(
        _get_field(update, ("filled_avg_price", "average_fill_price"))
    )
    if filled_avg_price is None:
        filled_avg_price = _coerce_float(
            _get_field(order, ("filled_avg_price", "average_fill_price", "avg_fill_price"))
        )

    status = _coerce_lower_str(_get_field(update, ("status", "order_status")))
    if status is None:
        status = _coerce_lower_str(_get_field(order, ("status", "order_status")))
    if event is None:
        event = status

    if (
        timestamp is None
        and event is None
        and order_id is None
        and symbol is None
        and status is None
    ):
        return None

    row: dict[str, Any] = {
        "timestamp": timestamp,
        "event": event,
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "filled_qty": filled_qty,
        "filled_avg_price": filled_avg_price,
        "status": status,
    }

    order_type = _coerce_lower_str(_get_field(update, ("type", "order_type")))
    if order_type is None:
        order_type = _coerce_lower_str(_get_field(order, ("type", "order_type")))
    if order_type is not None:
        row["type"] = order_type

    tif = _coerce_lower_str(_get_field(update, ("tif", "time_in_force")))
    if tif is None:
        tif = _coerce_lower_str(_get_field(order, ("time_in_force", "tif")))
    if tif is not None:
        row["tif"] = tif

    limit_price = _coerce_float(_get_field(update, ("limit_price",)))
    if limit_price is None:
        limit_price = _coerce_float(_get_field(order, ("limit_price",)))
    if limit_price is not None:
        row["limit_price"] = limit_price

    stop_price = _coerce_float(_get_field(update, ("stop_price",)))
    if stop_price is None:
        stop_price = _coerce_float(_get_field(order, ("stop_price",)))
    if stop_price is not None:
        row["stop_price"] = stop_price

    return row


__all__ = ["normalize_trade_update"]
