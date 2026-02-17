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


def _coalesce_field(
    *,
    update: Any,
    order: Any,
    update_names: tuple[str, ...],
    order_names: tuple[str, ...],
    caster,
) -> Any:
    value = caster(_get_field(update, update_names))
    if value is not None:
        return value
    return caster(_get_field(order, order_names))


def _normalize_trade_core_fields(*, update: Any, order: Any) -> dict[str, Any] | None:
    timestamp = _coalesce_field(
        update=update,
        order=order,
        update_names=("timestamp", "at", "time", "updated_at", "ts"),
        order_names=("updated_at", "filled_at", "submitted_at", "created_at"),
        caster=_coerce_timestamp,
    )
    event = _coerce_lower_str(_get_field(update, ("event", "event_type")))
    order_id = _coalesce_field(
        update=update,
        order=order,
        update_names=("order_id", "id"),
        order_names=("id", "order_id", "client_order_id"),
        caster=_coerce_str,
    )
    symbol = _coalesce_field(
        update=update,
        order=order,
        update_names=("symbol", "asset_symbol"),
        order_names=("symbol", "asset_symbol"),
        caster=_coerce_str,
    )
    if symbol is not None:
        symbol = symbol.upper()
    side = _coalesce_field(update=update, order=order, update_names=("side",), order_names=("side",), caster=_coerce_lower_str)
    qty = _coalesce_field(
        update=update,
        order=order,
        update_names=("qty", "quantity"),
        order_names=("qty", "quantity", "order_qty"),
        caster=_coerce_float,
    )
    filled_qty = _coalesce_field(
        update=update,
        order=order,
        update_names=("filled_qty", "filled_quantity"),
        order_names=("filled_qty", "filled_quantity", "filled"),
        caster=_coerce_float,
    )
    filled_avg_price = _coalesce_field(
        update=update,
        order=order,
        update_names=("filled_avg_price", "average_fill_price"),
        order_names=("filled_avg_price", "average_fill_price", "avg_fill_price"),
        caster=_coerce_float,
    )
    status = _coalesce_field(
        update=update,
        order=order,
        update_names=("status", "order_status"),
        order_names=("status", "order_status"),
        caster=_coerce_lower_str,
    )
    if event is None:
        event = status
    if timestamp is None and event is None and order_id is None and symbol is None and status is None:
        return None
    return {
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


def _set_optional_trade_fields(row: dict[str, Any], *, update: Any, order: Any) -> None:
    order_type = _coalesce_field(
        update=update,
        order=order,
        update_names=("type", "order_type"),
        order_names=("type", "order_type"),
        caster=_coerce_lower_str,
    )
    if order_type is not None:
        row["type"] = order_type
    tif = _coalesce_field(
        update=update,
        order=order,
        update_names=("tif", "time_in_force"),
        order_names=("time_in_force", "tif"),
        caster=_coerce_lower_str,
    )
    if tif is not None:
        row["tif"] = tif
    limit_price = _coalesce_field(
        update=update,
        order=order,
        update_names=("limit_price",),
        order_names=("limit_price",),
        caster=_coerce_float,
    )
    if limit_price is not None:
        row["limit_price"] = limit_price
    stop_price = _coalesce_field(
        update=update,
        order=order,
        update_names=("stop_price",),
        order_names=("stop_price",),
        caster=_coerce_float,
    )
    if stop_price is not None:
        row["stop_price"] = stop_price


def normalize_trade_update(update: Any) -> dict[str, Any] | None:
    if update is None:
        return None
    order = _get_field(update, ("order",))
    row = _normalize_trade_core_fields(update=update, order=order)
    if row is None:
        return None
    _set_optional_trade_fields(row, update=update, order=order)
    return row


__all__ = ["normalize_trade_update"]
