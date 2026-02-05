from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.flow import FlowArtifact


_GROUP_BY_VALUES = {"contract", "strike", "expiry", "expiry-strike"}
_FLOW_ROW_COLUMNS = [
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "delta_oi",
    "delta_oi_notional",
    "volume_notional",
    "delta_notional",
    "n_pairs",
]
_FLOW_LOAD_COLUMNS = [
    "symbol",
    "as_of",
    "from_date",
    "to_date",
    "window",
    "group_by",
    "row_key",
    *_FLOW_ROW_COLUMNS,
    "snapshot_dates",
    "generated_at",
    "updated_at",
]


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    text = str(value).strip()
    return text or None


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean_text(value)
    if text is None:
        raise ValueError("date value is required")
    return date.fromisoformat(text)


def _coerce_optional_date(value: object) -> date | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return _coerce_date(text)
    except Exception:  # noqa: BLE001
        return None


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = _clean_text(value)
    if text is None:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:  # noqa: BLE001
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _normalize_symbol(value: object) -> str:
    sym = (_clean_text(value) or "").upper()
    if not sym:
        raise ValueError("symbol is required")
    return sym


def _normalize_group_by(value: object) -> str:
    group_by = (_clean_text(value) or "").lower()
    if group_by not in _GROUP_BY_VALUES:
        allowed = ", ".join(sorted(_GROUP_BY_VALUES))
        raise ValueError(f"group_by must be one of: {allowed}")
    return group_by


def _snapshot_dates_json(value: object) -> str:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if _clean_text(item)]
        return json.dumps(out, sort_keys=True)
    text = _clean_text(value)
    if text is None:
        return "[]"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            out = [str(item).strip() for item in parsed if _clean_text(item)]
            return json.dumps(out, sort_keys=True)
    except Exception:  # noqa: BLE001
        pass
    return "[]"


def _snapshot_dates_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if _clean_text(item)]
    text = _clean_text(value)
    if text is None:
        return []
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if _clean_text(item)]


def _row_value(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _format_key_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _base_row_key(group_by: str, row: dict[str, Any], index: int) -> str:
    contract_symbol = _clean_text(_row_value(row, "contract_symbol", "contractSymbol"))
    expiry = _clean_text(_row_value(row, "expiry"))
    option_type = _clean_text(_row_value(row, "option_type", "optionType"))
    strike = _format_key_float(_coerce_float(_row_value(row, "strike")))

    if group_by == "contract":
        key_parts = [contract_symbol, expiry, option_type, strike]
    elif group_by == "strike":
        key_parts = [option_type, strike]
    elif group_by == "expiry":
        key_parts = [option_type, expiry]
    else:  # expiry-strike
        key_parts = [option_type, expiry, strike]

    cleaned = [part for part in key_parts if part]
    if not cleaned:
        return f"row-{index:06d}"
    return "|".join(cleaned)


def _unique_row_key(base: str, seen: set[str], index: int) -> str:
    if base not in seen:
        seen.add(base)
        return base
    candidate = f"{base}|{index}"
    suffix = index
    while candidate in seen:
        suffix += 1
        candidate = f"{base}|{suffix}"
    seen.add(candidate)
    return candidate


class FlowStore(Protocol):
    def upsert_artifact(self, artifact: FlowArtifact) -> int:
        ...

    def load_rows(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> pd.DataFrame:
        ...

    def load_artifact(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> FlowArtifact | None:
        ...

    def list_partitions(
        self,
        *,
        symbol: str | None = None,
        group_by: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class NoopFlowStore:
    root_dir: Path

    def upsert_artifact(self, artifact: FlowArtifact) -> int:
        del artifact
        return 0

    def load_rows(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> pd.DataFrame:
        del symbol, from_date, to_date, window, group_by
        return pd.DataFrame(columns=_FLOW_LOAD_COLUMNS)

    def load_artifact(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> FlowArtifact | None:
        del symbol, from_date, to_date, window, group_by
        return None

    def list_partitions(
        self,
        *,
        symbol: str | None = None,
        group_by: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        del symbol, group_by, limit
        return []


@dataclass(frozen=True)
class DuckDBFlowStore:
    root_dir: Path
    warehouse: DuckDBWarehouse

    def _partition(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> tuple[str, date, date, int, str]:
        return (
            _normalize_symbol(symbol),
            _coerce_date(from_date),
            _coerce_date(to_date),
            int(window),
            _normalize_group_by(group_by),
        )

    def upsert_artifact(self, artifact: FlowArtifact) -> int:
        ensure_schema(self.warehouse)
        payload = artifact.model_dump(mode="json")

        symbol, from_dt, to_dt, window_size, group_by = self._partition(
            symbol=str(payload.get("symbol") or ""),
            from_date=str(payload.get("from_date") or ""),
            to_date=str(payload.get("to_date") or ""),
            window=int(payload.get("window") or 0),
            group_by=str(payload.get("group_by") or ""),
        )

        as_of = _coerce_date(payload.get("as_of"))
        generated_at = _coerce_datetime(payload.get("generated_at"))
        snapshot_dates_json = _snapshot_dates_json(payload.get("snapshot_dates"))
        net_rows = payload.get("net") or []
        if not isinstance(net_rows, list):
            raise ValueError("artifact.net must be a list")

        written = 0
        seen_row_keys: set[str] = set()
        with self.warehouse.transaction() as tx:
            tx.execute(
                """
                DELETE FROM options_flow
                WHERE symbol = ?
                  AND from_date = ?
                  AND to_date = ?
                  AND window_size = ?
                  AND group_by = ?
                """,
                [symbol, from_dt, to_dt, window_size, group_by],
            )

            for idx, item in enumerate(net_rows):
                row = item if isinstance(item, dict) else {}
                row_key = _unique_row_key(_base_row_key(group_by, row, idx), seen_row_keys, idx)

                tx.execute(
                    """
                    INSERT INTO options_flow(
                      symbol, as_of, from_date, to_date, window_size, group_by, row_key,
                      contract_symbol, expiry, option_type, strike,
                      delta_oi, delta_oi_notional, volume_notional, delta_notional, n_pairs,
                      snapshot_dates_json, generated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        symbol,
                        as_of,
                        from_dt,
                        to_dt,
                        window_size,
                        group_by,
                        row_key,
                        _clean_text(_row_value(row, "contract_symbol", "contractSymbol")),
                        _coerce_optional_date(_row_value(row, "expiry")),
                        _clean_text(_row_value(row, "option_type", "optionType")),
                        _coerce_float(_row_value(row, "strike")),
                        _coerce_float(_row_value(row, "delta_oi", "deltaOI")),
                        _coerce_float(_row_value(row, "delta_oi_notional", "deltaOI_notional")),
                        _coerce_float(_row_value(row, "volume_notional")),
                        _coerce_float(_row_value(row, "delta_notional")),
                        _coerce_int(_row_value(row, "n_pairs", "size")),
                        snapshot_dates_json,
                        generated_at,
                    ],
                )
                written += 1
        return written

    def load_rows(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> pd.DataFrame:
        ensure_schema(self.warehouse)
        sym, from_dt, to_dt, window_size, group_by_norm = self._partition(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            window=window,
            group_by=group_by,
        )

        df = self.warehouse.fetch_df(
            """
            SELECT
              symbol,
              as_of,
              from_date,
              to_date,
              window_size AS window,
              group_by,
              row_key,
              contract_symbol,
              expiry,
              option_type,
              strike,
              delta_oi,
              delta_oi_notional,
              volume_notional,
              delta_notional,
              n_pairs,
              snapshot_dates_json,
              generated_at,
              updated_at
            FROM options_flow
            WHERE symbol = ?
              AND from_date = ?
              AND to_date = ?
              AND window_size = ?
              AND group_by = ?
            ORDER BY ABS(COALESCE(delta_oi_notional, 0)) DESC, row_key ASC
            """,
            [sym, from_dt, to_dt, window_size, group_by_norm],
        )

        if df is None or df.empty:
            return pd.DataFrame(columns=_FLOW_LOAD_COLUMNS)

        out = df.copy()
        out["snapshot_dates"] = out["snapshot_dates_json"].map(_snapshot_dates_list)
        out = out.drop(columns=["snapshot_dates_json"])

        for col in ("as_of", "from_date", "to_date", "expiry"):
            if col in out.columns:
                out[col] = out[col].astype(str)
        return out

    def load_artifact(
        self,
        *,
        symbol: str,
        from_date: date | str,
        to_date: date | str,
        window: int,
        group_by: str,
    ) -> FlowArtifact | None:
        rows = self.load_rows(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            window=window,
            group_by=group_by,
        )
        if rows.empty:
            return None

        first = rows.iloc[0]
        generated_at = _coerce_datetime(first.get("generated_at")) or datetime.now(timezone.utc)
        payload = {
            "schema_version": 1,
            "generated_at": generated_at,
            "as_of": str(first.get("as_of")),
            "symbol": str(first.get("symbol")),
            "from_date": str(first.get("from_date")),
            "to_date": str(first.get("to_date")),
            "window": int(first.get("window")),
            "group_by": str(first.get("group_by")),
            "snapshot_dates": list(first.get("snapshot_dates") or []),
            "net": rows[_FLOW_ROW_COLUMNS].where(pd.notna(rows[_FLOW_ROW_COLUMNS]), None).to_dict(orient="records"),
        }
        return FlowArtifact.model_validate(payload)

    def list_partitions(
        self,
        *,
        symbol: str | None = None,
        group_by: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        ensure_schema(self.warehouse)
        where_clauses: list[str] = []
        params: list[Any] = []

        if symbol is not None and _clean_text(symbol):
            where_clauses.append("symbol = ?")
            params.append(_normalize_symbol(symbol))

        if group_by is not None and _clean_text(group_by):
            where_clauses.append("group_by = ?")
            params.append(_normalize_group_by(group_by))

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(max(int(limit), 1))

        df = self.warehouse.fetch_df(
            f"""
            SELECT
              symbol,
              as_of,
              from_date,
              to_date,
              window_size AS window,
              group_by,
              COUNT(*) AS row_count,
              MAX(updated_at) AS updated_at,
              MAX(generated_at) AS generated_at
            FROM options_flow
            {where_sql}
            GROUP BY symbol, as_of, from_date, to_date, window_size, group_by
            ORDER BY as_of DESC, symbol ASC, group_by ASC, window_size ASC
            LIMIT ?
            """,
            params,
        )
        if df is None or df.empty:
            return []

        out = df.copy()
        for col in ("as_of", "from_date", "to_date"):
            if col in out.columns:
                out[col] = out[col].astype(str)
        return list(out.to_dict(orient="records"))
