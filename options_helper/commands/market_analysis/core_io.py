from __future__ import annotations

from datetime import date
import importlib
from pathlib import Path
from typing import Any

import pandas as pd

import options_helper.cli_deps as cli_deps
from options_helper.commands.common import _parse_date, _spot_from_meta
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.schemas.exposure import ExposureArtifact
from options_helper.schemas.iv_surface import IvSurfaceArtifact
from options_helper.schemas.levels import LevelsArtifact
from options_helper.schemas.tail_risk import TailRiskArtifact


def _select_close(history: pd.DataFrame | None) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    if "Close" in history.columns:
        return pd.to_numeric(history["Close"], errors="coerce")
    if "close" in history.columns:
        return pd.to_numeric(history["close"], errors="coerce")
    return pd.Series(dtype="float64")


def _latest_derived_row(df: pd.DataFrame) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    temp = df.copy()
    if "date" in temp.columns:
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.sort_values(by="date", kind="stable")
    latest = temp.iloc[-1]
    return dict(latest.to_dict())


def _save_artifact_json(
    out_root: Path,
    *,
    feature: str,
    symbol: str,
    as_of: str,
    artifact: TailRiskArtifact | IvSurfaceArtifact | ExposureArtifact | LevelsArtifact,
) -> Path:
    base = out_root / feature / symbol
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{as_of}.json"
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _resolve_snapshot_spot(*, meta: dict[str, Any], candles: pd.DataFrame, as_of: date) -> tuple[float, list[str]]:
    warnings: list[str] = []

    spot = _spot_from_meta(meta)
    if spot is not None and spot > 0:
        return float(spot), warnings

    fallback = _spot_from_candles(candles, as_of)
    if fallback is not None and fallback > 0:
        return float(fallback), warnings

    warnings.append("missing_spot")
    return 0.0, warnings


def _spot_from_candles(candles: pd.DataFrame, as_of: date) -> float | None:
    frame = _normalize_daily_history(candles)
    if frame.empty:
        return None
    filtered = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
    if filtered.empty:
        return None
    close = pd.to_numeric(filtered["close"], errors="coerce").dropna()
    if close.empty:
        return None
    value = float(close.iloc[-1])
    return value if value > 0 else None


def _normalize_daily_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = history.copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        date_col = pd.to_datetime(frame["date"], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        date_col = pd.to_datetime(frame.index, errors="coerce")
    else:
        date_col = pd.Series([pd.NaT] * len(frame), index=frame.index)

    out = pd.DataFrame(index=frame.index)
    out["date"] = date_col
    out["open"] = pd.to_numeric(frame.get("open"), errors="coerce")
    out["high"] = pd.to_numeric(frame.get("high"), errors="coerce")
    out["low"] = pd.to_numeric(frame.get("low"), errors="coerce")
    out["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    out["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _slice_history_to_as_of(history: pd.DataFrame, as_of_spec: str) -> tuple[pd.DataFrame, date]:
    if history.empty:
        raise ValueError("empty daily candle history")

    spec = str(as_of_spec or "").strip().lower()
    if spec == "latest":
        target = history["date"].iloc[-1].date()
    else:
        target = _parse_date(as_of_spec)

    sliced = history[history["date"] <= pd.Timestamp(target)].copy()
    if sliced.empty:
        raise ValueError(f"No candles available on or before {target.isoformat()}.")
    resolved = sliced["date"].iloc[-1].date()
    return sliced.reset_index(drop=True), resolved


def _duckdb_backend_enabled() -> bool:
    market_analysis_pkg = importlib.import_module("options_helper.commands.market_analysis")
    return market_analysis_pkg.get_storage_runtime_config().backend == "duckdb"


def _persist_iv_surface(tenor: pd.DataFrame, delta_buckets: pd.DataFrame, research_dir: Path) -> tuple[int, int]:
    store = cli_deps.build_research_metrics_store(research_dir)
    provider = get_default_provider_name()
    tenor_rows = int(store.upsert_iv_surface_tenor(tenor, provider=provider))
    delta_rows = int(store.upsert_iv_surface_delta_buckets(delta_buckets, provider=provider))
    return tenor_rows, delta_rows


def _persist_exposure_strikes(strike_rows: pd.DataFrame, research_dir: Path) -> int:
    store = cli_deps.build_research_metrics_store(research_dir)
    provider = get_default_provider_name()
    return int(store.upsert_dealer_exposure_strikes(strike_rows, provider=provider))


__all__ = [
    "_select_close",
    "_latest_derived_row",
    "_save_artifact_json",
    "_resolve_snapshot_spot",
    "_normalize_daily_history",
    "_slice_history_to_as_of",
    "_duckdb_backend_enabled",
    "_persist_iv_surface",
    "_persist_exposure_strikes",
]
