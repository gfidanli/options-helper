from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class TailRiskConfig(BaseModel):
    lookback_days: int = Field(default=252 * 6, ge=30)
    horizon_days: int = Field(default=60, ge=1)
    num_simulations: int = Field(default=25_000, ge=1_000)
    seed: int = 42
    var_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    end_percentiles: list[float] = Field(default_factory=lambda: [1, 5, 10, 25, 50, 75, 90, 95, 99])
    chart_percentiles: list[float] = Field(default_factory=lambda: [5, 25, 50, 75, 95])
    sample_paths: int = Field(default=200, ge=0)
    trading_days_per_year: int = Field(default=252, ge=1)

    @field_validator("end_percentiles", "chart_percentiles")
    @classmethod
    def _validate_percentiles(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("percentile list cannot be empty")
        out: list[float] = []
        for raw in value:
            pct = float(raw)
            if pct < 0.0 or pct > 100.0:
                raise ValueError("percentiles must be in [0, 100]")
            out.append(pct)
        # Deterministic ordering and no duplicates.
        return sorted(set(out))


@dataclass(frozen=True)
class TailRiskResult:
    as_of: str
    spot: float
    daily_logret_mean: float
    daily_logret_std: float
    realized_vol_annual: float
    expected_return_annual: float
    var_return: float
    cvar_return: float | None
    end_price_percentiles: dict[float, float]
    end_return_percentiles: dict[float, float]
    daily_price_bands: pd.DataFrame
    sample_price_paths: pd.DataFrame
    end_returns: np.ndarray
    warnings: list[str]


def _simulate_price_paths(
    *,
    spot: float,
    mean: float,
    std: float,
    horizon: int,
    sims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(horizon, sims))
    returns_matrix = mean + std * z
    log_paths = np.cumsum(returns_matrix, axis=0)
    simulated_paths = spot * np.exp(log_paths)
    full_paths = np.vstack([np.full(shape=(1, sims), fill_value=spot), simulated_paths])
    end_prices = full_paths[-1, :]
    end_returns = (end_prices / spot) - 1.0
    return full_paths, end_prices, end_returns


def _build_daily_price_bands(full_paths: np.ndarray, *, chart_percentiles: list[float]) -> pd.DataFrame:
    daily_price_bands = pd.DataFrame(index=np.arange(full_paths.shape[0]))
    daily_price_bands.index.name = "step"
    for pct in chart_percentiles:
        daily_price_bands[_percentile_label(pct)] = np.quantile(full_paths, pct / 100.0, axis=1)
    return daily_price_bands


def _build_sample_price_paths(full_paths: np.ndarray, *, sample_paths: int) -> pd.DataFrame:
    sample_count = min(max(int(sample_paths), 0), full_paths.shape[1])
    sample_path_columns = [f"path_{idx}" for idx in range(sample_count)]
    sample_price_paths = pd.DataFrame(
        full_paths[:, :sample_count],
        index=np.arange(full_paths.shape[0]),
        columns=sample_path_columns,
    )
    sample_price_paths.index.name = "step"
    return sample_price_paths


def compute_tail_risk(close: pd.Series, *, config: TailRiskConfig) -> TailRiskResult:
    cleaned = pd.to_numeric(close, errors="coerce")
    cleaned = cleaned.where(cleaned > 0).dropna()

    warnings: list[str] = []
    if cleaned.empty:
        warnings.append("invalid_inputs")
        return _empty_result(config=config, warnings=warnings)

    spot = float(cleaned.iloc[-1])
    if not np.isfinite(spot) or spot <= 0:
        warnings.append("invalid_inputs")
        return _empty_result(config=config, warnings=warnings)

    as_of = _as_of_from_index(cleaned)
    window = cleaned.tail(int(config.lookback_days) + 1)
    daily_log_returns = np.log(window).diff().dropna()

    if len(daily_log_returns) < 30:
        warnings.append("insufficient_history")

    if daily_log_returns.empty:
        warnings.append("degenerate_vol")
        return _empty_result(config=config, warnings=warnings, as_of=as_of, spot=spot)

    m = float(daily_log_returns.mean())
    s = float(daily_log_returns.std(ddof=1))
    if not np.isfinite(s) or s <= 0.0:
        warnings.append("degenerate_vol")
        s = 0.0

    horizon = int(config.horizon_days)
    sims = int(config.num_simulations)
    full_paths, end_prices, end_returns = _simulate_price_paths(
        spot=spot,
        mean=m,
        std=s,
        horizon=horizon,
        sims=sims,
        seed=int(config.seed),
    )

    end_price_percentiles = {
        float(pct): float(np.quantile(end_prices, pct / 100.0))
        for pct in config.end_percentiles
    }
    end_return_percentiles = {
        float(pct): float(np.quantile(end_returns, pct / 100.0))
        for pct in config.end_percentiles
    }

    var_return = float(np.quantile(end_returns, 1.0 - float(config.var_confidence)))
    cvar_tail = end_returns[end_returns <= var_return]
    cvar_return = float(cvar_tail.mean()) if cvar_tail.size else None

    if cvar_return is None:
        warnings.append("degenerate_vol")

    daily_price_bands = _build_daily_price_bands(full_paths, chart_percentiles=config.chart_percentiles)
    sample_price_paths = _build_sample_price_paths(full_paths, sample_paths=int(config.sample_paths))

    realized_vol_annual = float(s * math.sqrt(float(config.trading_days_per_year)))
    expected_return_annual = float(m * float(config.trading_days_per_year))

    return TailRiskResult(
        as_of=as_of,
        spot=spot,
        daily_logret_mean=float(m),
        daily_logret_std=float(s),
        realized_vol_annual=realized_vol_annual,
        expected_return_annual=expected_return_annual,
        var_return=var_return,
        cvar_return=cvar_return,
        end_price_percentiles=end_price_percentiles,
        end_return_percentiles=end_return_percentiles,
        daily_price_bands=daily_price_bands,
        sample_price_paths=sample_price_paths,
        end_returns=end_returns.astype("float64"),
        warnings=_dedupe_preserve_order(warnings),
    )


def _empty_result(
    *,
    config: TailRiskConfig,
    warnings: list[str],
    as_of: str = "",
    spot: float = float("nan"),
) -> TailRiskResult:
    horizon = int(config.horizon_days)
    sims = int(config.num_simulations)
    empty_bands = pd.DataFrame(index=np.arange(horizon + 1))
    empty_bands.index.name = "step"
    for pct in config.chart_percentiles:
        empty_bands[_percentile_label(pct)] = np.nan

    sample_count = min(max(int(config.sample_paths), 0), sims)
    sample_path_columns = [f"path_{idx}" for idx in range(sample_count)]
    empty_paths = pd.DataFrame(
        np.full(shape=(horizon + 1, sample_count), fill_value=np.nan),
        index=np.arange(horizon + 1),
        columns=sample_path_columns,
    )
    empty_paths.index.name = "step"

    end_percentiles = {float(pct): float("nan") for pct in config.end_percentiles}
    return TailRiskResult(
        as_of=as_of,
        spot=spot,
        daily_logret_mean=float("nan"),
        daily_logret_std=float("nan"),
        realized_vol_annual=float("nan"),
        expected_return_annual=float("nan"),
        var_return=float("nan"),
        cvar_return=None,
        end_price_percentiles=end_percentiles.copy(),
        end_return_percentiles=end_percentiles.copy(),
        daily_price_bands=empty_bands,
        sample_price_paths=empty_paths,
        end_returns=np.array([], dtype="float64"),
        warnings=_dedupe_preserve_order(warnings),
    )


def _as_of_from_index(series: pd.Series) -> str:
    if not isinstance(series.index, pd.Index) or series.empty:
        return ""
    last = series.index[-1]
    ts = pd.to_datetime(last, errors="coerce")
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts.date().isoformat()
    return str(last)


def _percentile_label(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile):02d}"
    text = f"{float(percentile):.2f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"p{text}"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[Any] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
