from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import market_analysis as market_analysis_command
from options_helper.schemas.exposure import ExposureArtifact
from options_helper.schemas.iv_surface import IvSurfaceArtifact
from options_helper.schemas.levels import LevelsArtifact
from options_helper.schemas.tail_risk import TailRiskArtifact


class _StubCandleStore:
    def __init__(self, histories: dict[str, pd.DataFrame] | pd.DataFrame) -> None:
        if isinstance(histories, pd.DataFrame):
            self._histories = {"*": histories.copy()}
        else:
            self._histories = {str(k).upper(): v.copy() for k, v in histories.items()}

    def load(self, symbol: str) -> pd.DataFrame:
        key = str(symbol or "").upper()
        if key in self._histories:
            return self._histories[key].copy()
        if "*" in self._histories:
            return self._histories["*"].copy()
        return pd.DataFrame()

    def get_daily_history(self, symbol: str, *, period: str = "max") -> pd.DataFrame:  # noqa: ARG002
        return self.load(symbol)


class _StubDerivedStore:
    def __init__(self, derived: pd.DataFrame) -> None:
        self._derived = derived

    def load(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        return self._derived.copy()


class _StubSnapshotStore:
    def __init__(self, daily_rows: dict[date, pd.DataFrame], metas: dict[date, dict[str, Any]]) -> None:
        self._daily_rows = {k: v.copy() for k, v in daily_rows.items()}
        self._metas = {k: dict(v) for k, v in metas.items()}
        self._dates = sorted(self._daily_rows)

    def resolve_date(self, symbol: str, spec: str) -> date:  # noqa: ARG002
        raw = str(spec or "").strip().lower()
        if raw == "latest":
            if not self._dates:
                raise RuntimeError("No snapshots available")
            return self._dates[-1]
        return date.fromisoformat(raw)

    def resolve_relative_date(self, symbol: str, *, to_date: date, offset: int) -> date:  # noqa: ARG002
        idx = self._dates.index(to_date)
        target = idx + int(offset)
        if target < 0 or target >= len(self._dates):
            raise RuntimeError("relative date out of range")
        return self._dates[target]

    def load_day(self, symbol: str, snapshot_date: date) -> pd.DataFrame:  # noqa: ARG002
        return self._daily_rows.get(snapshot_date, pd.DataFrame()).copy()

    def load_meta(self, symbol: str, snapshot_date: date) -> dict[str, Any]:  # noqa: ARG002
        return dict(self._metas.get(snapshot_date, {}))


class _StubResearchMetricsStore:
    def __init__(self) -> None:
        self.iv_tenor_rows = 0
        self.iv_delta_rows = 0
        self.exposure_rows = 0

    def upsert_iv_surface_tenor(self, df: pd.DataFrame, *, provider: str | None = None) -> int:  # noqa: ARG002
        self.iv_tenor_rows = int(len(df))
        return self.iv_tenor_rows

    def upsert_iv_surface_delta_buckets(self, df: pd.DataFrame, *, provider: str | None = None) -> int:  # noqa: ARG002
        self.iv_delta_rows = int(len(df))
        return self.iv_delta_rows

    def upsert_dealer_exposure_strikes(self, df: pd.DataFrame, *, provider: str | None = None) -> int:  # noqa: ARG002
        self.exposure_rows = int(len(df))
        return self.exposure_rows


def _build_history() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    values = (pd.Series(range(500), dtype="float64") + 100.0).to_numpy()
    return pd.DataFrame(
        {
            "Open": values - 1.0,
            "High": values + 1.5,
            "Low": values - 1.5,
            "Close": values,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _build_derived() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-31",
                "atm_iv_near": 0.22,
                "rv_20d": 0.18,
                "rv_60d": 0.20,
                "iv_rv_20d": 1.22,
                "atm_iv_near_percentile": 67.0,
                "iv_term_slope": 0.013,
            }
        ]
    )


def _build_snapshot_rows(*, iv_shift: float = 0.0, oi_scale: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expiries = ["2026-02-14", "2026-03-20"]
    for expiry in expiries:
        for strike, call_delta, put_delta, base_iv in (
            (95.0, 0.70, -0.30, 0.24),
            (100.0, 0.50, -0.50, 0.22),
            (105.0, 0.30, -0.70, 0.26),
        ):
            rows.append(
                {
                    "expiry": expiry,
                    "optionType": "call",
                    "strike": strike,
                    "impliedVolatility": base_iv + iv_shift,
                    "bs_delta": call_delta,
                    "bid": 1.0 + (strike - 95.0) * 0.1,
                    "ask": 1.2 + (strike - 95.0) * 0.1,
                    "lastPrice": 1.1 + (strike - 95.0) * 0.1,
                    "openInterest": int((120 - strike) * oi_scale + 80),
                    "bs_gamma": 0.01,
                }
            )
            rows.append(
                {
                    "expiry": expiry,
                    "optionType": "put",
                    "strike": strike,
                    "impliedVolatility": base_iv + 0.01 + iv_shift,
                    "bs_delta": put_delta,
                    "bid": 1.4 + (105.0 - strike) * 0.1,
                    "ask": 1.6 + (105.0 - strike) * 0.1,
                    "lastPrice": 1.5 + (105.0 - strike) * 0.1,
                    "openInterest": int((strike - 80) * oi_scale + 70),
                    "bs_gamma": 0.012,
                }
            )
    return pd.DataFrame(rows)


def _build_intraday_bars() -> pd.DataFrame:
    ts = pd.date_range("2026-02-07 14:30:00+00:00", periods=5, freq="min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [500.0, 500.5, 501.0, 501.2, 501.1],
            "high": [500.6, 501.0, 501.5, 501.7, 501.6],
            "low": [499.8, 500.2, 500.8, 501.0, 500.9],
            "close": [500.4, 500.9, 501.3, 501.4, 501.2],
            "vwap": [500.3, 500.8, 501.2, 501.3, 501.1],
            "volume": [1000, 1400, 1300, 1100, 900],
        }
    )


def test_market_analysis_cli_console_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore(_build_history()),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_derived_store",
        lambda *args, **kwargs: _StubDerivedStore(_build_derived()),
    )
    monkeypatch.setattr(market_analysis_command.cli_deps, "build_provider", lambda: object())

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "tail risk as-of" in res.output
    assert "Horizon End Percentiles" in res.output
    assert "IV context:" in res.output


def test_market_analysis_cli_json_output_validates_schema(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore(_build_history()),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_derived_store",
        lambda *args, **kwargs: _StubDerivedStore(_build_derived()),
    )
    monkeypatch.setattr(market_analysis_command.cli_deps, "build_provider", lambda: object())

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
            "--format",
            "json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = res.output.strip()
    artifact = TailRiskArtifact.model_validate_json(payload)
    assert artifact.symbol == "SPY"
    assert artifact.config.horizon_days == 20

    out_res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
            "--out",
            str(tmp_path),
        ],
    )
    assert out_res.exit_code == 0, out_res.output
    assert (tmp_path / "tail_risk" / "SPY").exists()


def test_market_analysis_iv_surface_json_and_duckdb_persist(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    as_of_prev = date(2026, 2, 6)
    as_of_cur = date(2026, 2, 7)
    snapshot_store = _StubSnapshotStore(
        {
            as_of_prev: _build_snapshot_rows(iv_shift=0.00, oi_scale=1.0),
            as_of_cur: _build_snapshot_rows(iv_shift=0.02, oi_scale=1.1),
        },
        {
            as_of_prev: {"spot": 501.0},
            as_of_cur: {"spot": 503.0},
        },
    )
    metrics_store = _StubResearchMetricsStore()

    monkeypatch.setattr(market_analysis_command.cli_deps, "build_snapshot_store", lambda *_a, **_k: snapshot_store)
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore({"SPY": _build_history()}),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_research_metrics_store",
        lambda *_a, **_k: metrics_store,
    )
    monkeypatch.setattr(
        market_analysis_command,
        "get_storage_runtime_config",
        lambda: SimpleNamespace(backend="duckdb"),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("network should not be used")),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "market-analysis",
            "iv-surface",
            "--symbol",
            "SPY",
            "--as-of",
            "2026-02-07",
            "--format",
            "json",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = IvSurfaceArtifact.model_validate_json(result.output)
    assert artifact.symbol == "SPY"
    assert artifact.as_of == "2026-02-07"
    assert artifact.tenor
    assert metrics_store.iv_tenor_rows > 0
    assert metrics_store.iv_delta_rows > 0
    assert (tmp_path / "iv_surface" / "SPY" / "2026-02-07.json").exists()


def test_market_analysis_exposure_json_is_offline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    as_of = date(2026, 2, 7)
    snapshot_store = _StubSnapshotStore(
        {as_of: _build_snapshot_rows(iv_shift=0.01, oi_scale=1.2)},
        {as_of: {"spot": 502.0}},
    )

    monkeypatch.setattr(market_analysis_command.cli_deps, "build_snapshot_store", lambda *_a, **_k: snapshot_store)
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore({"SPY": _build_history()}),
    )
    monkeypatch.setattr(
        market_analysis_command,
        "get_storage_runtime_config",
        lambda: SimpleNamespace(backend="filesystem"),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("network should not be used")),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "market-analysis",
            "exposure",
            "--symbol",
            "SPY",
            "--as-of",
            "2026-02-07",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = ExposureArtifact.model_validate_json(result.output)
    assert artifact.symbol == "SPY"
    assert [slice_row.mode for slice_row in artifact.slices] == ["near", "monthly", "all"]


def test_market_analysis_levels_json_from_cached_data(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    daily = _build_history().tail(40)
    benchmark = daily.copy()
    benchmark["Close"] = benchmark["Close"] * 0.9
    intraday_bars = _build_intraday_bars()

    class _StubIntradayStore:
        def __init__(self, root_dir: Path) -> None:  # noqa: ARG002
            pass

        def load_partition(
            self,
            kind: str,
            dataset: str,
            timeframe: str,
            symbol: str,
            day: date,
        ) -> pd.DataFrame:  # noqa: ARG002
            return intraday_bars.copy()

    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore({"SPY": daily, "QQQ": benchmark}),
    )
    monkeypatch.setattr(market_analysis_command, "IntradayStore", _StubIntradayStore)
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("network should not be used")),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "market-analysis",
            "levels",
            "--symbol",
            "SPY",
            "--benchmark",
            "QQQ",
            "--as-of",
            "latest",
            "--format",
            "json",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = LevelsArtifact.model_validate_json(result.output)
    assert artifact.symbol == "SPY"
    assert artifact.anchored_vwap
    assert artifact.volume_profile
    assert (tmp_path / "levels" / "SPY" / f"{artifact.as_of}.json").exists()
