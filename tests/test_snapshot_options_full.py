from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_snapshot_options_full_chain_from_all_watchlists(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"monitor":["AAA","BBB"],"positions":["BBB","CCC"]}}',
        encoding="utf-8",
    )

    candle_day = date(2026, 1, 28)

    def _stub_history(self, symbol: str, *, period: str = "10d"):  # noqa: ARG001
        # 2 daily closes ending at `candle_day`
        idx = pd.to_datetime([candle_day.replace(day=candle_day.day - 1), candle_day])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)

    expiries = [date(2026, 2, 20), date(2026, 3, 20)]

    class _StubProvider:
        name = "stub"
        version = "0.0"

        def get_underlying(self, symbol: str, *, period: str = "10d", interval: str = "1d"):  # noqa: ARG002
            raise AssertionError("Candle history should provide spot in this test")

        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return expiries

        def get_options_chain_raw(self, symbol: str, expiry: date):  # noqa: ARG002
            return {
                "expirationDate": int(pd.Timestamp(expiry).timestamp()),
                "underlying": {"regularMarketPrice": 101.0},
                "calls": [
                    {
                        "contractSymbol": "AAA_CALL",
                        "strike": 100.0,
                        "lastPrice": 2.0,
                        "impliedVolatility": 0.25,
                        "openInterest": 123,
                        "volume": 10,
                        # Extra Yahoo fields (dropped by yfinance's fixed-column DF).
                        "bidSize": 42,
                    }
                ],
                "puts": [
                    {
                        "contractSymbol": "AAA_PUT",
                        "strike": 100.0,
                        "lastPrice": 1.5,
                        "impliedVolatility": 0.25,
                        "openInterest": 321,
                        "volume": 5,
                        "bidSize": 7,
                    }
                ],
            }

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: _StubProvider())

    cache_dir = tmp_path / "snapshots"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "snapshot-options",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--all-watchlists",
            "--full-chain",
            "--cache-dir",
            str(cache_dir),
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )
    assert res.exit_code == 0, res.output

    # Union of watchlists: AAA, BBB, CCC
    for sym in ["AAA", "BBB", "CCC"]:
        day_dir = cache_dir / sym / candle_day.isoformat()
        assert day_dir.exists(), f"missing snapshot dir for {sym}"
        meta = json.loads((day_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["full_chain"] is True
        assert meta["symbol_source"] == "watchlists"
        assert "underlying" in meta
        assert meta["risk_free_rate"] == 0.0
        assert "quote_quality" in meta
        quality = meta["quote_quality"]
        assert quality["contracts"] > 0
        assert "missing_bid_ask_pct" in quality
        assert "spread_pct_median" in quality
        assert "spread_pct_worst" in quality
        assert "stale_quotes" in quality

        for exp in expiries:
            csv_path = day_dir / f"{exp.isoformat()}.csv"
            raw_path = day_dir / f"{exp.isoformat()}.raw.json"
            assert csv_path.exists()
            assert raw_path.exists()

            # Ensure we kept a column yfinance would normally drop.
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            assert "bidSize" in header
            assert "bs_delta" in header

            df = pd.read_csv(csv_path)
            assert df["bs_delta"].notna().any()
            assert df["bs_theta_per_day"].notna().any()


def test_snapshot_options_defaults_to_full_chain_and_all_expiries(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"monitor":["AAA"],"positions":["AAA"]}}',
        encoding="utf-8",
    )

    candle_day = date(2026, 1, 28)

    def _stub_history(self, symbol: str, *, period: str = "10d"):  # noqa: ARG001
        idx = pd.to_datetime([candle_day.replace(day=candle_day.day - 1), candle_day])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)

    expiries = [date(2026, 2, 20), date(2026, 3, 20), date(2026, 4, 17)]

    class _StubProvider:
        name = "stub"
        version = "0.0"

        def get_underlying(self, symbol: str, *, period: str = "10d", interval: str = "1d"):  # noqa: ARG002
            raise AssertionError("Candle history should provide spot in this test")

        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return expiries

        def get_options_chain_raw(self, symbol: str, expiry: date):  # noqa: ARG002
            return {
                "expirationDate": int(pd.Timestamp(expiry).timestamp()),
                "underlying": {"regularMarketPrice": 101.0},
                "calls": [
                    {
                        "contractSymbol": f"AAA_CALL_{expiry.isoformat()}",
                        "strike": 100.0,
                        "lastPrice": 2.0,
                        "impliedVolatility": 0.25,
                        "openInterest": 123,
                        "volume": 10,
                        "bidSize": 42,
                    }
                ],
                "puts": [
                    {
                        "contractSymbol": f"AAA_PUT_{expiry.isoformat()}",
                        "strike": 100.0,
                        "lastPrice": 1.5,
                        "impliedVolatility": 0.25,
                        "openInterest": 321,
                        "volume": 5,
                        "bidSize": 7,
                    }
                ],
            }

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: _StubProvider())

    cache_dir = tmp_path / "snapshots"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "snapshot-options",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--all-watchlists",
            "--cache-dir",
            str(cache_dir),
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )
    assert res.exit_code == 0, res.output

    day_dir = cache_dir / "AAA" / candle_day.isoformat()
    meta = json.loads((day_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["full_chain"] is True
    assert meta["all_expiries"] is True

    for exp in expiries:
        csv_path = day_dir / f"{exp.isoformat()}.csv"
        raw_path = day_dir / f"{exp.isoformat()}.raw.json"
        assert csv_path.exists()
        assert raw_path.exists()


def test_snapshot_options_position_expiries_caps_watchlists_by_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"monitor":["AAA"],"positions":["AAA"]}}',
        encoding="utf-8",
    )

    candle_day = date(2026, 1, 28)

    def _stub_history(self, symbol: str, *, period: str = "10d"):  # noqa: ARG001
        idx = pd.to_datetime([candle_day.replace(day=candle_day.day - 1), candle_day])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)

    expiries = [date(2026, 2, 20), date(2026, 3, 20), date(2026, 4, 17)]

    class _StubProvider:
        name = "stub"
        version = "0.0"

        def get_underlying(self, symbol: str, *, period: str = "10d", interval: str = "1d"):  # noqa: ARG002
            raise AssertionError("Candle history should provide spot in this test")

        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return expiries

        def get_options_chain_raw(self, symbol: str, expiry: date):  # noqa: ARG002
            return {
                "expirationDate": int(pd.Timestamp(expiry).timestamp()),
                "underlying": {"regularMarketPrice": 101.0},
                "calls": [
                    {
                        "contractSymbol": f"AAA_CALL_{expiry.isoformat()}",
                        "strike": 100.0,
                        "lastPrice": 2.0,
                        "impliedVolatility": 0.25,
                        "openInterest": 123,
                        "volume": 10,
                        "bidSize": 42,
                    }
                ],
                "puts": [
                    {
                        "contractSymbol": f"AAA_PUT_{expiry.isoformat()}",
                        "strike": 100.0,
                        "lastPrice": 1.5,
                        "impliedVolatility": 0.25,
                        "openInterest": 321,
                        "volume": 5,
                        "bidSize": 7,
                    }
                ],
            }

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: _StubProvider())

    cache_dir = tmp_path / "snapshots"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "snapshot-options",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--all-watchlists",
            "--position-expiries",
            "--cache-dir",
            str(cache_dir),
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )
    assert res.exit_code == 0, res.output

    day_dir = cache_dir / "AAA" / candle_day.isoformat()
    meta = json.loads((day_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["full_chain"] is True
    assert meta["all_expiries"] is False

    # Watchlists default cap is 2 expiries unless --max-expiries is set.
    for exp in expiries[:2]:
        assert (day_dir / f"{exp.isoformat()}.csv").exists()
        assert (day_dir / f"{exp.isoformat()}.raw.json").exists()
    assert not (day_dir / f"{expiries[2].isoformat()}.csv").exists()
    assert not (day_dir / f"{expiries[2].isoformat()}.raw.json").exists()
