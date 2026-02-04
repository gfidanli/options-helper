from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import json
import pandas as pd
from typer.testing import CliRunner

from options_helper.analysis.research import (
    Direction,
    UnderlyingSetup,
    choose_expiry,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.cli import app
from options_helper.data.market_types import OptionsChain
from options_helper.models import RiskProfile


def _stub_research_client(  # type: ignore[no-untyped-def]
    monkeypatch,
    *,
    expiry_strs: list[str],
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    class _StubProvider:
        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return [date.fromisoformat(s) for s in expiry_strs]

        def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:  # noqa: ARG002
            return OptionsChain(symbol="TEST", expiry=expiry, calls=calls, puts=puts)

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: _StubProvider())

    def _stub_history(self, symbol: str, *, period: str = "5y"):  # noqa: ARG001
        return history

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)


def test_choose_expiry_picks_closest_to_target() -> None:
    today = date(2026, 1, 1)
    expiries = ["2026-02-15", "2026-03-15", "2026-04-15"]
    # DTE: 45, 73, 104
    exp = choose_expiry(expiries, min_dte=30, max_dte=90, target_dte=60, today=today)
    assert exp == date(2026, 3, 15)


def test_select_option_candidate_prefers_target_delta() -> None:
    expiry = date.today() + timedelta(days=60)
    spot = 100.0

    # Constant IV, monotonic delta by strike.
    df = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "bid": [12.0, 6.0, 2.5],
            "ask": [13.0, 7.0, 3.0],
            "lastPrice": [12.5, 6.5, 2.75],
            "impliedVolatility": [0.25, 0.25, 0.25],
            "openInterest": [500, 500, 500],
            "volume": [50, 50, 50],
        }
    )

    # LEAPS-style (higher delta) should pick ITM 90 strike.
    itm = select_option_candidate(
        df,
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=0.70,
        window_pct=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert itm is not None
    assert itm.strike == 90.0

    # Short-dated momentum (lower delta) should lean toward higher strike.
    otm = select_option_candidate(
        df,
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=0.35,
        window_pct=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert otm is not None
    assert otm.strike in {100.0, 110.0}


def test_select_option_candidate_excludes_bad_spread() -> None:
    expiry = date.today() + timedelta(days=60)
    spot = 100.0

    df = pd.DataFrame(
        {
            "strike": [100.0, 105.0],
            "bid": [1.0, 4.5],
            "ask": [2.0, 4.7],
            "lastPrice": [1.5, 4.6],
            "impliedVolatility": [None, None],
            "openInterest": [500, 500],
            "volume": [50, 50],
        }
    )

    pick = select_option_candidate(
        df,
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=0.40,
        window_pct=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert pick is not None
    assert pick.strike == 105.0
    assert pick.execution_quality != "bad"
    assert pick.quality_label != "bad"


def test_suggest_trade_levels_breakout_uses_breakout_level() -> None:
    # 60 weeks of steadily rising closes -> breakout on the latest week.
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") * 0.1 + 100.0
    history = pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.0,
            "Low": close - 1.0,
        }
    )

    setup = UnderlyingSetup(
        symbol="TEST",
        spot=float(close.iloc[-1]),
        direction=Direction.BULLISH,
        reasons=[],
        daily_rsi=None,
        daily_stoch_rsi=None,
        weekly_rsi=None,
        weekly_breakout=True,
    )
    rp = RiskProfile(tolerance="high", support_proximity_pct=0.03, breakout_lookback_weeks=20)
    levels = suggest_trade_levels(setup, history=history, risk_profile=rp)

    close_w = close.resample("W-FRI").last().dropna()
    expected_breakout_level = float(close_w.iloc[-21:-1].max())
    assert levels.entry == setup.spot
    assert levels.pullback_entry == expected_breakout_level
    assert levels.stop is not None
    assert abs(levels.stop - expected_breakout_level * (1.0 - rp.support_proximity_pct)) < 1e-6


def test_research_cli_saves_report_and_includes_spreads(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    today = date.today()
    candle_day = today - timedelta(days=1)
    run_dt_1 = __import__("datetime").datetime(today.year, today.month, today.day, 11, 59, 3)
    run_dt_2 = __import__("datetime").datetime(today.year, today.month, today.day, 12, 0, 0)
    short_exp = today + timedelta(days=60)
    long_exp = today + timedelta(days=540)
    expiry_strs = [short_exp.isoformat(), long_exp.isoformat()]

    # 60+ weeks so weekly EMA50 is computable (required by analyze_underlying).
    idx = pd.date_range(end=pd.Timestamp(candle_day), periods=300, freq="B")
    # If `candle_day` lands on a weekend/holiday, the last business-day candle will be earlier.
    last_candle_day = idx.max().date()
    candle_dt_stamp = f"{last_candle_day.isoformat()}_000000"
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") * 0.1 + 100.0
    history = pd.DataFrame({"Close": close, "High": close + 1.0, "Low": close - 1.0})

    calls = pd.DataFrame(
        {
            "strike": [120.0, 130.0, 140.0],
            "bid": [8.0, 5.0, 3.0],
            "ask": [9.0, 6.0, 3.5],
            "lastPrice": [8.5, 5.5, 3.25],
            "impliedVolatility": [0.25, 0.25, 0.25],
            "openInterest": [500, 500, 500],
            "volume": [50, 50, 50],
        }
    )
    puts = calls.copy()

    class _StubProvider:
        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return [date.fromisoformat(s) for s in expiry_strs]

        def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:  # noqa: ARG002
            return OptionsChain(symbol="TEST", expiry=expiry, calls=calls, puts=puts)

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: _StubProvider())

    def _stub_history(self, symbol: str, *, period: str = "5y"):  # noqa: ARG001
        return history

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_history)

    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    import datetime as _dt

    class _FakeDateTime1(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return run_dt_1

    class _FakeDateTime2(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return run_dt_2

    runner = CliRunner()
    monkeypatch.setattr("options_helper.cli.datetime", _FakeDateTime1)
    res = runner.invoke(
        app,
        [
            "research",
            str(portfolio_path),
            "--symbol",
            "TEST",
            "--output-dir",
            str(tmp_path),
            "--derived-dir",
            str(tmp_path / "derived"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Saved research report to" in res.output

    expected_run_path_1 = tmp_path / f"research-{candle_dt_stamp}-{run_dt_1.strftime('%Y-%m-%d_%H%M%S')}.txt"
    assert expected_run_path_1.exists()
    assert f"Saved research report to {expected_run_path_1}" in res.output

    txt = expected_run_path_1.read_text(encoding="utf-8")
    assert f"candles_through: {last_candle_day.isoformat()} 00:00:00" in txt
    assert "Suggested entry (underlying)" in txt
    assert "Spr%" in txt
    assert "Exec" in txt
    assert "Quality" in txt
    assert "Stale" in txt
    assert "IV/RV20" in txt
    assert "IV pct" in txt
    assert "(+0.00%)" in txt
    assert "Confluence score" in txt
    assert "Components" in txt

    ticker_path = tmp_path / "tickers" / "TEST.txt"
    assert ticker_path.exists()
    ticker_txt = ticker_path.read_text(encoding="utf-8")
    assert f"=== {last_candle_day.isoformat()} ===" in ticker_txt
    assert f"run_at: {run_dt_1.strftime('%Y-%m-%d %H:%M:%S')}" in ticker_txt

    # Re-run later in the same day: overwrite that day's ticker entry.
    monkeypatch.setattr("options_helper.cli.datetime", _FakeDateTime2)
    res2 = runner.invoke(
        app,
        [
            "research",
            str(portfolio_path),
            "--symbol",
            "TEST",
            "--output-dir",
            str(tmp_path),
            "--derived-dir",
            str(tmp_path / "derived"),
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert (tmp_path / f"research-{candle_dt_stamp}-{run_dt_2.strftime('%Y-%m-%d_%H%M%S')}.txt").exists()

    ticker_txt2 = ticker_path.read_text(encoding="utf-8")
    assert ticker_txt2.count(f"=== {last_candle_day.isoformat()} ===") == 1
    assert f"run_at: {run_dt_2.strftime('%Y-%m-%d %H:%M:%S')}" in ticker_txt2
    assert f"run_at: {run_dt_1.strftime('%Y-%m-%d %H:%M:%S')}" not in ticker_txt2

    assert "Entry" in ticker_txt2


def test_research_cli_includes_earnings_warnings(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    today = date.today()
    candle_day = today - timedelta(days=1)
    short_exp = today + timedelta(days=60)
    long_exp = today + timedelta(days=540)
    expiry_strs = [short_exp.isoformat(), long_exp.isoformat()]

    idx = pd.date_range(end=pd.Timestamp(candle_day), periods=300, freq="B")
    last_candle_day = idx.max().date()
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") * 0.1 + 100.0
    history = pd.DataFrame({"Close": close, "High": close + 1.0, "Low": close - 1.0})

    calls = pd.DataFrame(
        {
            "strike": [120.0, 130.0, 140.0],
            "bid": [8.0, 5.0, 3.0],
            "ask": [9.0, 6.0, 3.5],
            "lastPrice": [8.5, 5.5, 3.25],
            "impliedVolatility": [0.25, 0.25, 0.25],
            "openInterest": [500, 500, 500],
            "volume": [50, 50, 50],
        }
    )
    puts = calls.copy()

    _stub_research_client(
        monkeypatch,
        expiry_strs=expiry_strs,
        calls=calls,
        puts=puts,
        history=history,
    )

    earnings_date = last_candle_day + timedelta(days=5)

    def _stub_next_earnings_date(store, symbol):  # type: ignore[no-untyped-def]
        return earnings_date if symbol.upper() == "TEST" else None

    monkeypatch.setattr("options_helper.cli.safe_next_earnings_date", _stub_next_earnings_date)

    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                    "earnings_warn_days": 21,
                    "earnings_avoid_days": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "research",
            str(portfolio_path),
            "--symbol",
            "TEST",
            "--no-save",
            "--derived-dir",
            str(tmp_path / "derived"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "earnings_within_21d" in res.output
    assert "expiry_crosses_earnings" in res.output


def test_research_cli_excludes_candidates_when_avoid_days_set(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    today = date.today()
    candle_day = today - timedelta(days=1)
    short_exp = today + timedelta(days=60)
    long_exp = today + timedelta(days=540)
    expiry_strs = [short_exp.isoformat(), long_exp.isoformat()]

    idx = pd.date_range(end=pd.Timestamp(candle_day), periods=300, freq="B")
    last_candle_day = idx.max().date()
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") * 0.1 + 100.0
    history = pd.DataFrame({"Close": close, "High": close + 1.0, "Low": close - 1.0})

    calls = pd.DataFrame(
        {
            "strike": [120.0, 130.0, 140.0],
            "bid": [8.0, 5.0, 3.0],
            "ask": [9.0, 6.0, 3.5],
            "lastPrice": [8.5, 5.5, 3.25],
            "impliedVolatility": [0.25, 0.25, 0.25],
            "openInterest": [500, 500, 500],
            "volume": [50, 50, 50],
        }
    )
    puts = calls.copy()

    _stub_research_client(
        monkeypatch,
        expiry_strs=expiry_strs,
        calls=calls,
        puts=puts,
        history=history,
    )

    earnings_date = last_candle_day + timedelta(days=5)

    def _stub_next_earnings_date(store, symbol):  # type: ignore[no-untyped-def]
        return earnings_date if symbol.upper() == "TEST" else None

    monkeypatch.setattr("options_helper.cli.safe_next_earnings_date", _stub_next_earnings_date)

    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                    "earnings_warn_days": 10,
                    "earnings_avoid_days": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "research",
            str(portfolio_path),
            "--symbol",
            "TEST",
            "--no-save",
            "--derived-dir",
            str(tmp_path / "derived"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Excluded 30–90d candidate due to earnings_avoid_days" in res.output
    assert "Excluded LEAPS candidate due to earnings_avoid_days" in res.output
    assert "earnings_within_10d" in res.output
