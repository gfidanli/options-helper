from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from options_helper.technicals_backtesting.backtest.batch_runner import run_backtest_batch


class _StepClock:
    def __init__(self, *, step: float = 1.0) -> None:
        self._value = -step
        self._step = step

    def __call__(self) -> float:
        self._value += self._step
        return self._value


def _ohlc_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.2, 10.1, 10.3],
            "High": [10.4, 10.5, 10.4, 10.6],
            "Low": [9.8, 9.9, 9.9, 10.0],
            "Close": [10.1, 10.0, 10.3, 10.5],
            "Volume": [1000, 1100, 1200, 1300],
            "feature_x": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )


def test_run_backtest_batch_single_symbol_emits_structured_outcome_and_progress() -> None:
    equity_curve = pd.DataFrame({"Equity": [100_000.0, 101_250.0]})
    trades = pd.DataFrame({"Size": [1], "EntryBar": [1], "ExitBar": [2]})
    progress_events = []

    def _run_strategy(_symbol: str, _frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "Return [%]": 1.25,
                "# Trades": 1,
                "_equity_curve": equity_curve,
                "_trades": trades,
            }
        )

    result = run_backtest_batch(
        symbols="spy",
        load_ohlc=lambda _symbol: _ohlc_frame(),
        compute_features=lambda _symbol, frame: frame.copy(),
        select_strategy_features=lambda _symbol, frame: frame.loc[:, ["Open", "High", "Low", "Close", "feature_x"]],
        run_strategy_backtest=_run_strategy,
        progress_callback=progress_events.append,
        clock=_StepClock(),
    )

    assert result.symbols == ("SPY",)
    assert result.success_count == 1
    assert result.failure_count == 0
    outcome = result.outcomes[0]
    assert outcome.ok is True
    assert outcome.symbol == "SPY"
    assert outcome.error is None
    assert outcome.warnings == ()
    assert outcome.stats == {"Return [%]": 1.25, "# Trades": 1}
    assert_frame_equal(outcome.equity_curve, equity_curve)
    assert_frame_equal(outcome.trades, trades)
    assert outcome.stage_timings == {
        "load_ohlc": 1.0,
        "compute_features": 1.0,
        "select_strategy_features": 1.0,
        "run_backtest": 1.0,
        "symbol_total": 9.0,
    }
    assert result.stage_timings == {
        "load_ohlc": 1.0,
        "compute_features": 1.0,
        "select_strategy_features": 1.0,
        "run_backtest": 1.0,
        "symbol_total": 9.0,
        "batch_total": 11.0,
    }

    progress_tuples = [(event.symbol, event.stage, event.state) for event in result.progress_events]
    assert progress_tuples == [
        (None, "batch", "started"),
        ("SPY", "symbol", "started"),
        ("SPY", "load_ohlc", "started"),
        ("SPY", "load_ohlc", "completed"),
        ("SPY", "compute_features", "started"),
        ("SPY", "compute_features", "completed"),
        ("SPY", "select_strategy_features", "started"),
        ("SPY", "select_strategy_features", "completed"),
        ("SPY", "run_backtest", "started"),
        ("SPY", "run_backtest", "completed"),
        ("SPY", "symbol", "completed"),
        (None, "batch", "completed"),
    ]
    assert progress_events == list(result.progress_events)


def test_run_backtest_batch_records_warnings_when_private_frames_are_missing() -> None:
    result = run_backtest_batch(
        symbols="spy",
        load_ohlc=lambda _symbol: _ohlc_frame(),
        compute_features=lambda _symbol, frame: frame.copy(),
        select_strategy_features=lambda _symbol, frame: frame,
        run_strategy_backtest=lambda _symbol, _frame: pd.Series({"Return [%]": 0.5}),
        clock=_StepClock(),
    )

    outcome = result.outcomes[0]
    assert outcome.ok is True
    assert outcome.warnings == ("missing_equity_curve", "missing_trades")
    assert outcome.equity_curve is None
    assert outcome.trades is None


def test_run_backtest_batch_continues_after_symbol_failure() -> None:
    loaded_symbols: list[str] = []

    def _load_ohlc(symbol: str) -> pd.DataFrame:
        loaded_symbols.append(symbol)
        if symbol == "QQQ":
            raise RuntimeError("missing OHLC for QQQ")
        return _ohlc_frame()

    def _run_strategy(symbol: str, _frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "Return [%]": 0.7 if symbol == "SPY" else 1.1,
                "_equity_curve": pd.DataFrame({"Equity": [100_000.0, 100_700.0]}),
                "_trades": pd.DataFrame({"Size": [1]}),
            }
        )

    result = run_backtest_batch(
        symbols=["spy", "qqq", "iwm"],
        load_ohlc=_load_ohlc,
        compute_features=lambda _symbol, frame: frame.copy(),
        select_strategy_features=lambda _symbol, frame: frame,
        run_strategy_backtest=_run_strategy,
        clock=_StepClock(),
    )

    assert loaded_symbols == ["SPY", "QQQ", "IWM"]
    assert result.success_count == 2
    assert result.failure_count == 1
    assert [outcome.symbol for outcome in result.outcomes] == ["SPY", "QQQ", "IWM"]
    assert result.outcomes[0].ok is True
    assert result.outcomes[2].ok is True

    failed = result.outcomes[1]
    assert failed.ok is False
    assert failed.error == "RuntimeError: missing OHLC for QQQ"
    assert failed.warnings == ("symbol_failed:RuntimeError",)
    assert failed.stats is None
    assert failed.stage_timings == {"load_ohlc": 1.0, "symbol_total": 3.0}

    symbol_events = [
        (event.symbol, event.state)
        for event in result.progress_events
        if event.stage == "symbol"
    ]
    assert symbol_events == [
        ("SPY", "started"),
        ("SPY", "completed"),
        ("QQQ", "started"),
        ("QQQ", "failed"),
        ("IWM", "started"),
        ("IWM", "completed"),
    ]
    assert any(
        event.symbol == "QQQ" and event.stage == "load_ohlc" and event.state == "failed"
        for event in result.progress_events
    )
    assert result.stage_timings["compute_features"] == 2.0
    assert result.stage_timings["run_backtest"] == 2.0
