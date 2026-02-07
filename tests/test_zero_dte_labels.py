from __future__ import annotations

import pandas as pd

from options_helper.analysis.zero_dte_labels import ZeroDTELabelConfig, build_zero_dte_labels


def _bars(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [row[0] for row in rows],
            "open": [row[1] for row in rows],
            "close": [row[2] for row in rows],
        }
    )


def test_build_zero_dte_labels_uses_next_bar_anchor_and_fails_closed_when_missing() -> None:
    bars = _bars(
        [
            ("2026-02-06T20:58:00Z", 10.0, 10.1),
            ("2026-02-06T20:59:00Z", 10.2, 10.4),
        ]
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06"],
            "decision_ts": ["2026-02-06T20:58:00Z", "2026-02-06T20:59:00Z"],
            "bar_ts": ["2026-02-06T20:58:00Z", "2026-02-06T20:59:00Z"],
            "status": ["ok", "ok"],
        }
    )

    out = build_zero_dte_labels(
        state_rows,
        bars,
        market_close_ts="2026-02-06T21:00:00Z",
        config=ZeroDTELabelConfig(max_close_lag_seconds=60),
    )

    first = out.iloc[0]
    second = out.iloc[1]

    assert str(first["entry_anchor_ts"]) == "2026-02-06 20:59:00+00:00"
    assert first["entry_anchor_price"] == 10.2
    assert first["label_status"] == "ok"
    assert first["skip_reason"] is None

    # Late-session decision has no next tradable bar, so it must fail closed.
    assert pd.isna(second["entry_anchor_ts"])
    assert second["label_status"] == "no_entry_anchor"
    assert second["skip_reason"] == "no_entry_anchor"


def test_build_zero_dte_labels_handles_early_close_sessions() -> None:
    bars = _bars(
        [
            ("2026-11-27T17:58:00Z", 5.0, 5.1),
            ("2026-11-27T17:59:00Z", 5.1, 5.2),
        ]
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-11-27"],
            "decision_ts": ["2026-11-27T17:58:00Z"],
            "bar_ts": ["2026-11-27T17:58:00Z"],
            "status": ["ok"],
        }
    )

    out = build_zero_dte_labels(
        state_rows,
        bars,
        market_close_ts="2026-11-27T18:00:00Z",
        config=ZeroDTELabelConfig(max_close_lag_seconds=60),
    )

    row = out.iloc[0]
    assert row["label_status"] == "ok"
    assert row["skip_reason"] is None
    assert str(row["close_label_ts"]) == "2026-11-27 17:59:00+00:00"


def test_build_zero_dte_labels_marks_missing_close_bar_as_insufficient_data() -> None:
    bars = _bars(
        [
            ("2026-02-06T15:00:00Z", 7.0, 7.1),
            ("2026-02-06T15:01:00Z", 7.1, 7.2),
            ("2026-02-06T20:40:00Z", 8.0, 8.1),
        ]
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-02-06"],
            "decision_ts": ["2026-02-06T15:00:00Z"],
            "bar_ts": ["2026-02-06T15:00:00Z"],
            "status": ["ok"],
        }
    )

    out = build_zero_dte_labels(
        state_rows,
        bars,
        market_close_ts="2026-02-06T21:00:00Z",
        config=ZeroDTELabelConfig(max_close_lag_seconds=60),
    )

    row = out.iloc[0]
    assert row["entry_anchor_price"] == 7.1
    assert row["label_status"] == "missing_close_bar"
    assert row["skip_reason"] == "insufficient_data"
    assert pd.isna(row["close_label_ts"])


def test_build_zero_dte_labels_respects_state_status_gate() -> None:
    bars = _bars(
        [
            ("2026-02-06T15:00:00Z", 10.0, 10.0),
            ("2026-02-06T15:01:00Z", 10.1, 10.2),
        ]
    )
    state_rows = pd.DataFrame(
        {
            "session_date": ["2026-02-06"],
            "decision_ts": ["2026-02-06T15:00:00Z"],
            "bar_ts": ["2026-02-06T15:00:00Z"],
            "status": ["outside_session"],
        }
    )

    out = build_zero_dte_labels(
        state_rows,
        bars,
        market_close_ts="2026-02-06T21:00:00Z",
    )

    row = out.iloc[0]
    assert row["label_status"] == "state_outside_session"
    assert row["skip_reason"] == "outside_decision_window"
