from __future__ import annotations

from datetime import date

from options_helper.analysis.events import earnings_event_risk, format_next_earnings_line
from options_helper.models import RiskProfile


def test_risk_profile_defaults_include_earnings_fields() -> None:
    rp = RiskProfile()
    assert rp.earnings_avoid_days == 0
    assert rp.earnings_warn_days == 21


def test_earnings_event_risk_unknown_date_warns_only() -> None:
    today = date(2026, 1, 2)
    expiry = date(2026, 2, 21)
    out = earnings_event_risk(today, expiry, None, warn_days=21, avoid_days=0)
    assert out["warnings"] == ["earnings_unknown"]
    assert out["exclude"] is False


def test_earnings_event_risk_warns_within_window_and_crosses() -> None:
    today = date(2026, 1, 2)
    expiry = date(2026, 2, 21)
    earnings = date(2026, 1, 20)
    out = earnings_event_risk(today, expiry, earnings, warn_days=21, avoid_days=0)
    assert out["warnings"] == ["earnings_within_21d", "expiry_crosses_earnings"]
    assert out["exclude"] is False


def test_earnings_event_risk_does_not_cross_before_expiry() -> None:
    today = date(2026, 1, 2)
    expiry = date(2026, 1, 10)
    earnings = date(2026, 1, 20)
    out = earnings_event_risk(today, expiry, earnings, warn_days=30, avoid_days=0)
    assert out["warnings"] == ["earnings_within_30d"]
    assert out["exclude"] is False


def test_earnings_event_risk_excludes_when_avoid_days_hit() -> None:
    today = date(2026, 1, 2)
    expiry = date(2026, 2, 21)
    earnings = date(2026, 1, 6)
    out = earnings_event_risk(today, expiry, earnings, warn_days=21, avoid_days=7)
    assert "earnings_within_21d" in out["warnings"]
    assert out["exclude"] is True


def test_earnings_event_risk_ignores_past_earnings() -> None:
    today = date(2026, 2, 1)
    expiry = date(2026, 3, 1)
    earnings = date(2026, 1, 15)
    out = earnings_event_risk(today, expiry, earnings, warn_days=21, avoid_days=10)
    assert out["warnings"] == []
    assert out["exclude"] is False


def test_format_next_earnings_line_formats_future_and_unknown() -> None:
    today = date(2026, 1, 2)
    assert format_next_earnings_line(today, None) == "Next earnings: unknown"
    assert format_next_earnings_line(today, date(2026, 1, 5)) == "Next earnings: 2026-01-05 (in 3 day(s))"
