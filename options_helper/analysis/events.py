from __future__ import annotations

from datetime import date


def earnings_event_risk(
    today: date,
    expiry: date | None,
    next_earnings_date: date | None,
    *,
    warn_days: int,
    avoid_days: int,
) -> dict[str, object]:
    warnings: list[str] = []
    exclude = False

    if next_earnings_date is None:
        warnings.append("earnings_unknown")
        return {"warnings": warnings, "exclude": exclude}

    delta_days = (next_earnings_date - today).days

    if 0 <= delta_days <= warn_days:
        warnings.append(f"earnings_within_{int(warn_days)}d")

    if expiry is not None and next_earnings_date >= today and expiry >= next_earnings_date:
        warnings.append("expiry_crosses_earnings")

    if avoid_days > 0 and 0 <= delta_days <= avoid_days:
        exclude = True

    return {"warnings": warnings, "exclude": exclude}


def format_next_earnings_line(today: date, next_earnings_date: date | None) -> str:
    if next_earnings_date is None:
        return "Next earnings: unknown"

    delta_days = (next_earnings_date - today).days
    if delta_days == 0:
        suffix = "today"
    elif delta_days > 0:
        suffix = f"in {delta_days} day(s)"
    else:
        suffix = f"{abs(delta_days)} day(s) ago"

    return f"Next earnings: {next_earnings_date.isoformat()} ({suffix})"
