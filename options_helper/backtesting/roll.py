from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from options_helper.analysis.roll_plan import RollCandidate, RollIntent, RollPlanReport, RollShape, compute_roll_plan
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.models import Position


@dataclass(frozen=True)
class RollPolicy:
    dte_threshold: int = 14
    intent: RollIntent = "max-upside"
    shape: RollShape = "out-same-strike"
    horizon_months: int = 2
    min_open_interest: int = 0
    min_volume: int = 0
    max_spread_pct: float = 0.35
    include_bad_quotes: bool = False


@dataclass(frozen=True)
class RollDecision:
    triggered: bool
    reason: str | None
    candidate: RollCandidate | None = None
    report: RollPlanReport | None = None
    warnings: list[str] = field(default_factory=list)


def should_roll(
    *,
    as_of: date,
    expiry: date,
    dte_threshold: int,
    thesis_ok: bool,
) -> bool:
    dte = (expiry - as_of).days
    if dte < 0:
        return False
    return dte <= int(dte_threshold) and thesis_ok


def select_roll_candidate(
    snapshot_store: OptionsSnapshotStore,
    *,
    symbol: str,
    as_of: date,
    spot: float,
    position: Position,
    policy: RollPolicy,
) -> RollDecision:
    day_df = snapshot_store.load_day(symbol, as_of)
    if day_df is None or day_df.empty:
        return RollDecision(triggered=True, reason="empty_snapshot", warnings=["empty_snapshot"])

    try:
        report = compute_roll_plan(
            day_df,
            symbol=symbol,
            as_of=as_of,
            spot=spot,
            position=position,
            intent=policy.intent,
            horizon_months=policy.horizon_months,
            shape=policy.shape,
            min_open_interest=policy.min_open_interest,
            min_volume=policy.min_volume,
            top=10,
            max_spread_pct=policy.max_spread_pct,
            include_bad_quotes=policy.include_bad_quotes,
        )
    except Exception as exc:  # noqa: BLE001
        return RollDecision(triggered=True, reason="roll_plan_error", warnings=[str(exc)])

    if not report.candidates:
        return RollDecision(triggered=True, reason="no_candidates", report=report)
    return RollDecision(
        triggered=True,
        reason=None,
        candidate=report.candidates[0],
        report=report,
        warnings=list(report.warnings),
    )

