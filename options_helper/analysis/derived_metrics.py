from __future__ import annotations

from pydantic import BaseModel, Field

from options_helper.analysis.chain_metrics import ChainReport


class DerivedRow(BaseModel):
    """
    A compact per-day row of derived metrics for a symbol.

    Notes:
    - This is intended to be persisted to a per-symbol CSV file (see `DerivedStore`).
    - Fields are best-effort: missing snapshot data should surface as nulls.
    """

    date: str
    spot: float
    pc_oi: float | None = None
    pc_vol: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    gamma_peak_strike: float | None = None
    atm_iv_near: float | None = None
    em_near_pct: float | None = None
    skew_near_pp: float | None = None

    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_chain_report(cls, report: ChainReport) -> "DerivedRow":
        warnings: list[str] = []

        call_wall = report.walls_overall.calls[0].strike if report.walls_overall.calls else None
        put_wall = report.walls_overall.puts[0].strike if report.walls_overall.puts else None

        near = report.expiries[0] if report.expiries else None
        if near is None:
            warnings.append("missing_near_expiry")

        return cls(
            date=report.as_of,
            spot=report.spot,
            pc_oi=report.totals.pc_oi_ratio,
            pc_vol=report.totals.pc_volume_ratio,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_peak_strike=report.gamma.peak_strike,
            atm_iv_near=None if near is None else near.atm_iv,
            em_near_pct=None if near is None else near.expected_move_pct,
            skew_near_pp=None if near is None else near.skew_25d_pp,
            warnings=warnings + list(report.warnings),
        )

