from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from options_helper.schemas.tail_risk import TailRiskArtifact


@dataclass(frozen=True)
class TailRiskArtifactPaths:
    run_dir: Path
    json_path: Path
    report_path: Path


def build_tail_risk_artifact_paths(
    out_dir: Path,
    *,
    symbol: str,
    as_of: str,
    horizon_days: int,
    num_simulations: int,
    seed: int,
) -> TailRiskArtifactPaths:
    run_dir = out_dir / "tail_risk" / symbol.upper()
    stem = f"tail_risk_{as_of}_h{horizon_days}_n{num_simulations}_seed{seed}"
    return TailRiskArtifactPaths(
        run_dir=run_dir,
        json_path=run_dir / f"{stem}.json",
        report_path=run_dir / f"{stem}.md",
    )


def write_tail_risk_artifacts(artifact: TailRiskArtifact, *, out_dir: Path) -> TailRiskArtifactPaths:
    paths = build_tail_risk_artifact_paths(
        out_dir,
        symbol=artifact.symbol,
        as_of=artifact.as_of,
        horizon_days=artifact.config.horizon_days,
        num_simulations=artifact.config.num_simulations,
        seed=artifact.config.seed,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    payload = artifact.to_dict()
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.report_path.write_text(_render_report_markdown(artifact), encoding="utf-8")
    return paths


def _render_report_markdown(artifact: TailRiskArtifact) -> str:
    lines: list[str] = [
        f"# {artifact.symbol} Tail Risk ({artifact.as_of})",
        "",
        f"- Spot: `{artifact.spot:.2f}`",
        f"- Realized Vol (annualized): `{artifact.realized_vol_annual:.2%}`",
        f"- Expected Return (annualized): `{artifact.expected_return_annual:.2%}`",
        f"- VaR ({artifact.config.var_confidence:.0%}): `{artifact.var_return:.2%}`",
        "- CVaR ({:.0%}): `{}`".format(
            artifact.config.var_confidence,
            "-" if artifact.cvar_return is None else f"{artifact.cvar_return:.2%}",
        ),
        "",
        "## Horizon End Percentiles",
        "",
        "| Percentile | End Price | End Return |",
        "|---:|---:|---:|",
    ]
    for row in artifact.end_percentiles:
        price = "-" if row.price is None else f"{row.price:.2f}"
        ret = "-" if row.return_pct is None else f"{row.return_pct:.2%}"
        lines.append(f"| {row.percentile:.0f} | {price} | {ret} |")

    if artifact.iv_context is not None:
        lines.extend(
            [
                "",
                "## IV Context",
                "",
                f"- Regime: `{artifact.iv_context.label}`",
                f"- Reason: {artifact.iv_context.reason}",
                f"- IV/RV20: {'-' if artifact.iv_context.iv_rv_20d is None else f'{artifact.iv_context.iv_rv_20d:.2f}x'}",
                f"- ATM IV near: {'-' if artifact.iv_context.atm_iv_near is None else f'{artifact.iv_context.atm_iv_near:.2%}'}",
                f"- RV20: {'-' if artifact.iv_context.rv_20d is None else f'{artifact.iv_context.rv_20d:.2%}'}",
                (
                    "- IV percentile: "
                    + (
                        "-"
                        if artifact.iv_context.atm_iv_near_percentile is None
                        else f"{artifact.iv_context.atm_iv_near_percentile:.0f}%"
                    )
                ),
                (
                    "- IV term slope: "
                    + (
                        "-"
                        if artifact.iv_context.iv_term_slope is None
                        else f"{artifact.iv_context.iv_term_slope:+.3f}"
                    )
                ),
            ]
        )

    if artifact.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in artifact.warnings:
            lines.append(f"- {warning}")

    lines.extend(["", artifact.disclaimer, ""])
    return "\n".join(lines)

