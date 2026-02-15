from __future__ import annotations

from rich.console import Console
from rich.table import Table

from options_helper.commands.market_analysis.core_helpers import _fmt_int, _fmt_num, _fmt_pct
from options_helper.schemas.exposure import ExposureArtifact
from options_helper.schemas.iv_surface import IvSurfaceArtifact
from options_helper.schemas.levels import LevelsArtifact
from options_helper.schemas.research_metrics_contracts import DELTA_BUCKET_ORDER
from options_helper.schemas.tail_risk import TailRiskArtifact


def _render_tail_risk_console(console: Console, artifact: TailRiskArtifact) -> None:
    console.print(f"[bold]{artifact.symbol}[/bold] tail risk as-of {artifact.as_of}")
    console.print(
        f"Spot={artifact.spot:.2f} | RV={artifact.realized_vol_annual:.2%} | "
        f"ExpRet={artifact.expected_return_annual:.2%} | "
        f"VaR({artifact.config.var_confidence:.0%})={artifact.var_return:.2%} | "
        f"CVaR({artifact.config.var_confidence:.0%})="
        + ("-" if artifact.cvar_return is None else f"{artifact.cvar_return:.2%}")
    )

    table = Table(title="Horizon End Percentiles")
    table.add_column("Pct", justify="right")
    table.add_column("End Price", justify="right")
    table.add_column("End Return", justify="right")
    for row in artifact.end_percentiles:
        table.add_row(
            f"{row.percentile:.0f}",
            "-" if row.price is None else f"{row.price:.2f}",
            "-" if row.return_pct is None else f"{row.return_pct:.2%}",
        )
    console.print(table)

    if artifact.iv_context is None:
        console.print(
            "IV context unavailable. Run `options-helper derived update --symbol "
            f"{artifact.symbol}` after snapshot ingestion."
        )
    else:
        iv = artifact.iv_context
        console.print(
            "IV context: "
            f"{iv.label} (IV/RV20={iv.iv_rv_20d:.2f}x; low={iv.low:.2f}, high={iv.high:.2f})"
        )
        console.print(iv.reason)
        if iv.atm_iv_near is not None or iv.rv_20d is not None:
            console.print(
                "ATM IV near="
                + ("-" if iv.atm_iv_near is None else f"{iv.atm_iv_near:.2%}")
                + ", RV20="
                + ("-" if iv.rv_20d is None else f"{iv.rv_20d:.2%}")
                + ", IV percentile="
                + ("-" if iv.atm_iv_near_percentile is None else f"{iv.atm_iv_near_percentile:.0f}%")
                + ", IV term slope="
                + ("-" if iv.iv_term_slope is None else f"{iv.iv_term_slope:+.3f}")
            )

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _render_iv_surface_console(console: Console, artifact: IvSurfaceArtifact) -> None:
    spot_text = "-" if artifact.spot is None else f"{artifact.spot:.2f}"
    console.print(f"[bold]{artifact.symbol}[/bold] IV surface as-of {artifact.as_of} (spot={spot_text})")

    tenor_table = Table(title="Tenor Surface")
    tenor_table.add_column("Target DTE", justify="right")
    tenor_table.add_column("Expiry")
    tenor_table.add_column("DTE", justify="right")
    tenor_table.add_column("ATM IV", justify="right")
    tenor_table.add_column("Straddle", justify="right")
    tenor_table.add_column("ExpMove%", justify="right")
    tenor_table.add_column("Skew25 (pp)", justify="right")
    tenor_table.add_column("Contracts", justify="right")
    tenor_table.add_column("Warnings")

    for row in artifact.tenor:
        tenor_table.add_row(
            str(row.tenor_target_dte),
            row.expiry or "-",
            _fmt_int(row.dte),
            _fmt_pct(row.atm_iv),
            _fmt_num(row.straddle_mark),
            _fmt_pct(row.expected_move_pct),
            _fmt_num(row.skew_25d_pp),
            str(row.contracts_used),
            ", ".join(row.warnings) if row.warnings else "-",
        )
    console.print(tenor_table)

    bucket_rank = {name: idx for idx, name in enumerate(DELTA_BUCKET_ORDER)}
    ordered_buckets = sorted(
        artifact.delta_buckets,
        key=lambda row: (
            row.tenor_target_dte,
            row.option_type,
            bucket_rank.get(row.delta_bucket, 999),
        ),
    )

    delta_table = Table(title="Delta Buckets")
    delta_table.add_column("Target DTE", justify="right")
    delta_table.add_column("Type")
    delta_table.add_column("Bucket")
    delta_table.add_column("Avg IV", justify="right")
    delta_table.add_column("Median IV", justify="right")
    delta_table.add_column("N", justify="right")
    delta_table.add_column("Warnings")

    for row in ordered_buckets:
        delta_table.add_row(
            str(row.tenor_target_dte),
            row.option_type,
            row.delta_bucket,
            _fmt_pct(row.avg_iv),
            _fmt_pct(row.median_iv),
            str(row.n_contracts),
            ", ".join(row.warnings) if row.warnings else "-",
        )
    console.print(delta_table)

    if artifact.tenor_changes:
        change_table = Table(title="Tenor Changes")
        change_table.add_column("Target DTE", justify="right")
        change_table.add_column("ATM IV Δpp", justify="right")
        change_table.add_column("Straddle Δ", justify="right")
        change_table.add_column("ExpMove Δpp", justify="right")
        for row in artifact.tenor_changes:
            change_table.add_row(
                str(row.tenor_target_dte),
                _fmt_num(row.atm_iv_change_pp),
                _fmt_num(row.straddle_mark_change),
                _fmt_num(row.expected_move_pct_change_pp),
            )
        console.print(change_table)

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _render_exposure_console(console: Console, artifact: ExposureArtifact) -> None:
    console.print(f"[bold]{artifact.symbol}[/bold] dealer exposure as-of {artifact.as_of} (spot={artifact.spot:.2f})")

    for slice_artifact in artifact.slices:
        summary = slice_artifact.summary
        console.print(
            f"\n[bold]{slice_artifact.mode}[/bold] expiries="
            + (", ".join(slice_artifact.included_expiries) if slice_artifact.included_expiries else "-")
        )
        console.print(
            "flip="
            + _fmt_num(summary.flip_strike)
            + " | total_call_gex="
            + _fmt_num(summary.total_call_gex)
            + " | total_put_gex="
            + _fmt_num(summary.total_put_gex)
            + " | total_net_gex="
            + _fmt_num(summary.total_net_gex)
        )

        top_table = Table(title=f"Top |Net GEX| Levels ({slice_artifact.mode})")
        top_table.add_column("Strike", justify="right")
        top_table.add_column("Net GEX", justify="right")
        top_table.add_column("Abs Net GEX", justify="right")
        for row in slice_artifact.top_abs_net_levels:
            top_table.add_row(_fmt_num(row.strike), _fmt_num(row.net_gex), _fmt_num(row.abs_net_gex))
        console.print(top_table)

        if summary.warnings:
            console.print("Warnings: " + ", ".join(summary.warnings))

    console.print(artifact.disclaimer)


def _render_levels_console(console: Console, artifact: LevelsArtifact) -> None:
    summary = artifact.summary
    console.print(f"[bold]{artifact.symbol}[/bold] levels as-of {artifact.as_of}")

    summary_table = Table(title="Summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Spot", _fmt_num(summary.spot))
    summary_table.add_row("Prev Close", _fmt_num(summary.prev_close))
    summary_table.add_row("Session Open", _fmt_num(summary.session_open))
    summary_table.add_row("Gap %", _fmt_pct(summary.gap_pct))
    summary_table.add_row("Prior High", _fmt_num(summary.prior_high))
    summary_table.add_row("Prior Low", _fmt_num(summary.prior_low))
    summary_table.add_row("Rolling High", _fmt_num(summary.rolling_high))
    summary_table.add_row("Rolling Low", _fmt_num(summary.rolling_low))
    summary_table.add_row("RS Ratio", _fmt_num(summary.rs_ratio))
    summary_table.add_row("Beta (20d)", _fmt_num(summary.beta_20d))
    summary_table.add_row("Corr (20d)", _fmt_num(summary.corr_20d))
    console.print(summary_table)

    anchor_table = Table(title="Anchored VWAP")
    anchor_table.add_column("Anchor")
    anchor_table.add_column("Anchor TS (UTC)")
    anchor_table.add_column("Anchor Px", justify="right")
    anchor_table.add_column("Anchored VWAP", justify="right")
    anchor_table.add_column("Distance %", justify="right")

    for row in artifact.anchored_vwap:
        anchor_ts = row.anchor_ts_utc.isoformat() if row.anchor_ts_utc is not None else "-"
        anchor_table.add_row(
            row.anchor_id,
            anchor_ts,
            _fmt_num(row.anchor_price),
            _fmt_num(row.anchored_vwap),
            _fmt_pct(row.distance_from_spot_pct),
        )
    console.print(anchor_table)

    profile_table = Table(title="Volume Profile")
    profile_table.add_column("Bin Low", justify="right")
    profile_table.add_column("Bin High", justify="right")
    profile_table.add_column("Volume", justify="right")
    profile_table.add_column("Vol %", justify="right")
    profile_table.add_column("POC")
    profile_table.add_column("HVN")
    profile_table.add_column("LVN")

    for row in artifact.volume_profile:
        profile_table.add_row(
            _fmt_num(row.price_bin_low),
            _fmt_num(row.price_bin_high),
            _fmt_num(row.volume),
            _fmt_pct(row.volume_pct),
            "Y" if row.is_poc else "",
            "Y" if row.is_hvn else "",
            "Y" if row.is_lvn else "",
        )
    console.print(profile_table)

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


__all__ = [
    "_render_tail_risk_console",
    "_render_iv_surface_console",
    "_render_exposure_console",
    "_render_levels_console",
]
