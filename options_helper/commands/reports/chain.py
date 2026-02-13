from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.roll_plan_multileg import compute_roll_plan_multileg
from options_helper.commands.common import _parse_date, _spot_from_meta
from options_helper.models import MultiLegPosition
from options_helper.reporting_chain import (
    render_chain_report_console,
    render_chain_report_markdown,
    render_compare_report_console,
)
from options_helper.reporting_roll import render_roll_plan_console, render_roll_plan_multileg_console
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.common import utc_now
from options_helper.schemas.compare import CompareArtifact
from options_helper.storage import load_portfolio


def chain_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to report on."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        help="Output format: console|md|json",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/chains/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    include_expiry: list[str] = typer.Option(
        [],
        "--include-expiry",
        help="Include a specific expiry date (repeatable). When provided, overrides --expiries selection.",
    ),
    expiries: str = typer.Option(
        "near",
        "--expiries",
        help="Expiry selection mode: near|monthly|all (ignored when --include-expiry is used).",
    ),
    best_effort: bool = typer.Option(
        False,
        "--best-effort",
        help="Don't fail hard on missing fields; emit warnings and partial outputs.",
    ),
) -> None:
    """Offline options chain dashboard from local snapshot files."""
    console = Console()
    store = cli_deps.build_snapshot_store(cache_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        fmt = format.strip().lower()
        if fmt not in {"console", "md", "json"}:
            raise typer.BadParameter("Invalid --format (use console|md|json)", param_hint="--format")

        expiries_mode = expiries.strip().lower()
        if expiries_mode not in {"near", "monthly", "all"}:
            raise typer.BadParameter("Invalid --expiries (use near|monthly|all)", param_hint="--expiries")

        include_dates = [_parse_date(x) for x in include_expiry] if include_expiry else None

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode=expiries_mode,  # type: ignore[arg-type]
            include_expiries=include_dates,
            top=top,
            best_effort=best_effort,
        )
        report_artifact = ChainReportArtifact(
            generated_at=utc_now(),
            **report.model_dump(),
        )
        if strict:
            ChainReportArtifact.model_validate(report_artifact.to_dict())

        if fmt == "console":
            render_chain_report_console(console, report)
        elif fmt == "md":
            console.print(render_chain_report_markdown(report))
        else:
            console.print(report_artifact.model_dump_json(indent=2))

        if out is not None:
            base = out / "chains" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            json_path = base / f"{as_of_date.isoformat()}.json"
            json_path.write_text(report_artifact.model_dump_json(indent=2), encoding="utf-8")
            md_path = base / f"{as_of_date.isoformat()}.md"
            md_path.write_text(render_chain_report_markdown(report), encoding="utf-8")
            console.print(f"\nSaved: {json_path}")
            console.print(f"Saved: {md_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def compare_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to compare."),
    from_spec: str = typer.Option(
        "-1",
        "--from",
        help="From snapshot date (YYYY-MM-DD) or a negative offset relative to --to (e.g. -1).",
    ),
    to_spec: str = typer.Option("latest", "--to", help="To snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/compare/{SYMBOL}/).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Compare two snapshot days for a symbol (delta in OI/IV/Greeks)."""
    console = Console()
    store = cli_deps.build_snapshot_store(cache_dir)

    try:
        to_dt = store.resolve_date(symbol, to_spec)
        from_spec_norm = from_spec.strip().lower()
        if from_spec_norm.startswith("-") and from_spec_norm[1:].isdigit():
            from_dt = store.resolve_relative_date(symbol, to_date=to_dt, offset=int(from_spec_norm))
        else:
            from_dt = store.resolve_date(symbol, from_spec_norm)

        if from_dt == to_dt:
            raise typer.BadParameter("--from and --to must be different dates.")

        df_from = store.load_day(symbol, from_dt)
        df_to = store.load_day(symbol, to_dt)
        meta_from = store.load_meta(symbol, from_dt)
        meta_to = store.load_meta(symbol, to_dt)
        spot_from = _spot_from_meta(meta_from)
        spot_to = _spot_from_meta(meta_to)
        if spot_from is None or spot_to is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        diff, report_from, report_to = compute_compare_report(
            symbol=symbol,
            from_date=from_dt,
            to_date=to_dt,
            from_df=df_from,
            to_df=df_to,
            spot_from=spot_from,
            spot_to=spot_to,
            top=top,
        )

        artifact = CompareArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=to_dt.isoformat(),
            symbol=symbol.upper(),
            from_report=report_from.model_dump(),
            to_report=report_to.model_dump(),
            diff=diff.model_dump(),
        )
        if strict:
            CompareArtifact.model_validate(artifact.to_dict())

        render_compare_report_console(console, diff)

        if out is not None:
            base = out / "compare" / symbol.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{from_dt.isoformat()}_to_{to_dt.isoformat()}.json"
            out_path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def roll_plan(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (positions + risk profile)."),
    position_id: str = typer.Option(..., "--id", help="Position id to plan a roll for."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    intent: str = typer.Option(
        "max-upside",
        "--intent",
        help="Intent: max-upside|reduce-theta|increase-delta|de-risk",
    ),
    horizon_months: int = typer.Option(..., "--horizon-months", min=1, max=60),
    shape: str = typer.Option(
        "out-same-strike",
        "--shape",
        help="Roll shape: out-same-strike|out-up|out-down",
    ),
    top: int = typer.Option(10, "--top", min=1, max=50, help="Number of candidates to display."),
    max_debit: float | None = typer.Option(
        None,
        "--max-debit",
        help="Max roll debit in dollars (total for position size).",
    ),
    min_credit: float | None = typer.Option(
        None,
        "--min-credit",
        help="Min roll credit in dollars (total for position size).",
    ),
    min_open_interest: int | None = typer.Option(
        None,
        "--min-open-interest",
        help="Override minimum open interest liquidity gate (default from risk profile).",
    ),
    min_volume: int | None = typer.Option(
        None,
        "--min-volume",
        help="Override minimum volume liquidity gate (default from risk profile).",
    ),
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
    ),
) -> None:
    """Propose and rank roll candidates for a single position using offline snapshots."""
    console = Console(width=200)

    import options_helper.commands.reports as reports_pkg

    portfolio = load_portfolio(portfolio_path)
    position = next((p for p in portfolio.positions if p.id == position_id), None)
    if position is None:
        raise typer.BadParameter(f"No position found with id: {position_id}", param_hint="--id")

    intent_norm = intent.strip().lower()
    if intent_norm not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
        raise typer.BadParameter(
            "Invalid --intent (use max-upside|reduce-theta|increase-delta|de-risk)",
            param_hint="--intent",
        )

    shape_norm = shape.strip().lower()
    if shape_norm not in {"out-same-strike", "out-up", "out-down"}:
        raise typer.BadParameter("Invalid --shape (use out-same-strike|out-up|out-down)", param_hint="--shape")

    rp = portfolio.risk_profile
    min_oi = rp.min_open_interest if min_open_interest is None else int(min_open_interest)
    min_vol = rp.min_volume if min_volume is None else int(min_volume)

    store = cli_deps.build_snapshot_store(cache_dir)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    next_earnings_date = reports_pkg.safe_next_earnings_date(earnings_store, position.symbol)

    try:
        as_of_date = store.resolve_date(position.symbol, as_of)
        df = store.load_day(position.symbol, as_of_date)
        meta = store.load_meta(position.symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        if isinstance(position, MultiLegPosition):
            report = compute_roll_plan_multileg(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                horizon_months=horizon_months,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
            )
            render_roll_plan_multileg_console(console, report)
        else:
            report = compute_roll_plan(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                intent=intent_norm,
                horizon_months=horizon_months,
                shape=shape_norm,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
            )
            render_roll_plan_console(console, report)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


__all__ = ["chain_report", "compare_report", "roll_plan"]
