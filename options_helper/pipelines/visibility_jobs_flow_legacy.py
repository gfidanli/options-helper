from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import options_helper.cli_deps as cli_deps
from rich.console import RenderableType
from rich.table import Table

from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.commands.common import _spot_from_meta
from options_helper.schemas.common import utc_now
from options_helper.schemas.flow import FlowArtifact
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


def run_flow_report_job_impl(
    *,
    portfolio_path: Path,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    cache_dir: Path,
    window: int,
    group_by: str,
    top: int,
    out: Path | None,
    strict: bool,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    flow_store_builder: Callable[[Path], Any] = cli_deps.build_flow_store,
    portfolio_loader: Callable[[Path], Any] = load_portfolio,
    watchlists_loader: Callable[[Path], Any] = load_watchlists,
    run_logger: Any | None = None,
    resolve_quality_run_logger_fn: Callable[[Any | None], Any | None],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_flow_quality_checks_fn: Callable[..., list[Any]],
    active_snapshot_store_fn: Callable[[Any], Any],
    parameter_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
) -> Any:
    import pandas as pd

    quality_logger = resolve_quality_run_logger_fn(run_logger)
    portfolio = portfolio_loader(portfolio_path)
    renderables: list[RenderableType] = []

    store = active_snapshot_store_fn(snapshot_store_builder(cache_dir))
    flow_store = flow_store_builder(cache_dir)
    use_watchlists = bool(watchlist) or all_watchlists
    if use_watchlists:
        wl = watchlists_loader(watchlists_path)
        if all_watchlists:
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                persist_quality_results_fn(
                    quality_logger,
                    run_flow_quality_checks_fn(
                        flow_store=flow_store,
                        symbols=[],
                        skip_reason="no_symbols",
                    ),
                )
                return result_factory(
                    renderables=[f"No watchlists in {watchlists_path}"],
                    no_symbols=True,
                )
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise parameter_error_factory(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
    else:
        symbols = sorted({p.symbol for p in portfolio.positions})
        if not symbols and symbol is None:
            persist_quality_results_fn(
                quality_logger,
                run_flow_quality_checks_fn(
                    flow_store=flow_store,
                    symbols=[],
                    skip_reason="no_symbols",
                ),
            )
            return result_factory(renderables=["No positions."], no_symbols=True)

    if symbol is not None:
        symbols = [symbol.upper()]

    pos_keys = {(p.symbol, p.expiry.isoformat(), float(p.strike), p.option_type) for p in portfolio.positions}

    group_by_norm = group_by.strip().lower()
    valid_group_by = {"contract", "strike", "expiry", "expiry-strike"}
    if group_by_norm not in valid_group_by:
        raise parameter_error_factory(
            f"Invalid --group-by (use {', '.join(sorted(valid_group_by))})",
            param_hint="--group-by",
        )
    group_by_val = cast(FlowGroupBy, group_by_norm)

    for sym in symbols:
        need = window + 1
        dates = store.latest_dates(sym, n=need)
        if len(dates) < need:
            renderables.append(f"[yellow]No flow data for {sym}:[/yellow] need at least {need} snapshots.")
            continue

        pair_flows: list[pd.DataFrame] = []
        for prev_date, today_date in zip(dates[:-1], dates[1:], strict=False):
            today_df = store.load_day(sym, today_date)
            prev_df = store.load_day(sym, prev_date)
            if today_df.empty or prev_df.empty:
                renderables.append(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s) in window.")
                pair_flows = []
                break

            spot = _spot_from_meta(store.load_meta(sym, today_date))
            pair_flows.append(compute_flow(today_df, prev_df, spot=spot))

        if not pair_flows:
            continue

        start_date, end_date = dates[0], dates[-1]

        if window == 1 and group_by_norm == "contract":
            prev_date, today_date = dates[-2], dates[-1]
            flow = pair_flows[-1]
            summary = summarize_flow(flow)

            renderables.append(
                f"\n[bold]{sym}[/bold] flow {prev_date.isoformat()} → {today_date.isoformat()} | "
                f"calls ΔOI$={summary['calls_delta_oi_notional']:,.0f} | puts ΔOI$={summary['puts_delta_oi_notional']:,.0f}"
            )

            if flow.empty:
                renderables.append("No flow rows.")
                continue

            if "deltaOI_notional" in flow.columns:
                flow = flow.assign(_abs=flow["deltaOI_notional"].abs())
                flow = flow.sort_values("_abs", ascending=False).drop(columns=["_abs"])

            table = Table(title=f"{sym} top {top} contracts by |ΔOI_notional|")
            table.add_column("*")
            table.add_column("Expiry")
            table.add_column("Type")
            table.add_column("Strike", justify="right")
            table.add_column("ΔOI", justify="right")
            table.add_column("OI", justify="right")
            table.add_column("Vol", justify="right")
            table.add_column("ΔOI$", justify="right")
            table.add_column("Class")

            for _, row in flow.head(top).iterrows():
                expiry = str(row.get("expiry", "-"))
                opt_type = str(row.get("optionType", "-"))
                strike = row.get("strike")
                strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
                key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
                in_port = key in pos_keys if strike_val is not None else False

                table.add_row(
                    "*" if in_port else "",
                    expiry,
                    opt_type,
                    "-" if strike_val is None else f"{strike_val:g}",
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("openInterest")) else f"{row.get('openInterest'):.0f}",
                    "-" if pd.isna(row.get("volume")) else f"{row.get('volume'):.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    str(row.get("flow_class", "-")),
                )

            renderables.append(table)

            net = aggregate_flow_window(pair_flows, group_by="contract")
            net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
            sort_cols = ["_abs"]
            ascending = [False]
            for c in ["expiry", "strike", "optionType", "contractSymbol"]:
                if c in net.columns:
                    sort_cols.append(c)
                    ascending.append(True)
            net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

            artifact_net = net.rename(
                columns={
                    "contractSymbol": "contract_symbol",
                    "optionType": "option_type",
                    "deltaOI": "delta_oi",
                    "deltaOI_notional": "delta_oi_notional",
                    "size": "n_pairs",
                }
            )
            artifact = FlowArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=today_date.isoformat(),
                symbol=sym.upper(),
                from_date=prev_date.isoformat(),
                to_date=today_date.isoformat(),
                window=1,
                group_by="contract",
                snapshot_dates=[prev_date.isoformat(), today_date.isoformat()],
                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
            )
            payload = artifact.to_dict()
            if strict:
                FlowArtifact.model_validate(payload)
            flow_store.upsert_artifact(artifact)

            if out is not None:
                base = out / "flow" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / f"{prev_date.isoformat()}_to_{today_date.isoformat()}_w1_contract.json"
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                renderables.append(f"\nSaved: {out_path}")
            continue

        net = aggregate_flow_window(pair_flows, group_by=group_by_val)
        if net.empty:
            renderables.append(
                f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()})"
            )
            renderables.append("No net flow rows.")
            continue

        calls_premium = (
            float(net[net["optionType"] == "call"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        )
        puts_premium = (
            float(net[net["optionType"] == "put"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        )

        renderables.append(
            f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()}) | "
            f"group-by={group_by_norm} | calls ΔOI$={calls_premium:,.0f} | puts ΔOI$={puts_premium:,.0f}"
        )

        net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
        sort_cols = ["_abs"]
        ascending = [False]
        for c in ["expiry", "strike", "optionType", "contractSymbol"]:
            if c in net.columns:
                sort_cols.append(c)
                ascending.append(True)

        net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

        def _render_zone_table(title: str) -> Table:
            t = Table(title=title)
            if group_by_norm == "contract":
                t.add_column("*")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                t.add_column("Expiry")
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                t.add_column("Strike", justify="right")
            t.add_column("Type")
            t.add_column("Net ΔOI", justify="right")
            t.add_column("Net ΔOI$", justify="right")
            t.add_column("Net Δ$", justify="right")
            t.add_column("N", justify="right")
            return t

        def _add_zone_row(t: Table, row: pd.Series) -> None:
            expiry = str(row.get("expiry", "-"))
            opt_type = str(row.get("optionType", "-"))
            strike = row.get("strike")
            strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
            key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
            in_port = key in pos_keys if strike_val is not None else False

            cells: list[str] = []
            if group_by_norm == "contract":
                cells.append("*" if in_port else "")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                cells.append(expiry)
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                cells.append("-" if strike_val is None else f"{strike_val:g}")
            cells.extend(
                [
                    opt_type,
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    "-" if pd.isna(row.get("delta_notional")) else f"{row.get('delta_notional'):+.0f}",
                    "-" if pd.isna(row.get("size")) else f"{int(row.get('size')):d}",
                ]
            )
            t.add_row(*cells)

        building = net[net["deltaOI_notional"] > 0].head(top)
        unwinding = net[net["deltaOI_notional"] < 0].head(top)

        t_build = _render_zone_table(f"{sym} building zones (top {top} by |net ΔOI$|)")
        for _, row in building.iterrows():
            _add_zone_row(t_build, row)
        renderables.append(t_build)

        t_unwind = _render_zone_table(f"{sym} unwinding zones (top {top} by |net ΔOI$|)")
        for _, row in unwinding.iterrows():
            _add_zone_row(t_unwind, row)
        renderables.append(t_unwind)

        artifact_net = net.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "optionType": "option_type",
                "deltaOI": "delta_oi",
                "deltaOI_notional": "delta_oi_notional",
                "size": "n_pairs",
            }
        )
        artifact = FlowArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=end_date.isoformat(),
            symbol=sym.upper(),
            from_date=start_date.isoformat(),
            to_date=end_date.isoformat(),
            window=window,
            group_by=group_by_norm,
            snapshot_dates=[d.isoformat() for d in dates],
            net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
        )
        payload = artifact.to_dict()
        if strict:
            FlowArtifact.model_validate(payload)
        flow_store.upsert_artifact(artifact)

        if out is not None:
            base = out / "flow" / sym.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{start_date.isoformat()}_to_{end_date.isoformat()}_w{window}_{group_by_norm}.json"
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            renderables.append(f"\nSaved: {out_path}")

    persist_quality_results_fn(
        quality_logger,
        run_flow_quality_checks_fn(flow_store=flow_store, symbols=symbols),
    )
    return result_factory(renderables=renderables, no_symbols=False)

