from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable, cast

import options_helper.cli_deps as cli_deps
import pandas as pd
from rich.console import RenderableType
from rich.table import Table

from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.commands.common import _spot_from_meta
from options_helper.schemas.common import utc_now
from options_helper.schemas.flow import FlowArtifact
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


@dataclass(frozen=True)
class _FlowRuntime:
    quality_logger: Any | None
    portfolio: Any
    store: Any
    flow_store: Any
    symbols: list[str]
    pos_keys: set[tuple[str, str, float, str]]
    group_by_norm: str
    group_by_val: FlowGroupBy


def _persist_flow_quality(
    *,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_flow_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    flow_store: Any,
    symbols: list[str],
    skip_reason: str | None = None,
) -> None:
    persist_quality_results_fn(
        quality_logger,
        run_flow_quality_checks_fn(
            flow_store=flow_store,
            symbols=symbols,
            skip_reason=skip_reason,
        ),
    )


def _no_symbols_result(
    *,
    message: str,
    result_factory: Callable[..., Any],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_flow_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    flow_store: Any,
) -> Any:
    _persist_flow_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_flow_quality_checks_fn=run_flow_quality_checks_fn,
        quality_logger=quality_logger,
        flow_store=flow_store,
        symbols=[],
        skip_reason="no_symbols",
    )
    return result_factory(
        renderables=[message],
        no_symbols=True,
    )


def _resolve_symbols(
    *,
    portfolio: Any,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    watchlists_loader: Callable[[Path], Any],
    parameter_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_flow_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    flow_store: Any,
) -> tuple[list[str], Any | None]:
    use_watchlists = bool(watchlist) or all_watchlists
    if use_watchlists:
        wl = watchlists_loader(watchlists_path)
        if all_watchlists:
            symbols = sorted({value for values in wl.watchlists.values() for value in values})
            if not symbols:
                return [], _no_symbols_result(
                    message=f"No watchlists in {watchlists_path}",
                    result_factory=result_factory,
                    persist_quality_results_fn=persist_quality_results_fn,
                    run_flow_quality_checks_fn=run_flow_quality_checks_fn,
                    quality_logger=quality_logger,
                    flow_store=flow_store,
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
        symbols = sorted({position.symbol for position in portfolio.positions})
        if not symbols and symbol is None:
            return [], _no_symbols_result(
                message="No positions.",
                result_factory=result_factory,
                persist_quality_results_fn=persist_quality_results_fn,
                run_flow_quality_checks_fn=run_flow_quality_checks_fn,
                quality_logger=quality_logger,
                flow_store=flow_store,
            )
    if symbol is not None:
        symbols = [symbol.upper()]
    return symbols, None


def _resolve_group_by(
    group_by: str,
    *,
    parameter_error_factory: Callable[..., Exception],
) -> tuple[str, FlowGroupBy]:
    group_by_norm = group_by.strip().lower()
    valid_group_by = {"contract", "strike", "expiry", "expiry-strike"}
    if group_by_norm not in valid_group_by:
        raise parameter_error_factory(
            f"Invalid --group-by (use {', '.join(sorted(valid_group_by))})",
            param_hint="--group-by",
        )
    return group_by_norm, cast(FlowGroupBy, group_by_norm)


def _position_keys(portfolio: Any) -> set[tuple[str, str, float, str]]:
    return {
        (position.symbol, position.expiry.isoformat(), float(position.strike), position.option_type)
        for position in portfolio.positions
    }


def _load_pair_flows_for_symbol(
    *,
    sym: str,
    window: int,
    store: Any,
    renderables: list[RenderableType],
) -> tuple[list[date], list[pd.DataFrame]]:
    need = window + 1
    dates = store.latest_dates(sym, n=need)
    if len(dates) < need:
        renderables.append(f"[yellow]No flow data for {sym}:[/yellow] need at least {need} snapshots.")
        return dates, []

    pair_flows: list[pd.DataFrame] = []
    for prev_date, today_date in zip(dates[:-1], dates[1:], strict=False):
        today_df = store.load_day(sym, today_date)
        prev_df = store.load_day(sym, prev_date)
        if today_df.empty or prev_df.empty:
            renderables.append(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s) in window.")
            return dates, []
        spot = _spot_from_meta(store.load_meta(sym, today_date))
        pair_flows.append(compute_flow(today_df, prev_df, spot=spot))
    return dates, pair_flows


def _sorted_flow_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "deltaOI_notional" in frame.columns:
        return frame.assign(_abs=frame["deltaOI_notional"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
    return frame


def _sorted_net_frame(frame: pd.DataFrame) -> pd.DataFrame:
    net = frame.assign(_abs=frame["deltaOI_notional"].abs() if "deltaOI_notional" in frame.columns else 0.0)
    sort_cols = ["_abs"]
    ascending = [False]
    for column in ["expiry", "strike", "optionType", "contractSymbol"]:
        if column in net.columns:
            sort_cols.append(column)
            ascending.append(True)
    return net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])


def _artifact_payload(
    *,
    net: pd.DataFrame,
    as_of: date,
    sym: str,
    from_date: date,
    to_date: date,
    window: int,
    group_by: str,
    snapshot_dates: list[date],
    strict: bool,
    flow_store: Any,
) -> dict[str, Any]:
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
        as_of=as_of.isoformat(),
        symbol=sym.upper(),
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        window=window,
        group_by=group_by,
        snapshot_dates=[value.isoformat() for value in snapshot_dates],
        net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
    )
    payload = artifact.to_dict()
    if strict:
        FlowArtifact.model_validate(payload)
    flow_store.upsert_artifact(artifact)
    return payload


def _write_flow_payload(
    *,
    out: Path | None,
    sym: str,
    filename: str,
    payload: dict[str, Any],
    renderables: list[RenderableType],
) -> None:
    if out is None:
        return
    base = out / "flow" / sym.upper()
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / filename
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    renderables.append(f"\nSaved: {out_path}")


def _contract_row_cells(
    *,
    sym: str,
    row: pd.Series,
    pos_keys: set[tuple[str, str, float, str]],
) -> list[str]:
    expiry = str(row.get("expiry", "-"))
    opt_type = str(row.get("optionType", "-"))
    strike = row.get("strike")
    strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
    key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
    in_port = key in pos_keys if strike_val is not None else False
    return [
        "*" if in_port else "",
        expiry,
        opt_type,
        "-" if strike_val is None else f"{strike_val:g}",
        "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
        "-" if pd.isna(row.get("openInterest")) else f"{row.get('openInterest'):.0f}",
        "-" if pd.isna(row.get("volume")) else f"{row.get('volume'):.0f}",
        "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
        str(row.get("flow_class", "-")),
    ]


def _render_contract_table(
    *,
    sym: str,
    flow: pd.DataFrame,
    top: int,
    pos_keys: set[tuple[str, str, float, str]],
) -> Table:
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
        table.add_row(*_contract_row_cells(sym=sym, row=row, pos_keys=pos_keys))
    return table


def _render_contract_window_one(
    *,
    sym: str,
    dates: list[date],
    pair_flows: list[pd.DataFrame],
    top: int,
    pos_keys: set[tuple[str, str, float, str]],
    strict: bool,
    out: Path | None,
    flow_store: Any,
    renderables: list[RenderableType],
) -> None:
    prev_date, today_date = dates[-2], dates[-1]
    flow = pair_flows[-1]
    summary = summarize_flow(flow)
    renderables.append(
        f"\n[bold]{sym}[/bold] flow {prev_date.isoformat()} → {today_date.isoformat()} | "
        f"calls ΔOI$={summary['calls_delta_oi_notional']:,.0f} | puts ΔOI$={summary['puts_delta_oi_notional']:,.0f}"
    )
    if flow.empty:
        renderables.append("No flow rows.")
        return

    flow = _sorted_flow_frame(flow)
    renderables.append(_render_contract_table(sym=sym, flow=flow, top=top, pos_keys=pos_keys))
    net = _sorted_net_frame(aggregate_flow_window(pair_flows, group_by="contract"))
    payload = _artifact_payload(
        net=net,
        as_of=today_date,
        sym=sym,
        from_date=prev_date,
        to_date=today_date,
        window=1,
        group_by="contract",
        snapshot_dates=[prev_date, today_date],
        strict=strict,
        flow_store=flow_store,
    )
    _write_flow_payload(
        out=out,
        sym=sym,
        filename=f"{prev_date.isoformat()}_to_{today_date.isoformat()}_w1_contract.json",
        payload=payload,
        renderables=renderables,
    )


def _render_zone_table(*, title: str, group_by_norm: str) -> Table:
    table = Table(title=title)
    if group_by_norm == "contract":
        table.add_column("*")
    if group_by_norm in {"expiry", "expiry-strike", "contract"}:
        table.add_column("Expiry")
    if group_by_norm in {"strike", "expiry-strike", "contract"}:
        table.add_column("Strike", justify="right")
    table.add_column("Type")
    table.add_column("Net ΔOI", justify="right")
    table.add_column("Net ΔOI$", justify="right")
    table.add_column("Net Δ$", justify="right")
    table.add_column("N", justify="right")
    return table


def _zone_row_cells(
    *,
    sym: str,
    row: pd.Series,
    group_by_norm: str,
    pos_keys: set[tuple[str, str, float, str]],
) -> list[str]:
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
    return cells


def _append_zone_tables(
    *,
    sym: str,
    top: int,
    net: pd.DataFrame,
    group_by_norm: str,
    pos_keys: set[tuple[str, str, float, str]],
    renderables: list[RenderableType],
) -> None:
    building = net[net["deltaOI_notional"] > 0].head(top)
    unwinding = net[net["deltaOI_notional"] < 0].head(top)

    t_build = _render_zone_table(title=f"{sym} building zones (top {top} by |net ΔOI$|)", group_by_norm=group_by_norm)
    for _, row in building.iterrows():
        t_build.add_row(*_zone_row_cells(sym=sym, row=row, group_by_norm=group_by_norm, pos_keys=pos_keys))
    renderables.append(t_build)

    t_unwind = _render_zone_table(
        title=f"{sym} unwinding zones (top {top} by |net ΔOI$|)",
        group_by_norm=group_by_norm,
    )
    for _, row in unwinding.iterrows():
        t_unwind.add_row(*_zone_row_cells(sym=sym, row=row, group_by_norm=group_by_norm, pos_keys=pos_keys))
    renderables.append(t_unwind)


def _render_grouped_window(
    *,
    sym: str,
    dates: list[date],
    pair_flows: list[pd.DataFrame],
    window: int,
    group_by_norm: str,
    group_by_val: FlowGroupBy,
    top: int,
    strict: bool,
    out: Path | None,
    pos_keys: set[tuple[str, str, float, str]],
    flow_store: Any,
    renderables: list[RenderableType],
) -> None:
    start_date, end_date = dates[0], dates[-1]
    net = aggregate_flow_window(pair_flows, group_by=group_by_val)
    if net.empty:
        renderables.append(
            f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()})"
        )
        renderables.append("No net flow rows.")
        return

    calls_premium = float(net[net["optionType"] == "call"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
    puts_premium = float(net[net["optionType"] == "put"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
    renderables.append(
        f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()}) | "
        f"group-by={group_by_norm} | calls ΔOI$={calls_premium:,.0f} | puts ΔOI$={puts_premium:,.0f}"
    )
    net = _sorted_net_frame(net)
    _append_zone_tables(
        sym=sym,
        top=top,
        net=net,
        group_by_norm=group_by_norm,
        pos_keys=pos_keys,
        renderables=renderables,
    )
    payload = _artifact_payload(
        net=net,
        as_of=end_date,
        sym=sym,
        from_date=start_date,
        to_date=end_date,
        window=window,
        group_by=group_by_norm,
        snapshot_dates=dates,
        strict=strict,
        flow_store=flow_store,
    )
    _write_flow_payload(
        out=out,
        sym=sym,
        filename=f"{start_date.isoformat()}_to_{end_date.isoformat()}_w{window}_{group_by_norm}.json",
        payload=payload,
        renderables=renderables,
    )


def _prepare_flow_runtime(params: dict[str, Any]) -> tuple[_FlowRuntime | None, Any | None]:
    quality_logger = params["resolve_quality_run_logger_fn"](params["run_logger"])
    portfolio = params["portfolio_loader"](params["portfolio_path"])
    store = params["active_snapshot_store_fn"](params["snapshot_store_builder"](params["cache_dir"]))
    flow_store = params["flow_store_builder"](params["cache_dir"])
    symbols, early_result = _resolve_symbols(
        portfolio=portfolio,
        symbol=params["symbol"],
        watchlists_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        all_watchlists=params["all_watchlists"],
        watchlists_loader=params["watchlists_loader"],
        parameter_error_factory=params["parameter_error_factory"],
        result_factory=params["result_factory"],
        persist_quality_results_fn=params["persist_quality_results_fn"],
        run_flow_quality_checks_fn=params["run_flow_quality_checks_fn"],
        quality_logger=quality_logger,
        flow_store=flow_store,
    )
    if early_result is not None:
        return None, early_result
    group_by_norm, group_by_val = _resolve_group_by(
        params["group_by"],
        parameter_error_factory=params["parameter_error_factory"],
    )
    return (
        _FlowRuntime(
            quality_logger=quality_logger,
            portfolio=portfolio,
            store=store,
            flow_store=flow_store,
            symbols=symbols,
            pos_keys=_position_keys(portfolio),
            group_by_norm=group_by_norm,
            group_by_val=group_by_val,
        ),
        None,
    )


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
    params = dict(locals())
    runtime, early_result = _prepare_flow_runtime(params)
    if early_result is not None:
        return early_result
    if runtime is None:
        raise RuntimeError("flow runtime unavailable")
    renderables: list[RenderableType] = []
    for sym in runtime.symbols:
        dates, pair_flows = _load_pair_flows_for_symbol(
            sym=sym,
            window=window,
            store=runtime.store,
            renderables=renderables,
        )
        if not pair_flows:
            continue
        if window == 1 and runtime.group_by_norm == "contract":
            _render_contract_window_one(
                sym=sym,
                dates=dates,
                pair_flows=pair_flows,
                top=top,
                pos_keys=runtime.pos_keys,
                strict=strict,
                out=out,
                flow_store=runtime.flow_store,
                renderables=renderables,
            )
            continue
        _render_grouped_window(
            sym=sym,
            dates=dates,
            pair_flows=pair_flows,
            window=window,
            group_by_norm=runtime.group_by_norm,
            group_by_val=runtime.group_by_val,
            top=top,
            strict=strict,
            out=out,
            pos_keys=runtime.pos_keys,
            flow_store=runtime.flow_store,
            renderables=renderables,
        )

    _persist_flow_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_flow_quality_checks_fn=run_flow_quality_checks_fn,
        quality_logger=runtime.quality_logger,
        flow_store=runtime.flow_store,
        symbols=runtime.symbols,
    )
    return result_factory(renderables=renderables, no_symbols=False)
