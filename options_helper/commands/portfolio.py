from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import options_helper.cli_deps as cli_deps
from options_helper.analysis.scenarios import compute_position_scenarios
from options_helper.commands.common import _parse_date, _spot_from_meta
from options_helper.commands.position_metrics import _extract_float, _mark_price
from options_helper.data.options_snapshots import find_snapshot_row
from options_helper.models import Leg, MultiLegPosition, OptionType, Position
from options_helper.reporting import render_summary
from options_helper.schemas.common import utc_now
from options_helper.schemas.scenarios import ScenarioGridRow, ScenarioSummaryRow, ScenariosArtifact
from options_helper.storage import load_portfolio, save_portfolio, write_template

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SCENARIO_PORTFOLIO_PATH_ARG = typer.Argument(..., help="Path to portfolio JSON.")
_SCENARIO_AS_OF_OPT = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'.")
_SCENARIO_CACHE_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--cache-dir",
    help="Directory containing options snapshot folders.",
)
_SCENARIO_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Output root for saved artifacts (writes under {out}/scenarios/{PORTFOLIO_DATE}/).",
)
_SCENARIO_STRICT_OPT = typer.Option(
    False,
    "--strict",
    help="Validate JSON artifacts against schemas before writing.",
)


@dataclass(frozen=True)
class _ScenarioTarget:
    key: str
    symbol: str
    option_type: OptionType
    side: str
    expiry: date
    strike: float
    contracts: int
    basis: float | None


@dataclass(frozen=True)
class _SymbolScenarioContext:
    as_of: date
    day_df: object
    spot: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ScenarioResult:
    target: _ScenarioTarget
    artifact: ScenariosArtifact


def register(app: typer.Typer) -> None:
    app.command()(init)
    app.command("list")(list_positions)
    app.command("scenarios")(position_scenarios)
    app.command("position-scenarios")(position_scenarios)
    app.command("add-position")(add_position)
    app.command("add-spread")(add_spread)
    app.command("remove-position")(remove_position)


def _default_position_id(symbol: str, expiry: date, strike: float, option_type: OptionType) -> str:
    suffix = "c" if option_type == "call" else "p"
    strike_str = f"{strike:g}".replace(".", "p")
    return f"{symbol.lower()}-{expiry.isoformat()}-{strike_str}{suffix}"


def _default_multileg_id(symbol: str, legs: list[Leg]) -> str:
    sorted_legs = sorted(legs, key=lambda leg: (leg.expiry, leg.option_type, leg.strike, leg.side))
    tokens: list[str] = []
    for leg in sorted_legs:
        strike_str = f"{leg.strike:g}".replace(".", "p")
        token = f"{leg.side[0]}{leg.option_type[0]}{strike_str}@{leg.expiry.isoformat()}"
        tokens.append(token)
    if len(tokens) > 2:
        tokens = tokens[:2] + [f"n{len(legs)}"]
    return f"{symbol.lower()}-ml-" + "-".join(tokens)


def _parse_leg_spec(value: str) -> Leg:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 5 or len(parts) > 6:
        raise typer.BadParameter(
            "Invalid --leg format. Use side,type,expiry,strike,contracts[,ratio].",
            param_hint="--leg",
        )
    side = parts[0].lower()
    if side not in {"long", "short"}:
        raise typer.BadParameter("Invalid leg side (use long|short).", param_hint="--leg")
    opt_type = parts[1].lower()
    if opt_type not in {"call", "put"}:
        raise typer.BadParameter("Invalid leg type (use call|put).", param_hint="--leg")
    expiry = _parse_date(parts[2])
    try:
        strike = float(parts[3])
    except ValueError as exc:
        raise typer.BadParameter("Invalid leg strike (use number).", param_hint="--leg") from exc
    try:
        contracts = int(parts[4])
    except ValueError as exc:
        raise typer.BadParameter("Invalid leg contracts (use integer).", param_hint="--leg") from exc
    ratio = None
    if len(parts) == 6:
        try:
            ratio = float(parts[5])
        except ValueError as exc:
            raise typer.BadParameter("Invalid leg ratio (use number).", param_hint="--leg") from exc

    return Leg(
        side=side,  # type: ignore[arg-type]
        option_type=opt_type,  # type: ignore[arg-type]
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        ratio=ratio,
    )


def _unique_id_with_suffix(existing_ids: set[str], base_id: str) -> str:
    if base_id not in existing_ids:
        return base_id
    for i in range(2, 1000):
        candidate = f"{base_id}-{i}"
        if candidate not in existing_ids:
            return candidate
    raise typer.BadParameter("Unable to generate a unique id; please supply --id.")


def init(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    force: bool = typer.Option(False, "--force", help="Overwrite if file exists."),
) -> None:
    """Create a starter portfolio JSON file."""
    write_template(portfolio_path, force=force)
    Console().print(f"Wrote template portfolio to {portfolio_path}")


def list_positions(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """List positions in the portfolio file."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        return

    from rich.table import Table

    table = Table(title="Portfolio Positions")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Type")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Cost", justify="right")
    for p in portfolio.positions:
        table.add_row(
            p.id,
            p.symbol,
            p.option_type,
            p.expiry.isoformat(),
            f"{p.strike:g}",
            str(p.contracts),
            f"${p.cost_basis:.2f}",
        )
    console.print(table)


def position_scenarios(
    portfolio_path: Path = _SCENARIO_PORTFOLIO_PATH_ARG,
    as_of: str = _SCENARIO_AS_OF_OPT,
    cache_dir: Path = _SCENARIO_CACHE_DIR_OPT,
    out: Path | None = _SCENARIO_OUT_OPT,
    strict: bool = _SCENARIO_STRICT_OPT,
) -> None:
    """Compute per-position scenario grids from local snapshot data (offline-first)."""
    portfolio = load_portfolio(portfolio_path)
    console = Console(width=200)
    render_summary(console, portfolio)

    targets = _build_scenario_targets(portfolio.positions)
    if not targets:
        console.print("No positions.")
        raise typer.Exit(0)

    explicit_as_of = _parse_explicit_as_of(as_of)
    fallback_as_of = explicit_as_of or date.today()
    contexts = _build_scenario_contexts(targets=targets, as_of_spec=as_of, fallback_as_of=fallback_as_of, cache_dir=cache_dir)
    results = [_build_position_scenario_result(target=target, context=contexts[target.symbol]) for target in targets]

    portfolio_date = _portfolio_date(results, fallback=fallback_as_of)
    _render_scenario_console(console, results, portfolio_date=portfolio_date)
    _write_scenario_artifacts(console, results=results, portfolio_date=portfolio_date, out=out, strict=strict)


def _build_scenario_contexts(
    *,
    targets: list[_ScenarioTarget],
    as_of_spec: str,
    fallback_as_of: date,
    cache_dir: Path,
) -> dict[str, _SymbolScenarioContext]:
    snapshot_store = cli_deps.build_snapshot_store(cache_dir)
    contexts: dict[str, _SymbolScenarioContext] = {}
    for symbol in sorted({target.symbol for target in targets}):
        contexts[symbol] = _resolve_symbol_context(
            store=snapshot_store,
            symbol=symbol,
            as_of_spec=as_of_spec,
            fallback_as_of=fallback_as_of,
        )
    return contexts


def _build_position_scenario_result(*, target: _ScenarioTarget, context: _SymbolScenarioContext) -> _ScenarioResult:
    row, extra_warnings = _resolve_target_snapshot_row(target=target, context=context)
    spot = _resolve_scenario_spot(context=context, row=row)
    contract_symbol = _resolve_scenario_contract_symbol(target=target, row=row)
    mark = _resolve_scenario_mark(row=row)
    iv = _row_float(row, "impliedVolatility", "implied_volatility")
    computed = compute_position_scenarios(
        symbol=target.symbol,
        as_of=context.as_of,
        contract_symbol=contract_symbol,
        option_type=target.option_type,
        side=target.side,
        contracts=target.contracts,
        spot=spot,
        strike=target.strike,
        expiry=target.expiry,
        mark=mark,
        iv=iv,
        basis=target.basis,
    )
    summary_payload = dict(computed.summary)
    summary_payload["warnings"] = _merge_warnings(base=summary_payload.get("warnings"), extra=extra_warnings)
    artifact = ScenariosArtifact(
        generated_at=utc_now(),
        as_of=context.as_of.isoformat(),
        symbol=target.symbol,
        contract_symbol=contract_symbol,
        summary=ScenarioSummaryRow.model_validate(summary_payload),
        grid=[ScenarioGridRow.model_validate(item) for item in computed.grid],
    )
    return _ScenarioResult(target=target, artifact=artifact)


def _resolve_target_snapshot_row(
    *,
    target: _ScenarioTarget,
    context: _SymbolScenarioContext,
) -> tuple[dict[str, object] | None, list[str]]:
    row = None
    extra_warnings = list(context.warnings)
    if not _is_empty_frame(context.day_df):
        row = find_snapshot_row(
            context.day_df,  # type: ignore[arg-type]
            expiry=target.expiry,
            strike=target.strike,
            option_type=target.option_type,
        )
    if row is None:
        extra_warnings.append("missing_snapshot_row")
    return row, extra_warnings


def _resolve_scenario_spot(*, context: _SymbolScenarioContext, row: dict[str, object] | None) -> float | None:
    spot = context.spot
    if spot is None and row is not None:
        spot = _row_float(row, "underlyingPrice", "underlying_price", "spot")
    return spot


def _resolve_scenario_contract_symbol(*, target: _ScenarioTarget, row: dict[str, object] | None) -> str:
    contract_symbol = _row_string(row, "contractSymbol", "contract_symbol")
    if contract_symbol:
        return contract_symbol
    return _fallback_contract_symbol(
        symbol=target.symbol,
        expiry=target.expiry,
        option_type=target.option_type,
        strike=target.strike,
    )


def _resolve_scenario_mark(*, row: dict[str, object] | None) -> float | None:
    bid = _row_float(row, "bid")
    ask = _row_float(row, "ask")
    last = _row_float(row, "lastPrice", "last_price")
    return _mark_price(bid=bid, ask=ask, last=last)


def _write_scenario_artifacts(
    console: Console,
    *,
    results: list[_ScenarioResult],
    portfolio_date: date,
    out: Path | None,
    strict: bool,
) -> None:
    if out is None:
        return
    out_dir = out / "scenarios" / portfolio_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if strict:
            ScenariosArtifact.model_validate(result.artifact.to_dict())
        out_path = out_dir / _scenario_filename(result)
        out_path.write_text(result.artifact.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Saved: {out_path}")


def add_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str = typer.Option(..., "--symbol"),
    expiry: str = typer.Option(..., "--expiry", help="Expiry date, e.g. 2026-04-17."),
    strike: float = typer.Option(..., "--strike"),
    option_type: OptionType = typer.Option(..., "--type", case_sensitive=False),
    contracts: int = typer.Option(1, "--contracts"),
    cost_basis: float = typer.Option(..., "--cost-basis", help="Premium per share (e.g. 0.45)."),
    position_id: str | None = typer.Option(None, "--id", help="Optional position id."),
    opened_at: str | None = typer.Option(None, "--opened-at", help="Optional open date (YYYY-MM-DD)."),
) -> None:
    """Add a position to the portfolio JSON."""
    portfolio = load_portfolio(portfolio_path)

    expiry_date = _parse_date(expiry)
    opened_at_date = _parse_date(opened_at) if opened_at else None

    symbol = symbol.upper()
    option_type = option_type.lower()  # type: ignore[assignment]

    pid = position_id or _default_position_id(symbol, expiry_date, strike, option_type)
    if any(p.id == pid for p in portfolio.positions):
        raise typer.BadParameter(f"Position id already exists: {pid}")

    position = Position(
        id=pid,
        symbol=symbol,
        option_type=option_type,
        expiry=expiry_date,
        strike=float(strike),
        contracts=int(contracts),
        cost_basis=float(cost_basis),
        opened_at=opened_at_date,
    )

    portfolio.positions.append(position)
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Added {pid}")


def add_spread(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str = typer.Option(..., "--symbol"),
    legs: list[str] = typer.Option(
        ...,
        "--leg",
        help="Repeatable leg spec: side,type,expiry,strike,contracts[,ratio].",
    ),
    net_debit: float | None = typer.Option(
        None,
        "--net-debit",
        help="Net debit in dollars for the whole structure (optional).",
    ),
    position_id: str | None = typer.Option(None, "--id", help="Optional position id."),
    opened_at: str | None = typer.Option(None, "--opened-at", help="Optional open date (YYYY-MM-DD)."),
) -> None:
    """Add a multi-leg (spread) position to the portfolio JSON."""
    portfolio = load_portfolio(portfolio_path)

    if len(legs) < 2:
        raise typer.BadParameter("Provide at least two --leg values.", param_hint="--leg")

    parsed_legs = [_parse_leg_spec(value) for value in legs]
    symbol = symbol.upper()
    opened_at_date = _parse_date(opened_at) if opened_at else None

    base_id = position_id or _default_multileg_id(symbol, parsed_legs)
    existing_ids = {p.id for p in portfolio.positions}
    if position_id is not None:
        if position_id in existing_ids:
            raise typer.BadParameter(f"Position id already exists: {position_id}", param_hint="--id")
        pid = position_id
    else:
        pid = _unique_id_with_suffix(existing_ids, base_id)

    position = MultiLegPosition(
        id=pid,
        symbol=symbol,
        legs=parsed_legs,
        net_debit=None if net_debit is None else float(net_debit),
        opened_at=opened_at_date,
    )

    portfolio.positions.append(position)
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Added {pid}")


def remove_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    position_id: str = typer.Argument(..., help="Position id to remove."),
) -> None:
    """Remove a position by id."""
    portfolio = load_portfolio(portfolio_path)
    before = len(portfolio.positions)
    portfolio.positions = [p for p in portfolio.positions if p.id != position_id]
    after = len(portfolio.positions)
    if before == after:
        raise typer.BadParameter(f"No position found with id: {position_id}")
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Removed {position_id}")


def _build_scenario_targets(positions: list[Position | MultiLegPosition]) -> list[_ScenarioTarget]:
    targets: list[_ScenarioTarget] = []
    for position in positions:
        if isinstance(position, MultiLegPosition):
            for index, leg in enumerate(position.legs, start=1):
                targets.append(
                    _ScenarioTarget(
                        key=f"{position.id}:leg{index}",
                        symbol=position.symbol.upper(),
                        option_type=leg.option_type,
                        side=leg.side,
                        expiry=leg.expiry,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        basis=None,
                    )
                )
            continue
        targets.append(
            _ScenarioTarget(
                key=position.id,
                symbol=position.symbol.upper(),
                option_type=position.option_type,
                side="long",
                expiry=position.expiry,
                strike=position.strike,
                contracts=position.contracts,
                basis=position.cost_basis,
            )
        )
    return targets


def _parse_explicit_as_of(value: str) -> date | None:
    spec = str(value or "").strip().lower()
    if spec == "latest":
        return None
    try:
        return date.fromisoformat(spec)
    except ValueError as exc:
        raise typer.BadParameter("Invalid --as-of (use YYYY-MM-DD or 'latest').", param_hint="--as-of") from exc


def _resolve_symbol_context(
    *,
    store: Any,
    symbol: str,
    as_of_spec: str,
    fallback_as_of: date,
) -> _SymbolScenarioContext:
    warnings: list[str] = []
    snapshot_date: date
    try:
        snapshot_date = store.resolve_date(symbol, as_of_spec)
    except Exception:  # noqa: BLE001
        snapshot_date = fallback_as_of
        warnings.append("missing_snapshot_date")

    try:
        day_df = store.load_day(symbol, snapshot_date)
    except Exception:  # noqa: BLE001
        day_df = {}
        warnings.append("snapshot_day_load_error")
    if _is_empty_frame(day_df):
        warnings.append("missing_snapshot_day")

    try:
        meta = store.load_meta(symbol, snapshot_date)
    except Exception:  # noqa: BLE001
        meta = {}
        warnings.append("snapshot_meta_load_error")

    spot = _spot_from_meta(meta)
    return _SymbolScenarioContext(as_of=snapshot_date, day_df=day_df, spot=spot, warnings=tuple(warnings))


def _row_float(row: object, *keys: str) -> float | None:
    if row is None:
        return None
    for key in keys:
        value = _extract_float(row, key)
        if value is not None:
            return value
    return None


def _row_string(row: object, *keys: str) -> str | None:
    if row is None:
        return None
    for key in keys:
        try:
            if key not in row:  # type: ignore[operator]
                continue
            raw = row[key]  # type: ignore[index]
        except Exception:  # noqa: BLE001
            continue
        if raw is None:
            continue
        token = str(raw).strip()
        if token:
            return token
    return None


def _is_empty_frame(value: object) -> bool:
    empty_flag = getattr(value, "empty", None)
    if isinstance(empty_flag, bool):
        return empty_flag
    return True


def _merge_warnings(*, base: object, extra: list[str]) -> list[str]:
    merged: list[str] = []
    for item in list(base) if isinstance(base, list) else []:
        token = str(item).strip()
        if token and token not in merged:
            merged.append(token)
    for token in extra:
        clean = str(token).strip()
        if clean and clean not in merged:
            merged.append(clean)
    return merged


def _fallback_contract_symbol(*, symbol: str, expiry: date, option_type: OptionType, strike: float) -> str:
    strike_scaled = int(round(float(strike) * 1000.0))
    side = "C" if option_type == "call" else "P"
    return f"{symbol.upper()}{expiry.strftime('%y%m%d')}{side}{strike_scaled:08d}"


def _portfolio_date(results: list[_ScenarioResult], *, fallback: date) -> date:
    dates: list[date] = []
    for result in results:
        try:
            dates.append(date.fromisoformat(result.artifact.as_of))
        except Exception:  # noqa: BLE001
            continue
    return max(dates) if dates else fallback


def _scenario_filename(result: _ScenarioResult) -> str:
    key = _sanitize_file_token(result.target.key)
    contract = _sanitize_file_token(result.artifact.contract_symbol)
    return f"{key}_{contract}.json"


def _sanitize_file_token(value: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "scenario"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def _render_scenario_console(console: Console, results: list[_ScenarioResult], *, portfolio_date: date) -> None:
    summary = Table(title=f"Position Scenarios (as-of {portfolio_date.isoformat()})")
    summary.add_column("ID")
    summary.add_column("Symbol")
    summary.add_column("As-Of")
    summary.add_column("Contract")
    summary.add_column("Mark", justify="right")
    summary.add_column("IV", justify="right")
    summary.add_column("Best PnL", justify="right")
    summary.add_column("Worst PnL", justify="right")
    summary.add_column("Rows", justify="right")
    summary.add_column("Warnings")

    for result in results:
        artifact = result.artifact
        warning_text = "-" if not artifact.summary.warnings else ",".join(artifact.summary.warnings)
        pnl_values = [row.pnl_position for row in artifact.grid if row.pnl_position is not None]
        best = max(pnl_values) if pnl_values else None
        worst = min(pnl_values) if pnl_values else None
        summary.add_row(
            result.target.key,
            artifact.symbol,
            artifact.as_of,
            artifact.contract_symbol,
            _fmt_money(artifact.summary.mark),
            _fmt_pct(artifact.summary.iv),
            _fmt_money(best),
            _fmt_money(worst),
            str(len(artifact.grid)),
            warning_text,
        )

    console.print(summary)

    for result in results:
        artifact = result.artifact
        if artifact.summary.warnings:
            console.print(
                f"[yellow]Warning:[/yellow] {result.target.key}: "
                + ", ".join(artifact.summary.warnings)
            )
        rows = [
            row
            for row in artifact.grid
            if abs(row.iv_change_pp) < 1e-12 and row.days_forward == 0
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row.spot_change_pct)

        table = Table(title=f"{result.target.key} spot-shock slice (iv=0pp, t+0d)")
        table.add_column("Spot Move", justify="right")
        table.add_column("Theo", justify="right")
        table.add_column("PnL/Contract", justify="right")
        table.add_column("PnL Position", justify="right")
        for row in rows:
            table.add_row(
                _fmt_pct(row.spot_change_pct),
                _fmt_money(row.theoretical_price),
                _fmt_money(row.pnl_per_contract),
                _fmt_money(row.pnl_position),
            )
        console.print(table)
