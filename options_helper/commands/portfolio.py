from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from options_helper.commands.common import _parse_date
from options_helper.models import Leg, MultiLegPosition, OptionType, Position
from options_helper.reporting import render_summary
from options_helper.storage import load_portfolio, save_portfolio, write_template


def register(app: typer.Typer) -> None:
    app.command()(init)
    app.command("list")(list_positions)
    app.command("add-position")(add_position)
    app.command("add-spread")(add_spread)
    app.command("remove-position")(remove_position)


def _default_position_id(symbol: str, expiry: date, strike: float, option_type: OptionType) -> str:
    suffix = "c" if option_type == "call" else "p"
    strike_str = f"{strike:g}".replace(".", "p")
    return f"{symbol.lower()}-{expiry.isoformat()}-{strike_str}{suffix}"


def _default_multileg_id(symbol: str, legs: list[Leg]) -> str:
    sorted_legs = sorted(legs, key=lambda l: (l.expiry, l.option_type, l.strike, l.side))
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
