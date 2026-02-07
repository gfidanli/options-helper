from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

import options_helper.cli_deps as cli_deps
from options_helper.analysis.intraday_flow import (
    classify_intraday_trades,
    summarize_intraday_contract_flow,
    summarize_intraday_time_buckets,
)
from options_helper.analysis.osi import normalize_underlying, parse_contract_symbol
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.market_types import DataFetchError
from options_helper.data.option_contracts import OptionContractsStore, OptionContractsStoreError
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.storage_runtime import get_storage_runtime_config
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.intraday_flow import (
    IntradayFlowArtifact,
    IntradayFlowContractRow,
    IntradayFlowTimeBucketRow,
)


app = typer.Typer(help="Intraday data capture (not financial advice).")


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


def _parse_symbols(symbols: list[str] | None, symbols_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in symbols or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if symbols_csv:
        tokens.extend([part for part in symbols_csv.split(",") if part.strip()])
    normalized = [normalize_underlying(tok) for tok in tokens if tok]
    return sorted({tok for tok in normalized if tok})


def _parse_contracts(contracts: list[str] | None, contracts_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in contracts or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if contracts_csv:
        tokens.extend([part for part in contracts_csv.split(",") if part.strip()])
    normalized = [str(tok).strip().upper() for tok in tokens if str(tok).strip()]
    return sorted({tok for tok in normalized if tok})


def _normalize_output_format(value: str) -> str:
    output_fmt = str(value or "").strip().lower()
    if output_fmt not in {"console", "json"}:
        raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")
    return output_fmt


def _latest_contracts_date(contracts_dir: Path, symbol: str) -> date | None:
    sym_dir = contracts_dir / symbol.upper()
    if not sym_dir.exists():
        return None
    dates: list[date] = []
    for entry in sym_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            dates.append(date.fromisoformat(entry.name))
        except ValueError:
            continue
    return max(dates) if dates else None


def _resolve_contracts_as_of(contracts_dir: Path, symbol: str, spec: str) -> date | None:
    spec = spec.strip().lower()
    if spec == "latest":
        return _latest_contracts_date(contracts_dir, symbol)
    return _parse_date(spec)


def _require_alpaca_provider() -> None:
    provider = cli_deps.build_provider()
    name = getattr(provider, "name", None)
    if name != "alpaca":
        raise typer.BadParameter("Intraday capture currently requires --provider alpaca.")


@app.command("pull-stocks-bars")
def pull_stocks_bars(
    symbol: list[str] | None = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Stock symbol to pull (repeatable or comma-separated).",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated stock symbols to pull.",
    ),
    day: str = typer.Option(
        date.today().isoformat(),
        "--day",
        help="Trading day to pull (YYYY-MM-DD).",
    ),
    timeframe: str = typer.Option(
        "1Min",
        "--timeframe",
        help="Intraday timeframe (1Min or 5Min).",
    ),
    feed: str | None = typer.Option(
        None,
        "--feed",
        help="Alpaca stock feed override (defaults to OH_ALPACA_STOCK_FEED).",
    ),
    out_dir: Path = typer.Option(
        Path("data/intraday"),
        "--out-dir",
        help="Base directory for intraday partitions.",
    ),
) -> None:
    """Pull intraday stock bars and persist CSV.gz partitions."""
    _require_alpaca_provider()
    console = Console()

    symbols_list = _parse_symbols(symbol, symbols)
    if not symbols_list:
        raise typer.BadParameter("Provide at least one --symbol or --symbols.")

    target_day = _parse_date(day)
    client = AlpacaClient()
    store = IntradayStore(out_dir)

    failures = 0
    for sym in symbols_list:
        try:
            df = client.get_stock_bars_intraday(
                sym,
                day=target_day,
                timeframe=timeframe,
                feed=feed,
            )
            meta = {
                "provider": "alpaca",
                "provider_version": client.provider_version,
                "request": {
                    "day": target_day.isoformat(),
                    "timeframe": timeframe,
                    "feed": feed or client.stock_feed,
                },
            }
            path = store.save_partition("stocks", "bars", timeframe, sym, target_day, df, meta)
            console.print(f"[green]Saved[/green] {sym} -> {path}")
        except DataFetchError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")

    if failures:
        console.print(f"[yellow]{failures} symbol(s) failed.[/yellow]")


@app.command("pull-options-bars")
def pull_options_bars(
    underlying: list[str] | None = typer.Option(
        None,
        "--underlying",
        "-u",
        help="Underlying symbol to pull (repeatable or comma-separated).",
    ),
    underlyings: str | None = typer.Option(
        None,
        "--underlyings",
        help="Comma-separated underlyings to pull.",
    ),
    expiry: list[str] | None = typer.Option(
        None,
        "--expiry",
        help="Filter to expiry date(s) (repeatable or comma-separated).",
    ),
    expiries: str | None = typer.Option(
        None,
        "--expiries",
        help="Comma-separated expiries (YYYY-MM-DD).",
    ),
    contracts_dir: Path = typer.Option(
        Path("data/option_contracts"),
        "--contracts-dir",
        help="Directory containing option contracts cache.",
    ),
    contracts_as_of: str = typer.Option(
        "latest",
        "--contracts-as-of",
        help="Contract cache date to use (YYYY-MM-DD or 'latest').",
    ),
    day: str = typer.Option(
        date.today().isoformat(),
        "--day",
        help="Trading day to pull (YYYY-MM-DD).",
    ),
    timeframe: str = typer.Option(
        "1Min",
        "--timeframe",
        help="Intraday timeframe (1Min or 5Min).",
    ),
    feed: str | None = typer.Option(
        None,
        "--feed",
        help="Alpaca options feed override (defaults to OH_ALPACA_OPTIONS_FEED).",
    ),
    out_dir: Path = typer.Option(
        Path("data/intraday"),
        "--out-dir",
        help="Base directory for intraday partitions.",
    ),
) -> None:
    """Pull intraday option bars for cached contracts and persist CSV.gz partitions."""
    _require_alpaca_provider()
    console = Console()

    underlyings_list = _parse_symbols(underlying, underlyings)
    if not underlyings_list:
        raise typer.BadParameter("Provide at least one --underlying or --underlyings.")

    expiry_dates = _parse_expiries(expiry, expiries)
    target_day = _parse_date(day)
    contracts_store = OptionContractsStore(contracts_dir)
    client = AlpacaClient()
    store = IntradayStore(out_dir)

    failures = 0
    for sym in underlyings_list:
        as_of = _resolve_contracts_as_of(contracts_dir, sym, contracts_as_of)
        if as_of is None:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no contracts cache found.")
            continue

        try:
            contracts_df = contracts_store.load(sym, as_of)
        except OptionContractsStoreError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")
            continue

        if contracts_df is None or contracts_df.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: contracts cache empty.")
            continue

        filtered = contracts_df
        if expiry_dates and "expiry" in filtered.columns:
            allowed = {exp.isoformat() for exp in expiry_dates}
            filtered = filtered[filtered["expiry"].astype(str).isin(allowed)]

        if filtered.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no contracts for requested expiry.")
            continue

        contract_symbols = [
            str(raw).strip()
            for raw in filtered.get("contractSymbol", pd.Series(dtype=str)).dropna().tolist()
            if str(raw).strip()
        ]
        if not contract_symbols:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: missing contract symbols in cache.")
            continue

        try:
            bars_df = client.get_option_bars_intraday(
                contract_symbols,
                day=target_day,
                timeframe=timeframe,
                feed=feed,
            )
        except DataFetchError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")
            continue

        if bars_df.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no intraday bars returned.")
            continue

        groups: dict[str, pd.DataFrame] = {}
        try:
            for contract_symbol, sub in bars_df.groupby("contractSymbol", sort=False):
                groups[str(contract_symbol)] = sub
        except Exception:  # noqa: BLE001
            groups = {}

        written = 0
        seen_contracts: set[str] = set()
        for contract in contract_symbols:
            if contract in seen_contracts:
                continue
            seen_contracts.add(contract)

            sub = groups.get(contract)
            if sub is None:
                sub = bars_df[bars_df["contractSymbol"] == contract]
            if sub.empty:
                continue
            meta = {
                "provider": "alpaca",
                "provider_version": client.provider_version,
                "underlying": sym,
                "contracts_as_of": as_of.isoformat(),
                "request": {
                    "day": target_day.isoformat(),
                    "timeframe": timeframe,
                    "feed": feed or client.options_feed,
                },
            }
            path = store.save_partition("options", "bars", timeframe, contract, target_day, sub, meta)
            written += 1
            console.print(f"[green]Saved[/green] {contract} -> {path}")

        if written == 0:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no matching intraday bars to save.")

    if failures:
        console.print(f"[yellow]{failures} warning(s) encountered.[/yellow]")


@app.command("flow")
def flow(
    underlying: list[str] | None = typer.Option(
        None,
        "--underlying",
        "-u",
        help="Underlying symbol to summarize (repeatable or comma-separated).",
    ),
    underlyings: str | None = typer.Option(
        None,
        "--underlyings",
        help="Comma-separated underlyings to summarize.",
    ),
    contract: list[str] | None = typer.Option(
        None,
        "--contract",
        "-c",
        help="Contract symbol to summarize (repeatable or comma-separated).",
    ),
    contracts: str | None = typer.Option(
        None,
        "--contracts",
        help="Comma-separated contract symbols to summarize.",
    ),
    day: str = typer.Option(
        "latest",
        "--day",
        help="Market day to summarize (YYYY-MM-DD) or 'latest' from local partitions.",
    ),
    timeframe: str = typer.Option(
        "tick",
        "--timeframe",
        help="Intraday partition timeframe (default: tick).",
    ),
    bucket_minutes: int = typer.Option(
        5,
        "--bucket-minutes",
        help="UTC time bucket size in minutes (5 or 15).",
    ),
    source: str = typer.Option(
        "offline_intraday",
        "--source",
        help="Source label stored in output rows.",
    ),
    out_dir: Path = typer.Option(
        Path("data/intraday"),
        "--out-dir",
        help="Directory containing captured intraday partitions.",
    ),
    research_dir: Path = typer.Option(
        Path("data/research_metrics"),
        "--research-dir",
        help="Research metrics root used for DuckDB persistence wiring.",
    ),
    persist: bool = typer.Option(
        True,
        "--persist/--no-persist",
        help="Persist contract-flow rows to DuckDB table when --storage duckdb.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for artifacts (writes under {out}/intraday_flow/{SYMBOL}/).",
    ),
) -> None:
    """Summarize captured options trades/quotes into intraday flow artifacts (offline-only)."""
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)

    underlying_filter = _parse_symbols(underlying, underlyings)
    contract_filter = _parse_contracts(contract, contracts)
    if not underlying_filter and not contract_filter:
        raise typer.BadParameter("Provide at least one --underlying/--underlyings or --contract/--contracts.")

    store = IntradayStore(out_dir)
    candidate_contracts: set[str] = set(contract_filter)
    if underlying_filter:
        candidate_contracts.update(
            _contracts_for_underlyings(store, underlyings=underlying_filter, timeframe=timeframe)
        )

    if not candidate_contracts:
        console.print("No matching option contract partitions found in local intraday data.")
        raise typer.Exit(1)

    target_day = _resolve_flow_day(
        store,
        day_spec=day,
        timeframe=timeframe,
        contract_symbols=sorted(candidate_contracts),
    )

    if underlying_filter:
        filtered_for_day = set(
            _contracts_for_underlyings(
                store,
                underlyings=underlying_filter,
                timeframe=timeframe,
                day=target_day,
            )
        )
        if contract_filter:
            filtered_for_day.update(contract_filter)
        candidate_contracts = filtered_for_day

    contracts_to_read = sorted(candidate_contracts)
    if not contracts_to_read:
        console.print(f"No matching contracts found for day {target_day.isoformat()}.")
        raise typer.Exit(1)

    trade_frames: list[pd.DataFrame] = []
    quote_frames: list[pd.DataFrame] = []
    missing_contracts: list[str] = []

    for contract_symbol in contracts_to_read:
        parsed = parse_contract_symbol(contract_symbol)
        trades = store.load_partition("options", "trades", timeframe, contract_symbol, target_day)
        quotes = store.load_partition("options", "quotes", timeframe, contract_symbol, target_day)

        if trades.empty and quotes.empty:
            missing_contracts.append(contract_symbol)
            continue

        if not trades.empty:
            trade_frames.append(
                _prepare_option_partition_rows(
                    trades,
                    contract_symbol=contract_symbol,
                    parsed_contract=parsed,
                )
            )
        if not quotes.empty:
            quote_frames.append(
                _prepare_option_partition_rows(
                    quotes,
                    contract_symbol=contract_symbol,
                    parsed_contract=parsed,
                )
            )

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    quotes_df = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()

    if trades_df.empty and quotes_df.empty:
        console.print("No trade/quote rows available for the requested filters/day.")
        raise typer.Exit(1)

    classified = classify_intraday_trades(trades_df, quotes_df)
    contract_flow_df = summarize_intraday_contract_flow(classified, source=source)
    time_bucket_df = summarize_intraday_time_buckets(classified, bucket_minutes=bucket_minutes)

    symbol_value, symbol_warning = _resolve_flow_symbol(underlying_filter, contracts_to_read)
    warnings: list[str] = []
    if symbol_warning:
        warnings.append(symbol_warning)
    if missing_contracts:
        warnings.append("missing_contract_partitions")
    if contract_flow_df.empty:
        warnings.append("empty_contract_flow")
    if time_bucket_df.empty:
        warnings.append("empty_time_buckets")
    warnings = _dedupe(warnings)

    artifact = IntradayFlowArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=target_day.isoformat(),
        symbol=symbol_value,
        market_date=target_day.isoformat(),
        source=source,
        bucket_minutes=bucket_minutes,
        disclaimer="Not financial advice.",
        contract_flow=[
            IntradayFlowContractRow.model_validate(row) for row in _records(contract_flow_df)
        ],
        time_buckets=[
            IntradayFlowTimeBucketRow.model_validate(row) for row in _records(time_bucket_df)
        ],
        warnings=warnings,
    )

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_flow_console(console, artifact)

    if out is not None:
        saved = _save_flow_artifact(out, artifact)
        if output_fmt != "json":
            console.print(f"Saved: {saved}")

    if persist and get_storage_runtime_config().backend == "duckdb":
        persisted = _persist_intraday_flow(contract_flow_df, research_dir)
        if output_fmt != "json":
            console.print(
                "Persisted intraday flow rows "
                f"(provider={get_default_provider_name()}): contracts={persisted}"
            )


def _prepare_option_partition_rows(
    df: pd.DataFrame,
    *,
    contract_symbol: str,
    parsed_contract: Any,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "contractSymbol" not in out.columns:
        out["contractSymbol"] = contract_symbol
    else:
        out["contractSymbol"] = out["contractSymbol"].where(out["contractSymbol"].notna(), contract_symbol)

    if parsed_contract is not None:
        if "underlying" not in out.columns:
            out["underlying"] = parsed_contract.underlying_norm
        if "expiry" not in out.columns:
            out["expiry"] = parsed_contract.expiry.isoformat()
        if "optionType" not in out.columns:
            out["optionType"] = parsed_contract.option_type
        if "strike" not in out.columns:
            out["strike"] = float(parsed_contract.strike)

    return out


def _contracts_for_underlyings(
    store: IntradayStore,
    *,
    underlyings: list[str],
    timeframe: str,
    day: date | None = None,
) -> list[str]:
    wanted = {normalize_underlying(symbol) for symbol in underlyings if symbol}
    if not wanted:
        return []

    contract_universe = set(store.list_symbols("options", "trades", timeframe))
    contract_universe.update(store.list_symbols("options", "quotes", timeframe))

    matches: list[str] = []
    for contract in sorted(contract_universe):
        parsed = parse_contract_symbol(contract)
        if parsed is None or parsed.underlying_norm not in wanted:
            continue
        if day is not None:
            has_trade = (store.partition_path("options", "trades", timeframe, contract, day)).exists()
            has_quote = (store.partition_path("options", "quotes", timeframe, contract, day)).exists()
            if not (has_trade or has_quote):
                continue
        matches.append(contract)
    return matches


def _resolve_flow_day(
    store: IntradayStore,
    *,
    day_spec: str,
    timeframe: str,
    contract_symbols: list[str],
) -> date:
    spec = str(day_spec or "").strip().lower()
    if spec != "latest":
        return _parse_date(day_spec)

    all_days: list[date] = []
    for contract in contract_symbols:
        all_days.extend(store.list_days("options", "trades", timeframe, contract))
        all_days.extend(store.list_days("options", "quotes", timeframe, contract))

    if not all_days:
        raise typer.BadParameter("No intraday partitions found to resolve --day latest.")

    return max(all_days)


def _resolve_flow_symbol(
    underlying_filter: list[str],
    contracts: list[str],
) -> tuple[str, str | None]:
    if len(underlying_filter) == 1:
        return underlying_filter[0], None

    underlyings: set[str] = set()
    for contract in contracts:
        parsed = parse_contract_symbol(contract)
        if parsed is not None:
            underlyings.add(parsed.underlying_norm)

    if len(underlyings) == 1:
        return next(iter(underlyings)), None
    if len(underlyings) > 1:
        return "MULTI", "multiple_underlyings"
    return "UNKNOWN", "missing_underlying"


def _persist_intraday_flow(contract_flow_df: pd.DataFrame, research_dir: Path) -> int:
    store = cli_deps.build_research_metrics_store(research_dir)
    provider = get_default_provider_name()
    return int(store.upsert_intraday_option_flow(contract_flow_df, provider=provider))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [clean_nan(row) for row in df.to_dict(orient="records")]


def _save_flow_artifact(out_root: Path, artifact: IntradayFlowArtifact) -> Path:
    base = out_root / "intraday_flow" / artifact.symbol
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{artifact.market_date}.json"
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "-"


def _render_flow_console(console: Console, artifact: IntradayFlowArtifact) -> None:
    console.print(
        f"[bold]{artifact.symbol}[/bold] intraday flow {artifact.market_date} "
        f"(source={artifact.source}, bucket={artifact.bucket_minutes}m)"
    )

    contract_table = Table(title="Contract Flow")
    contract_table.add_column("Contract")
    contract_table.add_column("Net Notional", justify="right")
    contract_table.add_column("Buy Notional", justify="right")
    contract_table.add_column("Sell Notional", justify="right")
    contract_table.add_column("Trades", justify="right")
    contract_table.add_column("Unknown %", justify="right")

    ordered_contracts = sorted(
        artifact.contract_flow,
        key=lambda row: (abs(float(row.net_notional)), row.contract_symbol),
        reverse=True,
    )
    for row in ordered_contracts:
        contract_table.add_row(
            row.contract_symbol,
            _fmt_num(row.net_notional),
            _fmt_num(row.buy_notional),
            _fmt_num(row.sell_notional),
            str(row.trade_count),
            _fmt_pct(row.unknown_trade_share),
        )
    console.print(contract_table)

    bucket_table = Table(title="Time Buckets (Top by |Net|)")
    bucket_table.add_column("UTC Bucket")
    bucket_table.add_column("Contract")
    bucket_table.add_column("Net Notional", justify="right")
    bucket_table.add_column("Trades", justify="right")

    ordered_buckets = sorted(
        artifact.time_buckets,
        key=lambda row: (abs(float(row.net_notional)), row.contract_symbol),
        reverse=True,
    )
    for row in ordered_buckets[:20]:
        bucket_table.add_row(
            row.bucket_start_utc.isoformat(),
            row.contract_symbol,
            _fmt_num(row.net_notional),
            str(row.trade_count),
        )
    console.print(bucket_table)

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _parse_expiries(expiries: list[str] | None, expiries_csv: str | None) -> list[date]:
    tokens: list[str] = []
    for item in expiries or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if expiries_csv:
        tokens.extend([part for part in expiries_csv.split(",") if part.strip()])
    parsed = [_parse_date(tok) for tok in tokens]
    return sorted({val for val in parsed if val})
