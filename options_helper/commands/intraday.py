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


_PULL_OPTIONS_UNDERLYING_OPT = typer.Option(
    None,
    "--underlying",
    "-u",
    help="Underlying symbol to pull (repeatable or comma-separated).",
)
_PULL_OPTIONS_UNDERLYINGS_OPT = typer.Option(
    None,
    "--underlyings",
    help="Comma-separated underlyings to pull.",
)
_PULL_OPTIONS_EXPIRY_OPT = typer.Option(
    None,
    "--expiry",
    help="Filter to expiry date(s) (repeatable or comma-separated).",
)
_PULL_OPTIONS_EXPIRIES_OPT = typer.Option(
    None,
    "--expiries",
    help="Comma-separated expiries (YYYY-MM-DD).",
)
_PULL_OPTIONS_CONTRACTS_DIR_OPT = typer.Option(
    Path("data/option_contracts"),
    "--contracts-dir",
    help="Directory containing option contracts cache.",
)
_PULL_OPTIONS_CONTRACTS_AS_OF_OPT = typer.Option(
    "latest",
    "--contracts-as-of",
    help="Contract cache date to use (YYYY-MM-DD or 'latest').",
)
_PULL_OPTIONS_DAY_OPT = typer.Option(date.today().isoformat(), "--day", help="Trading day to pull (YYYY-MM-DD).")
_PULL_OPTIONS_TIMEFRAME_OPT = typer.Option("1Min", "--timeframe", help="Intraday timeframe (1Min or 5Min).")
_PULL_OPTIONS_FEED_OPT = typer.Option(
    None,
    "--feed",
    help="Alpaca options feed override (defaults to OH_ALPACA_OPTIONS_FEED).",
)
_PULL_OPTIONS_OUT_DIR_OPT = typer.Option(
    Path("data/intraday"),
    "--out-dir",
    help="Base directory for intraday partitions.",
)


@app.command("pull-options-bars")
def pull_options_bars(
    underlying: list[str] | None = _PULL_OPTIONS_UNDERLYING_OPT,
    underlyings: str | None = _PULL_OPTIONS_UNDERLYINGS_OPT,
    expiry: list[str] | None = _PULL_OPTIONS_EXPIRY_OPT,
    expiries: str | None = _PULL_OPTIONS_EXPIRIES_OPT,
    contracts_dir: Path = _PULL_OPTIONS_CONTRACTS_DIR_OPT,
    contracts_as_of: str = _PULL_OPTIONS_CONTRACTS_AS_OF_OPT,
    day: str = _PULL_OPTIONS_DAY_OPT,
    timeframe: str = _PULL_OPTIONS_TIMEFRAME_OPT,
    feed: str | None = _PULL_OPTIONS_FEED_OPT,
    out_dir: Path = _PULL_OPTIONS_OUT_DIR_OPT,
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
        prepared = _prepare_option_contract_symbols(
            console=console,
            symbol=sym,
            contracts_dir=contracts_dir,
            contracts_as_of=contracts_as_of,
            contracts_store=contracts_store,
            expiry_dates=expiry_dates,
        )
        if prepared is None:
            failures += 1
            continue
        as_of, contract_symbols = prepared
        bars_df = _fetch_intraday_option_bars(
            console=console,
            client=client,
            symbol=sym,
            contract_symbols=contract_symbols,
            target_day=target_day,
            timeframe=timeframe,
            feed=feed,
        )
        if bars_df is None:
            failures += 1
            continue

        written = _write_option_bar_partitions(
            console=console,
            store=store,
            bars_df=bars_df,
            contract_symbols=contract_symbols,
            symbol=sym,
            as_of=as_of,
            target_day=target_day,
            timeframe=timeframe,
            feed=feed,
            provider_version=client.provider_version,
            options_feed=client.options_feed,
        )
        if written == 0:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no matching intraday bars to save.")

    if failures:
        console.print(f"[yellow]{failures} warning(s) encountered.[/yellow]")


def _prepare_option_contract_symbols(
    *,
    console: Console,
    symbol: str,
    contracts_dir: Path,
    contracts_as_of: str,
    contracts_store: OptionContractsStore,
    expiry_dates: list[date],
) -> tuple[date, list[str]] | None:
    as_of = _resolve_contracts_as_of(contracts_dir, symbol, contracts_as_of)
    if as_of is None:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: no contracts cache found.")
        return None

    try:
        contracts_df = contracts_store.load(symbol, as_of)
    except OptionContractsStoreError as exc:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: {exc}")
        return None
    if contracts_df is None or contracts_df.empty:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: contracts cache empty.")
        return None

    filtered = _filter_contracts_by_expiry(contracts_df, expiry_dates=expiry_dates)
    if filtered.empty:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: no contracts for requested expiry.")
        return None

    contract_symbols = _extract_contract_symbols(filtered)
    if not contract_symbols:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: missing contract symbols in cache.")
        return None
    return as_of, contract_symbols


def _filter_contracts_by_expiry(contracts_df: pd.DataFrame, *, expiry_dates: list[date]) -> pd.DataFrame:
    filtered = contracts_df
    if expiry_dates and "expiry" in filtered.columns:
        allowed = {exp.isoformat() for exp in expiry_dates}
        filtered = filtered[filtered["expiry"].astype(str).isin(allowed)]
    return filtered


def _extract_contract_symbols(contracts_df: pd.DataFrame) -> list[str]:
    return [
        str(raw).strip()
        for raw in contracts_df.get("contractSymbol", pd.Series(dtype=str)).dropna().tolist()
        if str(raw).strip()
    ]


def _fetch_intraday_option_bars(
    *,
    console: Console,
    client: AlpacaClient,
    symbol: str,
    contract_symbols: list[str],
    target_day: date,
    timeframe: str,
    feed: str | None,
) -> pd.DataFrame | None:
    try:
        bars_df = client.get_option_bars_intraday(
            contract_symbols,
            day=target_day,
            timeframe=timeframe,
            feed=feed,
        )
    except DataFetchError as exc:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: {exc}")
        return None
    if bars_df.empty:
        console.print(f"[yellow]Warning:[/yellow] {symbol}: no intraday bars returned.")
        return None
    return bars_df


def _write_option_bar_partitions(
    *,
    console: Console,
    store: IntradayStore,
    bars_df: pd.DataFrame,
    contract_symbols: list[str],
    symbol: str,
    as_of: date,
    target_day: date,
    timeframe: str,
    feed: str | None,
    provider_version: str,
    options_feed: str,
) -> int:
    groups = _group_option_bars_by_contract(bars_df)
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
            "provider_version": provider_version,
            "underlying": symbol,
            "contracts_as_of": as_of.isoformat(),
            "request": {
                "day": target_day.isoformat(),
                "timeframe": timeframe,
                "feed": feed or options_feed,
            },
        }
        path = store.save_partition("options", "bars", timeframe, contract, target_day, sub, meta)
        written += 1
        console.print(f"[green]Saved[/green] {contract} -> {path}")
    return written


def _group_option_bars_by_contract(bars_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    try:
        return {str(contract_symbol): sub for contract_symbol, sub in bars_df.groupby("contractSymbol", sort=False)}
    except Exception:  # noqa: BLE001
        return {}


_FLOW_UNDERLYING_OPT = typer.Option(
    None,
    "--underlying",
    "-u",
    help="Underlying symbol to summarize (repeatable or comma-separated).",
)
_FLOW_UNDERLYINGS_OPT = typer.Option(None, "--underlyings", help="Comma-separated underlyings to summarize.")
_FLOW_CONTRACT_OPT = typer.Option(
    None,
    "--contract",
    "-c",
    help="Contract symbol to summarize (repeatable or comma-separated).",
)
_FLOW_CONTRACTS_OPT = typer.Option(None, "--contracts", help="Comma-separated contract symbols to summarize.")
_FLOW_DAY_OPT = typer.Option(
    "latest",
    "--day",
    help="Market day to summarize (YYYY-MM-DD) or 'latest' from local partitions.",
)
_FLOW_TIMEFRAME_OPT = typer.Option("tick", "--timeframe", help="Intraday partition timeframe (default: tick).")
_FLOW_BUCKET_MINUTES_OPT = typer.Option(5, "--bucket-minutes", help="UTC time bucket size in minutes (5 or 15).")
_FLOW_SOURCE_OPT = typer.Option("offline_intraday", "--source", help="Source label stored in output rows.")
_FLOW_OUT_DIR_OPT = typer.Option(
    Path("data/intraday"),
    "--out-dir",
    help="Directory containing captured intraday partitions.",
)
_FLOW_RESEARCH_DIR_OPT = typer.Option(
    Path("data/research_metrics"),
    "--research-dir",
    help="Research metrics root used for DuckDB persistence wiring.",
)
_FLOW_PERSIST_OPT = typer.Option(
    True,
    "--persist/--no-persist",
    help="Persist contract-flow rows to DuckDB table when --storage duckdb.",
)
_FLOW_FORMAT_OPT = typer.Option("console", "--format", help="Output format: console|json")
_FLOW_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Output root for artifacts (writes under {out}/intraday_flow/{SYMBOL}/).",
)


@app.command("flow")
def flow(
    underlying: list[str] | None = _FLOW_UNDERLYING_OPT,
    underlyings: str | None = _FLOW_UNDERLYINGS_OPT,
    contract: list[str] | None = _FLOW_CONTRACT_OPT,
    contracts: str | None = _FLOW_CONTRACTS_OPT,
    day: str = _FLOW_DAY_OPT,
    timeframe: str = _FLOW_TIMEFRAME_OPT,
    bucket_minutes: int = _FLOW_BUCKET_MINUTES_OPT,
    source: str = _FLOW_SOURCE_OPT,
    out_dir: Path = _FLOW_OUT_DIR_OPT,
    research_dir: Path = _FLOW_RESEARCH_DIR_OPT,
    persist: bool = _FLOW_PERSIST_OPT,
    format: str = _FLOW_FORMAT_OPT,
    out: Path | None = _FLOW_OUT_OPT,
) -> None:
    """Summarize captured options trades/quotes into intraday flow artifacts (offline-only)."""
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)

    underlying_filter, contract_filter = _resolve_flow_filters(
        underlying=underlying,
        underlyings=underlyings,
        contract=contract,
        contracts=contracts,
    )
    store = IntradayStore(out_dir)
    try:
        contracts_to_read, target_day = _resolve_flow_contracts(
            store,
            underlying_filter=underlying_filter,
            contract_filter=contract_filter,
            day_spec=day,
            timeframe=timeframe,
        )
        trades_df, quotes_df, missing_contracts = _load_flow_frames(
            store,
            contracts_to_read=contracts_to_read,
            timeframe=timeframe,
            target_day=target_day,
        )
    except ValueError as exc:
        console.print(str(exc))
        raise typer.Exit(1) from exc
    classified = classify_intraday_trades(trades_df, quotes_df)
    contract_flow_df = summarize_intraday_contract_flow(classified, source=source)
    time_bucket_df = summarize_intraday_time_buckets(classified, bucket_minutes=bucket_minutes)
    artifact = _build_intraday_flow_artifact(
        target_day=target_day,
        source=source,
        bucket_minutes=bucket_minutes,
        underlying_filter=underlying_filter,
        contracts_to_read=contracts_to_read,
        missing_contracts=missing_contracts,
        contract_flow_df=contract_flow_df,
        time_bucket_df=time_bucket_df,
    )
    _render_intraday_flow_output(console, artifact=artifact, output_fmt=output_fmt)
    _persist_intraday_flow_outputs(
        console,
        artifact=artifact,
        output_fmt=output_fmt,
        out=out,
        persist=persist,
        contract_flow_df=contract_flow_df,
        research_dir=research_dir,
    )


def _resolve_flow_filters(
    *,
    underlying: list[str] | None,
    underlyings: str | None,
    contract: list[str] | None,
    contracts: str | None,
) -> tuple[list[str], list[str]]:
    underlying_filter = _parse_symbols(underlying, underlyings)
    contract_filter = _parse_contracts(contract, contracts)
    if not underlying_filter and not contract_filter:
        raise typer.BadParameter("Provide at least one --underlying/--underlyings or --contract/--contracts.")
    return underlying_filter, contract_filter


def _resolve_flow_contracts(
    store: IntradayStore,
    *,
    underlying_filter: list[str],
    contract_filter: list[str],
    day_spec: str,
    timeframe: str,
) -> tuple[list[str], date]:
    candidate_contracts = _collect_candidate_contracts(
        store,
        underlying_filter=underlying_filter,
        contract_filter=contract_filter,
        timeframe=timeframe,
    )
    if not candidate_contracts:
        raise ValueError("No matching option contract partitions found in local intraday data.")
    target_day = _resolve_flow_day(
        store,
        day_spec=day_spec,
        timeframe=timeframe,
        contract_symbols=sorted(candidate_contracts),
    )
    if underlying_filter:
        candidate_contracts = _filter_flow_contracts_for_day(
            store,
            underlying_filter=underlying_filter,
            contract_filter=contract_filter,
            timeframe=timeframe,
            target_day=target_day,
        )
    contracts_to_read = sorted(candidate_contracts)
    if not contracts_to_read:
        raise ValueError(f"No matching contracts found for day {target_day.isoformat()}.")
    return contracts_to_read, target_day


def _collect_candidate_contracts(
    store: IntradayStore,
    *,
    underlying_filter: list[str],
    contract_filter: list[str],
    timeframe: str,
) -> set[str]:
    candidate_contracts: set[str] = set(contract_filter)
    if underlying_filter:
        candidate_contracts.update(_contracts_for_underlyings(store, underlyings=underlying_filter, timeframe=timeframe))
    return candidate_contracts


def _filter_flow_contracts_for_day(
    store: IntradayStore,
    *,
    underlying_filter: list[str],
    contract_filter: list[str],
    timeframe: str,
    target_day: date,
) -> set[str]:
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
    return filtered_for_day


def _load_flow_frames(
    store: IntradayStore,
    *,
    contracts_to_read: list[str],
    timeframe: str,
    target_day: date,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
            trade_frames.append(_prepare_option_partition_rows(trades, contract_symbol=contract_symbol, parsed_contract=parsed))
        if not quotes.empty:
            quote_frames.append(_prepare_option_partition_rows(quotes, contract_symbol=contract_symbol, parsed_contract=parsed))
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    quotes_df = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
    if trades_df.empty and quotes_df.empty:
        raise ValueError("No trade/quote rows available for the requested filters/day.")
    return trades_df, quotes_df, missing_contracts


def _build_intraday_flow_artifact(
    *,
    target_day: date,
    source: str,
    bucket_minutes: int,
    underlying_filter: list[str],
    contracts_to_read: list[str],
    missing_contracts: list[str],
    contract_flow_df: pd.DataFrame,
    time_bucket_df: pd.DataFrame,
) -> IntradayFlowArtifact:
    symbol_value, symbol_warning = _resolve_flow_symbol(underlying_filter, contracts_to_read)
    warnings = _build_intraday_flow_warnings(
        symbol_warning=symbol_warning,
        missing_contracts=missing_contracts,
        contract_flow_df=contract_flow_df,
        time_bucket_df=time_bucket_df,
    )
    return IntradayFlowArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=target_day.isoformat(),
        symbol=symbol_value,
        market_date=target_day.isoformat(),
        source=source,
        bucket_minutes=bucket_minutes,
        disclaimer="Not financial advice.",
        contract_flow=[IntradayFlowContractRow.model_validate(row) for row in _records(contract_flow_df)],
        time_buckets=[IntradayFlowTimeBucketRow.model_validate(row) for row in _records(time_bucket_df)],
        warnings=warnings,
    )


def _build_intraday_flow_warnings(
    *,
    symbol_warning: str | None,
    missing_contracts: list[str],
    contract_flow_df: pd.DataFrame,
    time_bucket_df: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    if symbol_warning:
        warnings.append(symbol_warning)
    if missing_contracts:
        warnings.append("missing_contract_partitions")
    if contract_flow_df.empty:
        warnings.append("empty_contract_flow")
    if time_bucket_df.empty:
        warnings.append("empty_time_buckets")
    return _dedupe(warnings)


def _render_intraday_flow_output(console: Console, *, artifact: IntradayFlowArtifact, output_fmt: str) -> None:
    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
        return
    _render_flow_console(console, artifact)


def _persist_intraday_flow_outputs(
    console: Console,
    *,
    artifact: IntradayFlowArtifact,
    output_fmt: str,
    out: Path | None,
    persist: bool,
    contract_flow_df: pd.DataFrame,
    research_dir: Path,
) -> None:
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
