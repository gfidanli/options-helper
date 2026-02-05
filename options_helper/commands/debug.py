from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console


app = typer.Typer(help="Diagnostics helpers (not financial advice).")


def _latest_log_file(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = [path for path in log_dir.glob("*.log") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_kv_tokens(message: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for token in message.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def _iter_tail_lines(path: Path, *, max_lines: int) -> list[str]:
    out: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            out.append(line.rstrip("\n"))
    return list(out)


@app.command("rate-limits")
def rate_limits(
    provider: str = typer.Option(
        "alpaca",
        "--provider",
        help="Provider name (currently only 'alpaca' supported).",
    ),
    log_dir: Path = typer.Option(
        Path("data/logs"),
        "--log-dir",
        help="Directory containing CLI logs.",
    ),
    log_path: Path | None = typer.Option(
        None,
        "--log-path",
        help="Specific log file to inspect (overrides --log-dir).",
    ),
    tail: int = typer.Option(
        50,
        "--tail",
        help="How many recent ratelimit lines to show.",
    ),
    scan_lines: int = typer.Option(
        5000,
        "--scan-lines",
        help="How many recent log lines to scan for ratelimit info.",
    ),
) -> None:
    """Show recent Alpaca rate-limit header snapshots recorded in logs."""
    console = Console()
    if provider.strip().lower() != "alpaca":
        console.print("[red]Error:[/red] only --provider alpaca is supported for now.")
        raise typer.Exit(2)

    path = log_path or _latest_log_file(log_dir)
    if path is None:
        console.print("[red]Error:[/red] no log files found.")
        console.print("Tip: run any command with OH_ALPACA_LOG_RATE_LIMITS=1 to emit ratelimit lines.")
        raise typer.Exit(2)

    lines = _iter_tail_lines(path, max_lines=max(scan_lines, tail, 1))
    matched: list[dict[str, Any]] = []
    for line in lines:
        if "ALPACA_RATELIMIT" not in line:
            continue
        try:
            message = line.split(": ", 1)[1]
        except IndexError:
            message = line
        if "ALPACA_RATELIMIT" in message:
            message = message.split("ALPACA_RATELIMIT", 1)[1].strip()
        payload = _parse_kv_tokens(message)
        if payload:
            matched.append(payload)

    if not matched:
        console.print(f"[yellow]No ratelimit lines found in[/yellow] {path}")
        console.print("Enable with: `OH_ALPACA_LOG_RATE_LIMITS=1 options-helper ...`")
        raise typer.Exit(1)

    last = matched[-1]
    remaining = last.get("remaining")
    limit = last.get("limit")
    reset_epoch = last.get("reset_epoch")
    reset_in_s = last.get("reset_in_s")

    reset_text = "?"
    try:
        if reset_epoch and str(reset_epoch).isdigit():
            reset_dt = datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
            reset_text = reset_dt.isoformat()
    except Exception:  # noqa: BLE001
        reset_text = "?"

    console.print(f"[green]Log:[/green] {path}")
    console.print(
        f"[green]Last:[/green] remaining={remaining} limit={limit} reset_at={reset_text} reset_in_s={reset_in_s}"
    )
    console.print(f"[green]Recent:[/green] showing last {min(tail, len(matched))} snapshots")
    for item in matched[-tail:]:
        console.print(
            "client={client} method={method} path={path} status={status} remaining={remaining} limit={limit} reset_epoch={reset_epoch} reset_in_s={reset_in_s}".format(
                client=item.get("client", "?"),
                method=item.get("method", "?"),
                path=item.get("path", "?"),
                status=item.get("status", "?"),
                remaining=item.get("remaining", "?"),
                limit=item.get("limit", "?"),
                reset_epoch=item.get("reset_epoch", "?"),
                reset_in_s=item.get("reset_in_s", "?"),
            )
        )

