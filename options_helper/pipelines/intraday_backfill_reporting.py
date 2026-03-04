from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BackfillPaths:
    run_root: Path
    results_jsonl: Path
    failures_csv: Path
    overall_status_json: Path
    current_symbol_json: Path
    symbols_status_dir: Path
    checkpoint_md: Path
    checkpoint_json: Path


def build_paths(*, status_dir: Path, run_id: str) -> BackfillPaths:
    run_root = status_dir / run_id
    return BackfillPaths(
        run_root=run_root,
        results_jsonl=run_root / "results_symbol_month.jsonl",
        failures_csv=run_root / "failures.csv",
        overall_status_json=run_root / "status" / "overall.json",
        current_symbol_json=run_root / "status" / "current_symbol.json",
        symbols_status_dir=run_root / "status" / "symbols",
        checkpoint_md=run_root / "performance_checkpoint_25_symbols.md",
        checkpoint_json=run_root / "performance_checkpoint_25_symbols.json",
    )


def ensure_run_dirs(paths: BackfillPaths) -> None:
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.symbols_status_dir.mkdir(parents=True, exist_ok=True)
    if not paths.failures_csv.exists():
        paths.failures_csv.write_text("symbol,year_month,error_type,error_message\n", encoding="utf-8")


def write_run_config(*, paths: BackfillPaths, payload: dict[str, Any], symbols: list[str]) -> None:
    atomic_write_json(paths.run_root / "run_config.json", payload)
    (paths.run_root / "targets.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")


def initial_totals(*, symbols_total: int) -> dict[str, Any]:
    return {
        "symbols_total": symbols_total,
        "symbols_processed": 0,
        "symbols_failed": 0,
        "symbols_no_data": 0,
        "months_processed": 0,
        "months_ok": 0,
        "months_skipped_existing": 0,
        "months_no_data": 0,
        "months_error": 0,
        "rows_loaded": 0,
        "days_written": 0,
        "days_skipped_existing": 0,
        "fetch_seconds": 0.0,
        "write_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def apply_symbol_summary(*, totals: dict[str, Any], summary: dict[str, Any], month_records: list[dict[str, Any]]) -> None:
    totals["symbols_processed"] += 1
    if summary.get("status") == "error":
        totals["symbols_failed"] += 1
    if summary.get("status") == "no_data":
        totals["symbols_no_data"] += 1
    for rec in month_records:
        totals["months_processed"] += 1
        status = rec.get("status")
        if status == "ok":
            totals["months_ok"] += 1
        elif status == "skipped_existing":
            totals["months_skipped_existing"] += 1
        elif status == "no_data":
            totals["months_no_data"] += 1
        elif status == "error":
            totals["months_error"] += 1
        totals["rows_loaded"] += int(rec.get("rows", 0) or 0)
        totals["days_written"] += int(rec.get("written_days", 0) or 0)
        totals["days_skipped_existing"] += int(rec.get("skipped_days", 0) or 0)
        totals["fetch_seconds"] += float(rec.get("fetch_seconds", 0.0) or 0.0)
        totals["write_seconds"] += float(rec.get("write_seconds", 0.0) or 0.0)


def write_overall_status(*, paths: BackfillPaths, totals: dict[str, Any]) -> None:
    payload = dict(totals)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["throughput_rows_per_sec"] = safe_rate(payload["rows_loaded"], payload["elapsed_seconds"])
    payload["throughput_months_per_min"] = safe_rate(payload["months_processed"], payload["elapsed_seconds"] / 60.0)
    atomic_write_json(paths.overall_status_json, payload)


def write_symbol_status(*, paths: BackfillPaths, symbol: str, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = {
        "symbol": symbol,
        "summary": summary,
        "months": records,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(paths.symbols_status_dir / f"{symbol}.json", payload)


def write_current_symbol_status(*, paths: BackfillPaths, symbol: str, year_month: str, rec: dict[str, Any]) -> None:
    payload = {
        "symbol": symbol,
        "year_month": year_month,
        "status": rec.get("status"),
        "rows": rec.get("rows"),
        "fetch_seconds": rec.get("fetch_seconds"),
        "write_seconds": rec.get("write_seconds"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(paths.current_symbol_json, payload)


def record_failure(
    *,
    paths: BackfillPaths,
    symbol: str,
    year_month: str,
    error: Exception,
    error_type: str | None = None,
) -> None:
    clean = str(error).replace("\n", " ").replace("\r", " ").replace(",", ";").strip()
    row = f"{symbol},{year_month},{error_type or error.__class__.__name__},{clean}\n"
    with paths.failures_csv.open("a", encoding="utf-8") as handle:
        handle.write(row)


def write_checkpoint_reports(
    *,
    paths: BackfillPaths,
    symbol_summaries: list[dict[str, Any]],
    month_records: list[dict[str, Any]],
    totals: dict[str, Any],
) -> None:
    checkpoint = build_checkpoint_payload(
        symbol_summaries=symbol_summaries,
        month_records=month_records,
        totals=totals,
    )
    atomic_write_json(paths.checkpoint_json, checkpoint)
    paths.checkpoint_md.write_text(checkpoint_markdown(checkpoint), encoding="utf-8")


def build_checkpoint_payload(
    *,
    symbol_summaries: list[dict[str, Any]],
    month_records: list[dict[str, Any]],
    totals: dict[str, Any],
) -> dict[str, Any]:
    durations = [float(rec.get("total_seconds", 0.0) or 0.0) for rec in month_records]
    fetch_seconds = sum(float(rec.get("fetch_seconds", 0.0) or 0.0) for rec in month_records)
    write_seconds = sum(float(rec.get("write_seconds", 0.0) or 0.0) for rec in month_records)
    total_seconds = sum(durations)
    status_counts = count_values(rec.get("status") for rec in month_records)
    error_counts = count_values(rec.get("error_type") for rec in month_records if rec.get("status") == "error")
    slowest = sorted(month_records, key=lambda rec: float(rec.get("total_seconds", 0.0) or 0.0), reverse=True)[:10]
    findings = build_bottleneck_findings(
        total_seconds=total_seconds,
        fetch_seconds=fetch_seconds,
        write_seconds=write_seconds,
        status_counts=status_counts,
        error_counts=error_counts,
        durations=durations,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols_processed": len(symbol_summaries),
        "months_observed": len(month_records),
        "status_counts": status_counts,
        "error_type_counts": error_counts,
        "timing": {
            "total_seconds": round(total_seconds, 3),
            "fetch_seconds": round(fetch_seconds, 3),
            "write_seconds": round(write_seconds, 3),
            "p50_month_seconds": round(percentile(durations, 50), 3),
            "p95_month_seconds": round(percentile(durations, 95), 3),
        },
        "totals_snapshot": totals,
        "slowest_symbol_months": slowest,
        "key_bottlenecks": findings,
    }


def checkpoint_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Performance checkpoint (25 symbols)",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Symbols processed: `{payload['symbols_processed']}`",
        f"Symbol-month rows observed: `{payload['months_observed']}`",
        "",
        "## Key bottlenecks",
    ]
    for item in payload.get("key_bottlenecks") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Timing", ""])
    timing = payload.get("timing") or {}
    lines.append(f"- total_seconds: `{timing.get('total_seconds')}`")
    lines.append(f"- fetch_seconds: `{timing.get('fetch_seconds')}`")
    lines.append(f"- write_seconds: `{timing.get('write_seconds')}`")
    lines.append(f"- p50_month_seconds: `{timing.get('p50_month_seconds')}`")
    lines.append(f"- p95_month_seconds: `{timing.get('p95_month_seconds')}`")
    lines.extend(["", "## Slowest symbol-months", ""])
    lines.append("| symbol | year_month | status | total_seconds | rows |")
    lines.append("|---|---|---|---:|---:|")
    for rec in payload.get("slowest_symbol_months") or []:
        lines.append(
            "| {symbol} | {year_month} | {status} | {total_seconds:.3f} | {rows} |".format(
                symbol=rec.get("symbol"),
                year_month=rec.get("year_month"),
                status=rec.get("status"),
                total_seconds=float(rec.get("total_seconds", 0.0) or 0.0),
                rows=int(rec.get("rows", 0) or 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_bottleneck_findings(
    *,
    total_seconds: float,
    fetch_seconds: float,
    write_seconds: float,
    status_counts: dict[str, int],
    error_counts: dict[str, int],
    durations: list[float],
) -> list[str]:
    findings: list[str] = []
    fetch_share = safe_ratio(fetch_seconds, total_seconds)
    write_share = safe_ratio(write_seconds, total_seconds)
    if fetch_share >= 0.65:
        findings.append(f"Network/API fetch dominates runtime ({fetch_share:.1%} of measured time).")
    if write_share >= 0.35:
        findings.append(f"Local write/compression is material ({write_share:.1%} of measured time).")
    if sum(error_counts.values()) > 0:
        top = max(error_counts.items(), key=lambda item: item[1])
        findings.append(f"Error pressure observed: {top[0]} occurred {top[1]} times.")
    no_data = int(status_counts.get("no_data", 0) or 0)
    skipped = int(status_counts.get("skipped_existing", 0) or 0)
    observed = max(1, sum(status_counts.values()))
    if (no_data + skipped) / observed >= 0.5:
        findings.append("A high share of months were empty or already covered; narrow date scope if runtime is critical.")
    if len(durations) >= 5 and percentile(durations, 95) >= (percentile(durations, 50) * 3.0):
        findings.append("Long-tail month latency is high (p95 >= 3x p50); investigate slow symbols and retries.")
    if not findings:
        findings.append("No single dominant bottleneck detected in first checkpoint window.")
    return findings


def count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = max(0, min(99, pct - 1))
    return float(statistics.quantiles(values, n=100, method="inclusive")[idx])


def safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return round(float(num) / float(den), 6)


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
