from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from options_helper.commands.market_analysis.zero_dte_types import _ZeroDTEStudyResult
from options_helper.commands.market_analysis.zero_dte_utils import _as_clean_text
from options_helper.data.zero_dte_dataset import DEFAULT_PROXY_UNDERLYING
from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import ZeroDtePutStudyArtifact


def _upsert_forward_snapshot_records(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> int:
    existing = _read_jsonl_records(path)
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in existing:
        key = _row_key(row, key_fields=key_fields)
        if key is not None:
            by_key[key] = row
    for row in rows:
        key = _row_key(row, key_fields=key_fields)
        if key is not None:
            by_key[key] = clean_nan(row)

    ordered = [by_key[key] for key in sorted(by_key.keys())]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(clean_nan(row), sort_keys=True) + "\n")
    return len(ordered)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _row_key(row: dict[str, Any], *, key_fields: tuple[str, ...]) -> tuple[str, ...] | None:
    values: list[str] = []
    for field in key_fields:
        text = _as_clean_text(row.get(field))
        if text is None:
            return None
        values.append(text)
    return tuple(values)


def _save_zero_dte_study_artifact(out: Path, artifact: ZeroDtePutStudyArtifact, *, symbol: str) -> Path:
    base = out / "zero_dte_put_study" / symbol
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{artifact.as_of.isoformat()}.json"
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _save_zero_dte_active_model(out: Path, *, symbol: str, payload: dict[str, Any]) -> Path:
    path = _default_zero_dte_active_model_path(out, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_nan(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _default_zero_dte_active_model_path(out: Path, symbol: str) -> Path:
    return out / "zero_dte_put_study" / symbol / "active_model.json"


def _default_zero_dte_forward_snapshot_path(out: Path, symbol: str) -> Path:
    return out / "zero_dte_put_study" / symbol / "forward_snapshots.jsonl"


def _render_zero_dte_study_console(console: Console, result: _ZeroDTEStudyResult) -> None:
    artifact = result.artifact
    console.print(
        f"[bold]{artifact.assumptions.proxy_underlying_symbol}[/bold] 0DTE put study as-of {artifact.as_of}"
    )
    console.print(
        "Rows: "
        f"probabilities={len(artifact.probability_rows)}, "
        f"strike_ladders={len(artifact.strike_ladder_rows)}, "
        f"simulations={len(artifact.simulation_rows)}"
    )
    console.print("Preflight: " + ("passed" if result.preflight_passed else "failed (soft)"))
    if result.preflight_messages:
        console.print("Preflight diagnostics: " + "; ".join(result.preflight_messages))
    if artifact.warnings:
        console.print("Warnings: " + "; ".join(artifact.warnings))
    console.print("Proxy notice: " + artifact.disclaimer.spy_proxy_caveat)
    console.print(artifact.disclaimer.not_financial_advice)


def _render_zero_dte_forward_console(
    console: Console,
    *,
    payload: dict[str, Any],
    persisted_rows: int,
    snapshot_path: Path,
) -> None:
    console.print(
        f"[bold]{payload.get('symbol', DEFAULT_PROXY_UNDERLYING)}[/bold] forward snapshot session "
        f"{payload.get('session_date', '-')}"
    )
    console.print(
        "Rows: "
        f"total={payload.get('rows', 0)}, "
        f"pending_close={payload.get('pending_close_rows', 0)}, "
        f"finalized={payload.get('finalized_rows', 0)}"
    )
    console.print(f"Persisted unique rows: {persisted_rows}")
    console.print(f"Snapshot file: {snapshot_path}")
    disclaimer = payload.get("disclaimer") if isinstance(payload.get("disclaimer"), dict) else {}
    proxy = _as_clean_text(disclaimer.get("spy_proxy_caveat"))
    nfa = _as_clean_text(disclaimer.get("not_financial_advice"))
    if proxy:
        console.print("Proxy notice: " + proxy)
    if nfa:
        console.print(nfa)


__all__ = [
    "_upsert_forward_snapshot_records",
    "_read_jsonl_records",
    "_row_key",
    "_save_zero_dte_study_artifact",
    "_save_zero_dte_active_model",
    "_default_zero_dte_active_model_path",
    "_default_zero_dte_forward_snapshot_path",
    "_render_zero_dte_study_console",
    "_render_zero_dte_forward_console",
]
