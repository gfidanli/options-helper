from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ArtifactPaths:
    params_path: Path
    report_path: Path
    heatmap_path: Path | None


def _render_path(template: str, *, base_dir: Path, ticker: str, strategy: str) -> Path:
    rel = template.format(ticker=ticker.upper(), strategy=strategy)
    return base_dir / rel


def build_artifact_paths(cfg: dict, *, ticker: str, strategy: str) -> ArtifactPaths:
    base_dir = Path(cfg["artifacts"]["base_dir"])
    params_path = _render_path(cfg["artifacts"]["params_path_template"], base_dir=base_dir, ticker=ticker, strategy=strategy)
    report_path = _render_path(cfg["artifacts"]["report_path_template"], base_dir=base_dir, ticker=ticker, strategy=strategy)
    heatmap_path = None
    if cfg["artifacts"].get("write_heatmap", False):
        heatmap_path = _render_path(
            cfg["artifacts"]["heatmap_path_template"],
            base_dir=base_dir,
            ticker=ticker,
            strategy=strategy,
        )
    return ArtifactPaths(params_path=params_path, report_path=report_path, heatmap_path=heatmap_path)


def _serialize_stats(stats: Any) -> dict:
    if isinstance(stats, pd.Series):
        return {k: _jsonable(v) for k, v in stats.to_dict().items() if not k.startswith("_")}
    if isinstance(stats, dict):
        return {k: _jsonable(v) for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": _jsonable(stats)}


def _jsonable(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return [_jsonable(v) for v in val.tolist()]
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.isoformat()
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_jsonable(v) for v in val]
    if isinstance(val, dict):
        return {k: _jsonable(v) for k, v in val.items()}
    try:
        return float(val)
    except Exception:  # noqa: BLE001
        return str(val)


def write_artifacts(
    cfg: dict,
    *,
    ticker: str,
    strategy: str,
    params: dict,
    train_stats: Any | None,
    walk_forward_result: Any | None,
    optimize_meta: dict,
    data_meta: dict,
    heatmap: pd.DataFrame | None = None,
) -> ArtifactPaths:
    paths = build_artifact_paths(cfg, ticker=ticker, strategy=strategy)
    overwrite = bool(cfg["artifacts"].get("overwrite", False))

    for path in (paths.params_path, paths.report_path, paths.heatmap_path):
        if path is None:
            continue
        if path.exists() and not overwrite:
            raise FileExistsError(f"Artifact exists and overwrite=false: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "ticker": ticker.upper(),
        "strategy": strategy,
        "asof": datetime.now(timezone.utc).isoformat(),
        "params": _jsonable(params),
        "yfinance_adjustment": _jsonable(cfg["data"]["candles"]["price_adjustment"]),
        "optimization": _jsonable(optimize_meta),
        "data": _jsonable(data_meta),
        "train_stats": _serialize_stats(train_stats) if train_stats is not None else None,
        "walk_forward": _jsonable(walk_forward_result) if walk_forward_result is not None else None,
    }

    paths.params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    report_lines = _render_summary_markdown(
        ticker=ticker,
        strategy=strategy,
        params=params,
        train_stats=train_stats,
        walk_forward_result=walk_forward_result,
        optimize_meta=optimize_meta,
    )
    paths.report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if heatmap is not None and paths.heatmap_path is not None:
        heatmap.to_csv(paths.heatmap_path)

    return paths


def _render_summary_markdown(
    *,
    ticker: str,
    strategy: str,
    params: dict,
    train_stats: Any | None,
    walk_forward_result: Any | None,
    optimize_meta: dict,
) -> list[str]:
    lines = [
        f"# {ticker.upper()} — {strategy} summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Optimization: {optimize_meta.get('method')} | maximize={optimize_meta.get('maximize')}",
        f"- Constraints: {', '.join(optimize_meta.get('constraints', [])) or 'None'}",
        "",
        "## Chosen params",
        "```json",
        json.dumps(_jsonable(params), indent=2),
        "```",
    ]

    if train_stats is not None:
        lines.append("")
        lines.append("## Best train stats")
        for key, val in _serialize_stats(train_stats).items():
            lines.append(f"- {key}: {val}")

    if walk_forward_result:
        lines.append("")
        lines.append("## Walk-forward")
        stability = walk_forward_result.get("stability", {})
        lines.append(f"- Stability: {stability}")
        folds = walk_forward_result.get("folds", [])
        for i, fold in enumerate(folds, start=1):
            lines.append(
                f"- Fold {i}: train {fold['train_start'].date()}→{fold['train_end'].date()}, "
                f"validate {fold['validate_start'].date()}→{fold['validate_end'].date()}, "
                f"score {fold['validate_score']:.3f}"
            )
    return lines
