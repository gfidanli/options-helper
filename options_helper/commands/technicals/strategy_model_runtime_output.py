from __future__ import annotations

from typing import Any

from rich.console import Console

from .strategy_model_helpers_legacy import _coerce_int, _extract_directional_headline, _mapping_view


def emit_filter_summary(*, console: Console, run_result: Any) -> None:
    filter_summary = _mapping_view(getattr(run_result, "filter_summary", None))
    if not filter_summary:
        return

    base_events = _coerce_int(filter_summary.get("base_event_count"))
    kept_events = _coerce_int(filter_summary.get("kept_event_count"))
    rejected_events = _coerce_int(filter_summary.get("rejected_event_count"))
    if base_events is not None and kept_events is not None and rejected_events is not None:
        console.print(f"filters base={base_events} kept={kept_events} rejected={rejected_events}")

    reject_counts = _mapping_view(filter_summary.get("reject_counts"))
    reject_parts: list[str] = []
    for reason, count in reject_counts.items():
        parsed = _coerce_int(count)
        if parsed is None or parsed <= 0:
            continue
        reject_parts.append(f"{reason}={parsed}")
    if reject_parts:
        console.print(f"filter_rejects {', '.join(reject_parts)}")


def emit_directional_summary(*, console: Console, run_result: Any) -> None:
    directional_metrics = _mapping_view(getattr(run_result, "directional_metrics", None))
    if not directional_metrics:
        return

    parts: list[str] = []
    for label in ("combined", "long_only", "short_only"):
        trade_count_value, total_return_value = _extract_directional_headline(directional_metrics.get(label))
        if trade_count_value is None and total_return_value is None:
            continue
        part = label
        if trade_count_value is not None:
            part = f"{part} trades={trade_count_value}"
        if total_return_value is not None:
            part = f"{part} return={total_return_value:.2f}%"
        parts.append(part)
    if parts:
        console.print(f"directional {' | '.join(parts)}")


def emit_artifact_paths(*, console: Console, write_json: bool, write_csv: bool, write_md: bool, paths: Any) -> None:
    if write_json:
        console.print(f"Wrote summary JSON: {paths.summary_json}")
    if write_csv:
        console.print(f"Wrote trades CSV: {paths.trade_log_csv}")
        console.print(f"Wrote R ladder CSV: {paths.r_ladder_csv}")
        console.print(f"Wrote segments CSV: {paths.segments_csv}")
    if write_md:
        console.print(f"Wrote summary Markdown: {paths.summary_md}")
        console.print(f"Wrote LLM analysis prompt: {paths.llm_analysis_prompt_md}")


__all__ = [
    "emit_filter_summary",
    "emit_directional_summary",
    "emit_artifact_paths",
]
