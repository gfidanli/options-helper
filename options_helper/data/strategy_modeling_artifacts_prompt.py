from __future__ import annotations

from typing import Any, Mapping


def render_llm_analysis_prompt(payload: Mapping[str, Any], *, disclaimer_text: str) -> str:
    strategy = str(payload.get("strategy", "")).upper() or "UNKNOWN"
    generated_at = str(payload.get("generated_at", ""))
    requested_symbols = _to_string_list(payload.get("requested_symbols"))
    modeled_symbols = _to_string_list(payload.get("modeled_symbols"))
    summary = _to_mapping(payload.get("summary"))
    policy = _to_mapping(payload.get("policy_metadata"))
    filters = _to_mapping(payload.get("filter_summary"))
    directional = _to_mapping(payload.get("directional_metrics"))
    reject_counts = _to_mapping(filters.get("reject_counts"))
    nonzero_rejects = [
        f"{reason}={count}"
        for reason, count in reject_counts.items()
        if (_as_int_or_none(count) or 0) > 0
    ]
    lines: list[str] = []
    lines.extend(_render_llm_prompt_header_lines(disclaimer_text=disclaimer_text))
    lines.extend(
        _render_llm_prompt_context_lines(
            strategy=strategy,
            generated_at=generated_at,
            requested_symbols=requested_symbols,
            modeled_symbols=modeled_symbols,
            summary=summary,
            policy=policy,
            nonzero_rejects=nonzero_rejects,
        )
    )
    lines.extend(_render_llm_prompt_file_lines())
    lines.extend(_render_llm_prompt_required_output_lines())
    lines.extend(_render_llm_prompt_constraint_lines())
    if directional:
        lines.extend(
            [
                "",
                "## Directional Context",
                "",
                "- Use directional metrics in `summary.json -> directional_metrics` to compare long-only vs short-only behavior before recommending directional constraints.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_llm_prompt_header_lines(*, disclaimer_text: str) -> list[str]:
    return [
        "# Strategy Modeling LLM Analysis Prompt",
        "",
        disclaimer_text,
        "",
        "Use the companion report files in this same folder to evaluate strategy quality and propose improvements.",
        "",
    ]


def _render_llm_prompt_context_lines(
    *,
    strategy: str,
    generated_at: str,
    requested_symbols: list[str],
    modeled_symbols: list[str],
    summary: Mapping[str, Any],
    policy: Mapping[str, Any],
    nonzero_rejects: list[str],
) -> list[str]:
    return [
        "## Context Snapshot",
        "",
        f"- Strategy: `{strategy}`",
        f"- Generated: `{generated_at}`",
        f"- Requested symbols: `{', '.join(requested_symbols) if requested_symbols else '-'}`",
        f"- Modeled symbols: `{', '.join(modeled_symbols) if modeled_symbols else '-'}`",
        f"- Signal events: `{summary.get('signal_event_count', 0)}`",
        f"- Simulated trades: `{summary.get('trade_count', 0)}`",
        f"- Accepted / skipped trades: `{summary.get('accepted_trade_count', 0)}` / `{summary.get('skipped_trade_count', 0)}`",
        f"- Losses below -1.0R: `{summary.get('losses_below_minus_one_r', 0)}`",
        f"- Entry anchor: `{policy.get('entry_anchor', 'next_bar_open')}`",
        f"- Gap policy: `{policy.get('gap_fill_policy', '-')}`",
        f"- Intraday timeframe: `{policy.get('intraday_timeframe', '-')}`",
        f"- Filter rejects (non-zero): `{', '.join(nonzero_rejects) if nonzero_rejects else 'none'}`",
        "",
    ]


def _render_llm_prompt_file_lines() -> list[str]:
    return [
        "## Files To Read",
        "",
        "- `summary.json`: full machine-readable run payload and metrics.",
        "- `summary.md`: human summary and caveats.",
        "- `trades.csv`: per-trade outcomes (`realized_r`, stop slippage, gap exits).",
        "- `top_20_best_trades.csv`: top accepted/fallback closed trades ranked by `realized_r`.",
        "- `top_20_worst_trades.csv`: bottom accepted/fallback closed trades ranked by `realized_r`.",
        "- `segments.csv`: grouped performance by dimension/value.",
        "- `r_ladder.csv`: target hit rates and expectancy by R target.",
        "",
    ]


def _render_llm_prompt_required_output_lines() -> list[str]:
    return [
        "## Analysis Task",
        "",
        "Identify concrete, testable changes that could improve risk-adjusted performance while preserving anti-lookahead rules.",
        "",
        "## Required Output",
        "",
        "1. Baseline diagnosis",
        "- Describe main weaknesses and strengths using explicit evidence from files/columns.",
        "",
        "2. Ranked improvement ideas",
        "- Provide 5-10 changes, ranked by expected impact and confidence.",
        "- For each change include: rationale, likely affected metrics, and failure modes.",
        "",
        "3. Experiment plan",
        "- Define backtest experiments for the top 3 ideas with precise config deltas.",
        "- Include acceptance criteria and guardrails against overfitting/lookahead bias.",
        "",
        "4. Risk controls",
        "- Explain downside risks (for example regime concentration, tail-loss behavior, sparse segments).",
        "- Recommend monitoring metrics and trigger thresholds.",
        "",
        "5. Portfolio/action summary",
        "- Summarize whether the current strategy should be: keep, modify, or pause for further testing.",
        "",
    ]


def _render_llm_prompt_constraint_lines() -> list[str]:
    return [
        "## Non-negotiable Constraints",
        "",
        "- Treat this as informational research only, not financial advice.",
        "- Do not use same-bar-close entry assumptions for close-confirmed signals.",
        "- Explicitly call out data-quality limitations and sample-size caveats.",
        "- Prefer robust changes that generalize across symbols/regimes over single-slice optimizations.",
        "",
        "Not financial advice.",
    ]


def _to_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _to_string_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidate = values.strip()
        return [candidate] if candidate else []
    out: list[str] = []
    if isinstance(values, list | tuple | set):
        for value in values:
            candidate = str(value).strip()
            if candidate:
                out.append(candidate)
    return out


def _as_int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
