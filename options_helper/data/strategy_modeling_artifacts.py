from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from options_helper.analysis.strategy_modeling_trade_review import rank_trades_for_review
from options_helper.schemas.common import clean_nan

DISCLAIMER_TEXT = "Informational output only; this tool is not financial advice."
_ENTRY_ANCHOR_LABEL = "next_bar_open"
_INTRABAR_TIE_BREAK_DEFAULT = "stop_first"
_DEFAULT_OUTPUT_TIMEZONE = "UTC"
_TRADE_REVIEW_TOP_N = 20
_OUTPUT_TIMEZONE_ALIASES: dict[str, str] = {
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
}
_LLM_PROMPT_FILENAME = "llm_analysis_prompt.md"

_TRADE_LOG_FIELDS: tuple[str, ...] = (
    "trade_id",
    "event_id",
    "symbol",
    "direction",
    "status",
    "signal_confirmed_ts",
    "entry_ts",
    "entry_price_source",
    "entry_price",
    "stop_price",
    "stop_price_final",
    "target_price",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "initial_risk",
    "realized_r",
    "gap_fill_applied",
    "gap_exit",
    "stop_slippage_r",
    "loss_below_1r",
    "reject_code",
)

_TRADE_REVIEW_CSV_FIELDS: tuple[str, ...] = ("rank", *_TRADE_LOG_FIELDS)
_TRADE_REVIEW_HIGHLIGHT_FIELDS: tuple[str, ...] = (
    "rank",
    "trade_id",
    "symbol",
    "direction",
    "entry_ts",
    "exit_ts",
    "realized_r",
    "exit_reason",
)

_R_LADDER_PREFERRED_FIELDS: tuple[str, ...] = (
    "target_label",
    "target_r",
    "trade_count",
    "hit_count",
    "hit_rate",
    "avg_bars_to_hit",
    "median_bars_to_hit",
    "expectancy_r",
)

_SEGMENT_PREFERRED_FIELDS: tuple[str, ...] = (
    "segment_dimension",
    "segment_value",
    "trade_count",
    "win_rate",
    "avg_realized_r",
    "expectancy_r",
    "profit_factor",
    "sharpe_ratio",
    "max_drawdown_pct",
)


@dataclass(frozen=True)
class StrategyModelingArtifactPaths:
    run_dir: Path
    summary_json: Path
    trade_log_csv: Path
    r_ladder_csv: Path
    segments_csv: Path
    top_best_trades_csv: Path
    top_worst_trades_csv: Path
    summary_md: Path
    llm_analysis_prompt_md: Path


def build_strategy_modeling_artifact_paths(
    base_dir: Path,
    *,
    strategy: str,
    as_of: str,
) -> StrategyModelingArtifactPaths:
    run_dir = base_dir / strategy.lower() / as_of
    return StrategyModelingArtifactPaths(
        run_dir=run_dir,
        summary_json=run_dir / "summary.json",
        trade_log_csv=run_dir / "trades.csv",
        r_ladder_csv=run_dir / "r_ladder.csv",
        segments_csv=run_dir / "segments.csv",
        top_best_trades_csv=run_dir / "top_20_best_trades.csv",
        top_worst_trades_csv=run_dir / "top_20_worst_trades.csv",
        summary_md=run_dir / "summary.md",
        llm_analysis_prompt_md=run_dir / _LLM_PROMPT_FILENAME,
    )


def write_strategy_modeling_artifacts(
    *,
    out_dir: Path,
    strategy: str,
    request: object,
    run_result: object,
    write_json: bool = True,
    write_csv: bool = True,
    write_md: bool = True,
    generated_at: datetime | None = None,
) -> StrategyModelingArtifactPaths:
    generated_ts = generated_at or datetime.now(timezone.utc)
    if generated_ts.tzinfo is None:
        generated_ts = generated_ts.replace(tzinfo=timezone.utc)
    generated_ts = generated_ts.astimezone(timezone.utc)
    output_timezone = _resolve_output_timezone(getattr(request, "output_timezone", None))
    output_timezone_name = _output_timezone_name(output_timezone)
    generated_output_ts = generated_ts.astimezone(output_timezone)

    as_of_label = _resolve_as_of_label(request=request, run_result=run_result, generated_at=generated_ts)
    paths = build_strategy_modeling_artifact_paths(out_dir, strategy=strategy, as_of=as_of_label)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _to_mapping(getattr(run_result, "portfolio_metrics", None))
    r_ladder_rows = _normalize_records(getattr(run_result, "target_hit_rates", ()) or ())
    segment_rows = _normalize_records(getattr(run_result, "segment_records", ()) or ())
    filter_metadata = _to_mapping(getattr(run_result, "filter_metadata", None))
    filter_summary = _to_mapping(getattr(run_result, "filter_summary", None))
    directional_metrics = _to_mapping(getattr(run_result, "directional_metrics", None))
    trade_rows = [
        _build_trade_log_row(row, output_timezone=output_timezone)
        for row in _normalize_records(getattr(run_result, "trade_simulations", ()) or ())
    ]

    requested_symbols = _to_string_list(
        getattr(run_result, "requested_symbols", None) or getattr(request, "symbols", None)
    )
    modeled_symbols = _to_string_list(
        getattr(run_result, "modeled_symbols", None) or getattr(request, "symbols", None)
    )

    signal_events = getattr(run_result, "signal_events", ()) or ()
    accepted_trade_ids = getattr(run_result, "accepted_trade_ids", ()) or ()
    accepted_trade_ids_for_review: Sequence[str] | None = None
    if hasattr(run_result, "accepted_trade_ids"):
        accepted_trade_ids_for_review = tuple(
            _normalize_string_sequence(getattr(run_result, "accepted_trade_ids", None))
        )
    skipped_trade_ids = getattr(run_result, "skipped_trade_ids", ()) or ()
    losses_below_one_r = sum(1 for row in trade_rows if row.get("loss_below_1r") is True)
    trade_review = rank_trades_for_review(
        trade_rows,
        accepted_trade_ids=accepted_trade_ids_for_review,
        top_n=_TRADE_REVIEW_TOP_N,
    )
    top_best_trade_rows = [clean_nan(_to_mapping(row)) for row in trade_review.top_best_rows]
    top_worst_trade_rows = [clean_nan(_to_mapping(row)) for row in trade_review.top_worst_rows]

    payload = clean_nan(
        {
            "schema_version": 1,
            "generated_at": generated_output_ts.isoformat(),
            "strategy": strategy.lower(),
            "requested_symbols": requested_symbols,
            "modeled_symbols": modeled_symbols,
            "disclaimer": DISCLAIMER_TEXT,
            "summary": {
                "signal_event_count": len(signal_events),
                "trade_count": len(trade_rows),
                "accepted_trade_count": len(accepted_trade_ids),
                "skipped_trade_count": len(skipped_trade_ids),
                "losses_below_minus_one_r": losses_below_one_r,
            },
            "policy_metadata": _build_policy_metadata(
                request=request,
                run_result=run_result,
                output_timezone_name=output_timezone_name,
            ),
            "filter_metadata": filter_metadata,
            "filter_summary": filter_summary,
            "directional_metrics": directional_metrics,
            "metrics": metrics,
            "r_ladder": r_ladder_rows,
            "segments": segment_rows,
            "trade_log": trade_rows,
            "trade_review": _build_trade_review_payload(
                metric=trade_review.metric,
                scope=trade_review.scope,
                candidate_trade_count=trade_review.candidate_trade_count,
                top_best_rows=top_best_trade_rows,
                top_worst_rows=top_worst_trade_rows,
            ),
        }
    )

    if write_json:
        paths.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if write_csv:
        _write_csv(path=paths.trade_log_csv, fieldnames=_TRADE_LOG_FIELDS, rows=trade_rows)
        _write_csv(
            path=paths.r_ladder_csv,
            fieldnames=_csv_fields_for_rows(r_ladder_rows, _R_LADDER_PREFERRED_FIELDS),
            rows=r_ladder_rows,
        )
        _write_csv(
            path=paths.segments_csv,
            fieldnames=_csv_fields_for_rows(segment_rows, _SEGMENT_PREFERRED_FIELDS),
            rows=segment_rows,
        )
        _write_csv(
            path=paths.top_best_trades_csv,
            fieldnames=_TRADE_REVIEW_CSV_FIELDS,
            rows=top_best_trade_rows,
        )
        _write_csv(
            path=paths.top_worst_trades_csv,
            fieldnames=_TRADE_REVIEW_CSV_FIELDS,
            rows=top_worst_trade_rows,
        )

    if write_md:
        markdown = _render_summary_markdown(payload)
        paths.summary_md.write_text(markdown, encoding="utf-8")
        llm_prompt = _render_llm_analysis_prompt(payload)
        paths.llm_analysis_prompt_md.write_text(llm_prompt, encoding="utf-8")

    return paths


def _resolve_as_of_label(*, request: object, run_result: object, generated_at: datetime) -> str:
    run_as_of = getattr(run_result, "as_of", None)
    if run_as_of is not None:
        label = _date_label(run_as_of)
        if label:
            return label

    end_date = getattr(request, "end_date", None)
    if end_date is not None:
        label = _date_label(end_date)
        if label:
            return label

    return generated_at.date().isoformat()


def _build_policy_metadata(
    *,
    request: object,
    run_result: object,
    output_timezone_name: str,
) -> dict[str, Any]:
    policy_payload = _to_mapping(getattr(request, "policy", None))
    preflight = getattr(run_result, "intraday_preflight", None)
    preflight_payload = _to_mapping(preflight)

    require_intraday = _as_bool_or_none(policy_payload.get("require_intraday_bars"))
    if require_intraday is None:
        require_intraday = _as_bool_or_none(preflight_payload.get("require_intraday_bars"))
    if require_intraday is None:
        require_intraday = True

    signal_confirmation_lag = _as_int_or_none(
        getattr(request, "signal_confirmation_lag_bars", None)
        or getattr(request, "swing_right_bars", None)
    )

    tie_break_rule = (
        _as_str_or_none(getattr(request, "intra_bar_tie_break_rule", None))
        or _INTRABAR_TIE_BREAK_DEFAULT
    )
    gap_policy = (
        _as_str_or_none(policy_payload.get("gap_fill_policy"))
        or _as_str_or_none(getattr(request, "gap_fill_policy", None))
        or "fill_at_open"
    )
    entry_anchor_policy = _as_str_or_none(policy_payload.get("entry_ts_anchor_policy"))
    max_hold_bars = _as_int_or_none(policy_payload.get("max_hold_bars"))
    if max_hold_bars is None:
        max_hold_bars = _as_int_or_none(getattr(request, "max_hold_bars", None))
    max_hold_timeframe = (
        _as_str_or_none(policy_payload.get("max_hold_timeframe"))
        or _as_str_or_none(getattr(request, "max_hold_timeframe", None))
        or "entry"
    )
    stop_move_rules = _normalize_records(policy_payload.get("stop_move_rules") or ())

    return clean_nan(
        {
            "entry_anchor": _ENTRY_ANCHOR_LABEL,
            "entry_ts_anchor_policy": entry_anchor_policy,
            "signal_confirmation_lag_bars": signal_confirmation_lag,
            "require_intraday_bars": require_intraday,
            "intraday_timeframe": _as_str_or_none(getattr(request, "intraday_timeframe", None)),
            "intraday_source": _as_str_or_none(getattr(request, "intraday_source", None)),
            "gap_fill_policy": gap_policy,
            "intra_bar_tie_break_rule": tie_break_rule,
            "output_timezone": output_timezone_name,
            "max_hold_bars": max_hold_bars,
            "max_hold_timeframe": max_hold_timeframe,
            "risk_per_trade_pct": _as_float_or_none(policy_payload.get("risk_per_trade_pct")),
            "sizing_rule": _as_str_or_none(policy_payload.get("sizing_rule")),
            "one_open_per_symbol": _as_bool_or_none(policy_payload.get("one_open_per_symbol")),
            "price_adjustment_policy": _as_str_or_none(policy_payload.get("price_adjustment_policy")),
            "stop_move_rules": stop_move_rules,
            "anti_lookahead_note": (
                "Signals confirmed at bar close are modeled from the next tradable bar open."
            ),
        }
    )


def _build_trade_review_payload(
    *,
    metric: str,
    scope: str,
    candidate_trade_count: int,
    top_best_rows: Sequence[Mapping[str, Any]],
    top_worst_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    best_rows = [clean_nan(_to_mapping(row)) for row in top_best_rows]
    worst_rows = [clean_nan(_to_mapping(row)) for row in top_worst_rows]
    return clean_nan(
        {
            "metric": metric,
            "scope": scope,
            "candidate_trade_count": candidate_trade_count,
            "top_best_count": len(best_rows),
            "top_worst_count": len(worst_rows),
            "top_best": _trade_review_highlights(best_rows),
            "top_worst": _trade_review_highlights(worst_rows),
        }
    )


def _trade_review_highlights(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for row in rows:
        src = _to_mapping(row)
        highlight = {field: src.get(field) for field in _TRADE_REVIEW_HIGHLIGHT_FIELDS}
        highlights.append(clean_nan(highlight))
    return highlights


def _render_summary_markdown(payload: Mapping[str, Any]) -> str:
    summary = _to_mapping(payload.get("summary"))
    policy = _to_mapping(payload.get("policy_metadata"))
    filter_metadata = _to_mapping(payload.get("filter_metadata"))
    filter_summary = _to_mapping(payload.get("filter_summary"))
    directional_metrics = _to_mapping(payload.get("directional_metrics"))
    metrics = _to_mapping(payload.get("metrics"))
    r_ladder_rows = _normalize_records(payload.get("r_ladder", []))
    segments = _normalize_records(payload.get("segments", []))
    trade_rows = _normalize_records(payload.get("trade_log", []))
    trade_review = _to_mapping(payload.get("trade_review"))

    lines: list[str] = [
        f"# Strategy Modeling Summary ({str(payload.get('strategy', '')).upper()})",
        "",
        DISCLAIMER_TEXT,
        "",
        f"- Generated: `{payload.get('generated_at')}`",
        f"- Requested symbols: `{len(_to_string_list(payload.get('requested_symbols')))}`",
        f"- Modeled symbols: `{len(_to_string_list(payload.get('modeled_symbols')))}`",
        f"- Signal events: `{summary.get('signal_event_count', 0)}`",
        f"- Simulated trades: `{summary.get('trade_count', 0)}`",
        f"- Accepted / skipped trades: `{summary.get('accepted_trade_count', 0)}` / `{summary.get('skipped_trade_count', 0)}`",
        f"- Realized losses below `-1.0R`: `{summary.get('losses_below_minus_one_r', 0)}`",
        "",
        "## Policy Metadata",
        "",
    ]

    for key in (
        "entry_anchor",
        "signal_confirmation_lag_bars",
        "require_intraday_bars",
        "intraday_timeframe",
        "intraday_source",
        "gap_fill_policy",
        "intra_bar_tie_break_rule",
        "output_timezone",
    ):
        if key in policy:
            lines.append(f"- {key}: `{policy.get(key)}`")

    lines.extend(["", "## Filters", ""])
    if filter_summary:
        lines.append(
            (
                "- Base / kept / rejected events: "
                f"`{filter_summary.get('base_event_count', 0)}` / "
                f"`{filter_summary.get('kept_event_count', 0)}` / "
                f"`{filter_summary.get('rejected_event_count', 0)}`"
            )
        )
        reject_counts = _to_mapping(filter_summary.get("reject_counts"))
        rejects_with_count = [
            f"{reason}={count}"
            for reason, count in reject_counts.items()
            if (_as_int_or_none(count) or 0) > 0
        ]
        if rejects_with_count:
            lines.append(f"- Reject reasons: `{', '.join(rejects_with_count)}`")
        else:
            lines.append("- Reject reasons: `none`")
    else:
        lines.append("- No filter summary returned.")

    active_filters = _normalize_string_sequence(filter_metadata.get("active_filters"))
    if active_filters:
        lines.append(f"- Active filters: `{', '.join(active_filters)}`")
    elif filter_metadata:
        lines.append("- Active filters: `none`")

    lines.extend(["", "## Directional Results", ""])
    if directional_metrics:
        counts = _to_mapping(directional_metrics.get("counts"))
        if counts:
            lines.append(
                (
                    "- Portfolio subset trades: "
                    f"all=`{counts.get('portfolio_subset_trade_count', 0)}` "
                    f"closed=`{counts.get('portfolio_subset_closed_trade_count', 0)}` "
                    f"long=`{counts.get('portfolio_subset_long_trade_count', 0)}` "
                    f"short=`{counts.get('portfolio_subset_short_trade_count', 0)}`"
                )
            )

        portfolio_target = _to_mapping(directional_metrics.get("portfolio_target"))
        if portfolio_target:
            target_label = portfolio_target.get("target_label") or portfolio_target.get("target_r")
            lines.append(
                (
                    f"- Portfolio target: `{target_label}` "
                    f"(source=`{portfolio_target.get('selection_source')}`)"
                )
            )

        lines.extend(_render_directional_bucket_lines(directional_metrics))
    else:
        lines.append("- No directional metrics returned.")

    lines.extend(
        [
            "",
            "## Portfolio Metrics",
            "",
        ]
    )
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No portfolio metrics returned.")

    lines.extend(["", "## R Ladder", ""])
    if r_ladder_rows:
        for row in r_ladder_rows:
            target = row.get("target_label") or row.get("target_r")
            hit_rate = row.get("hit_rate")
            expectancy = row.get("expectancy_r")
            lines.append(f"- {target}: hit_rate=`{hit_rate}` expectancy_r=`{expectancy}`")
    else:
        lines.append("- No R-ladder records returned.")

    lines.extend(["", "## Segments", ""])
    if segments:
        for row in segments[:10]:
            lines.append(
                (
                    f"- {row.get('segment_dimension')}={row.get('segment_value')}: "
                    f"trades=`{row.get('trade_count')}` avg_realized_r=`{row.get('avg_realized_r')}`"
                )
            )
    else:
        lines.append("- No segment records returned.")

    lines.extend(["", "## Trade Review", ""])
    if trade_review:
        lines.append(f"- Scope: `{trade_review.get('scope', '-')}`")
        lines.append(f"- Metric: `{trade_review.get('metric', 'realized_r')}`")
        lines.append(
            (
                "- Candidate trades / best / worst: "
                f"`{trade_review.get('candidate_trade_count', 0)}` / "
                f"`{trade_review.get('top_best_count', 0)}` / "
                f"`{trade_review.get('top_worst_count', 0)}`"
            )
        )

        top_best = _normalize_records(trade_review.get("top_best", []))
        if top_best:
            lines.append(
                f"- Best highlight: {_format_trade_review_highlight(top_best[0])}"
            )

        top_worst = _normalize_records(trade_review.get("top_worst", []))
        if top_worst:
            lines.append(
                f"- Worst highlight: {_format_trade_review_highlight(top_worst[0])}"
            )
    else:
        lines.append("- No trade-review rows returned.")

    lines.extend(
        [
            "",
            "## Trade Log Notes",
            "",
            f"- Trades logged: `{len(trade_rows)}`",
            "- Trade rows include `realized_r`, `gap_exit`, and `stop_slippage_r`.",
            "- Gap-through stop fills can produce realized losses below `-1.0R`; these are explicit in `trades.csv`.",
            "",
            "Not financial advice.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def _format_trade_review_highlight(row: Mapping[str, Any]) -> str:
    src = _to_mapping(row)
    rank = _as_int_or_none(src.get("rank")) or 1
    trade_id = _as_str_or_none(src.get("trade_id")) or "-"
    symbol = _as_str_or_none(src.get("symbol")) or "-"
    direction = _as_str_or_none(src.get("direction")) or "-"
    realized_r = _as_float_or_none(src.get("realized_r"))
    exit_reason = _as_str_or_none(src.get("exit_reason")) or "-"
    realized_r_label = "-" if realized_r is None else f"{realized_r:.6g}"
    return (
        f"`#{rank}` trade=`{trade_id}` symbol=`{symbol}` direction=`{direction}` "
        f"realized_r=`{realized_r_label}` exit_reason=`{exit_reason}`"
    )


def _render_directional_bucket_lines(directional_metrics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, key in (
        ("Combined", "combined"),
        ("Long-only", "long_only"),
        ("Short-only", "short_only"),
    ):
        bucket = _to_mapping(directional_metrics.get(key))
        if not bucket:
            lines.append(f"- {label}: no metrics returned.")
            continue

        portfolio_metrics = _to_mapping(bucket.get("portfolio_metrics"))
        lines.append(
            (
                f"- {label}: "
                f"simulated=`{bucket.get('simulated_trade_count', 0)}` "
                f"closed=`{bucket.get('closed_trade_count', 0)}` "
                f"accepted/skipped=`{bucket.get('accepted_trade_count', 0)}`/"
                f"`{bucket.get('skipped_trade_count', 0)}` "
                f"total_return_pct=`{portfolio_metrics.get('total_return_pct')}` "
                f"avg_realized_r=`{portfolio_metrics.get('avg_realized_r')}`"
            )
        )
    return lines


def _render_llm_analysis_prompt(payload: Mapping[str, Any]) -> str:
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

    lines: list[str] = [
        "# Strategy Modeling LLM Analysis Prompt",
        "",
        DISCLAIMER_TEXT,
        "",
        "Use the companion report files in this same folder to evaluate strategy quality and propose improvements.",
        "",
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
        (
            f"- Filter rejects (non-zero): "
            f"`{', '.join(nonzero_rejects) if nonzero_rejects else 'none'}`"
        ),
        "",
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
        "## Non-negotiable Constraints",
        "",
        "- Treat this as informational research only, not financial advice.",
        "- Do not use same-bar-close entry assumptions for close-confirmed signals.",
        "- Explicitly call out data-quality limitations and sample-size caveats.",
        "- Prefer robust changes that generalize across symbols/regimes over single-slice optimizations.",
        "",
        "Not financial advice.",
    ]

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


def _build_trade_log_row(
    payload: Mapping[str, Any],
    *,
    output_timezone: ZoneInfo,
) -> dict[str, Any]:
    src = _to_mapping(payload)
    realized_r = _coerce_realized_r(src)
    exit_reason = _as_str_or_none(src.get("exit_reason"))
    gap_fill_applied = bool(src.get("gap_fill_applied") is True)
    gap_exit = gap_fill_applied or _is_gap_exit_reason(exit_reason)

    stop_slippage_r = _as_float_or_none(src.get("stop_slippage_r"))
    if stop_slippage_r is None and realized_r is not None and _is_stop_reason(exit_reason):
        entry_price = _as_float_or_none(src.get("entry_price"))
        initial_risk = _as_float_or_none(src.get("initial_risk"))
        direction = (_as_str_or_none(src.get("direction")) or "").strip().lower()
        stop_price = _as_float_or_none(src.get("stop_price_final"))
        if stop_price is None:
            stop_price = _as_float_or_none(src.get("stop_price"))

        slippage: float | None = None
        if entry_price is not None and initial_risk is not None and initial_risk > 0.0 and stop_price is not None:
            if direction == "short":
                stop_r = (entry_price - stop_price) / initial_risk
            else:
                stop_r = (stop_price - entry_price) / initial_risk
            slippage = realized_r - float(stop_r)

        if slippage is None:
            slippage = realized_r + 1.0

        if slippage < 0.0:
            stop_slippage_r = round(slippage, 6)

    return clean_nan(
        {
            "trade_id": _as_str_or_none(src.get("trade_id")),
            "event_id": _as_str_or_none(src.get("event_id")),
            "symbol": _as_str_or_none(src.get("symbol")),
            "direction": _as_str_or_none(src.get("direction")),
            "status": _as_str_or_none(src.get("status")),
            "signal_confirmed_ts": _to_output_timezone_iso(
                src.get("signal_confirmed_ts"),
                output_timezone=output_timezone,
            ),
            "entry_ts": _to_output_timezone_iso(
                src.get("entry_ts"),
                output_timezone=output_timezone,
            ),
            "entry_price_source": _as_str_or_none(src.get("entry_price_source")),
            "entry_price": _as_float_or_none(src.get("entry_price")),
            "stop_price": _as_float_or_none(src.get("stop_price")),
            "stop_price_final": _as_float_or_none(src.get("stop_price_final")),
            "target_price": _as_float_or_none(src.get("target_price")),
            "exit_ts": _to_output_timezone_iso(
                src.get("exit_ts"),
                output_timezone=output_timezone,
            ),
            "exit_price": _as_float_or_none(src.get("exit_price")),
            "exit_reason": exit_reason,
            "initial_risk": _as_float_or_none(src.get("initial_risk")),
            "realized_r": realized_r,
            "gap_fill_applied": gap_fill_applied,
            "gap_exit": gap_exit,
            "stop_slippage_r": stop_slippage_r,
            "loss_below_1r": bool(realized_r is not None and realized_r < -1.0),
            "reject_code": _as_str_or_none(src.get("reject_code")),
        }
    )


def _coerce_realized_r(row: Mapping[str, Any]) -> float | None:
    realized_r = _as_float_or_none(row.get("realized_r"))
    if realized_r is not None:
        return round(realized_r, 6)

    direction = (_as_str_or_none(row.get("direction")) or "").strip().lower()
    entry_price = _as_float_or_none(row.get("entry_price"))
    exit_price = _as_float_or_none(row.get("exit_price"))
    initial_risk = _as_float_or_none(row.get("initial_risk"))
    if entry_price is None or exit_price is None or initial_risk is None or initial_risk <= 0.0:
        return None

    if direction == "short":
        value = (entry_price - exit_price) / initial_risk
    else:
        value = (exit_price - entry_price) / initial_risk
    return round(value, 6)


def _is_stop_reason(reason: str | None) -> bool:
    if not reason:
        return False
    normalized = reason.lower()
    return "stop" in normalized


def _is_gap_exit_reason(reason: str | None) -> bool:
    if not reason:
        return False
    normalized = reason.lower()
    return "gap" in normalized or "open_fill" in normalized


def _write_csv(path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_mapping(row))


def _csv_fields_for_rows(
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str],
) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for name in preferred_fields:
        seen[str(name)] = None
    for row in rows:
        for key in row.keys():
            k = str(key)
            if k not in seen:
                seen[k] = None
    return tuple(seen.keys())


def _normalize_records(records: object) -> list[dict[str, Any]]:
    if records is None:
        return []
    if isinstance(records, (str, bytes)):
        return []
    if isinstance(records, Mapping):
        return [clean_nan(_to_mapping(records))]

    try:
        iterator = iter(records)  # type: ignore[arg-type]
    except TypeError:
        return [clean_nan(_to_mapping(records))]

    normalized: list[dict[str, Any]] = []
    for item in iterator:
        normalized.append(clean_nan(_to_mapping(item)))
    return normalized


def _to_mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return {str(k): v for k, v in dumped.items()}
    if hasattr(value, "__dict__"):
        return {
            str(k): v
            for k, v in vars(value).items()
            if not str(k).startswith("_") and not callable(v)
        }
    return {"value": value}


def _to_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().upper() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        items: list[str] = []
        for item in value:
            token = _as_str_or_none(item)
            if token:
                items.append(token.upper())
        return items
    token = _as_str_or_none(value)
    return [token.upper()] if token else []


def _normalize_string_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = _as_str_or_none(value)
        return [token] if token else []
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            token = _as_str_or_none(item)
            if token:
                out.append(token)
        return out
    token = _as_str_or_none(value)
    return [token] if token else []


def _date_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _as_str_or_none(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text.split("T")[0]


def _resolve_output_timezone(value: object) -> ZoneInfo:
    token = _as_str_or_none(value) or _DEFAULT_OUTPUT_TIMEZONE
    canonical = _OUTPUT_TIMEZONE_ALIASES.get(token.upper(), token)
    try:
        return ZoneInfo(canonical)
    except ZoneInfoNotFoundError:
        return ZoneInfo(_DEFAULT_OUTPUT_TIMEZONE)


def _output_timezone_name(value: ZoneInfo) -> str:
    key = getattr(value, "key", None)
    if isinstance(key, str) and key.strip():
        return key
    return str(value)


def _to_output_timezone_iso(value: object, *, output_timezone: ZoneInfo) -> str | None:
    dt = _coerce_datetime(value)
    if dt is None:
        return _as_str_or_none(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(output_timezone).isoformat()


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = _as_str_or_none(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


__all__ = [
    "DISCLAIMER_TEXT",
    "StrategyModelingArtifactPaths",
    "build_strategy_modeling_artifact_paths",
    "write_strategy_modeling_artifacts",
]
