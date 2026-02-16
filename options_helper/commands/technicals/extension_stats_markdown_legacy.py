from __future__ import annotations

from typing import Any, Callable, Mapping


def _append_current_percentiles(md_lines: list[str], *, report: Any, heading: str) -> None:
    md_lines.append("")
    md_lines.append(heading)
    if report.current_percentiles:
        for years, pct in sorted(report.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
        return
    md_lines.append("- No percentile windows available (insufficient history).")


def _append_rolling_quantiles(md_lines: list[str], *, report: Any, heading: str) -> None:
    md_lines.append("")
    md_lines.append(heading)
    if report.quantiles_by_window:
        for years, q in sorted(report.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
        return
    md_lines.append("- Not available.")


def _format_divergence_delta(divergence: Mapping[str, Any], key: str, *, pct: bool = False) -> str:
    try:
        value = divergence.get(key)
        if value is None:
            return "-"
        return f"{float(value):+.2f}%" if pct else f"{float(value):+.2f}"
    except Exception:  # noqa: BLE001
        return "-"


def _append_current_divergence(
    md_lines: list[str],
    *,
    label: str,
    divergence: dict[str, Any] | None,
) -> None:
    if divergence is None:
        return
    drsi = _format_divergence_delta(divergence, "rsi_delta", pct=False)
    dpct = _format_divergence_delta(divergence, "price_delta_pct", pct=True)
    md_lines.append(
        f"- Current {label} divergence: `{divergence.get('swing1_date')} → {divergence.get('swing2_date')}` "
        f"(RSI tag `{divergence.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
    )


def _format_summary_move(value: object) -> str:
    try:
        return "-" if value is None else f"{float(value):+.1f}%"
    except Exception:  # noqa: BLE001
        return "-"


def _format_summary_percentile(value: object) -> str:
    try:
        return "-" if value is None else f"{float(value):.1f}"
    except Exception:  # noqa: BLE001
        return "-"


def _append_daily_tail_outcome_summary(
    md_lines: list[str],
    *,
    rsi_divergence_daily: dict[str, Any],
) -> None:
    summary = rsi_divergence_daily.get("tail_event_summary")
    if not isinstance(summary, dict):
        return
    md_lines.append("")
    md_lines.append("### Tail Outcomes With vs Without Divergence (Daily)")
    md_lines.append("| Tail | Divergence | N | Med max move (5d/15d) | Med fwd pctl (5d/15d) |")
    md_lines.append("|---|---|---:|---|---|")
    for tail, want in (("high", "bearish"), ("low", "bullish")):
        tail_bucket = summary.get(tail)
        if not isinstance(tail_bucket, dict):
            continue
        for bucket_name, label in (("with_divergence", f"with {want}"), ("without_divergence", f"without {want}")):
            bucket = tail_bucket.get(bucket_name)
            if not isinstance(bucket, dict):
                continue
            n = bucket.get("n", 0)
            r_str = (
                f"{_format_summary_move(bucket.get('median_fwd_max_fav_move_pct_5d'))} / "
                f"{_format_summary_move(bucket.get('median_fwd_max_fav_move_pct_15d'))}"
            )
            p_str = (
                f"{_format_summary_percentile(bucket.get('median_fwd_extension_percentile_5d'))} / "
                f"{_format_summary_percentile(bucket.get('median_fwd_extension_percentile_15d'))}"
            )
            md_lines.append(f"| {tail} | {label} | {n} | {r_str} | {p_str} |")


def _append_daily_rsi_divergence(
    md_lines: list[str],
    *,
    rsi_divergence_cfg: dict[str, Any] | None,
    rsi_divergence_daily: dict[str, Any] | None,
) -> None:
    md_lines.append("")
    md_lines.append("## RSI Divergence (Daily)")
    if rsi_divergence_daily is None or rsi_divergence_cfg is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
        return
    cfg_line = (
        f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
        f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
        f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
        f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
    )
    md_lines.append(cfg_line)
    cur = rsi_divergence_daily.get("current")
    if not isinstance(cur, dict):
        cur = {}
    cur_bear = cur.get("bearish") if isinstance(cur.get("bearish"), dict) else None
    cur_bull = cur.get("bullish") if isinstance(cur.get("bullish"), dict) else None
    if cur_bear is None and cur_bull is None:
        md_lines.append("- No divergences detected in the most recent window.")
    _append_current_divergence(md_lines, label="bearish", divergence=cur_bear)
    _append_current_divergence(md_lines, label="bullish", divergence=cur_bull)
    _append_daily_tail_outcome_summary(md_lines, rsi_divergence_daily=rsi_divergence_daily)


def _format_pair(median: object, p75: object) -> str:
    try:
        if median is None or p75 is None:
            return "-"
        return f"{float(median):+.1f}% / {float(p75):+.1f}%"
    except Exception:  # noqa: BLE001
        return "-"


def _format_max_move_cell(stats: Mapping[str, Any]) -> str:
    fav = _format_pair(stats.get("fav_median"), stats.get("fav_p75"))
    dd = _format_pair(stats.get("dd_median"), stats.get("dd_p75"))
    if fav == "-" and dd == "-":
        return "-"
    if dd == "-":
        return fav
    if fav == "-":
        return dd
    return f"{fav}; {dd}"


def _append_daily_max_move_section(md_lines: list[str], *, payload: Mapping[str, Any]) -> None:
    md_lines.append("")
    md_lines.append("## Max Favorable Move (Daily)")
    md_lines.append(
        "Directional metrics: fav is in the mean-reversion direction; dd is max adverse move against it (both use High/Low vs entry Close)."
    )
    md_lines.append("Cells: fav (med/p75); dd (med/p75).")
    md_lines.append("Descriptive (not financial advice).")
    max_up = payload.get("max_move_summary_daily")
    buckets = max_up.get("buckets", []) if isinstance(max_up, dict) else []
    if not buckets:
        md_lines.append("- Not available (insufficient history).")
        return
    md_lines.append("| Bucket | N | 1w | 4w | 3m | 6m | 9m | 1y |")
    md_lines.append("|---|---:|---|---|---|---|---|---|")
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        n = int(bucket.get("n", 0) or 0)
        stats = bucket.get("stats")
        stats_map = stats if isinstance(stats, dict) else {}
        cells = []
        for label in ("1w", "4w", "3m", "6m", "9m", "1y"):
            row_stats = stats_map.get(label)
            cells.append(_format_max_move_cell(row_stats if isinstance(row_stats, dict) else {}))
        md_lines.append(f"| {bucket.get('label', '-')} | {n} | {' | '.join(cells)} |")


def _safe_extension_atr(value: object) -> str:
    try:
        return f"{float(value):+.2f}"
    except Exception:  # noqa: BLE001
        return "-"


def _safe_percentile(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except Exception:  # noqa: BLE001
        return "-"


def _format_tail_event_pct_list(values: list[object], *, pct: bool = False) -> str:
    if pct:
        return ", ".join("-" if value is None else f"{float(value) * 100.0:+.1f}%" for value in values)
    return ", ".join("-" if value is None else f"{float(value):.1f}" for value in values)


def _append_daily_tail_events_section(
    md_lines: list[str],
    *,
    daily_tail_events: list[dict[str, Any]],
    forward_days_daily: list[int],
    max_return_horizons_days: Mapping[str, int],
) -> None:
    md_lines.append("")
    md_lines.append("## Tail Events (Daily, all)")
    if not daily_tail_events:
        md_lines.append("- No tail events found.")
        return
    horiz = "/".join(str(int(day)) for day in forward_days_daily)
    max_horiz = "/".join(max_return_horizons_days.keys())
    md_lines.append(
        f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | W pctl | W RSI | W div | Fwd pctl ({horiz}) | Max ret% ({max_horiz}) |"
    )
    md_lines.append("|---|---|---:|---:|---|---|---|---:|---|---|---|---|")
    for event in daily_tail_events:
        pcts = [(event.get("forward_extension_percentiles") or {}).get(int(day)) for day in forward_days_daily]
        maxrets = [(event.get("max_fav_returns") or {}).get(str(label)) for label in max_return_horizons_days.keys()]
        divergence = event.get("rsi_divergence") if isinstance(event.get("rsi_divergence"), dict) else None
        wctx = event.get("weekly_context") if isinstance(event.get("weekly_context"), dict) else {}
        w_pct = wctx.get("extension_percentile")
        w_pct_str = "-" if w_pct is None else _safe_percentile(w_pct)
        div_type = "-" if not divergence else (divergence.get("divergence") or "-")
        div_tag = "-" if not divergence else (divergence.get("rsi_regime") or "-")
        md_lines.append(
            f"| {event.get('date')} | {event.get('direction')} | {_safe_extension_atr(event.get('extension_atr'))} | "
            f"{_safe_percentile(event.get('percentile'))} | {event.get('rsi_regime') or '-'} | {div_type} | {div_tag} | "
            f"{w_pct_str} | {wctx.get('rsi_regime') or '-'} | {wctx.get('divergence') or '-'} | "
            f"{_format_tail_event_pct_list(pcts, pct=False)} | {_format_tail_event_pct_list(maxrets, pct=True)} |"
        )


def _latest_weekly_rsi(
    *,
    weekly_rsi_series: Any,
    rsi_regime_tag: Callable[..., str],
    rsi_overbought: float,
    rsi_oversold: float,
) -> tuple[float | None, str | None]:
    if weekly_rsi_series is None:
        return None, None
    try:
        values = weekly_rsi_series.dropna()
        if values.empty:
            return None, None
        weekly_rsi_val = float(values.iloc[-1])
        weekly_rsi_label = rsi_regime_tag(
            rsi_value=weekly_rsi_val,
            rsi_overbought=float(rsi_overbought),
            rsi_oversold=float(rsi_oversold),
        )
        return weekly_rsi_val, weekly_rsi_label
    except Exception:  # noqa: BLE001
        return None, None


def _append_weekly_context(
    md_lines: list[str],
    *,
    report_weekly: Any,
    weekly_rsi_series: Any,
    rsi_regime_tag: Callable[..., str],
    rsi_overbought: float,
    rsi_oversold: float,
) -> None:
    md_lines.append("")
    md_lines.append("## Weekly Context")
    md_lines.append(f"- As-of (weekly): `{report_weekly.asof}`")
    md_lines.append(
        f"- Extension (weekly, ATR units): `{'-' if report_weekly.extension_atr is None else f'{report_weekly.extension_atr:+.2f}'}`"
    )
    weekly_rsi_val, weekly_rsi_label = _latest_weekly_rsi(
        weekly_rsi_series=weekly_rsi_series,
        rsi_regime_tag=rsi_regime_tag,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    if weekly_rsi_val is not None and weekly_rsi_label is not None:
        md_lines.append(f"- RSI (weekly): `{weekly_rsi_val:.1f}` (tag `{weekly_rsi_label}`)")


def _append_weekly_rsi_divergence(
    md_lines: list[str],
    *,
    rsi_divergence_cfg: dict[str, Any] | None,
    rsi_divergence_weekly: dict[str, Any] | None,
) -> None:
    md_lines.append("")
    md_lines.append("## RSI Divergence (Weekly)")
    if rsi_divergence_weekly is None or rsi_divergence_cfg is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
        return
    cfg_line = (
        f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
        f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
        f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
        f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
    )
    md_lines.append(cfg_line)
    cur = rsi_divergence_weekly.get("current")
    if not isinstance(cur, dict):
        cur = {}
    cur_bear = cur.get("bearish") if isinstance(cur.get("bearish"), dict) else None
    cur_bull = cur.get("bullish") if isinstance(cur.get("bullish"), dict) else None
    if cur_bear is None and cur_bull is None:
        md_lines.append("- No divergences detected in the most recent window.")
    if cur_bear is not None:
        md_lines.append(
            f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
            f"(RSI tag `{cur_bear.get('rsi_regime')}`)"
        )
    if cur_bull is not None:
        md_lines.append(
            f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
            f"(RSI tag `{cur_bull.get('rsi_regime')}`)"
        )


def _append_weekly_tail_events_section(
    md_lines: list[str],
    *,
    weekly_tail_events: list[dict[str, Any]],
    forward_days_weekly: list[int],
) -> None:
    md_lines.append("")
    md_lines.append("## Tail Events (Weekly, all)")
    if not weekly_tail_events:
        md_lines.append("- No tail events found.")
        return
    horiz = "/".join(str(int(day)) for day in forward_days_weekly)
    md_lines.append(f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | Fwd pctl ({horiz}) | Max move% ({horiz}) |")
    md_lines.append("|---|---|---:|---:|---|---|---|---|---|")
    for event in weekly_tail_events:
        pcts = [(event.get("forward_extension_percentiles") or {}).get(int(day)) for day in forward_days_weekly]
        maxrets = [(event.get("forward_max_fav_returns") or {}).get(int(day)) for day in forward_days_weekly]
        divergence = event.get("rsi_divergence") if isinstance(event.get("rsi_divergence"), dict) else None
        div_type = "-" if not divergence else (divergence.get("divergence") or "-")
        div_tag = "-" if not divergence else (divergence.get("rsi_regime") or "-")
        md_lines.append(
            f"| {event.get('date')} | {event.get('direction')} | {_safe_extension_atr(event.get('extension_atr'))} | "
            f"{_safe_percentile(event.get('percentile'))} | {event.get('rsi_regime') or '-'} | {div_type} | {div_tag} | "
            f"{_format_tail_event_pct_list(pcts, pct=False)} | {_format_tail_event_pct_list(maxrets, pct=True)} |"
        )


def build_extension_stats_markdown(
    *,
    sym_label: str,
    report_daily: Any,
    report_weekly: Any,
    payload: dict[str, Any],
    rsi_divergence_cfg: dict[str, Any] | None,
    rsi_divergence_daily: dict[str, Any] | None,
    rsi_divergence_weekly: dict[str, Any] | None,
    daily_tail_events: list[dict[str, Any]],
    weekly_tail_events: list[dict[str, Any]],
    forward_days_daily: list[int],
    forward_days_weekly: list[int],
    max_return_horizons_days: dict[str, int],
    weekly_rsi_series: Any,
    rsi_regime_tag: Callable[..., str],
    rsi_overbought: float,
    rsi_oversold: float,
) -> str:
    md_lines: list[str] = [
        f"# {sym_label} — Extension Percentile Stats",
        "",
        f"- As-of (daily): `{report_daily.asof}`",
        f"- Extension (daily, ATR units): `{'-' if report_daily.extension_atr is None else f'{report_daily.extension_atr:+.2f}'}`",
    ]
    _append_current_percentiles(md_lines, report=report_daily, heading="## Current Percentiles")
    _append_rolling_quantiles(md_lines, report=report_daily, heading="## Rolling Quantiles (Daily p5 / p50 / p95)")
    _append_daily_rsi_divergence(
        md_lines,
        rsi_divergence_cfg=rsi_divergence_cfg,
        rsi_divergence_daily=rsi_divergence_daily,
    )
    _append_daily_max_move_section(md_lines, payload=payload)
    _append_daily_tail_events_section(
        md_lines,
        daily_tail_events=daily_tail_events,
        forward_days_daily=forward_days_daily,
        max_return_horizons_days=max_return_horizons_days,
    )
    _append_weekly_context(
        md_lines,
        report_weekly=report_weekly,
        weekly_rsi_series=weekly_rsi_series,
        rsi_regime_tag=rsi_regime_tag,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    _append_current_percentiles(md_lines, report=report_weekly, heading="## Current Percentiles (Weekly)")
    _append_rolling_quantiles(md_lines, report=report_weekly, heading="## Rolling Quantiles (Weekly p5 / p50 / p95)")
    _append_weekly_rsi_divergence(
        md_lines,
        rsi_divergence_cfg=rsi_divergence_cfg,
        rsi_divergence_weekly=rsi_divergence_weekly,
    )
    _append_weekly_tail_events_section(
        md_lines,
        weekly_tail_events=weekly_tail_events,
        forward_days_weekly=forward_days_weekly,
    )
    return "\n".join(md_lines).rstrip() + "\n"


__all__ = ["build_extension_stats_markdown"]
