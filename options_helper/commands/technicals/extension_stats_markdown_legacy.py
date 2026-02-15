from __future__ import annotations

from typing import Any, Callable


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
        "",
        "## Current Percentiles",
    ]
    if report_daily.current_percentiles:
        for years, pct in sorted(report_daily.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Daily p5 / p50 / p95)")
    if report_daily.quantiles_by_window:
        for years, q in sorted(report_daily.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Daily)")
    if rsi_divergence_daily is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_daily.get("current") or {}) if isinstance(rsi_divergence_daily, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
        if cur_bear is None and cur_bull is None:
            md_lines.append("- No divergences detected in the most recent window.")
        if cur_bear is not None:
            try:
                drsi = "-" if cur_bear.get("rsi_delta") is None else f"{float(cur_bear.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = (
                    "-"
                    if cur_bear.get("price_delta_pct") is None
                    else f"{float(cur_bear.get('price_delta_pct')):+.2f}%"
                )
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
                f"(RSI tag `{cur_bear.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )
        if cur_bull is not None:
            try:
                drsi = "-" if cur_bull.get("rsi_delta") is None else f"{float(cur_bull.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = (
                    "-"
                    if cur_bull.get("price_delta_pct") is None
                    else f"{float(cur_bull.get('price_delta_pct')):+.2f}%"
                )
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
                f"(RSI tag `{cur_bull.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )

        # Compact conditional summary table (focus on the most commonly used horizons).
        summ = rsi_divergence_daily.get("tail_event_summary") if isinstance(rsi_divergence_daily, dict) else None
        if isinstance(summ, dict):
            md_lines.append("")
            md_lines.append("### Tail Outcomes With vs Without Divergence (Daily)")
            md_lines.append("| Tail | Divergence | N | Med max move (5d/15d) | Med fwd pctl (5d/15d) |")
            md_lines.append("|---|---|---:|---|---|")
            for tail, want in (("high", "bearish"), ("low", "bullish")):
                for bucket_name, label in (
                    ("with_divergence", f"with {want}"),
                    ("without_divergence", f"without {want}"),
                ):
                    b = (summ.get(tail) or {}).get(bucket_name) if isinstance(summ.get(tail), dict) else None
                    if not isinstance(b, dict):
                        continue
                    n = b.get("n", 0)
                    r5 = b.get("median_fwd_max_fav_move_pct_5d")
                    r15 = b.get("median_fwd_max_fav_move_pct_15d")
                    p5 = b.get("median_fwd_extension_percentile_5d")
                    p15 = b.get("median_fwd_extension_percentile_15d")

                    def _fmt_ret(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):+.1f}%"
                        except Exception:  # noqa: BLE001
                            return "-"

                    def _fmt_pct(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):.1f}"
                        except Exception:  # noqa: BLE001
                            return "-"

                    r_str = f"{_fmt_ret(r5)} / {_fmt_ret(r15)}"
                    p_str = f"{_fmt_pct(p5)} / {_fmt_pct(p15)}"
                    md_lines.append(f"| {tail} | {label} | {n} | {r_str} | {p_str} |")

    md_lines.append("")
    md_lines.append("## Max Favorable Move (Daily)")
    md_lines.append(
        "Directional metrics: fav is in the mean-reversion direction; dd is max adverse move against it (both use High/Low vs entry Close)."
    )
    md_lines.append("Cells: fav (med/p75); dd (med/p75).")
    md_lines.append("Descriptive (not financial advice).")
    max_up = payload.get("max_move_summary_daily", {}) if isinstance(payload, dict) else {}
    buckets = max_up.get("buckets", []) if isinstance(max_up, dict) else []
    if buckets:
        md_lines.append("| Bucket | N | 1w | 4w | 3m | 6m | 9m | 1y |")
        md_lines.append("|---|---:|---|---|---|---|---|---|")

        def _fmt_pair(med: object, p75: object) -> str:
            try:
                if med is None or p75 is None:
                    return "-"
                return f"{float(med):+.1f}% / {float(p75):+.1f}%"
            except Exception:  # noqa: BLE001
                return "-"

        def _fmt_cell(fav_med: object, fav_p75: object, dd_med: object, dd_p75: object) -> str:
            fav = _fmt_pair(fav_med, fav_p75)
            dd = _fmt_pair(dd_med, dd_p75)
            if fav == "-" and dd == "-":
                return "-"
            if dd == "-":
                return fav
            if fav == "-":
                return dd
            return f"{fav}; {dd}"

        for b in buckets:
            if not isinstance(b, dict):
                continue
            n = int(b.get("n", 0) or 0)
            stats = b.get("stats", {}) if isinstance(b.get("stats"), dict) else {}

            def _get(label: str) -> str:
                s = stats.get(label, {}) if isinstance(stats.get(label), dict) else {}
                return _fmt_cell(s.get("fav_median"), s.get("fav_p75"), s.get("dd_median"), s.get("dd_p75"))

            md_lines.append(
                f"| {b.get('label', '-')} | {n} | {_get('1w')} | {_get('4w')} | {_get('3m')} | {_get('6m')} | {_get('9m')} | {_get('1y')} |"
            )
    else:
        md_lines.append("- Not available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Tail Events (Daily, all)")
    if daily_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_daily)
        max_horiz = "/".join(max_return_horizons_days.keys())
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | W pctl | W RSI | W div | Fwd pctl ({horiz}) | Max ret% ({max_horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---:|---|---|---|---|")
        for ev in daily_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_daily]
            maxrets = [(ev.get("max_fav_returns") or {}).get(str(lbl)) for lbl in max_return_horizons_days.keys()]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            wctx = ev.get("weekly_context") if isinstance(ev.get("weekly_context"), dict) else {}
            w_pct = wctx.get("extension_percentile")
            w_pct_str = "-" if w_pct is None else f"{float(w_pct):.1f}"
            w_rsi = wctx.get("rsi_regime") or "-"
            w_div = wctx.get("divergence") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {w_pct_str} | {w_rsi} | {w_div} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    md_lines.append("")
    md_lines.append("## Weekly Context")
    md_lines.append(f"- As-of (weekly): `{report_weekly.asof}`")
    md_lines.append(
        f"- Extension (weekly, ATR units): `{'-' if report_weekly.extension_atr is None else f'{report_weekly.extension_atr:+.2f}'}`"
    )
    weekly_rsi_val = None
    weekly_rsi_tag = None
    if weekly_rsi_series is not None:
        try:
            v = weekly_rsi_series.dropna()
            if not v.empty:
                weekly_rsi_val = float(v.iloc[-1])
                weekly_rsi_tag = rsi_regime_tag(
                    rsi_value=weekly_rsi_val,
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
        except Exception:  # noqa: BLE001
            weekly_rsi_val = None
            weekly_rsi_tag = None
    if weekly_rsi_val is not None and weekly_rsi_tag is not None:
        md_lines.append(f"- RSI (weekly): `{weekly_rsi_val:.1f}` (tag `{weekly_rsi_tag}`)")

    md_lines.append("")
    md_lines.append("## Current Percentiles (Weekly)")
    if report_weekly.current_percentiles:
        for years, pct in sorted(report_weekly.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Weekly p5 / p50 / p95)")
    if report_weekly.quantiles_by_window:
        for years, q in sorted(report_weekly.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Weekly)")
    if rsi_divergence_weekly is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_weekly.get("current") or {}) if isinstance(rsi_divergence_weekly, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
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

    md_lines.append("")
    md_lines.append("## Tail Events (Weekly, all)")
    if weekly_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_weekly)
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | Fwd pctl ({horiz}) | Max move% ({horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---|---|")
        for ev in weekly_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_weekly]
            maxrets = [(ev.get("forward_max_fav_returns") or {}).get(int(d)) for d in forward_days_weekly]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    return "\n".join(md_lines).rstrip() + "\n"


__all__ = ["build_extension_stats_markdown"]
