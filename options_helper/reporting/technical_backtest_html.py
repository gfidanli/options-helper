from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from typing import Any, Mapping, Sequence

from options_helper.schemas.technical_backtest_batch import (
    TechnicalBacktestBatchEquityPoint,
    TechnicalBacktestBatchSummaryArtifact,
    validate_technical_backtest_batch_summary_payload,
)

_CHART_WIDTH = 860
_CHART_HEIGHT = 260
_CHART_PADDING = 28


@dataclass(frozen=True)
class _ChartSeries:
    label: str
    color: str
    values: list[float | None]


def _coerce_summary(
    payload: TechnicalBacktestBatchSummaryArtifact | Mapping[str, Any],
) -> TechnicalBacktestBatchSummaryArtifact:
    if isinstance(payload, TechnicalBacktestBatchSummaryArtifact):
        return payload
    return validate_technical_backtest_batch_summary_payload(payload)


def _fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _fmt_number(value: float | None, *, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_money(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"${value:,.2f}"


def _render_metric_cards(summary: TechnicalBacktestBatchSummaryArtifact) -> str:
    cards = [
        ("Total Return", _fmt_pct(summary.aggregate_metrics.total_return_pct)),
        ("Win Rate", _fmt_pct(summary.aggregate_metrics.win_rate)),
        ("Profit Factor", _fmt_number(summary.aggregate_metrics.profit_factor)),
        ("Max Drawdown", _fmt_pct(summary.aggregate_metrics.max_drawdown_pct)),
    ]
    chunks = ['<div class="card-grid">']
    for label, value in cards:
        chunks.append(
            "<article class=\"card\">"
            f"<h3>{escape(label)}</h3>"
            f"<p>{escape(value)}</p>"
            "</article>"
        )
    chunks.append("</div>")
    return "".join(chunks)


def _extract_dates(points: Sequence[TechnicalBacktestBatchEquityPoint]) -> list[str]:
    return [point.session_date.isoformat() for point in points]


def _series_min_max(series: Sequence[_ChartSeries]) -> tuple[float, float] | None:
    values: list[float] = []
    for item in series:
        values.extend(value for value in item.values if value is not None and math.isfinite(value))
    if not values:
        return None
    lower = min(values)
    upper = max(values)
    if lower == upper:
        pad = abs(lower) * 0.05 or 1.0
        return lower - pad, upper + pad
    return lower, upper


def _point_coordinates(
    *,
    index: int,
    total: int,
    value: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    x_span = _CHART_WIDTH - (2 * _CHART_PADDING)
    y_span = _CHART_HEIGHT - (2 * _CHART_PADDING)
    x = _CHART_PADDING if total <= 1 else _CHART_PADDING + (index / (total - 1)) * x_span
    scale = 0.0 if upper == lower else (value - lower) / (upper - lower)
    y = (_CHART_HEIGHT - _CHART_PADDING) - (scale * y_span)
    return x, y


def _series_polyline(
    *,
    values: Sequence[float | None],
    lower: float,
    upper: float,
) -> str:
    points: list[str] = []
    total = len(values)
    for idx, value in enumerate(values):
        if value is None or not math.isfinite(value):
            continue
        x, y = _point_coordinates(index=idx, total=total, value=float(value), lower=lower, upper=upper)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_chart(
    *,
    title: str,
    dates: Sequence[str],
    series: Sequence[_ChartSeries],
) -> str:
    if not dates or not _series_min_max(series):
        return f"<div class=\"chart\"><h3>{escape(title)}</h3><p class=\"empty\">No data available.</p></div>"

    bounds = _series_min_max(series)
    assert bounds is not None
    lower, upper = bounds
    lines: list[str] = [
        "<div class=\"chart\">",
        f"<h3>{escape(title)}</h3>",
        f"<svg viewBox=\"0 0 {_CHART_WIDTH} {_CHART_HEIGHT}\" role=\"img\" aria-label=\"{escape(title)}\">",
        f"<line class=\"axis\" x1=\"{_CHART_PADDING}\" y1=\"{_CHART_HEIGHT - _CHART_PADDING}\" "
        f"x2=\"{_CHART_WIDTH - _CHART_PADDING}\" y2=\"{_CHART_HEIGHT - _CHART_PADDING}\" />",
        f"<line class=\"axis\" x1=\"{_CHART_PADDING}\" y1=\"{_CHART_PADDING}\" "
        f"x2=\"{_CHART_PADDING}\" y2=\"{_CHART_HEIGHT - _CHART_PADDING}\" />",
    ]
    for item in series:
        polyline = _series_polyline(values=item.values, lower=lower, upper=upper)
        if not polyline:
            continue
        lines.append(
            f"<polyline points=\"{polyline}\" stroke=\"{item.color}\" stroke-width=\"2.5\" fill=\"none\" />"
        )
    lines.append("</svg>")
    lines.append("<p class=\"axis-label\">")
    lines.append(f"{escape(dates[0])} to {escape(dates[-1])}")
    lines.append("</p>")
    legend = " · ".join(
        f"<span><i style=\"background:{escape(item.color)}\"></i>{escape(item.label)}</span>"
        for item in series
    )
    lines.append(f"<p class=\"legend\">{legend}</p>")
    lines.append("</div>")
    return "".join(lines)


def _render_equity_drawdown(summary: TechnicalBacktestBatchSummaryArtifact) -> str:
    points = summary.equity_curve
    if not points:
        return (
            "<section id=\"equity-drawdown\">"
            "<h2>Equity + Drawdown (Strategy vs SPY)</h2>"
            "<p class=\"empty\">No equity history available.</p>"
            "</section>"
        )

    dates = _extract_dates(points)
    equity_series = [
        _ChartSeries(
            label="Strategy Equity",
            color="#0d9488",
            values=[float(point.aggregate_equity) for point in points],
        ),
        _ChartSeries(
            label="SPY Equity",
            color="#2563eb",
            values=[
                None if point.benchmark_equity is None else float(point.benchmark_equity)
                for point in points
            ],
        ),
    ]
    drawdown_series = [
        _ChartSeries(
            label="Strategy Drawdown",
            color="#be123c",
            values=[
                None if point.aggregate_drawdown_pct is None else float(point.aggregate_drawdown_pct)
                for point in points
            ],
        ),
        _ChartSeries(
            label="SPY Drawdown",
            color="#7c3aed",
            values=[
                None if point.benchmark_drawdown_pct is None else float(point.benchmark_drawdown_pct)
                for point in points
            ],
        ),
    ]
    equity_chart = _render_chart(title="Equity Curves", dates=dates, series=equity_series)
    drawdown_chart = _render_chart(title="Drawdown Curves", dates=dates, series=drawdown_series)
    return (
        "<section id=\"equity-drawdown\">"
        "<h2>Equity + Drawdown (Strategy vs SPY)</h2>"
        f"{equity_chart}{drawdown_chart}"
        "</section>"
    )


def _render_returns_table(
    *,
    title: str,
    section_id: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    if not rows:
        return f"<section id=\"{escape(section_id)}\"><h2>{escape(title)}</h2><p class=\"empty\">No rows available.</p></section>"

    table = ["<table><thead><tr>"]
    table.extend(f"<th>{escape(header)}</th>" for header in headers)
    table.append("</tr></thead><tbody>")
    for row in rows:
        table.append("<tr>")
        table.extend(f"<td>{escape(cell)}</td>" for cell in row)
        table.append("</tr>")
    table.append("</tbody></table>")
    table_html = "".join(table)
    return f"<section id=\"{escape(section_id)}\"><h2>{escape(title)}</h2>{table_html}</section>"


def _monthly_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in summary.monthly_returns:
        rows.append(
            [
                f"{item.year:04d}-{item.month:02d}",
                _fmt_pct(item.aggregate_return_pct),
                _fmt_pct(item.benchmark_return_pct),
            ]
        )
    return rows


def _yearly_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in summary.yearly_returns:
        rows.append(
            [
                f"{item.year:04d}",
                _fmt_pct(item.aggregate_return_pct),
                _fmt_pct(item.benchmark_return_pct),
            ]
        )
    return rows


def _render_document(summary: TechnicalBacktestBatchSummaryArtifact) -> str:
    metrics = _render_metric_cards(summary)
    equity_drawdown = _render_equity_drawdown(summary)
    monthly = _render_returns_table(
        title="Monthly Returns",
        section_id="monthly-returns",
        headers=("Month", "Strategy", "SPY"),
        rows=_monthly_rows(summary),
    )
    yearly = _render_returns_table(
        title="Yearly Returns",
        section_id="yearly-returns",
        headers=("Year", "Strategy", "SPY"),
        rows=_yearly_rows(summary),
    )
    generated = escape(summary.generated_at.isoformat())
    run_id = escape(summary.run_id)
    strategy = escape(summary.strategy)
    period = f"{escape(summary.period_start.isoformat())} to {escape(summary.period_end.isoformat())}"
    ending = _fmt_money(summary.aggregate_metrics.ending_equity)
    disclaimer = escape(summary.disclaimer)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Technical Backtest Batch Report</title>
  <style>
    :root {{ --bg:#f6f8fc; --card:#ffffff; --ink:#0f172a; --muted:#475569; --border:#dbe1ea; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:linear-gradient(180deg,#eef4ff 0%,var(--bg) 220px); color:var(--ink); font:15px/1.45 "Avenir Next", "Segoe UI", sans-serif; }}
    main {{ max-width:980px; margin:0 auto; padding:20px 16px 28px; }}
    h1,h2,h3 {{ margin:0 0 10px; }}
    p {{ margin:6px 0; }}
    .meta {{ color:var(--muted); }}
    section {{ margin-top:18px; border:1px solid var(--border); background:var(--card); border-radius:12px; padding:14px; }}
    .card-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
    .card {{ border:1px solid var(--border); border-radius:10px; padding:10px; background:#fbfdff; }}
    .card h3 {{ font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .card p {{ font-size:23px; margin-top:4px; }}
    .chart {{ margin-top:14px; border:1px solid var(--border); border-radius:10px; padding:10px; background:#fcfdff; }}
    svg {{ width:100%; height:auto; display:block; }}
    .axis {{ stroke:#cbd5e1; stroke-width:1; }}
    .axis-label,.legend {{ margin-top:6px; color:var(--muted); font-size:12px; }}
    .legend span {{ margin-right:12px; white-space:nowrap; }}
    .legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:8px 6px; border-bottom:1px solid var(--border); text-align:left; }}
    th {{ font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }}
    .empty {{ color:var(--muted); }}
    footer {{ margin-top:14px; color:var(--muted); font-size:12px; }}
  </style>
</head>
<body>
  <main>
    <h1>Technical Backtest Batch Report</h1>
    <p class="meta">Run ID: <strong>{run_id}</strong> · Strategy: <strong>{strategy}</strong></p>
    <p class="meta">Period: <strong>{period}</strong> · Ending Equity: <strong>{escape(ending)}</strong> · Generated: {generated}</p>
    <section id="headline-metrics">
      <h2>Headline Metrics</h2>
      {metrics}
    </section>
    {equity_drawdown}
    {monthly}
    {yearly}
    <footer>{disclaimer}</footer>
  </main>
</body>
</html>
"""


def render_technical_backtest_html(
    payload: TechnicalBacktestBatchSummaryArtifact | Mapping[str, Any],
) -> str:
    summary = _coerce_summary(payload)
    return _render_document(summary)


__all__ = ["render_technical_backtest_html"]
