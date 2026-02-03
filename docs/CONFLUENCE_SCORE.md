# Confluence Score (Explainable Conviction)

The **confluence score** is a simple, explainable blend of technical direction, extension context,
flow alignment, and volatility regime. It is a **decision-support heuristic** — not financial advice.

## What it is
- **Total score**: 0–100 (higher = more aligned signals).
- **Coverage**: 0–100% (how much of the score is based on available inputs).
- **Components**: per-signal contributions with human-readable reasons.

Missing inputs **do not crash** the score; they reduce coverage and keep the score near neutral.

## Inputs (v1)
- **Weekly trend**: up / down / flat
- **Extension percentile**: tail vs neutral (low/high tails are interpreted vs trend)
- **Flow alignment**: net ΔOI notional direction vs trend
- **RSI divergence**: bullish/bearish vs trend (optional; best-effort)
- **IV/RV regime**: high vs low IV/RV20

## Default weights (points)
| Component | Weight |
|---|---:|
| Weekly trend | 25 |
| Extension percentile | 10 |
| Flow alignment | 20 |
| RSI divergence | 10 |
| IV/RV regime | 5 |

Exact thresholds + weights are configurable in `config/confluence.yaml`.

## Output locations
- **Research**: printed per symbol (score + components); watchlist ordering uses the score.
- **Scanner**: shortlist ordering uses the score; `shortlist.md` includes score + coverage.
- **Briefing JSON**: `sections[].confluence` contains the full object (components + warnings).

## Config
Default config lives at:
- `config/confluence.yaml`
- `config/confluence.schema.json`

Tune weights/thresholds there; no code changes required.

## Notes / caveats
- The score is **not a prediction** — it’s a transparent summary of signal alignment.
- Yahoo data can be stale; flow + IV inputs are **best-effort**.
- Use `coverage` to gauge how much of the score is supported by real inputs.
