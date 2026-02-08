# Strategy Modeling Technical Notes

This project is **not financial advice**. Strategy-modeling outputs are informational research artifacts for decision support only.

## Scope

Strategy-modeling documentation in this repo currently covers:
- CLI modeling runs via `options-helper technicals strategy-model`.
- Artifact outputs (`summary.json`, `trades.csv`, `r_ladder.csv`, `segments.csv`, `summary.md`).
- Streamlit research pages used to review related SFP/MSB signal context (`09_SFP.py`, `10_MSB.py`).

## Methodology and Policy Assumptions

Baseline policy contract: `options_helper.schemas.strategy_modeling_policy.StrategyModelingPolicyConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `require_intraday_bars` | `bool` | `true` | Modeling runs require intraday bars for requested scope; missing coverage blocks simulation. |
| `max_hold_bars` | `int` | `20` | Time-stop after 20 bars if stop/target logic does not exit earlier. |
| `sizing_rule` | `Literal["risk_pct_of_equity"]` | `"risk_pct_of_equity"` | Size each trade from a fixed risk percent of current equity. |
| `risk_per_trade_pct` | `float` | `1.0` | Per-trade risk budget as a percent of current equity. |
| `one_open_per_symbol` | `bool` | `true` | At most one open position per symbol at a time. |
| `gap_fill_policy` | `Literal["fill_at_open"]` | `"fill_at_open"` | If price gaps through stop/target, fill at that bar open (realized `R` can be below `-1.0R`). |
| `entry_ts_anchor_policy` | `Literal["first_tradable_bar_open_after_signal_confirmed_ts"]` | `"first_tradable_bar_open_after_signal_confirmed_ts"` | Entry timestamp anchor policy for anti-lookahead simulation. |
| `price_adjustment_policy` | `Literal["adjusted_ohlc"]` | `"adjusted_ohlc"` | Use adjusted OHLC consistently for signal generation and simulation. |

### Anti-lookahead anchor (required)

`signal_confirmed_ts` is when a signal becomes knowable.

For close-confirmed signals:
- `entry_ts` must be the first tradable bar open strictly after `signal_confirmed_ts`.
- Same-bar close fills are disallowed.
- When immediate next bars are non-tradable or missing, anchor at the next tradable bar open.

### Strict override parsing

CLI/Streamlit override payloads parse through:
- `options_helper.analysis.strategy_modeling_policy.parse_strategy_modeling_policy_config`

Validation behavior:
- Unknown fields rejected (`extra="forbid"`).
- Domain constraints enforced (`max_hold_bars >= 1`, `0 < risk_per_trade_pct <= 100`).

## CLI Usage

Command:

```bash
./.venv/bin/options-helper technicals strategy-model --help
```

Example run:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --strategy sfp \
  --symbols SPY,QQQ \
  --exclude-symbols QQQ \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --intraday-timeframe 5Min \
  --intraday-source alpaca \
  --r-ladder-min-tenths 10 \
  --r-ladder-max-tenths 20 \
  --r-ladder-step-tenths 1 \
  --starting-capital 100000 \
  --risk-per-trade-pct 1.0 \
  --gap-fill-policy fill_at_open \
  --segment-dimensions symbol,direction \
  --segment-values SPY,long \
  --segment-min-trades 2 \
  --segment-limit 10 \
  --out data/reports/technicals/strategy_modeling
```

Implemented option behavior:
- `--strategy` accepts only `sfp` or `msb`.
- `--start-date` / `--end-date` must be ISO (`YYYY-MM-DD`) and start must be <= end.
- `--gap-fill-policy` currently supports only `fill_at_open`.
- `--risk-per-trade-pct` must be `> 0` and `<= 100`.
- `--universe-limit`, `--segment-min-trades`, `--segment-limit` must be `>= 1` when provided.
- Intraday preflight failures stop the run with blocked symbol coverage details.

Output location:
- `data/reports/technicals/strategy_modeling/{strategy}/{as_of}/`
- `as_of` resolves from run result `as_of`, then request `end_date`, then run date.

Generated files:
- `summary.json`
- `trades.csv`
- `r_ladder.csv`
- `segments.csv`
- `summary.md`

## Dashboard Usage

Launch Streamlit portal:

```bash
./.venv/bin/options-helper ui
```

Current related pages:
- `09_SFP.py` (SFP research)
- `10_MSB.py` (MSB research)

These pages are read-only research views and include:
- DuckDB path selector.
- Lookback window controls.
- Extension-threshold filters.
- Swing/RSI advanced settings.

Important:
- These pages are signal-context dashboards, not portfolio strategy-modeling simulators.
- Use CLI artifacts (`summary.json`, CSV outputs) as the source of truth for strategy-modeling performance metrics.

## Metric and Artifact Definitions

`summary.json` top-level:
- `schema_version`: artifact schema version (`1`).
- `generated_at`: UTC timestamp.
- `strategy`: normalized strategy id.
- `requested_symbols`, `modeled_symbols`.
- `disclaimer`: informational-only / not financial advice text.
- `summary`: aggregate run counts.
- `policy_metadata`: effective policy + anti-lookahead metadata.
- `metrics`: portfolio-level metric payload returned by service.
- `r_ladder`: target-level hit-rate/expectancy rows.
- `segments`: segmented performance rows.
- `trade_log`: per-trade simulation rows.

`summary` fields:
- `signal_event_count`
- `trade_count`
- `accepted_trade_count`
- `skipped_trade_count`
- `losses_below_minus_one_r`

`policy_metadata` fields include:
- `entry_anchor=next_bar_open`
- `entry_ts_anchor_policy`
- `signal_confirmation_lag_bars`
- `require_intraday_bars`
- `intraday_timeframe`, `intraday_source`
- `gap_fill_policy`
- `intra_bar_tie_break_rule`
- `risk_per_trade_pct`, `sizing_rule`
- `anti_lookahead_note`

`trades.csv` includes:
- timestamps: `signal_confirmed_ts`, `entry_ts`, `exit_ts`
- prices: `entry_price`, `stop_price`, `target_price`, `exit_price`
- risk/outcome: `initial_risk`, `realized_r`, `stop_slippage_r`
- gap diagnostics: `gap_fill_applied`, `gap_exit`, `loss_below_1r`
- status/debug: `status`, `exit_reason`, `reject_code`

`r_ladder.csv` preferred fields:
- `target_label`, `target_r`
- `trade_count`, `hit_count`, `hit_rate`
- `avg_bars_to_hit`, `median_bars_to_hit`
- `expectancy_r`

`segments.csv` preferred fields:
- `segment_dimension`, `segment_value`
- `trade_count`, `win_rate`
- `avg_realized_r`, `expectancy_r`
- `profit_factor`, `sharpe_ratio`, `max_drawdown_pct`

## Segmentation Caveats

- Segment metrics are descriptive, not causal.
- Small samples are unstable; use `--segment-min-trades` and inspect `trade_count`.
- Segment filters (`--segment-dimensions`, `--segment-values`) prioritize output slices; they do not create out-of-sample validation.
- Compare segment results against overall metrics to avoid overfitting single slices.

## Guardrails

- Explicitly treat outputs as informational research, not trade advice.
- Preserve anti-lookahead semantics: execute from next tradable bar open after confirmation.
- Keep gap-through-stop outcomes visible (`realized_r < -1.0R` can occur with `fill_at_open`).
- Treat missing/stale market data as a reliability risk and validate coverage before interpretation.

## Universe-Scale Performance Smoke Gate (T18)

Performance smoke test: `tests/test_strategy_modeling_performance.py`

Deterministic benchmark setup:
- `300` symbols (`SYM000..SYM299`)
- `12` simulated trades per symbol (`3,600` total)
- Full artifact write enabled (`summary.json`, `trades.csv`, `r_ladder.csv`, `segments.csv`, `summary.md`)
- Fixed timestamps and fixed synthetic trade/segment payloads (no randomness, no network)

CI smoke thresholds (single test run):
- Runtime: `<= 1.5s` for one universe-scale artifact write
- Peak memory (Python `tracemalloc`): `<= 32 MiB`

Validation command:
- `./.venv/bin/python -m pytest tests/test_strategy_modeling_performance.py -q`
