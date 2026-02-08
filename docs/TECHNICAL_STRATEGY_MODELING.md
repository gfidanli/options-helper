# Strategy Modeling Technical Notes

This project is **not financial advice**. Strategy-modeling outputs are informational research artifacts for decision support only.

## Scope

This page documents the implemented behavior of:
- `options-helper technicals strategy-model`
- Strategy-modeling artifacts (`summary.json`, `trades.csv`, `r_ladder.csv`, `segments.csv`, `summary.md`)
- Streamlit strategy-modeling page parity (`apps/streamlit/pages/11_Strategy_Modeling.py`)

## CLI Contract

Command:

```bash
./.venv/bin/options-helper technicals strategy-model --help
```

Primary strategy options:
- `--strategy sfp|msb|orb` (default `sfp`)
- `--allow-shorts/--no-allow-shorts` (default `--allow-shorts`)

ORB/filter options:
- `--enable-orb-confirmation/--no-enable-orb-confirmation` (default off)
- `--orb-range-minutes` (default `15`, valid `1..120`)
- `--orb-confirmation-cutoff-et` (default `10:30`, strict `HH:MM` ET 24-hour)
- `--orb-stop-policy base|orb_range|tighten` (default `base`)
- `--enable-atr-stop-floor/--no-enable-atr-stop-floor` (default off)
- `--atr-stop-floor-multiple` (default `0.5`, must be `> 0`)
- `--enable-rsi-extremes/--no-enable-rsi-extremes` (default off)
- `--enable-ema9-regime/--no-enable-ema9-regime` (default off)
- `--ema9-slope-lookback-bars` (default `3`, must be `>= 1`)
- `--enable-volatility-regime/--no-enable-volatility-regime` (default off)
- `--allowed-volatility-regimes` (default `low,normal,high`; must be non-empty, unique, values in `{low,normal,high}`)

General modeling options:
- `--symbols`, `--exclude-symbols`, `--universe-limit`
- `--start-date`, `--end-date` (ISO `YYYY-MM-DD`; start must be `<=` end)
- `--intraday-timeframe` (default `5Min`)
- `--intraday-source` (default `alpaca`)
- `--r-ladder-min-tenths`, `--r-ladder-max-tenths`, `--r-ladder-step-tenths`
- `--starting-capital` (must be `> 0`)
- `--risk-per-trade-pct` (must be `> 0` and `<= 100`)
- `--gap-fill-policy` (currently only `fill_at_open`)
- `--signal-confirmation-lag-bars`
- `--segment-dimensions`, `--segment-values`, `--segment-min-trades`, `--segment-limit`
- `--output-timezone` (default `America/Chicago`; `CST`/`CDT` alias to `America/Chicago`)
- `--out`, `--write-json/--no-write-json`, `--write-csv/--no-write-csv`, `--write-md/--no-write-md`
- `--show-progress/--no-show-progress`

Example:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --strategy msb \
  --symbols SPY,QQQ \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --intraday-timeframe 5Min \
  --enable-orb-confirmation \
  --orb-range-minutes 15 \
  --orb-confirmation-cutoff-et 10:30 \
  --orb-stop-policy tighten \
  --enable-rsi-extremes \
  --enable-ema9-regime \
  --enable-volatility-regime \
  --allowed-volatility-regimes low,normal \
  --r-ladder-min-tenths 10 \
  --r-ladder-max-tenths 20 \
  --r-ladder-step-tenths 1 \
  --out data/reports/technicals/strategy_modeling
```

## Anti-Lookahead Semantics

Global rule:
- `signal_confirmed_ts` is when a signal is knowable.
- `entry_ts` must be the first tradable bar open strictly after `signal_confirmed_ts`.
- Same-bar close entry is not allowed.

Daily close-confirmed paths (`sfp`/`msb`):
- Events are normalized with `entry_ts = next bar timestamp` in the signal frame.
- Simulation enforces next tradable regular-session open anchoring for daily workflows.

ORB primary strategy (`--strategy orb`):
- Uses regular-session intraday bars only (`09:30 <= time < 16:00` America/New_York).
- Opening range window: `09:30 <= ts < 09:30 + orb_range_minutes`.
- `opening_range_high = max(high)` and `opening_range_low = min(low)` in that window.
- Long breakout: first bar after range window with `close > opening_range_high`.
- Short breakout: first bar after range window with `close < opening_range_low`.
- `signal_ts` is the breakout bar open timestamp.
- `signal_confirmed_ts = signal_ts + (bar_duration - 1 microsecond)` with a floor of `1 microsecond`.
- Cutoff check is on confirmation time (`signal_confirmed_ts <= cutoff_et` to pass).
- `entry_ts` is the next bar open; must be strictly after confirmation.
- Stop assignment:
  - Long ORB: `stop_price = opening_range_low`
  - Short ORB: `stop_price = opening_range_high`
- At most one ORB event per session (earliest confirmed breakout wins).

ORB confirmation gate (only for base `sfp`/`msb` events):
- Evaluated on the event entry session day (derived from event `entry_ts`).
- Requires opening-range availability for that session.
- Requires a directional ORB breakout by cutoff.
- On pass, mutates event timing:
  - `signal_confirmed_ts -> orb_breakout_confirmed_ts`
  - `entry_ts -> next bar open after breakout confirmation`
- Stop policy:
  - `base`: keep base stop
  - `orb_range`: replace with ORB range stop
  - `tighten`: long uses `max(base_stop, orb_stop)`, short uses `min(base_stop, orb_stop)`
- On miss, rejects with `orb_opening_range_missing` or `orb_breakout_missing`.

## Filter Engine

Filter config model: `options_helper.schemas.strategy_modeling_filters.StrategyEntryFilterConfig`

Defaults:
- `allow_shorts=True`
- all optional filters/gates off
- `orb_range_minutes=15`
- `orb_confirmation_cutoff_et="10:30"`
- `orb_stop_policy="base"`
- `atr_stop_floor_multiple=0.5`
- `ema9_slope_lookback_bars=3`
- `allowed_volatility_regimes=("low","normal","high")`

Evaluation order (stable deterministic order):
1. `allow_shorts` gate
2. Daily-context availability check (only if daily filters are enabled)
3. RSI extremes
4. EMA9 regime
5. Volatility regime allowlist
6. ORB confirmation gate (`sfp`/`msb` only)
7. ATR stop floor

Daily indicator anchoring:
- `sfp`/`msb`: anchor on event `signal_ts` day.
- `orb`: anchor on most recent completed daily bar strictly before ORB signal day.

Filter formulas:
- RSI extremes:
  - Long passes if `rsi <= rsi_oversold`
  - Short passes if `rsi >= rsi_overbought`
- EMA9 regime:
  - Long passes if `close >= ema9` and `ema9_slope_filter >= 0`
  - Short passes if `close <= ema9` and `ema9_slope_filter <= 0`
- Volatility regime:
  - Resolve `volatility_regime`, fallback `realized_vol_regime`
  - Pass if resolved regime is in `allowed_volatility_regimes`
- ATR floor:
  - `initial_risk = abs(entry_price_estimate - stop_price)`
  - Pass if `(initial_risk / atr) >= atr_stop_floor_multiple`
  - Entry price estimate prefers first intraday open at/after `entry_ts` in that session, then falls back to `signal_close`, then `signal_open`

Stable reject codes:
- `shorts_disabled`
- `missing_daily_context`
- `rsi_not_extreme`
- `ema9_regime_mismatch`
- `volatility_regime_disallowed`
- `orb_opening_range_missing`
- `orb_breakout_missing`
- `atr_floor_failed`

## Directional Metrics and Target-Ladder Semantics

Simulation and portfolio semantics:
- Trade simulation runs for the full target ladder (`trade_simulations` and `r_ladder` include all targets).
- Portfolio construction uses one selected target subset.
- Preferred portfolio target is the first ladder entry (`target_ladder[0]`, label and `target_r`).
- If preferred label/r cannot be matched, selection falls back deterministically (`inferred_first_target`, else `all_trades`).

Portfolio-target subset drives:
- portfolio ledger
- accepted/skipped trade ids
- portfolio metrics
- segmentation
- directional counterfactual reruns

`directional_metrics` payload:
- `counts`
- `portfolio_target` (`target_label`, `target_r`, `selection_source`)
- `combined`
- `long_only`
- `short_only`

Each directional bucket contains serialized metrics (`portfolio_metrics`) plus:
- `simulated_trade_count`
- `closed_trade_count`
- `accepted_trade_count`
- `skipped_trade_count`

## Artifact Contract

Output path:
- `data/reports/technicals/strategy_modeling/{strategy}/{as_of}/`
- `as_of` resolution order: `run_result.as_of` -> request `end_date` -> run date

Files:
- `summary.json`
- `trades.csv`
- `r_ladder.csv`
- `segments.csv`
- `summary.md`

`summary.json` top-level keys:
- `schema_version`
- `generated_at`
- `strategy`
- `requested_symbols`
- `modeled_symbols`
- `disclaimer`
- `summary`
- `policy_metadata`
- `filter_metadata`
- `filter_summary`
- `directional_metrics`
- `metrics`
- `r_ladder`
- `segments`
- `trade_log`

`summary` keys:
- `signal_event_count`
- `trade_count`
- `accepted_trade_count`
- `skipped_trade_count`
- `losses_below_minus_one_r`

`policy_metadata` keys include:
- `entry_anchor` (`next_bar_open`)
- `entry_ts_anchor_policy`
- `signal_confirmation_lag_bars`
- `require_intraday_bars`
- `intraday_timeframe`
- `intraday_source`
- `gap_fill_policy`
- `intra_bar_tie_break_rule`
- `output_timezone`
- `max_hold_bars`
- `risk_per_trade_pct`
- `sizing_rule`
- `one_open_per_symbol`
- `price_adjustment_policy`
- `anti_lookahead_note`

`filter_metadata` keys include:
- full effective filter config values
- `active_filters`
- `reject_reason_order`
- `parsed_orb_range_minutes`
- `parsed_orb_confirmation_cutoff_et` (`hour`, `minute`, `minutes_since_midnight`)
- `orb_cache_symbol_count`
- `orb_cache_session_count`

`filter_summary` keys:
- `base_event_count`
- `kept_event_count`
- `rejected_event_count`
- `reject_counts`

`trades.csv` fields include:
- `trade_id`, `event_id`, `symbol`, `direction`, `status`
- `signal_confirmed_ts`, `entry_ts`, `exit_ts`
- `entry_price_source`, `entry_price`, `stop_price`, `target_price`, `exit_price`
- `initial_risk`, `realized_r`, `stop_slippage_r`
- `gap_fill_applied`, `gap_exit`, `loss_below_1r`
- `exit_reason`, `reject_code`

## Guardrails

- Treat outputs as informational research artifacts, not trade instructions.
- Preserve anti-lookahead entry anchoring in all modeling and interpretation.
- Keep gap-through-stop outcomes visible (`realized_r < -1.0R` can occur under `fill_at_open`).
- Treat missing/stale intraday data as a hard reliability signal for strategy-model runs.
