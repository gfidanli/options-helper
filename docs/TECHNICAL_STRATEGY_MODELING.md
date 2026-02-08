# Strategy Modeling Technical Notes

This project is **not financial advice**. Strategy-modeling outputs are research tooling for position-management decisions.

## T0 Policy Contract (Locked Defaults)

Baseline policy contract: `options_helper.schemas.strategy_modeling_policy.StrategyModelingPolicyConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `require_intraday_bars` | `bool` | `true` | Modeling runs require intraday bars for the requested scope; missing coverage blocks simulation. |
| `max_hold_bars` | `int` | `20` | Time-stop after 20 bars if stop/target logic does not exit earlier. |
| `sizing_rule` | `Literal["risk_pct_of_equity"]` | `"risk_pct_of_equity"` | Size each trade from a fixed risk percent of current equity. |
| `risk_per_trade_pct` | `float` | `1.0` | Per-trade risk budget as a percent of current equity. |
| `one_open_per_symbol` | `bool` | `true` | At most one open position per symbol at a time. |
| `gap_fill_policy` | `Literal["fill_at_open"]` | `"fill_at_open"` | If price gaps through stop/target, fill at that bar open (realized `R` can be below `-1.0R`). |
| `entry_ts_anchor_policy` | `Literal["first_tradable_bar_open_after_signal_confirmed_ts"]` | `"first_tradable_bar_open_after_signal_confirmed_ts"` | Entry timestamp anchor policy for anti-lookahead simulation. |
| `price_adjustment_policy` | `Literal["adjusted_ohlc"]` | `"adjusted_ohlc"` | Use adjusted OHLC consistently for signal generation and simulation. |

## Anti-Lookahead Session Anchor

`signal_confirmed_ts` is the timestamp when a signal becomes knowable.

For all close-confirmed signals:
- `entry_ts` must be the **first tradable intraday bar open strictly after `signal_confirmed_ts`**.
- If the immediate next calendar bar is not tradable (weekend, holiday, missing bar), anchor to the next available tradable bar open.
- Same-bar fills at signal close are disallowed by policy.

This anchor rule is required to avoid lookahead bias in both CLI and Streamlit modeling flows.

## Override Contract

CLI/Streamlit override payloads are parsed through:
- `options_helper.analysis.strategy_modeling_policy.parse_strategy_modeling_policy_config`

Parsing is strict:
- Unknown fields are rejected (`extra="forbid"`).
- Type/domain validation is enforced (`max_hold_bars >= 1`, `0 < risk_per_trade_pct <= 100`).

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
