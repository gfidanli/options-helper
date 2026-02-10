# Strategy Modeling Profiles

This project is **not financial advice**. Profiles are local convenience settings for repeatable research runs.

## Purpose

Strategy-modeling profiles let you save a named set of modeling inputs once and reuse them from:
- CLI: `options-helper technicals strategy-model`
- Streamlit: `apps/streamlit/pages/11_Strategy_Modeling.py`

Profiles store run inputs only. Export/report toggles (output paths, write flags, timezone/export preferences) are intentionally excluded.

## Store Path and Schema

Default path:

- `config/strategy_modeling_profiles.json`

Top-level schema (`schema_version=1`):

- `schema_version`
- `updated_at` (UTC ISO timestamp)
- `profiles` (mapping of profile name -> normalized profile payload)

Each profile stores validated strategy-modeling inputs including:

- strategy + symbols + date range
- intraday timeframe/source
- portfolio policy and target-ladder settings
- filter gates (ORB/ATR/RSI/EMA9/volatility regimes)
- MA/trend signal parameters

## CLI Usage

Load a profile:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --profile swing_orb
```

Load a profile from a custom file:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --profile-path /tmp/strategy_profiles.json \
  --profile swing_orb
```

Save effective inputs as a profile:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --strategy orb \
  --symbols SPY \
  --save-profile orb_fast
```

Overwrite an existing profile explicitly:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --profile-path config/strategy_modeling_profiles.json \
  --save-profile orb_fast \
  --overwrite-profile
```

## Dashboard Workflow

In the Strategy Modeling sidebar:

1. Set **Profile store path**
2. Choose a saved profile from **Saved profiles**
3. Click **Load Profile** to populate widget state
4. Enter a profile name and click **Save Profile** to persist current settings
5. Use **Overwrite existing profile** only when replacing a profile intentionally

## Precedence and Safety Rules

- Load order: profile values are applied first.
- Explicit CLI flags override loaded profile values.
- Dashboard edits after loading override loaded values for that run/save action.
- Saving to an existing profile name fails unless overwrite is explicitly enabled.
