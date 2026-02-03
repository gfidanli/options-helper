# Market Data Providers

This repo’s market data access is routed through a small **provider interface**, so we can swap Yahoo/yfinance for a higher-quality data source later without rewriting analysis code.

> Reminder: this project is informational/educational only — not financial advice.

## CLI usage

All commands accept a global `--provider` option:

```bash
options-helper --provider yahoo analyze portfolio.json
options-helper --provider yahoo snapshot-options portfolio.json
```

Currently implemented providers:
- `yahoo` (default)

Aliases:
- `yf`
- `yfinance`

## Provider contract (dev)

The interface lives in:
- `options_helper/data/providers/base.py`

Key requirements:
- `list_option_expiries(symbol) -> list[date]`
- `get_options_chain(symbol, expiry) -> OptionsChain`
- `get_history(symbol, start/end, interval, auto_adjust, back_adjust) -> DataFrame`

Option chains should include a normalized set of columns (missing values are allowed):
- `contractSymbol`, `expiry`, `strike`, `optionType`, `bid`, `ask`, `lastPrice`, `openInterest`, `volume`, `impliedVolatility`

## Adding a new provider

1) Implement the protocol (see `MarketDataProvider`).
2) Register it in `options_helper/data/providers/__init__.py`.
3) Ensure option-chain normalization and add offline tests by monkeypatching `options_helper.cli.get_provider`.
