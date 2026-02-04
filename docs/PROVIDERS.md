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
- `alpaca` (stocks + options chain snapshots; volume/intraday follow-on IMPs)

Aliases:
- `yf`
- `yfinance`
- `apca`

## Alpaca setup

Install extras:
```bash
pip install -e ".[alpaca]"
```

Required env vars:
- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

Local (repo) secret file (recommended; not committed):
- Copy `config/alpaca.env.example` → `config/alpaca.env` and fill in your keys.
- The app auto-loads `config/alpaca.env` and `.env` (only `APCA_` / `OH_ALPACA_` keys), without overriding existing env vars.

Optional env vars:
- `APCA_API_BASE_URL`
- `OH_ALPACA_STOCK_FEED` (e.g., `sip` or `iex`)
- `OH_ALPACA_OPTIONS_FEED`
- `OH_ALPACA_RECENT_BARS_BUFFER_MINUTES` (default 16)

Current Alpaca coverage:
- Stock candles/history + spot quotes
- Option expiry listing + contract metadata cache
- Options chain snapshots (bid/ask/last, greeks when available)

Not yet implemented:
- Options volume from bars (IMP-025)
- Intraday trades/quotes stores (IMP-027)

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
