# Watchlists — Design & Usage

## Overview
Watchlists are named collections of ticker symbols stored locally as JSON.

Use them to:
- keep a reusable list of symbols to research
- separate “symbols I already own” from “symbols I’m watching”

## Storage
Default path used by the CLI:
- `data/watchlists.json`

Format:

```json
{
  "watchlists": {
    "positions": ["CVX", "UROY"],
    "watchlist": ["IREN"]
  }
}
```

Symbols are normalized to uppercase and de-duplicated.

## Commands
Initialize the two starter watchlists:

```bash
options-helper watchlists init portfolio.json
```

This creates:
- `positions`: unique symbols from `portfolio.json`
- `watchlist`: seeded with `IREN`

List all watchlists:

```bash
options-helper watchlists list
```

Show a specific watchlist:

```bash
options-helper watchlists show watchlist
```

Add/remove symbols:

```bash
options-helper watchlists add watchlist IREN
options-helper watchlists remove watchlist IREN
```

Keep the `positions` watchlist synced to your portfolio:

```bash
options-helper watchlists sync-positions portfolio.json
```

## Custom path
All commands accept `--path` to use a different file:

```bash
options-helper watchlists list --path /some/where/watchlists.json
```

