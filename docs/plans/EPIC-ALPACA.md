# EPIC — Alpaca market data provider integration (stocks + options)

- **Status:** draft
- **Effort:** L–XL (split across IMPs)
- **Alpha potential:** Very High

## Summary
Add **Alpaca** as a first-class market data provider for **stocks + listed options**. The goal is not only to replace
yfinance for existing workflows (candles, quotes, option chains, snapshots), but also to **pull and persist richer market data**
(quotes/trades, greeks, corporate actions, news, streaming) so new features can be built on top with minimal additional provider work.

This EPIC is intentionally decomposed into small, PR-sized IMPs with clear dependencies and acceptance criteria.

## Provider stack
- **SDK:** `alpaca-py` (official Alpaca Python SDK)
- **APIs used:**
  - Market Data API (historical + snapshots)
  - Trading API (option contracts metadata; open interest is exposed here)

## Plan inventory (implementation order + deps)

| Order | Plan | Status | Effort | Depends on | Key outputs |
|---:|---|---|:---:|---|---|
| 1 | IMP-021 — Alpaca provider scaffold + config surface | done | S | — | Provider can be selected; SDK wired; helpful errors |
| 2 | IMP-022 — Stocks: candles/history + quote + underlying | done | S–M | IMP-021 | CandleStore + spot workflows work with Alpaca |
| 3 | IMP-023 — Options contract universe + expiry listing + contract cache | done | M | IMP-021 | Fast expiry listing; contract metadata cache (OI, strike, etc.) |
| 4 | IMP-024 — Options chain snapshots → normalized calls/puts + raw payload | done | M | IMP-021, IMP-022, IMP-023 | Snapshotter works end-to-end with Alpaca |
| 5 | IMP-025 — Options volume + “recent bars restriction” hardening | draft | M | IMP-024 | Better liquidity metrics; fewer provider-plan footguns |
| 6 | IMP-026 — Snapshot enrichment: store more fields (greeks, timestamps, OI dates) | draft | S–M | IMP-024 | Future features unlocked without re-snapshotting |
| 7 | IMP-027 — Intraday microstructure stores (stock/option trades+quotes+bars) | draft | L | IMP-021 | New offline analytics: spreads, staleness, RV intraday |
| 8 | IMP-028 — Corporate actions + news ingestion (event-aware analytics) | draft | M | IMP-021 | Split/dividend awareness; news context (when permitted) |
| 9 | IMP-029 — Streaming adapters (optional) | draft | M–L | IMP-021 (and ideally IMP-027) | Real-time watchlists, alerts, continuous capture |

## Critical design decisions (make these once)
1. **Symbol normalization**
   - Repo canonical for underlyings is currently “dash” style (e.g. `BRK-B`).
   - Alpaca commonly uses “dot” style (e.g. `BRK.B`).
   - Standardize on repo canonical (`options_helper.analysis.osi.normalize_underlying`) in all downstream analysis and caches.
   - Add a provider-local mapper to translate repo symbols → Alpaca request symbols.

2. **Options identifiers**
   - Alpaca option contract symbols are OSI-like and should parse with existing `options_helper.analysis.osi.parse_contract_symbol`.
   - We will keep storing `contractSymbol` and (already implemented) `osi` for stable joins.

3. **Open interest availability**
   - Alpaca exposes OI via option contracts endpoints (Trading API). In snapshots, store both:
     - `openInterest`
     - `openInterestDate` (to communicate staleness)

4. **Plan-based data limitations**
   - Some accounts/plans cannot query “most recent ~15 minutes” of historical bars.
   - Build a single configurable mitigation: `end = now - buffer_minutes`, default 16 minutes, used in any bar-based fetch.

## Execution guidelines
- Keep tests offline and deterministic (mock Alpaca SDK clients).
- Ensure provider failures raise `DataFetchError` with actionable hints (missing keys, plan limitation, symbol format).
- Prefer *caching* (contract metadata, intraday partitions) to avoid cost + rate limits.
- Ensure snapshot artifacts remain backward compatible (extra columns are additive; meta is versioned).

## Definition of done
- Existing CLI workflows run with `--provider alpaca`:
  - `analyze`, `research`, `scan`, snapshotting, technical backtests
- Snapshots include required normalized schema columns plus Alpaca-enriched optional fields
- New stores exist for intraday quotes/trades/bars (even if not used by all commands yet)
- Clear documentation for setup and limitations
