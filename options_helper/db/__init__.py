"""DuckDB warehouse (embedded, local-first).

This package holds:
- connection helpers
- schema migrations
- (v1) core tables used by DuckDB-backed stores
"""

from __future__ import annotations

__all__ = [
    "DuckDBWarehouse",
    "ensure_schema",
    "current_schema_version",
]

from .warehouse import DuckDBWarehouse
from .migrations import ensure_schema, current_schema_version
