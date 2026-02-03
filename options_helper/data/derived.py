from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from options_helper.analysis.derived_metrics import DerivedRow


DERIVED_SCHEMA_VERSION = 2
DERIVED_COLUMNS_V1 = [
    "date",
    "spot",
    "pc_oi",
    "pc_vol",
    "call_wall",
    "put_wall",
    "gamma_peak_strike",
    "atm_iv_near",
    "em_near_pct",
    "skew_near_pp",
]
DERIVED_COLUMNS_V2 = DERIVED_COLUMNS_V1 + [
    "rv_20d",
    "rv_60d",
    "iv_rv_20d",
    "atm_iv_near_percentile",
    "iv_term_slope",
]
DERIVED_COLUMNS = DERIVED_COLUMNS_V2


class DerivedStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class DerivedStore:
    root_dir: Path

    def _symbol_path(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}.csv"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._symbol_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=DERIVED_COLUMNS)
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise DerivedStoreError(f"Failed to read derived file: {path}") from exc

        # Normalize columns (future-proofing).
        for col in DERIVED_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[DERIVED_COLUMNS].copy()

        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        return df

    def upsert(self, symbol: str, row: DerivedRow) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._symbol_path(symbol)

        df = self.load(symbol)
        date_str = row.date

        # Replace existing day row if present (idempotent update).
        if not df.empty:
            df = df[df["date"] != date_str].copy()

        payload = row.model_dump()
        data = {k: payload.get(k) for k in DERIVED_COLUMNS}
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        # Keep deterministic ordering.
        df = df.sort_values(["date"], ascending=True, na_position="last")
        df = df[DERIVED_COLUMNS]

        # Stable-ish float formatting while preserving readability.
        df.to_csv(path, index=False, float_format="%.8g")
        return path
