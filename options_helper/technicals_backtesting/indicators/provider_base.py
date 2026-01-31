from __future__ import annotations

from typing import Protocol

import pandas as pd


FeatureFrame = pd.DataFrame


class IndicatorProvider(Protocol):
    def compute_indicators(self, df: pd.DataFrame, cfg: dict) -> FeatureFrame:
        ...

