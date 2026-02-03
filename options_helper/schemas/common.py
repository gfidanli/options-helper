from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_nan(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value)) or math.isinf(float(value))
    try:
        return bool(pd.isna(value))
    except Exception:  # noqa: BLE001
        return False


def clean_nan(value: Any) -> Any:
    if _is_nan(value):
        return None
    if isinstance(value, BaseModel):
        return clean_nan(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value):
        return clean_nan(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_nan(v) for v in value]
    if isinstance(value, pd.Series):
        return clean_nan(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return clean_nan(value.to_dict(orient="records"))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


class ArtifactMixin:
    def to_dict(self) -> dict[str, Any]:  # noqa: D401 - concise helper
        return clean_nan(self.model_dump(mode="json", by_alias=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):  # noqa: ANN001
        return cls.model_validate(payload)

    @classmethod
    def from_json(cls, payload: str):  # noqa: ANN001
        return cls.model_validate_json(payload)


class ArtifactBase(BaseModel, ArtifactMixin):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
