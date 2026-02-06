from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IVRegimeLabel = Literal["cheap", "fair", "expensive"]


class IVRegimeResult(BaseModel):
    label: IVRegimeLabel
    iv_rv: float
    low: float = Field(ge=0.0)
    high: float = Field(ge=0.0)
    reason: str


def classify_iv_rv(iv_rv: float | None, *, low: float = 0.8, high: float = 1.2) -> IVRegimeResult | None:
    if iv_rv is None:
        return None
    try:
        ratio = float(iv_rv)
        low_threshold = float(low)
        high_threshold = float(high)
    except Exception:  # noqa: BLE001
        return None
    if low_threshold >= high_threshold:
        return None
    if ratio >= high_threshold:
        return IVRegimeResult(
            label="expensive",
            iv_rv=ratio,
            low=low_threshold,
            high=high_threshold,
            reason="IV/RV high (premium rich).",
        )
    if ratio <= low_threshold:
        return IVRegimeResult(
            label="cheap",
            iv_rv=ratio,
            low=low_threshold,
            high=high_threshold,
            reason="IV/RV low (premium cheap).",
        )
    return IVRegimeResult(
        label="fair",
        iv_rv=ratio,
        low=low_threshold,
        high=high_threshold,
        reason="IV/RV neutral.",
    )

