from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from options_helper.schemas.common import ArtifactBase


class CompareArtifact(ArtifactBase):
    schema_version: int = 1
    generated_at: datetime
    as_of: str
    symbol: str
    from_report: dict[str, Any] = Field(alias="from")
    to_report: dict[str, Any] = Field(alias="to")
    diff: dict[str, Any]
