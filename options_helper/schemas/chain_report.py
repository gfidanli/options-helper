from __future__ import annotations

from datetime import datetime

from options_helper.analysis.chain_metrics import ChainReport
from options_helper.schemas.common import ArtifactMixin


class ChainReportArtifact(ChainReport, ArtifactMixin):
    generated_at: datetime
