from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.schemas.scanner_shortlist import ScannerShortlistArtifact


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "artifacts"


@pytest.mark.parametrize(
    "filename,model",
    [
        ("briefing.json", BriefingArtifact),
        ("chain_report.json", ChainReportArtifact),
        ("compare.json", CompareArtifact),
        ("flow.json", FlowArtifact),
        ("scanner_shortlist.json", ScannerShortlistArtifact),
    ],
)
def test_artifact_fixture_validates(filename: str, model) -> None:  # noqa: ANN001
    payload = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
    model.model_validate(payload)
