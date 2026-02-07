from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.exposure import ExposureArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.schemas.intraday_flow import IntradayFlowArtifact
from options_helper.schemas.iv_surface import IvSurfaceArtifact
from options_helper.schemas.levels import LevelsArtifact
from options_helper.schemas.scenarios import ScenariosArtifact
from options_helper.schemas.scanner_shortlist import ScannerShortlistArtifact
from options_helper.schemas.tail_risk import TailRiskArtifact


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "artifacts"


@pytest.mark.parametrize(
    "filename,model",
    [
        ("briefing.json", BriefingArtifact),
        ("chain_report.json", ChainReportArtifact),
        ("compare.json", CompareArtifact),
        ("exposure_minimal.json", ExposureArtifact),
        ("flow.json", FlowArtifact),
        ("intraday_flow_minimal.json", IntradayFlowArtifact),
        ("iv_surface_minimal.json", IvSurfaceArtifact),
        ("levels_minimal.json", LevelsArtifact),
        ("scanner_shortlist.json", ScannerShortlistArtifact),
        ("scenarios_minimal.json", ScenariosArtifact),
        ("tail_risk.json", TailRiskArtifact),
    ],
)
def test_artifact_fixture_validates(filename: str, model) -> None:  # noqa: ANN001
    payload = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
    model.model_validate(payload)
