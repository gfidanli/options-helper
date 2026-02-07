from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "docs" / "diagrams" / "src"
OUTPUT_ROOT = REPO_ROOT / "docs" / "assets" / "diagrams" / "generated"


def _iter_source_files() -> list[Path]:
    if not SOURCE_ROOT.exists():
        return []
    return sorted(SOURCE_ROOT.rglob("*.mmd"))


def _expected_output_path(source_path: Path) -> Path:
    return OUTPUT_ROOT / source_path.relative_to(SOURCE_ROOT).with_suffix(".svg")


def _relative_paths(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(REPO_ROOT)) for path in paths]


def test_each_mermaid_source_has_generated_svg() -> None:
    missing_outputs = []
    for source_path in _iter_source_files():
        output_path = _expected_output_path(source_path)
        if not output_path.exists():
            missing_outputs.append(output_path)

    assert not missing_outputs, (
        "Missing generated diagram SVG(s). Run `npm run render-diagrams` and commit updates:\n"
        + "\n".join(_relative_paths(missing_outputs))
    )


def test_generated_svg_files_match_mermaid_sources() -> None:
    expected_outputs = {_expected_output_path(source_path) for source_path in _iter_source_files()}
    existing_outputs = set(OUTPUT_ROOT.rglob("*.svg")) if OUTPUT_ROOT.exists() else set()
    stale_outputs = sorted(existing_outputs - expected_outputs)

    assert not stale_outputs, (
        "Stale generated diagram SVG(s) found without matching source `.mmd` file:\n"
        + "\n".join(_relative_paths(stale_outputs))
    )
