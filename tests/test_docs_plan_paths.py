from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_plan_index_exists() -> None:
    assert (REPO_ROOT / "docs/plans/LEGACY_INDEX.md").exists()


def test_docs_plan_directory_is_retired() -> None:
    assert not (REPO_ROOT / "docs/plan").exists()


def test_no_root_plan_markdown_files() -> None:
    root_plan_files = list(REPO_ROOT.glob("*-PLAN.md")) + list(REPO_ROOT.glob("*-plan.md"))
    assert not root_plan_files


def test_no_docs_plan_references_outside_legacy_archive() -> None:
    offenders: list[Path] = []
    for path in (REPO_ROOT / "docs").rglob("*.md"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("docs/plans/"):
            continue
        if "docs/plan/" in path.read_text(encoding="utf-8"):
            offenders.append(path)
    assert not offenders
