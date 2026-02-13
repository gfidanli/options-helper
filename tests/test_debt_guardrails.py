from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "debt_guardrails.py"
    spec = importlib.util.spec_from_file_location("debt_guardrails", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_command_import_boundary_flags_cross_command(monkeypatch, tmp_path: Path) -> None:
    debt = _load_module()
    monkeypatch.setattr(debt, "REPO_ROOT", tmp_path)

    _write(
        tmp_path / "options_helper/commands/a.py",
        "from options_helper.commands.technicals import technicals_extension_stats\n",
    )
    _write(
        tmp_path / "options_helper/commands/b.py",
        "from options_helper.commands.common import _parse_date\n",
    )

    violations = debt._command_import_violations(
        ["options_helper/commands/a.py", "options_helper/commands/b.py"]
    )
    assert any(v.code == "CMD_IMPORT" and v.path.endswith("a.py") for v in violations)
    assert not any(v.path.endswith("b.py") for v in violations)


def test_analysis_data_boundary_allows_adapter_seam(monkeypatch, tmp_path: Path) -> None:
    debt = _load_module()
    monkeypatch.setattr(debt, "REPO_ROOT", tmp_path)

    _write(
        tmp_path / "options_helper/analysis/strategy_modeling_io_adapter.py",
        "from options_helper.data.strategy_modeling_io import normalize_symbol\n",
    )
    _write(
        tmp_path / "options_helper/analysis/bad.py",
        "from options_helper.data.strategy_modeling_io import normalize_symbol\n",
    )

    violations = debt._analysis_data_import_violations(
        [
            "options_helper/analysis/strategy_modeling_io_adapter.py",
            "options_helper/analysis/bad.py",
        ]
    )
    assert any(v.code == "ANALYSIS_DATA_IMPORT" and v.path.endswith("bad.py") for v in violations)
    assert not any(v.path.endswith("strategy_modeling_io_adapter.py") for v in violations)


def test_new_file_size_and_function_size_checks(monkeypatch, tmp_path: Path) -> None:
    debt = _load_module()
    monkeypatch.setattr(debt, "REPO_ROOT", tmp_path)

    long_body = "\n".join(["    x = 1" for _ in range(85)])
    filler = "\n".join(["# filler" for _ in range(320)])
    contents = f"def too_long():\n{long_body}\n\n{filler}\n"

    rel = "options_helper/new_module.py"
    _write(tmp_path / rel, contents)

    changed = [debt.ChangedFile(status="A", path=rel)]
    violations = debt._check_new_file_size(changed)
    codes = {v.code for v in violations}

    assert "NEW_FILE_LINES" in codes
    assert "NEW_FUNCTION_LINES" in codes


def test_legacy_growth_check(monkeypatch, tmp_path: Path) -> None:
    debt = _load_module()
    monkeypatch.setattr(debt, "REPO_ROOT", tmp_path)

    rel = "options_helper/legacy_big.py"
    _write(tmp_path / rel, "\n".join(["# now" for _ in range(1105)]) + "\n")

    changed = [debt.ChangedFile(status="M", path=rel)]
    monkeypatch.setattr(debt, "_file_text_at_ref", lambda ref, path: "\n".join(["# old" for _ in range(1090)]))

    violations = debt._check_legacy_growth(changed, base_ref="HEAD~1")
    assert any(v.code == "LEGACY_GROWTH" and v.path == rel for v in violations)
