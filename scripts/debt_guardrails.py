#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = ("options_helper", "apps", "scripts")
MAX_NEW_FILE_LINES = 400
MAX_NEW_FUNCTION_LINES = 80
LEGACY_FILE_THRESHOLD = 1000

ALLOWED_COMMAND_IMPORTS = {
    "options_helper.commands.common",
    "options_helper.commands.position_metrics",
}
ALLOWED_ANALYSIS_DATA_SEAMS = {
    "options_helper/analysis/strategy_modeling_io_adapter.py",
}


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    message: str


def _run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _path_is_production_python(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    rel = path.replace("\\", "/")
    return rel.startswith(PRODUCTION_ROOTS)


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _parse_tree(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return None


def _function_lengths(path: Path) -> list[tuple[str, int]]:
    tree = _parse_tree(path)
    if tree is None:
        return []
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            out.append((node.name, end - node.lineno + 1))
    return out


def _module_name_from_path(path: str) -> str:
    rel = path.replace("\\", "/")
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def _imported_modules(path: Path) -> set[str]:
    tree = _parse_tree(path)
    if tree is None:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "options_helper.commands":
                for alias in node.names:
                    modules.add(f"{node.module}.{alias.name}")
            else:
                modules.add(node.module)
    return modules


def _is_allowed_command_import(imported: str) -> bool:
    if imported in ALLOWED_COMMAND_IMPORTS:
        return True
    last = imported.rsplit(".", 1)[-1]
    return last == "common" or last.endswith("_common") or last.endswith("_legacy")


def _command_import_violations(paths: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for rel in paths:
        rel_norm = rel.replace("\\", "/")
        if not rel_norm.startswith("options_helper/commands/") or not rel_norm.endswith(".py"):
            continue
        imported = _imported_modules(REPO_ROOT / rel_norm)
        own_module = _module_name_from_path(rel_norm)
        own_family = _command_family(own_module)
        for module in sorted(imported):
            if not module.startswith("options_helper.commands."):
                continue
            if module == own_module or module.startswith(f"{own_module}."):
                continue
            if own_family is not None and _command_family(module) == own_family:
                continue
            if _is_allowed_command_import(module):
                continue
            violations.append(
                Violation(
                    code="CMD_IMPORT",
                    path=rel_norm,
                    message=(
                        "cross-command import is not allowed: "
                        f"`{module}` (use analysis/pipelines/common seam instead)"
                    ),
                )
            )
    return violations


def _command_family(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 3:
        return None
    if parts[0] != "options_helper" or parts[1] != "commands":
        return None
    return parts[2]


def _analysis_data_import_violations(paths: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for rel in paths:
        rel_norm = rel.replace("\\", "/")
        if not rel_norm.startswith("options_helper/analysis/") or not rel_norm.endswith(".py"):
            continue
        if rel_norm in ALLOWED_ANALYSIS_DATA_SEAMS:
            continue
        imported = _imported_modules(REPO_ROOT / rel_norm)
        offenders = sorted(m for m in imported if m.startswith("options_helper.data"))
        for module in offenders:
            violations.append(
                Violation(
                    code="ANALYSIS_DATA_IMPORT",
                    path=rel_norm,
                    message=(
                        "analysis module imports data layer directly: "
                        f"`{module}` (route via documented adapter seam)"
                    ),
                )
            )
    return violations


def _changed_files(base_ref: str | None) -> list[ChangedFile]:
    # Prefer working-tree changes first so local runs enforce only active edits.
    raw = _run_git(["diff", "--name-status", "--diff-filter=AMR", "--relative"])

    changed = _parse_changed_file_lines(raw)
    changed.extend(_untracked_files())
    changed = _dedupe_changed_files(changed)

    if changed:
        return changed

    if base_ref:
        diff_target = f"{base_ref}...HEAD"
        raw = _run_git(["diff", "--name-status", "--diff-filter=AMR", "--relative", diff_target])
        changed = _parse_changed_file_lines(raw)
        changed.extend(_untracked_files())
        changed = _dedupe_changed_files(changed)
        if changed:
            return changed

    # Fallback for local runs with no resolvable merge base.
    tracked = _run_git(["ls-files"]).splitlines()
    return [ChangedFile(status="M", path=p) for p in tracked if p]


def _parse_changed_file_lines(raw: str) -> list[ChangedFile]:
    changed: list[ChangedFile] = []
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            path = parts[2]
        else:
            path = parts[1]
        changed.append(ChangedFile(status=status[0], path=path))
    return changed


def _untracked_files() -> list[ChangedFile]:
    raw = _run_git(["ls-files", "--others", "--exclude-standard"])
    out: list[ChangedFile] = []
    for path in raw.splitlines():
        if path:
            out.append(ChangedFile(status="A", path=path))
    return out


def _dedupe_changed_files(changed: list[ChangedFile]) -> list[ChangedFile]:
    by_path: dict[str, ChangedFile] = {}
    for item in changed:
        existing = by_path.get(item.path)
        if existing is None:
            by_path[item.path] = item
            continue
        if item.status == "A" and existing.status != "A":
            by_path[item.path] = item
    return sorted(by_path.values(), key=lambda item: item.path)


def _resolve_base_ref(cli_base_ref: str | None) -> str | None:
    if cli_base_ref:
        return cli_base_ref

    gh_base = os.environ.get("GITHUB_BASE_REF")
    if gh_base:
        candidate = f"origin/{gh_base}"
        if _run_git(["rev-parse", "--verify", candidate]).strip():
            return candidate

    if _run_git(["rev-parse", "--verify", "HEAD~1"]).strip():
        return "HEAD~1"

    return None


def _file_text_at_ref(ref: str | None, path: str) -> str | None:
    if not ref:
        return None
    out = _run_git(["show", f"{ref}:{path}"])
    return out if out else None


def _check_new_file_size(changed: list[ChangedFile]) -> list[Violation]:
    violations: list[Violation] = []
    for item in changed:
        if item.status != "A" or not _path_is_production_python(item.path):
            continue
        if item.path.endswith("_legacy.py"):
            # Transitional monolith moves are tracked as "new" files locally.
            continue
        abs_path = REPO_ROOT / item.path
        if not abs_path.exists():
            continue
        text = abs_path.read_text(encoding="utf-8")
        lines = _line_count(text)
        if lines > MAX_NEW_FILE_LINES:
            violations.append(
                Violation(
                    code="NEW_FILE_LINES",
                    path=item.path,
                    message=f"new file has {lines} lines (> {MAX_NEW_FILE_LINES})",
                )
            )
        for name, length in _function_lengths(abs_path):
            if length > MAX_NEW_FUNCTION_LINES:
                violations.append(
                    Violation(
                        code="NEW_FUNCTION_LINES",
                        path=item.path,
                        message=(
                            f"new function `{name}` has {length} lines "
                            f"(> {MAX_NEW_FUNCTION_LINES})"
                        ),
                    )
                )
    return violations


def _check_legacy_growth(changed: list[ChangedFile], base_ref: str | None) -> list[Violation]:
    violations: list[Violation] = []
    for item in changed:
        if item.status != "M" or not _path_is_production_python(item.path):
            continue
        abs_path = REPO_ROOT / item.path
        if not abs_path.exists():
            continue
        current_text = abs_path.read_text(encoding="utf-8")
        current_lines = _line_count(current_text)
        if current_lines <= LEGACY_FILE_THRESHOLD:
            continue

        prev_text = _file_text_at_ref(base_ref, item.path)
        if prev_text is None:
            continue
        prev_lines = _line_count(prev_text)
        if current_lines > prev_lines:
            violations.append(
                Violation(
                    code="LEGACY_GROWTH",
                    path=item.path,
                    message=(
                        f"legacy file grew from {prev_lines} to {current_lines} lines "
                        "(must be non-increasing unless split in same PR)"
                    ),
                )
            )
    return violations


def _report_mode() -> int:
    production = [
        p
        for p in REPO_ROOT.rglob("*.py")
        if _path_is_production_python(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    ]
    total = len(production)
    large_files = []
    long_functions = 0
    for path in production:
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        lines = _line_count(path.read_text(encoding="utf-8"))
        if lines > LEGACY_FILE_THRESHOLD:
            large_files.append((lines, rel))
        long_functions += sum(1 for _, n in _function_lengths(path) if n > MAX_NEW_FUNCTION_LINES)

    print("Debt guardrails report")
    print(f"- production python files: {total}")
    print(f"- files > {LEGACY_FILE_THRESHOLD} lines: {len(large_files)}")
    print(f"- functions > {MAX_NEW_FUNCTION_LINES} lines: {long_functions}")
    if large_files:
        top = sorted(large_files, reverse=True)[:10]
        print("- top large files:")
        for lines, rel in top:
            print(f"  - {rel}: {lines}")
    return 0


def _enforce_mode(base_ref: str | None) -> int:
    changed = _changed_files(base_ref)
    changed_paths = [c.path for c in changed if _path_is_production_python(c.path)]

    violations: list[Violation] = []
    violations.extend(_check_new_file_size(changed))
    violations.extend(_check_legacy_growth(changed, base_ref=base_ref))
    violations.extend(_command_import_violations(changed_paths))
    violations.extend(_analysis_data_import_violations(changed_paths))

    if not violations:
        print("Debt guardrails: no violations in changed files.")
        return 0

    print("Debt guardrails violations:")
    for violation in violations:
        print(f"- [{violation.code}] {violation.path}: {violation.message}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tech debt guardrail checks")
    parser.add_argument("mode", choices=["report", "enforce-changed"])
    parser.add_argument("--base-ref", default=None, help="Git base ref for changed-file comparison")
    args = parser.parse_args(argv)

    if args.mode == "report":
        return _report_mode()

    base_ref = _resolve_base_ref(args.base_ref)
    return _enforce_mode(base_ref=base_ref)


if __name__ == "__main__":
    sys.exit(main())
