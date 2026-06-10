"""Import-safety tests for the StateGraph skeleton."""

from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
MODULE_ROOTS = [PROJECT_ROOT / "src" / "stategraph", PROJECT_ROOT / "scripts"]


def module_name_for(path: Path) -> str:
    """Convert a project Python file path into an importable module name."""
    if path.is_relative_to(PACKAGE_ROOT):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    else:
        relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def iter_main_modules() -> list[str]:
    """Return all Python modules under the package and scripts."""
    modules: list[str] = []
    for module_root in MODULE_ROOTS:
        for path in sorted(module_root.rglob("*.py")):
            modules.append(module_name_for(path))
    return modules


def test_main_modules_import_without_side_effects() -> None:
    """All main skeleton modules should be import safe."""
    for module_name in iter_main_modules():
        importlib.import_module(module_name)
