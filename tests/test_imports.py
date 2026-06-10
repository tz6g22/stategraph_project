"""Import-safety tests for the StateGraph skeleton."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

MODULE_DIRS = ["src", "baselines", "scripts"]


def module_name_for(path: Path) -> str:
    """Convert a project Python file path into an importable module name."""
    relative = path.relative_to(PROJECT_ROOT.parent).with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def iter_main_modules() -> list[str]:
    """Return all Python modules under src, baselines, and scripts."""
    modules: list[str] = []
    for module_dir in MODULE_DIRS:
        for path in sorted((PROJECT_ROOT / module_dir).glob("*.py")):
            modules.append(module_name_for(path))
    return modules


def test_main_modules_import_without_side_effects() -> None:
    """All main skeleton modules should be import safe."""
    for module_name in iter_main_modules():
        importlib.import_module(module_name)
