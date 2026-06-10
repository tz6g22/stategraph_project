"""Path helper stubs."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for the src-layout package."""
    return Path(__file__).resolve().parents[4]

