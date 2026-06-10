"""Build StateChangeBench-style JSONL files.

This script is a placeholder and does not download datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_statechangebench(output_dir: Path) -> None:
    """Create the output directory for future generated JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Add dataset conversion logic once the schema is finalized.


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/statechangebench"))
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    build_statechangebench(args.output_dir)


if __name__ == "__main__":
    main()

