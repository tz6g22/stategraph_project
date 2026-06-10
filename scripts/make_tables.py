"""Table generation stub for paper or report outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def make_tables(metrics_path: Path, output_dir: Path) -> None:
    """Create the table output directory for future tables."""
    _ = metrics_path
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Add CSV or LaTeX table generation once metrics are implemented.


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/metrics/metrics.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    make_tables(args.metrics, args.output_dir)


if __name__ == "__main__":
    main()

