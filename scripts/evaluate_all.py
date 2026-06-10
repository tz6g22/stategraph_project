"""Evaluation orchestration stub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stategraph_project.src.metrics import METRIC_REGISTRY


def evaluate_all(predictions_dir: Path, output_path: Path) -> None:
    """Write placeholder metric names for future evaluation outputs."""
    _ = predictions_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": metric_name, "value": None, "todo": "not implemented"}
        for metric_name in sorted(METRIC_REGISTRY)
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/metrics.jsonl"),
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    evaluate_all(args.predictions_dir, args.output)


if __name__ == "__main__":
    main()

