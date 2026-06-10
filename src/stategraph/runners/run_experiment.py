"""JSONL runner for MVP baseline stubs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from stategraph.core.graph_store import dataset_example_from_dict
from stategraph.methods.cupmem_method import CupMemBaseline
from stategraph.methods.summary_memory_method import SummaryMemoryBaseline
from stategraph.methods.time_decay_rag_method import TimeDecayRagBaseline
from stategraph.methods.vector_rag_method import VectorRagBaseline
from stategraph.schemas import DatasetExample


class Baseline(Protocol):
    """Minimal baseline interface for future experiments."""

    name: str

    def predict(self, example: DatasetExample) -> str:
        """Return a prediction for an example."""


BASELINE_REGISTRY: dict[str, type[Baseline]] = {
    "vector_rag": VectorRagBaseline,
    "summary_memory": SummaryMemoryBaseline,
    "time_decay_rag": TimeDecayRagBaseline,
    "cupmem_reimpl": CupMemBaseline,
}


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    """Yield dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_baseline(baseline_name: str, input_path: Path, output_path: Path) -> None:
    """Run a baseline stub over JSONL examples."""
    if baseline_name not in BASELINE_REGISTRY:
        allowed = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unknown baseline '{baseline_name}'. Allowed: {allowed}")

    baseline = BASELINE_REGISTRY[baseline_name]()
    rows: list[dict[str, object]] = []
    for payload in read_jsonl(input_path):
        example = dataset_example_from_dict(payload)
        rows.append(
            {
                "case_id": example.case_id,
                "baseline": baseline.name,
                "prediction": baseline.predict(example),
            }
        )
    write_jsonl(output_path, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINE_REGISTRY))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    run_baseline(args.baseline, args.input, args.output)


if __name__ == "__main__":
    main()
