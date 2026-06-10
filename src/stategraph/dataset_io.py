"""JSONL dataset I/O utilities for StateGraph examples."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from stategraph.schemas import DatasetExample


def validate_example(raw: dict[str, Any]) -> DatasetExample:
    """Validate a raw dictionary against DatasetExample."""
    try:
        if hasattr(DatasetExample, "model_validate"):
            return DatasetExample.model_validate(raw)
        return DatasetExample(**raw)
    except Exception as exc:
        raise ValueError(f"DatasetExample validation failed: {exc}") from exc


def load_jsonl(path: str | Path) -> list[DatasetExample]:
    """Load a JSONL file and return validated DatasetExample objects."""
    jsonl_path = Path(path)
    examples: list[DatasetExample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Invalid example in {jsonl_path} at line {line_number}: "
                    "expected a JSON object"
                )
            try:
                examples.append(validate_example(raw))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid DatasetExample in {jsonl_path} at line {line_number}: "
                    f"{exc}"
                ) from exc
    return examples


def _example_to_dict(example: DatasetExample) -> dict[str, Any]:
    """Convert a DatasetExample to a JSON-serializable dictionary."""
    if hasattr(example, "model_dump"):
        return example.model_dump(mode="json")
    if is_dataclass(example):
        return asdict(example)
    if isinstance(example, dict):
        return example
    raise TypeError(f"Unsupported example type: {type(example).__name__}")


def write_jsonl(path: str | Path, examples: list[DatasetExample]) -> None:
    """Write DatasetExample objects to JSONL."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = _example_to_dict(example)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_dataset_split(dataset_dir: str | Path, split: str) -> list[DatasetExample]:
    """Load a split such as dev/test from dataset_dir/{split}.jsonl."""
    return load_jsonl(Path(dataset_dir) / f"{split}.jsonl")
