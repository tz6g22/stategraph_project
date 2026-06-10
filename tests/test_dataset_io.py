"""Tests for JSONL dataset I/O utilities."""

from __future__ import annotations

import pytest

from stategraph.dataset_io import (
    load_dataset_split,
    load_jsonl,
    validate_example,
    write_jsonl,
)
from stategraph.schemas import DatasetExample


def make_example(case_id: str = "IO_001") -> DatasetExample:
    """Construct a small valid DatasetExample."""
    return DatasetExample(
        case_id=case_id,
        history=[{"evidence_id": "e1", "text": "The user is free on Friday."}],
        new_observation={"evidence_id": "e2", "text": "The user now has a flight."},
        query="Is the user still free?",
        gold_current_states=["state_has_flight"],
        gold_invalidated_states=["state_free_friday"],
        gold_keep_states=[],
        gold_answer="No.",
        expected_behavior="Reject stale availability.",
    )


def test_construct_small_dataset_example() -> None:
    """A small valid example should pass validation."""
    example = validate_example(make_example().model_dump())

    assert example.case_id == "IO_001"


def test_write_and_load_jsonl_round_trip(tmp_path) -> None:
    """A written JSONL file should load back into DatasetExample objects."""
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [make_example()])

    loaded = load_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].case_id == "IO_001"


def test_empty_lines_are_ignored(tmp_path) -> None:
    """Blank JSONL lines should be skipped."""
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [make_example("IO_002")])
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    loaded = load_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].case_id == "IO_002"


def test_invalid_json_raises_exception(tmp_path) -> None:
    """Invalid JSON should raise a clear exception."""
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(path)


def test_invalid_schema_raises_exception(tmp_path) -> None:
    """JSON objects that fail DatasetExample validation should raise."""
    path = tmp_path / "bad_schema.jsonl"
    path.write_text('{"case_id": "", "query": ""}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid DatasetExample"):
        load_jsonl(path)


def test_load_dataset_split(tmp_path) -> None:
    """load_dataset_split should read dataset_dir/{split}.jsonl."""
    split_path = tmp_path / "dev.jsonl"
    write_jsonl(split_path, [make_example("IO_DEV")])

    loaded = load_dataset_split(tmp_path, "dev")

    assert len(loaded) == 1
    assert loaded[0].case_id == "IO_DEV"
