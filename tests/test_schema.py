"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stategraph.core.graph_store import (
    InMemoryStateGraph,
    dataset_example_from_dict,
)
from stategraph.schemas import (
    ConflictDecision,
    DatasetExample,
    EvidenceNode,
    StateEdge,
    StateNode,
)


def test_construct_valid_state_node() -> None:
    """A valid StateNode should construct."""
    state = StateNode(
        state_id="s1",
        entity="Alice",
        attribute="location",
        value="Paris",
        time_scope="current",
        condition_scope=None,
        status="current",
        evidence_id="e1",
        confidence=0.9,
    )

    assert state.state_id == "s1"
    assert state.status == "current"


def test_reject_invalid_status() -> None:
    """Pydantic should reject unsupported state statuses."""
    with pytest.raises(ValidationError):
        StateNode(
            state_id="bad",
            entity="Alice",
            attribute="location",
            value="Paris",
            status="invalid",  # type: ignore[arg-type]
        )


def test_construct_valid_state_edge() -> None:
    """A valid StateEdge should construct."""
    edge = StateEdge(
        source="s1",
        target="s2",
        edge_type="updates",
        reason="newer observation",
    )

    assert edge.edge_type == "updates"


def test_reject_invalid_edge_type() -> None:
    """Pydantic should reject unsupported edge types."""
    with pytest.raises(ValidationError):
        StateEdge(
            source="s1",
            target="s2",
            edge_type="invalid",  # type: ignore[arg-type]
        )


def test_construct_valid_dataset_example() -> None:
    """A valid DatasetExample should construct."""
    example = DatasetExample(
        case_id="case-1",
        history=[{"evidence_id": "e1", "text": "Alice lives in Paris."}],
        new_observation={"evidence_id": "e2", "text": "Alice moved to Rome."},
        query="Where does Alice live?",
        gold_current_states=["s2"],
        gold_invalidated_states=["s1"],
        gold_keep_states=[],
        gold_answer="Rome",
        expected_behavior="answer with the latest current state",
    )

    assert example.case_id == "case-1"
    assert example.history[0]["text"] == "Alice lives in Paris."


def test_construct_valid_conflict_decision() -> None:
    """A valid ConflictDecision should construct."""
    decision = ConflictDecision(
        label="update",
        reason="candidate has a newer timestamp",
        confidence=0.8,
    )

    assert decision.label == "update"


def test_graph_store_uses_central_schemas() -> None:
    """The graph store should accept central schema objects."""
    evidence = EvidenceNode(
        evidence_id="e1",
        text="Alice lives in Paris.",
        source="unit-test",
        timestamp="2026-01-01T00:00:00Z",
    )
    state = StateNode(
        state_id="s1",
        entity="Alice",
        attribute="location",
        value="Paris",
        status="current",
        evidence_id=evidence.evidence_id,
        confidence=0.9,
    )
    edge = StateEdge(source="s1", target="s2", edge_type="updates")

    graph = InMemoryStateGraph()
    graph.add_evidence(evidence)
    graph.add_state(state)
    graph.add_edge(edge)

    assert graph.get_state("s1") == state
    assert graph.get_current_states() == [state]
    assert graph.edges_from("s1") == [edge]


def test_dataset_example_from_dict_accepts_legacy_history_strings() -> None:
    """The JSONL helper should keep old string history inputs usable."""
    example = dataset_example_from_dict(
        {
            "case_id": "case-2",
            "history": ["Bob owns a blue car."],
            "new_observation": "Bob sold the blue car.",
            "query": "What car does Bob own?",
            "gold_current_states": [],
            "gold_invalidated_states": [],
            "gold_keep_states": [],
            "gold_answer": "Unknown",
        }
    )

    assert example.history[0]["text"] == "Bob owns a blue car."
    assert example.history[0]["source"] == "history"
    assert example.new_observation == {
        "text": "Bob sold the blue car.",
        "source": "new_observation",
    }
