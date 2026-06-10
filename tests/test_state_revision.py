"""Tests for deterministic state revision policies."""

from __future__ import annotations

import pytest

from stategraph.core.graph_store import GraphStore
from stategraph.core.state_revision import StateReviser, revise_state
from stategraph.schemas import (
    ConflictDecision,
    EvidenceNode,
    RevisionItem,
    RevisionReport,
    StateEdge,
    StateNode,
)


def evidence(evidence_id: str, text: str | None = None) -> EvidenceNode:
    """Create test evidence."""
    default_text = {
        "E1": "User said she is free on Friday afternoon.",
        "E2": "Assistant planned a meeting for Friday afternoon.",
        "E3": "User said she has a flight to Melbourne on Friday afternoon.",
        "E4": "Project deadline was changed.",
    }
    return EvidenceNode(
        evidence_id=evidence_id,
        text=text or default_text.get(evidence_id, "Synthetic revision evidence."),
        source="test",
        timestamp=None,
    )


def graph_with_evidence() -> GraphStore:
    """Create a graph with common evidence nodes."""
    graph = GraphStore()
    for evidence_id in ["E1", "E2", "E3", "E4"]:
        graph.add_evidence(evidence(evidence_id))
    return graph


def state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    time_scope: str | None = "Friday afternoon",
    status: str = "current",
    evidence_id: str | None = "E1",
    condition_scope: str | None = None,
) -> StateNode:
    """Create a test state node."""
    return StateNode(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope=time_scope,
        condition_scope=condition_scope,
        status=status,  # type: ignore[arg-type]
        evidence_id=evidence_id,
        confidence=1.0,
    )


def availability_free(state_id: str = "S1") -> StateNode:
    """Existing free availability state."""
    return state(
        state_id=state_id,
        entity="user",
        attribute="availability",
        value="free",
        evidence_id="E1",
    )


def unavailable_candidate(state_id: str = "C1") -> StateNode:
    """Candidate unavailable availability state."""
    return state(
        state_id=state_id,
        entity="user",
        attribute="availability",
        value="unavailable",
        status="uncertain",
        evidence_id="E3",
    )


def decision(label: str, reason: str = "test revision rule") -> ConflictDecision:
    """Create a conflict decision."""
    return ConflictDecision(
        label=label,  # type: ignore[arg-type]
        reason=reason,
        confidence=0.8,
    )


def item(
    existing_state_id: str = "S1",
    label: str = "update",
    reason: str = "test revision rule",
) -> RevisionItem:
    """Create a revision item."""
    return RevisionItem(
        existing_state_id=existing_state_id,
        decision=decision(label, reason),
    )


def test_no_linked_states_adds_candidate_current() -> None:
    """A candidate with no linked states should be inserted as current."""
    graph = graph_with_evidence()
    candidate = unavailable_candidate()

    report = StateReviser().revise(graph, candidate, [])

    assert graph.get_state("C1").status == "current"
    assert candidate.status == "uncertain"
    assert report.candidate_added is True
    assert report.candidate_final_status == "current"
    assert any("no linked states" in note for note in report.notes)


def test_duplicate_default_skips_candidate() -> None:
    """Duplicate candidates should be skipped by default."""
    graph = graph_with_evidence()
    graph.add_state(availability_free())
    candidate = state(
        "C1",
        "user",
        "availability",
        "free",
        status="uncertain",
        evidence_id="E3",
    )

    report = StateReviser().revise(graph, candidate, [item("S1", "duplicate")])

    assert graph.get_state("S1").status == "current"
    assert not graph.has_state("C1")
    assert report.candidate_added is False
    assert report.candidate_final_status is None
    assert report.duplicate_state_ids == ["S1"]


def test_consistent_adds_candidate_and_keeps_existing() -> None:
    """Consistent states should coexist without an edge by default."""
    graph = graph_with_evidence()
    graph.add_state(
        state("S1", "user", "location", "London", time_scope="today")
    )
    candidate = state(
        "C1",
        "user",
        "availability",
        "busy",
        time_scope="today",
        status="uncertain",
        evidence_id="E3",
    )

    report = StateReviser().revise(graph, candidate, [item("S1", "consistent")])

    assert graph.get_state("C1").status == "current"
    assert graph.get_state("S1").status == "current"
    assert graph.list_edges() == []
    assert report.unchanged_state_ids == ["S1"]


def test_consistent_can_create_support_edge() -> None:
    """Consistent states can optionally receive supports edges."""
    graph = graph_with_evidence()
    graph.add_state(state("S1", "user", "location", "London", time_scope="today"))
    candidate = state(
        "C1",
        "user",
        "availability",
        "busy",
        time_scope="today",
        status="uncertain",
        evidence_id="E3",
    )

    report = StateReviser(add_support_edges_for_consistent=True).revise(
        graph,
        candidate,
        [item("S1", "consistent")],
    )

    assert graph.has_edge("C1", "S1", "supports")
    assert [edge.edge_type for edge in report.created_edges] == ["supports"]


def test_update_marks_existing_historical_and_adds_updates_edge() -> None:
    """Updates should make the old state historical and link the new state."""
    graph = graph_with_evidence()
    graph.add_state(
        state(
            "S1",
            "project",
            "deadline",
            "Monday",
            time_scope=None,
            evidence_id="E1",
        )
    )
    candidate = state(
        "C1",
        "project",
        "deadline",
        "Wednesday",
        time_scope=None,
        status="uncertain",
        evidence_id="E4",
    )

    report = StateReviser().revise(graph, candidate, [item("S1", "update")])

    assert graph.get_state("C1").status == "current"
    assert graph.get_state("S1").status == "historical"
    assert graph.has_edge("C1", "S1", "updates")
    assert report.updated_state_ids == ["S1"]
    assert report.historical_state_ids == ["S1"]


def test_explicit_conflict_marks_existing_stale_and_invalidates() -> None:
    """Explicit conflicts should stale the old state and add invalidation."""
    graph = graph_with_evidence()
    graph.add_state(availability_free())

    report = StateReviser().revise(
        graph,
        unavailable_candidate(),
        [item("S1", "explicit_conflict")],
    )

    assert graph.get_state("C1").status == "current"
    assert graph.get_state("S1").status == "stale"
    assert graph.has_edge("C1", "S1", "invalidates")
    assert report.invalidated_state_ids == ["S1"]


def test_implicit_invalidation_marks_existing_stale_and_invalidates() -> None:
    """Implicit invalidations should stale the old state and add invalidation."""
    graph = graph_with_evidence()
    graph.add_state(
        state(
            "S1",
            "meeting_plan",
            "feasibility",
            "valid",
            evidence_id="E2",
        )
    )

    report = StateReviser().revise(
        graph,
        unavailable_candidate(),
        [item("S1", "implicit_invalidation")],
    )

    assert graph.get_state("S1").status == "stale"
    assert graph.has_edge("C1", "S1", "invalidates")
    assert report.invalidated_state_ids == ["S1"]


def test_temporary_exception_keeps_existing_and_adds_affects_action() -> None:
    """Temporary exceptions should keep the general state unchanged."""
    graph = graph_with_evidence()
    graph.add_state(
        state(
            "S1",
            "user",
            "preference",
            "likes Italian food",
            time_scope=None,
            evidence_id="E1",
        )
    )
    candidate = state(
        "C1",
        "user",
        "preference",
        "does not want Italian",
        time_scope="tonight",
        status="uncertain",
        evidence_id="E3",
    )

    report = StateReviser().revise(
        graph,
        candidate,
        [item("S1", "temporary_exception")],
    )

    assert graph.get_state("C1").status == "current"
    assert graph.get_state("S1").status == "current"
    assert graph.has_edge("C1", "S1", "affects-action")
    assert report.unchanged_state_ids == ["S1"]


def test_uncertain_adds_candidate_uncertain_without_revision_edge() -> None:
    """Uncertain relations should not update or invalidate old states."""
    graph = graph_with_evidence()
    graph.add_state(
        state("S1", "project", "priority", "medium", time_scope=None)
    )
    candidate = state(
        "C1",
        "user",
        "location",
        "Manchester",
        time_scope=None,
        status="current",
        evidence_id="E3",
    )

    report = StateReviser().revise(graph, candidate, [item("S1", "uncertain")])

    assert graph.get_state("C1").status == "uncertain"
    assert graph.get_state("S1").status == "current"
    assert graph.list_edges() == []
    assert report.uncertain_state_ids == ["S1"]


def test_multiple_decisions_add_candidate_once_and_records_all_changes() -> None:
    """A candidate should be inserted once while all decisions are applied."""
    graph = graph_with_evidence()
    graph.add_state(
        state("S1", "project", "deadline", "Monday", time_scope=None)
    )
    graph.add_state(
        state(
            "S2",
            "meeting_plan",
            "feasibility",
            "valid",
            evidence_id="E2",
        )
    )

    report = StateReviser().revise(
        graph,
        unavailable_candidate(),
        [
            item("S1", "update", "new deadline supersedes old deadline"),
            item("S2", "implicit_invalidation", "new availability invalidates plan"),
        ],
    )

    assert graph.get_state("C1").status == "current"
    assert graph.get_state("S1").status == "historical"
    assert graph.get_state("S2").status == "stale"
    assert len([state for state in graph.list_states() if state.state_id == "C1"]) == 1
    assert {edge.edge_type for edge in report.created_edges} == {
        "updates",
        "invalidates",
    }


def test_duplicate_candidate_id_records_note_without_crashing() -> None:
    """An existing candidate id should be noted rather than reinserted."""
    graph = graph_with_evidence()
    candidate = unavailable_candidate()
    graph.add_state(candidate)
    graph.add_state(availability_free())

    report = StateReviser().revise(graph, candidate, [item("S1", "consistent")])

    assert report.candidate_added is False
    assert graph.get_state("C1").status == "uncertain"
    assert any("already exists" in note for note in report.notes)


def test_missing_existing_state_raises_key_error() -> None:
    """Revision items must point to an existing state."""
    graph = graph_with_evidence()

    with pytest.raises(KeyError):
        StateReviser().revise(
            graph,
            unavailable_candidate(),
            [item("missing", "update")],
        )


def test_duplicate_edge_prevention_records_skipped_edge() -> None:
    """Repeating a revision should not duplicate typed edges."""
    graph = graph_with_evidence()
    graph.add_state(availability_free())
    candidate = unavailable_candidate()
    revision_item = item("S1", "explicit_conflict")
    first_report = StateReviser().revise(graph, candidate, [revision_item])

    second_report = StateReviser().revise(graph, candidate, [revision_item])

    assert len(graph.list_edges(edge_type="invalidates")) == 1
    assert [edge.edge_type for edge in first_report.created_edges] == ["invalidates"]
    assert second_report.created_edges == []
    assert second_report.skipped_edges == [
        {
            "source": "C1",
            "target": "S1",
            "edge_type": "invalidates",
            "reason": "duplicate edge already exists",
        }
    ]


def test_wrapper_function_returns_revision_report() -> None:
    """Functional wrapper should return a RevisionReport."""
    graph = graph_with_evidence()
    graph.add_state(availability_free())

    report = revise_state(graph, unavailable_candidate(), [item("S1", "update")])

    assert isinstance(report, RevisionReport)
    assert report.candidate_state_id == "C1"


def test_report_serialization() -> None:
    """Revision reports should serialize through Pydantic."""
    graph = graph_with_evidence()
    graph.add_state(availability_free())

    report = StateReviser().revise(
        graph,
        unavailable_candidate(),
        [item("S1", "explicit_conflict")],
    )
    payload = report.model_dump()

    assert payload["candidate_state_id"] == "C1"
    assert payload["created_edges"][0]["edge_type"] == "invalidates"
    assert payload["decisions"][0]["decision"]["label"] == "explicit_conflict"


def test_revision_does_not_propagate_to_dependent_states() -> None:
    """Revision should not recursively update dependent states."""
    graph = graph_with_evidence()
    graph.add_state(availability_free("S1"))
    graph.add_state(
        state(
            "S2",
            "meeting_plan",
            "feasibility",
            "valid",
            evidence_id="E2",
        )
    )
    graph.add_edge(
        StateEdge(
            source="S2",
            target="S1",
            edge_type="depends-on",
            reason="meeting plan depends on user availability",
        )
    )

    StateReviser().revise(
        graph,
        unavailable_candidate(),
        [item("S1", "explicit_conflict")],
    )

    assert graph.get_state("S1").status == "stale"
    assert graph.get_state("S2").status == "current"
    assert graph.has_edge("S2", "S1", "depends-on")
    assert graph.has_edge("C1", "S1", "invalidates")
