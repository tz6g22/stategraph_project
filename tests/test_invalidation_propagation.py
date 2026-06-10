"""Tests for dependency-based invalidation propagation."""

from __future__ import annotations

import pytest

from stategraph.core.graph_store import GraphStore
from stategraph.core.invalidation_propagation import (
    InvalidationPropagator,
    propagate_invalidation,
)
from stategraph.schemas import (
    EvidenceNode,
    PropagationReport,
    PropagationStep,
    StateEdge,
    StateNode,
)


def evidence(evidence_id: str, text: str) -> EvidenceNode:
    """Create test evidence."""
    return EvidenceNode(
        evidence_id=evidence_id,
        text=text,
        source="test",
        timestamp=None,
    )


def state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    status: str,
    evidence_id: str,
) -> StateNode:
    """Create a test state."""
    return StateNode(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope="Friday afternoon",
        condition_scope=None,
        status=status,  # type: ignore[arg-type]
        evidence_id=evidence_id,
        confidence=1.0,
    )


def edge(
    source: str,
    target: str,
    edge_type: str,
    reason: str = "test dependency",
) -> StateEdge:
    """Create a test edge."""
    return StateEdge(
        source=source,
        target=target,
        edge_type=edge_type,  # type: ignore[arg-type]
        reason=reason,
    )


def make_graph(
    *,
    s1_status: str = "stale",
    s2_status: str = "current",
    s3_status: str = "current",
    s4_status: str = "current",
) -> GraphStore:
    """Build the canonical propagation test graph."""
    graph = GraphStore()
    graph.add_evidence(
        evidence("E1", "User said she is free on Friday afternoon.")
    )
    graph.add_evidence(
        evidence("E2", "Assistant planned a meeting for Friday afternoon.")
    )
    graph.add_evidence(evidence("E3", "Assistant booked a room for the meeting."))
    graph.add_evidence(
        evidence("E4", "Assistant prepared an agenda for the meeting.")
    )
    graph.add_state(
        state(
            "S1",
            "user",
            "availability",
            "free",
            s1_status,
            "E1",
        )
    )
    graph.add_state(
        state(
            "S2",
            "meeting_plan",
            "feasibility",
            "valid",
            s2_status,
            "E2",
        )
    )
    graph.add_state(
        state(
            "S3",
            "room_booking",
            "status",
            "booked",
            s3_status,
            "E3",
        )
    )
    graph.add_state(
        state(
            "S4",
            "agenda",
            "status",
            "prepared",
            s4_status,
            "E4",
        )
    )
    graph.add_edge(
        edge("S2", "S1", "depends-on", "meeting feasibility depends on availability")
    )
    graph.add_edge(
        edge("S3", "S2", "depends-on", "room booking depends on meeting plan")
    )
    graph.add_edge(
        edge("S4", "S2", "derived-from", "agenda derives from meeting plan")
    )
    return graph


def add_extra_state(
    graph: GraphStore,
    state_id: str,
    edge_type: str,
) -> None:
    """Add an extra current state pointing at S1 through edge_type."""
    evidence_id = f"E_{state_id}"
    graph.add_evidence(evidence(evidence_id, f"Evidence for {state_id}."))
    graph.add_state(
        state(
            state_id,
            f"entity_{state_id}",
            "status",
            "active",
            "current",
            evidence_id,
        )
    )
    graph.add_edge(edge(state_id, "S1", edge_type, f"{state_id} {edge_type} S1"))


def step_pairs(report: PropagationReport) -> list[tuple[str, str, int]]:
    """Return source, target, and depth triples for report steps."""
    return [
        (step.source_state_id, step.target_state_id, step.depth)
        for step in report.propagation_steps
    ]


def test_direct_dependency_propagation() -> None:
    """Invalidating a prerequisite should stale its direct dependent."""
    graph = make_graph()

    report = InvalidationPropagator(max_depth=1).propagate(graph, ["S1"])

    assert graph.get_state("S2").status == "stale"
    assert "S2" in report.propagated_state_ids
    assert ("S1", "S2", 1) in step_pairs(report)


def test_multi_hop_propagation() -> None:
    """Propagation should continue across dependent states."""
    graph = make_graph()

    report = InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S2").status == "stale"
    assert graph.get_state("S3").status == "stale"
    assert graph.get_state("S4").status == "stale"
    assert report.propagated_state_ids == ["S2", "S3", "S4"]
    assert step_pairs(report) == [
        ("S1", "S2", 1),
        ("S2", "S3", 2),
        ("S2", "S4", 2),
    ]


def test_max_depth_one_stops_after_direct_dependents() -> None:
    """max_depth=1 should affect only direct dependents."""
    graph = make_graph()

    report = InvalidationPropagator(max_depth=1).propagate(graph, ["S1"])

    assert graph.get_state("S2").status == "stale"
    assert graph.get_state("S3").status == "current"
    assert graph.get_state("S4").status == "current"
    assert report.max_depth_reached is True
    assert report.propagated_state_ids == ["S2"]


def test_max_depth_zero_stops_all_propagation() -> None:
    """max_depth=0 should not affect dependent states."""
    graph = make_graph()

    report = InvalidationPropagator(max_depth=0).propagate(graph, ["S1"])

    assert graph.get_state("S1").status == "stale"
    assert graph.get_state("S2").status == "current"
    assert graph.get_state("S3").status == "current"
    assert graph.get_state("S4").status == "current"
    assert report.propagated_state_ids == []
    assert report.propagation_steps == []
    assert report.max_depth_reached is True


def test_dry_run_reports_without_mutating_graph() -> None:
    """dry_run should compute propagation without status updates."""
    graph = make_graph()

    report = InvalidationPropagator(dry_run=True).propagate(graph, ["S1"])

    assert report.dry_run is True
    assert report.propagated_state_ids == ["S2", "S3", "S4"]
    assert graph.get_state("S2").status == "current"
    assert graph.get_state("S3").status == "current"
    assert graph.get_state("S4").status == "current"
    assert report.propagation_steps[0].old_status == "current"
    assert report.propagation_steps[0].new_status == "stale"


def test_historical_state_is_skipped() -> None:
    """Historical states should not be changed by propagation."""
    graph = make_graph(s3_status="historical")

    report = InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S3").status == "historical"
    assert "S3" in report.skipped_state_ids
    assert "S3" not in report.propagated_state_ids


def test_uncertain_state_remains_uncertain() -> None:
    """Uncertain states should remain unchanged by default."""
    graph = make_graph(s3_status="uncertain")

    report = InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S3").status == "uncertain"
    assert "S3" in report.unchanged_state_ids
    assert "S3" not in report.propagated_state_ids
    assert any("uncertain state left unchanged" in note for note in report.notes)


def test_stale_state_remains_stale_and_propagation_continues() -> None:
    """Already stale dependents should remain stale while downstream continues."""
    graph = make_graph(s2_status="stale")

    report = InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S2").status == "stale"
    assert graph.get_state("S3").status == "stale"
    assert graph.get_state("S4").status == "stale"
    assert "S2" in report.unchanged_state_ids
    assert report.propagated_state_ids == ["S3", "S4"]


def test_supports_edge_does_not_propagate_by_default() -> None:
    """supports edges are not dependency edges by default."""
    graph = make_graph()
    add_extra_state(graph, "S5", "supports")

    InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S5").status == "current"


def test_invalidates_edge_does_not_propagate_by_default() -> None:
    """invalidates edges are not dependency edges by default."""
    graph = make_graph()
    add_extra_state(graph, "S6", "invalidates")

    InvalidationPropagator().propagate(graph, ["S1"])

    assert graph.get_state("S6").status == "current"


def test_custom_propagation_edge_types() -> None:
    """Custom edge type configuration should be honored."""
    graph = make_graph()
    add_extra_state(graph, "S5", "supports")

    report = InvalidationPropagator(
        propagation_edge_types={"supports"}
    ).propagate(graph, ["S1"])

    assert graph.get_state("S5").status == "stale"
    assert report.propagated_state_ids == ["S5"]


def test_missing_seed_raises_key_error() -> None:
    """Missing seed ids should fail explicitly."""
    graph = make_graph()

    with pytest.raises(KeyError):
        InvalidationPropagator().propagate(graph, ["missing"])


def test_invalid_propagation_edge_type_raises_value_error() -> None:
    """Unsupported propagation edge types should be rejected."""
    with pytest.raises(ValueError, match="Unsupported propagation edge"):
        InvalidationPropagator(propagation_edge_types={"not-an-edge"})


def test_invalid_target_status_raises_value_error() -> None:
    """Unsupported configured target statuses should be rejected."""
    with pytest.raises(ValueError, match="Unsupported state status"):
        InvalidationPropagator(stale_status="invalid")


def test_cycle_detection() -> None:
    """Cycles should terminate and be reported."""
    graph = make_graph()
    graph.add_edge(edge("S1", "S2", "depends-on", "cycle edge"))

    report = InvalidationPropagator().propagate(graph, ["S1"])

    assert report.cycle_detected is True
    assert graph.get_state("S2").status == "stale"
    assert any("cycle detected" in note for note in report.notes)


def test_deterministic_ordering() -> None:
    """Equivalent graphs should produce equivalent ordered reports."""
    report_a = InvalidationPropagator().propagate(make_graph(), ["S1"])
    report_b = InvalidationPropagator().propagate(make_graph(), ["S1"])

    assert report_a.propagated_state_ids == report_b.propagated_state_ids
    assert report_a.model_dump()["propagation_steps"] == report_b.model_dump()[
        "propagation_steps"
    ]


def test_report_serialization() -> None:
    """Propagation reports should serialize through Pydantic."""
    report = InvalidationPropagator().propagate(make_graph(), ["S1"])
    payload = report.model_dump()

    assert payload["seed_state_ids"] == ["S1"]
    assert payload["propagated_state_ids"] == ["S2", "S3", "S4"]
    assert payload["propagation_steps"][0]["edge_type"] == "depends-on"


def test_wrapper_function_returns_report() -> None:
    """Functional wrapper should return a PropagationReport."""
    graph = make_graph()

    report = propagate_invalidation(graph, ["S1"], max_depth=1)

    assert isinstance(report, PropagationReport)
    assert report.propagated_state_ids == ["S2"]


def test_propagation_step_schema_instantiates() -> None:
    """The step schema should be directly constructible."""
    step = PropagationStep(
        source_state_id="S1",
        target_state_id="S2",
        edge_type="depends-on",
        old_status="current",
        new_status="stale",
        depth=1,
        reason="S2 depends on S1",
    )

    assert step.model_dump()["target_state_id"] == "S2"
