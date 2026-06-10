"""Tests for the GraphStore storage layer."""

from __future__ import annotations

import json

import pytest

from stategraph.core.graph_store import GraphStore
from stategraph.schemas import EvidenceNode, StateEdge, StateNode


def evidence_1() -> EvidenceNode:
    """User availability evidence."""
    return EvidenceNode(
        evidence_id="E1",
        text="User said she is free on Friday afternoon.",
        source="test",
        timestamp=None,
    )


def evidence_2() -> EvidenceNode:
    """Assistant plan evidence."""
    return EvidenceNode(
        evidence_id="E2",
        text="Assistant planned a meeting for Friday afternoon.",
        source="test",
        timestamp=None,
    )


def evidence_3() -> EvidenceNode:
    """User flight evidence."""
    return EvidenceNode(
        evidence_id="E3",
        text="User said she has a flight to Melbourne on Friday afternoon.",
        source="test",
        timestamp=None,
    )


def state_1() -> StateNode:
    """User availability state."""
    return StateNode(
        state_id="S1",
        entity="user",
        attribute="availability",
        value="free",
        time_scope="Friday afternoon",
        condition_scope=None,
        status="current",
        evidence_id="E1",
        confidence=1.0,
    )


def state_2() -> StateNode:
    """Meeting plan state."""
    return StateNode(
        state_id="S2",
        entity="meeting_plan",
        attribute="feasibility",
        value="valid",
        time_scope="Friday afternoon",
        condition_scope=None,
        status="current",
        evidence_id="E2",
        confidence=1.0,
    )


def state_3() -> StateNode:
    """Updated user availability state."""
    return StateNode(
        state_id="S3",
        entity="user",
        attribute="availability",
        value="unavailable",
        time_scope="Friday afternoon",
        condition_scope=None,
        status="current",
        evidence_id="E3",
        confidence=1.0,
    )


def state_4() -> StateNode:
    """User location state."""
    return StateNode(
        state_id="S4",
        entity="user",
        attribute="location",
        value="Melbourne",
        time_scope="Friday afternoon",
        condition_scope=None,
        status="current",
        evidence_id="E3",
        confidence=1.0,
    )


def edge_depends_on() -> StateEdge:
    """S2 depends on S1."""
    return StateEdge(
        source="S2",
        target="S1",
        edge_type="depends-on",
        reason="meeting feasibility depends on user availability",
    )


def edge_invalidates() -> StateEdge:
    """S3 invalidates S1."""
    return StateEdge(
        source="S3",
        target="S1",
        edge_type="invalidates",
        reason="newer availability update",
    )


def edge_supports() -> StateEdge:
    """S4 supports S3."""
    return StateEdge(
        source="S4",
        target="S3",
        edge_type="supports",
        reason="flight destination explains unavailability",
    )


def make_store_with_states() -> GraphStore:
    """Create a graph store with all evidence and states."""
    store = GraphStore()
    for evidence in [evidence_2(), evidence_3(), evidence_1()]:
        store.add_evidence(evidence)
    for state in [state_3(), state_1(), state_4(), state_2()]:
        store.add_state(state)
    return store


def make_store_with_edges() -> GraphStore:
    """Create a graph store with all test edges."""
    store = make_store_with_states()
    for edge in [edge_supports(), edge_invalidates(), edge_depends_on()]:
        store.add_edge(edge)
    return store


def ids(states: list[StateNode]) -> list[str]:
    """Return state ids for assertions."""
    return [state.state_id for state in states]


def test_empty_graph_store_creation() -> None:
    """A new GraphStore should be empty."""
    store = GraphStore()

    assert store.list_evidence() == []
    assert store.list_states() == []
    assert store.list_edges() == []
    assert store.graph.number_of_nodes() == 0


def test_add_and_retrieve_evidence() -> None:
    """Evidence should be stored and retrieved by evidence_id."""
    store = GraphStore()
    store.add_evidence(evidence_1())

    assert store.has_evidence("E1")
    assert store.get_evidence("E1") == evidence_1()
    assert store.list_evidence() == [evidence_1()]


def test_add_and_retrieve_states() -> None:
    """States should be stored and retrieved by state_id."""
    store = GraphStore()
    store.add_state(state_1())

    assert store.has_state("S1")
    assert store.get_state("S1") == state_1()
    assert store.list_states() == [state_1()]


def test_duplicate_state_id_raises_value_error() -> None:
    """Duplicate state IDs should be rejected."""
    store = GraphStore()
    store.add_state(state_1())

    with pytest.raises(ValueError, match="Duplicate state_id"):
        store.add_state(state_1())


def test_duplicate_evidence_id_raises_value_error() -> None:
    """Duplicate evidence IDs should be rejected."""
    store = GraphStore()
    store.add_evidence(evidence_1())

    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        store.add_evidence(evidence_1())


def test_missing_state_and_evidence_raise_key_error() -> None:
    """Missing direct lookups should raise KeyError."""
    store = GraphStore()

    with pytest.raises(KeyError, match="Unknown state_id"):
        store.get_state("missing")

    with pytest.raises(KeyError, match="Unknown evidence_id"):
        store.get_evidence("missing")


def test_add_valid_edges() -> None:
    """Valid typed edges should be stored."""
    store = make_store_with_states()
    store.add_edge(edge_depends_on())

    assert store.list_edges() == [edge_depends_on()]


def test_add_edge_with_missing_source_raises_key_error() -> None:
    """Edges with missing source states should be rejected."""
    store = make_store_with_states()

    with pytest.raises(KeyError, match="Unknown source"):
        store.add_edge(StateEdge(source="missing", target="S1", edge_type="depends-on"))


def test_add_edge_with_missing_target_raises_key_error() -> None:
    """Edges with missing target states should be rejected."""
    store = make_store_with_states()

    with pytest.raises(KeyError, match="Unknown target"):
        store.add_edge(StateEdge(source="S1", target="missing", edge_type="depends-on"))


def test_list_states_filters() -> None:
    """State listing should support all requested filters."""
    store = make_store_with_states()

    assert ids(store.list_states()) == ["S1", "S2", "S3", "S4"]
    assert ids(store.list_states(status="current")) == ["S1", "S2", "S3", "S4"]
    assert ids(store.list_states(entity="user")) == ["S1", "S3", "S4"]
    assert ids(store.list_states(attribute="availability")) == ["S1", "S3"]
    assert ids(store.list_states(entity="user", attribute="availability")) == [
        "S1",
        "S3",
    ]
    assert ids(store.list_states(time_scope="Friday afternoon")) == [
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    assert ids(store.list_states(evidence_id="E3")) == ["S3", "S4"]


def test_status_specific_state_listing_and_update() -> None:
    """Status updates should affect status-specific listing only."""
    store = make_store_with_states()

    store.update_state_status("S1", "stale")
    store.update_state_status("S2", "uncertain")
    store.update_state_status("S4", "historical")

    assert ids(store.list_current_states()) == ["S3"]
    assert ids(store.list_stale_states()) == ["S1"]
    assert ids(store.list_uncertain_states()) == ["S2"]
    assert ids(store.list_historical_states()) == ["S4"]
    assert store.get_state("S1").status == "stale"


def test_invalid_status_raises_error() -> None:
    """Invalid status updates should be rejected."""
    store = make_store_with_states()

    with pytest.raises(ValueError, match="Unsupported state status"):
        store.update_state_status("S1", "invalid")


def test_entity_and_attribute_indexes() -> None:
    """Entity and attribute helper methods should return deterministic data."""
    store = make_store_with_states()

    assert store.list_entities() == ["meeting_plan", "user"]
    assert store.list_attributes() == ["availability", "feasibility", "location"]
    assert store.list_attributes(entity="user") == ["availability", "location"]
    assert ids(store.find_states_by_entity("user")) == ["S1", "S3", "S4"]
    assert ids(store.find_states_by_attribute("availability")) == ["S1", "S3"]
    assert ids(store.find_states_by_entity_attribute("user", "availability")) == [
        "S1",
        "S3",
    ]


def test_list_edges_filters() -> None:
    """Edge listing should support edge_type, source, and target filters."""
    store = make_store_with_edges()

    assert store.list_edges(edge_type="depends-on") == [edge_depends_on()]
    assert store.list_edges(source="S3") == [edge_invalidates()]
    assert store.list_edges(target="S1") == [edge_depends_on(), edge_invalidates()]
    assert store.list_edges(edge_type="supports", source="S4", target="S3") == [
        edge_supports()
    ]


def test_has_edge_with_and_without_edge_type() -> None:
    """has_edge should support optional edge_type filtering."""
    store = make_store_with_edges()

    assert store.has_edge("S2", "S1")
    assert store.has_edge("S2", "S1", edge_type="depends-on")
    assert not store.has_edge("S2", "S1", edge_type="invalidates")
    assert not store.has_edge("S1", "S2")


def test_successors_and_predecessors_with_filters() -> None:
    """Traversal helpers should support optional edge-type filters."""
    store = make_store_with_edges()

    assert ids(store.successors("S2")) == ["S1"]
    assert ids(store.successors("S2", edge_type="depends-on")) == ["S1"]
    assert store.successors("S2", edge_type="invalidates") == []
    assert ids(store.predecessors("S1")) == ["S2", "S3"]
    assert ids(store.predecessors("S1", edge_type="invalidates")) == ["S3"]
    assert store.predecessors("S1", edge_type="supports") == []


def test_incoming_and_outgoing_edges() -> None:
    """Incoming and outgoing edge helpers should return sorted edges."""
    store = make_store_with_edges()

    assert store.outgoing_edges("S4") == [edge_supports()]
    assert store.outgoing_edges("S4", edge_type="supports") == [edge_supports()]
    assert store.incoming_edges("S1") == [edge_depends_on(), edge_invalidates()]
    assert store.incoming_edges("S1", edge_type="depends-on") == [edge_depends_on()]
    assert store.edges_from("S4") == [edge_supports()]
    assert store.edges_to("S1") == [edge_depends_on(), edge_invalidates()]


def test_evidence_state_helpers() -> None:
    """Evidence-state helpers should use StateNode.evidence_id links."""
    store = make_store_with_states()

    assert ids(store.states_supported_by("E3")) == ["S3", "S4"]
    assert store.evidence_for_state("S3") == evidence_3()

    store.add_state(
        StateNode(
            state_id="S5",
            entity="note",
            attribute="status",
            value="unverified",
            status="uncertain",
            confidence=0.1,
        )
    )
    assert store.evidence_for_state("S5") is None


def test_validate_integrity_passes_on_valid_graph() -> None:
    """A valid graph should pass integrity validation."""
    store = make_store_with_edges()

    store.validate_integrity()


def test_validate_integrity_catches_missing_edge_reference() -> None:
    """Integrity validation should catch broken edge references."""
    store = make_store_with_edges()
    store.edges.append(
        StateEdge(source="missing", target="S1", edge_type="depends-on")
    )

    with pytest.raises(ValueError, match="missing source"):
        store.validate_integrity()


def test_validate_integrity_catches_missing_state_evidence_reference() -> None:
    """Integrity validation should catch missing state evidence references."""
    store = GraphStore()
    store.add_state(state_1())

    with pytest.raises(ValueError, match="missing evidence_id"):
        store.validate_integrity()


def test_to_dict_returns_json_serializable_data() -> None:
    """Serialized graph data should be JSON serializable and deterministic."""
    store = make_store_with_edges()

    payload = store.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert set(payload) == {"states", "evidence", "edges"}
    assert [state["state_id"] for state in payload["states"]] == [
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    assert [evidence["evidence_id"] for evidence in payload["evidence"]] == [
        "E1",
        "E2",
        "E3",
    ]
    assert [edge["edge_type"] for edge in payload["edges"]] == [
        "depends-on",
        "invalidates",
        "supports",
    ]
    assert "Friday afternoon" in encoded


def test_from_dict_restores_all_graph_data() -> None:
    """GraphStore should restore states, evidence, and edges."""
    original = make_store_with_edges()

    restored = GraphStore.from_dict(original.to_dict())

    assert restored.list_evidence() == [evidence_1(), evidence_2(), evidence_3()]
    assert ids(restored.list_states()) == ["S1", "S2", "S3", "S4"]
    assert restored.list_edges() == [
        edge_depends_on(),
        edge_invalidates(),
        edge_supports(),
    ]
    assert ids(restored.successors("S4")) == ["S3"]


def test_save_json_and_load_json_round_trip(tmp_path) -> None:
    """GraphStore should save and load pretty JSON files."""
    original = make_store_with_edges()
    path = tmp_path / "graphs" / "stategraph.json"

    original.save_json(path)
    restored = GraphStore.load_json(path)

    assert path.exists()
    assert restored.to_dict() == original.to_dict()


def test_copy_creates_independent_graph() -> None:
    """copy should create an independent graph store."""
    original = make_store_with_edges()
    copied = original.copy()

    copied.update_state_status("S1", "stale")

    assert copied.get_state("S1").status == "stale"
    assert original.get_state("S1").status == "current"
    assert copied.to_dict() != original.to_dict()


def test_clear_removes_all_data() -> None:
    """clear should remove states, evidence, edges, indexes, and graph nodes."""
    store = make_store_with_edges()

    store.clear()

    assert store.list_evidence() == []
    assert store.list_states() == []
    assert store.list_edges() == []
    assert store.list_entities() == []
    assert store.graph.number_of_nodes() == 0


def test_deterministic_ordering() -> None:
    """Lists and serialization should use deterministic ordering."""
    store = make_store_with_edges()

    assert [evidence.evidence_id for evidence in store.list_evidence()] == [
        "E1",
        "E2",
        "E3",
    ]
    assert ids(store.list_states()) == ["S1", "S2", "S3", "S4"]
    assert store.list_edges() == [
        edge_depends_on(),
        edge_invalidates(),
        edge_supports(),
    ]
    assert [edge["source"] for edge in store.to_dict()["edges"]] == ["S2", "S3", "S4"]

