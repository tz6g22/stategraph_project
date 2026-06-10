"""Tests for deterministic rule-based state linking."""

from __future__ import annotations

from stategraph.core.graph_store import GraphStore
from stategraph.core.state_linking import StateLinker, find_linked_states
from stategraph.schemas import EvidenceNode, StateEdge, StateLink, StateNode


def evidence(evidence_id: str = "E1") -> EvidenceNode:
    """Create a simple evidence node."""
    text_by_id = {
        "E1": "User said she is free on Friday afternoon.",
        "E2": "Assistant planned a meeting for Friday afternoon.",
        "E3": "User said she has a flight to Melbourne on Friday afternoon.",
        "E4": "Project deadline is Monday.",
    }
    return EvidenceNode(
        evidence_id=evidence_id,
        text=text_by_id.get(evidence_id, "Synthetic test evidence."),
        source="test",
        timestamp=None,
    )


def state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    time_scope: str | None = "Friday afternoon",
    status: str = "current",
    evidence_id: str | None = "E1",
) -> StateNode:
    """Create a test state node."""
    return StateNode(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope=time_scope,
        condition_scope=None,
        status=status,  # type: ignore[arg-type]
        evidence_id=evidence_id,
        confidence=1.0,
    )


def availability_state(
    state_id: str = "S1",
    status: str = "current",
    evidence_id: str | None = "E1",
) -> StateNode:
    """Create the canonical existing availability state."""
    return state(
        state_id=state_id,
        entity="user",
        attribute="availability",
        value="free",
        status=status,
        evidence_id=evidence_id,
    )


def candidate(
    state_id: str = "C1",
    entity: str = "user",
    attribute: str = "availability",
    value: str = "unavailable",
    time_scope: str | None = "Friday afternoon",
    evidence_id: str | None = "E3",
) -> StateNode:
    """Create a candidate state."""
    return state(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope=time_scope,
        status="current",
        evidence_id=evidence_id,
    )


def store_with(*states: StateNode) -> GraphStore:
    """Create a graph store containing common evidence and provided states."""
    graph = GraphStore()
    for evidence_id in ["E1", "E2", "E3", "E4"]:
        graph.add_evidence(evidence(evidence_id))
    for existing in states:
        graph.add_state(existing)
    return graph


def test_same_entity_attribute_time_gets_strongest_link() -> None:
    """Same entity, attribute, and time should produce the strongest type."""
    graph = store_with(availability_state())

    links = StateLinker().find_linked_states(candidate(), graph)

    assert len(links) == 1
    assert links[0].matched_state_id == "S1"
    assert links[0].match_type == "entity_attribute_time"
    assert links[0].score >= 0.95


def test_same_entity_different_attribute_gets_medium_link() -> None:
    """Same entity with different attribute should still link."""
    graph = store_with(availability_state())
    cand = candidate(entity="user", attribute="location", value="Melbourne")

    links = StateLinker().find_linked_states(cand, graph)

    assert len(links) == 1
    assert links[0].match_type == "same_entity"
    assert 0.5 <= links[0].score < 0.8


def test_same_attribute_different_entity_gets_weak_to_medium_link() -> None:
    """Same attribute with different entity should produce a weaker link."""
    graph = store_with(availability_state())
    cand = candidate(entity="assistant", attribute="availability", value="unavailable")

    links = StateLinker().find_linked_states(cand, graph)

    assert len(links) == 1
    assert links[0].match_type == "same_attribute"
    assert 0.4 <= links[0].score < 0.75


def test_same_time_scope_only_gets_weak_link() -> None:
    """Same time scope alone should be enough for a weak link."""
    graph = store_with(availability_state())
    cand = candidate(
        entity="meeting_plan",
        attribute="feasibility",
        value="invalid",
    )

    links = StateLinker().find_linked_states(cand, graph)

    assert len(links) == 1
    assert links[0].match_type == "same_time_scope"
    assert 0.0 < links[0].score < 0.3


def test_unrelated_state_returns_no_link() -> None:
    """Unrelated states should not be linked."""
    existing = state(
        state_id="S_deadline",
        entity="project",
        attribute="deadline",
        value="Monday",
        time_scope=None,
        evidence_id="E4",
    )
    graph = store_with(existing)
    cand = candidate(entity="user", attribute="location", value="Melbourne")

    links = StateLinker().find_linked_states(cand, graph)

    assert links == []


def test_status_filtering_defaults_and_explicit_statuses() -> None:
    """Default linking should include current and uncertain only."""
    graph = store_with(
        availability_state("S_current", status="current"),
        availability_state("S_uncertain", status="uncertain"),
        availability_state("S_stale", status="stale"),
        availability_state("S_historical", status="historical"),
    )

    default_links = StateLinker().find_linked_states(candidate(), graph)
    explicit_links = StateLinker(
        include_statuses={"current", "uncertain", "stale"}
    ).find_linked_states(candidate(), graph)

    assert [link.matched_state_id for link in default_links] == [
        "S_current",
        "S_uncertain",
    ]
    assert [link.matched_state_id for link in explicit_links] == [
        "S_current",
        "S_stale",
        "S_uncertain",
    ]


def test_links_sorted_by_score_then_state_id() -> None:
    """Links should sort by descending score then deterministic state id."""
    graph = store_with(
        state("S3", "user", "location", "London"),
        state("S4", "user", "timezone", "UTC"),
        state("S1", "user", "availability", "free"),
        state("S2", "assistant", "availability", "busy"),
    )

    links = StateLinker().find_linked_states(candidate(), graph)

    assert [link.matched_state_id for link in links] == ["S1", "S3", "S4", "S2"]
    assert [link.score for link in links] == sorted(
        [link.score for link in links],
        reverse=True,
    )
    assert links[1].score == links[2].score


def test_max_links_limits_after_sorting() -> None:
    """max_links should keep only the strongest links."""
    graph = store_with(
        availability_state("S1"),
        state("S2", "user", "location", "Melbourne"),
    )

    links = StateLinker(max_links=1).find_linked_states(candidate(), graph)

    assert len(links) == 1
    assert links[0].matched_state_id == "S1"


def test_min_score_filters_weak_links() -> None:
    """A high min_score should remove weak time-only links."""
    graph = store_with(availability_state())
    cand = candidate(
        entity="meeting_plan",
        attribute="feasibility",
        value="invalid",
    )

    links = StateLinker(min_score=0.5).find_linked_states(cand, graph)

    assert links == []


def test_self_link_prevention() -> None:
    """A candidate already in the graph should not link to itself."""
    existing = availability_state("S1")
    graph = store_with(existing)

    links = StateLinker().find_linked_states(existing, graph)

    assert links == []


def test_shared_evidence_bonus() -> None:
    """Shared evidence should create an evidence-neighbor link."""
    graph = store_with(availability_state(evidence_id="E1"))
    cand = candidate(
        entity="project",
        attribute="deadline",
        value="Monday",
        time_scope=None,
        evidence_id="E1",
    )

    link = StateLinker().score_pair(cand, graph.get_state("S1"), graph)

    assert link is not None
    assert link.match_type == "evidence_neighbor"
    assert link.features["shared_evidence"] is True
    assert link.score >= 0.10


def test_graph_neighbor_bonus_is_deterministic_and_safe() -> None:
    """Dependency-neighbor scoring should be deterministic and not crash."""
    s1 = availability_state("S1")
    s2 = state(
        "S2",
        entity="meeting_plan",
        attribute="feasibility",
        value="valid",
        evidence_id="E2",
    )
    graph = store_with(s1, s2)
    graph.add_edge(StateEdge(source="S2", target="S1", edge_type="depends-on"))
    cand = candidate(
        entity="meeting_plan",
        attribute="status",
        value="invalid",
        time_scope=None,
        evidence_id=None,
    )

    link = StateLinker().score_pair(cand, s1, graph)

    assert link is not None
    assert link.match_type == "dependency_neighbor"
    assert link.features["graph_neighbor"] is True
    assert link.score >= 0.10


def test_functional_wrapper_returns_links() -> None:
    """The functional wrapper should use StateLinker."""
    graph = store_with(availability_state())

    links = find_linked_states(candidate(), graph)

    assert len(links) == 1
    assert isinstance(links[0], StateLink)


def test_state_link_serialization_safety() -> None:
    """StateLink should be JSON-serializable via model_dump."""
    graph = store_with(availability_state())
    link = StateLinker().find_linked_states(candidate(), graph)[0]

    payload = link.model_dump()

    assert payload["candidate_state_id"] == "C1"
    assert payload["matched_state_id"] == "S1"
    assert isinstance(payload["features"], dict)
