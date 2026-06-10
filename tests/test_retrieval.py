"""Tests for premise-aware state retrieval."""

from __future__ import annotations

from stategraph.core.graph_store import GraphStore
from stategraph.core.premise_checking import PremiseChecker
from stategraph.core.retrieval import StateRetriever, retrieve_context
from stategraph.schemas import (
    EvidenceNode,
    PremiseCheckReport,
    PremiseCheckResult,
    QueryPremise,
    RetrievedEvidence,
    RetrievedState,
    RetrievalResult,
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
    evidence_id: str | None,
    time_scope: str | None = None,
) -> StateNode:
    """Create a test state."""
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


def make_graph() -> GraphStore:
    """Build a reusable retrieval graph."""
    graph = GraphStore()
    for item in [
        evidence("E1", "User said she is free on Friday afternoon."),
        evidence("E2", "User said she has a flight to Melbourne on Friday afternoon."),
        evidence("E3", "Assistant planned a meeting for Friday afternoon."),
        evidence("E4", "User said she likes Italian food."),
        evidence("E5", "User said she does not want Italian tonight."),
        evidence("E6", "The project deadline is Wednesday."),
    ]:
        graph.add_evidence(item)

    for item in [
        state(
            "S1",
            "user",
            "availability",
            "free",
            "stale",
            "E1",
            "Friday afternoon",
        ),
        state(
            "S2",
            "user",
            "availability",
            "unavailable",
            "current",
            "E2",
            "Friday afternoon",
        ),
        state(
            "S3",
            "user",
            "location",
            "Melbourne",
            "current",
            "E2",
            "Friday afternoon",
        ),
        state(
            "S4",
            "meeting_plan",
            "feasibility",
            "invalid",
            "stale",
            "E3",
            "Friday afternoon",
        ),
        state(
            "S5",
            "user",
            "preference",
            "likes Italian food",
            "current",
            "E4",
            None,
        ),
        state(
            "S6",
            "user",
            "preference",
            "does not want Italian",
            "current",
            "E5",
            "tonight",
        ),
        state(
            "S7",
            "project",
            "deadline",
            "Wednesday",
            "current",
            "E6",
            None,
        ),
        state(
            "S8",
            "user",
            "availability",
            "maybe free",
            "uncertain",
            "E1",
            "Saturday",
        ),
    ]:
        graph.add_state(item)
    return graph


def ids(states: list[RetrievedState]) -> list[str]:
    """Return retrieved state ids."""
    return [state.state_id for state in states]


def warning_types(result: RetrievalResult) -> set[str]:
    """Return warning types from a result."""
    return {warning.warning_type for warning in result.warnings}


def stale_premise_report() -> PremiseCheckReport:
    """Create a report where a premise is supported only by stale S1."""
    premise = QueryPremise(
        premise_id="P1",
        text="because I am free Friday afternoon",
        entity="user",
        attribute="availability",
        value="free",
        time_scope="Friday afternoon",
        condition_scope=None,
        confidence=0.9,
    )
    result = PremiseCheckResult(
        premise=premise,
        status="supported_by_stale",
        supporting_current_state_ids=[],
        conflicting_current_state_ids=[],
        stale_support_state_ids=["S1"],
        historical_support_state_ids=[],
        uncertain_state_ids=[],
        reason="premise supported by stale state",
        recommended_response_policy="correct_stale_premise",
    )
    return PremiseCheckReport(
        query="Schedule because I am free Friday afternoon.",
        premises=[premise],
        results=[result],
        has_stale_premise=True,
        has_contradiction=False,
        has_uncertainty=False,
        recommended_response_policy="correct_stale_premise",
        notes=[],
    )


def test_retrieves_current_states_only_by_default() -> None:
    """Default retrieval should place only current states in main context."""
    result = StateRetriever().retrieve(
        "Friday availability Melbourne Italian deadline tonight",
        make_graph(),
    )

    assert set(ids(result.current_states)) >= {"S2", "S3", "S5", "S6", "S7"}
    assert "S1" not in ids(result.current_states)
    assert "S4" not in ids(result.current_states)
    assert {state.status for state in result.current_states} == {"current"}


def test_excludes_stale_and_historical_from_current_context() -> None:
    """Stale and historical states should not enter current context."""
    graph = make_graph()
    graph.add_evidence(evidence("E7", "Old deadline was Monday."))
    graph.add_state(
        state("S9", "project", "deadline", "Monday", "historical", "E7", None)
    )

    result = StateRetriever().retrieve("deadline Monday Friday", graph)

    assert "S1" in result.excluded_stale_state_ids
    assert "S4" in result.excluded_stale_state_ids
    assert "S9" in result.excluded_historical_state_ids
    assert "S1" not in ids(result.current_states)
    assert "S9" not in ids(result.current_states)


def test_retrieves_supporting_evidence_without_duplicates() -> None:
    """Evidence should be grouped by evidence id and support all retrieved states."""
    result = StateRetriever().retrieve("Friday availability Melbourne", make_graph())

    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in result.supporting_evidence
    }

    assert len(evidence_by_id) == len(result.supporting_evidence)
    assert "E2" in evidence_by_id
    assert evidence_by_id["E2"].supporting_state_ids == ["S2", "S3"]


def test_missing_evidence_warning() -> None:
    """Missing evidence references should warn without crashing."""
    graph = make_graph()
    graph.add_state(
        state("S_missing", "user", "timezone", "UTC", "current", "E_missing", None)
    )

    result = StateRetriever().retrieve("timezone UTC", graph)

    assert "missing_evidence" in warning_types(result)


def test_premise_contradiction_behavior() -> None:
    """Contradicted premise reports should retrieve current correction context."""
    graph = make_graph()
    premise_report = PremiseChecker().check_query(
        "Schedule the meeting on Friday afternoon because I am free then.",
        graph,
    )

    result = StateRetriever().retrieve(
        "Schedule the meeting on Friday afternoon because I am free then.",
        graph,
        premise_report,
    )

    assert "contradicted_premise" in warning_types(result)
    assert "S2" in ids(result.current_states)
    assert "S1" in ids(result.correction_states)
    assert "S1" not in ids(result.current_states)


def test_stale_premise_behavior() -> None:
    """Stale premise reports should include stale states only as correction context."""
    result = StateRetriever().retrieve(
        "Schedule because I am free Friday afternoon.",
        make_graph(),
        stale_premise_report(),
    )

    assert "stale_premise" in warning_types(result)
    assert "S1" in ids(result.correction_states)
    assert "S1" not in ids(result.current_states)


def test_uncertain_premise_behavior() -> None:
    """Uncertain premise reports should include uncertain states separately."""
    graph = make_graph()
    premise_report = PremiseChecker().check_query(
        "Schedule it Saturday because I may be free.",
        graph,
    )

    result = StateRetriever().retrieve(
        "Schedule it Saturday because I may be free.",
        graph,
        premise_report,
    )

    assert "uncertain_premise" in warning_types(result)
    assert "S8" in ids(result.uncertain_states)


def test_include_uncertain_false() -> None:
    """include_uncertain=False should omit uncertain context."""
    graph = make_graph()
    premise_report = PremiseChecker().check_query(
        "Schedule it Saturday because I may be free.",
        graph,
    )

    result = StateRetriever(include_uncertain=False).retrieve(
        "Schedule it Saturday because I may be free.",
        graph,
        premise_report,
    )

    assert result.uncertain_states == []
    assert "uncertain_premise" in warning_types(result)


def test_no_premise_report_still_retrieves_by_overlap() -> None:
    """Retrieval should work without a premise report."""
    result = StateRetriever().retrieve("What is my Friday availability?", make_graph())

    assert ids(result.current_states)[0] == "S2"
    assert result.premise_policy is None


def test_max_states() -> None:
    """max_states should keep only the strongest current state."""
    result = StateRetriever(max_states=1).retrieve(
        "What is my Friday availability?",
        make_graph(),
    )

    assert ids(result.current_states) == ["S2"]
    assert any("max_states=1" in note for note in result.notes)


def test_max_evidence() -> None:
    """max_evidence should limit grouped evidence."""
    result = StateRetriever(max_evidence=1).retrieve(
        "Friday availability Melbourne Italian deadline tonight",
        make_graph(),
    )

    assert len(result.supporting_evidence) == 1
    assert any("max_evidence=1" in note for note in result.notes)


def test_token_budget() -> None:
    """A small token budget should truncate retrieved context."""
    result = StateRetriever(token_budget=8).retrieve(
        "Friday availability Melbourne Italian deadline tonight",
        make_graph(),
    )

    assert result.truncated is True
    assert "token_budget_truncated" in warning_types(result)


def test_empty_query() -> None:
    """Empty queries should return deterministic safe context."""
    result = StateRetriever().retrieve("", make_graph())

    assert isinstance(result, RetrievalResult)
    assert result.query == ""
    assert all(state.status == "current" for state in result.current_states)


def test_deterministic_output() -> None:
    """Equivalent retrievals should produce identical dumps."""
    first = StateRetriever().retrieve(
        "Friday availability Melbourne",
        make_graph(),
    ).model_dump()
    second = StateRetriever().retrieve(
        "Friday availability Melbourne",
        make_graph(),
    ).model_dump()

    assert first == second


def test_wrapper_function_returns_result() -> None:
    """Functional wrapper should return a RetrievalResult."""
    result = retrieve_context("What is my Friday availability?", make_graph())

    assert isinstance(result, RetrievalResult)
    assert result.current_states


def test_report_serialization() -> None:
    """Retrieval reports should serialize through Pydantic."""
    result = StateRetriever().retrieve("What is my Friday availability?", make_graph())
    payload = result.model_dump()

    assert payload["query"] == "What is my Friday availability?"
    assert payload["current_states"][0]["state_id"] == "S2"


def test_no_mutation() -> None:
    """Retrieval should not mutate graph state statuses."""
    graph = make_graph()
    before = {state.state_id: state.status for state in graph.list_states()}

    StateRetriever().retrieve("Friday availability Melbourne", graph)

    after = {state.state_id: state.status for state in graph.list_states()}
    assert after == before


def test_retrieval_schemas_instantiate() -> None:
    """Retrieval schemas should be directly constructible."""
    retrieved_state = RetrievedState(
        state_id="S1",
        entity="user",
        attribute="availability",
        value="free",
        time_scope="Friday afternoon",
        condition_scope=None,
        status="current",
        evidence_id="E1",
        relevance_score=0.9,
        retrieval_reason="test",
    )
    retrieved_evidence = RetrievedEvidence(
        evidence_id="E1",
        text="User said she is free.",
        source="test",
        timestamp=None,
        supporting_state_ids=["S1"],
    )

    assert retrieved_state.model_dump()["relevance_score"] == 0.9
    assert retrieved_evidence.supporting_state_ids == ["S1"]
