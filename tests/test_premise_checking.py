"""Tests for deterministic query premise checking."""

from __future__ import annotations

from stategraph.core.graph_store import GraphStore
from stategraph.core.premise_checking import (
    PremiseChecker,
    check_query_premises,
)
from stategraph.schemas import (
    EvidenceNode,
    PremiseCheckReport,
    PremiseCheckResult,
    QueryPremise,
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
    """Build the reusable premise checking graph."""
    graph = GraphStore()
    for item in [
        evidence("E0", "The task was previously marked completed."),
        evidence("E1", "User said she is free on Friday afternoon."),
        evidence("E2", "User said she has a flight to Melbourne on Friday afternoon."),
        evidence("E3", "User said she likes Italian food."),
        evidence("E4", "User said she does not want Italian tonight."),
        evidence("E5", "The project deadline is Wednesday."),
        evidence("E6", "The task result has a bug."),
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
            "user",
            "preference",
            "likes Italian food",
            "current",
            "E3",
            None,
        ),
        state(
            "S5",
            "user",
            "preference",
            "does not want Italian",
            "current",
            "E4",
            "tonight",
        ),
        state(
            "S6",
            "project",
            "deadline",
            "Wednesday",
            "current",
            "E5",
            None,
        ),
        state(
            "S7",
            "task",
            "status",
            "needs-review",
            "current",
            "E6",
            None,
        ),
        state(
            "S8",
            "task",
            "status",
            "completed",
            "stale",
            "E0",
            None,
        ),
    ]:
        graph.add_state(item)
    return graph


def first_result(report: PremiseCheckReport) -> PremiseCheckResult:
    """Return the first premise check result."""
    assert report.results
    return report.results[0]


def test_no_premise_query() -> None:
    """A direct information request should not introduce a premise."""
    report = PremiseChecker().check_query("What is my current availability?", make_graph())

    assert report.premises == []
    assert report.results == []
    assert report.recommended_response_policy == "proceed_without_premise"


def test_stale_availability_premise() -> None:
    """A stale availability premise should be flagged before answering."""
    report = PremiseChecker().check_query(
        "Schedule the meeting on Friday afternoon because I am free then.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.attribute == "availability"
    assert result.status == "contradicted_by_current"
    assert result.conflicting_current_state_ids == ["S2"]
    assert result.stale_support_state_ids == ["S1"]
    assert report.has_contradiction is True
    assert report.recommended_response_policy == "correct_stale_premise"


def test_current_availability_premise() -> None:
    """A premise aligned with current availability should be accepted."""
    report = PremiseChecker().check_query(
        "Do not schedule the meeting on Friday afternoon because I am unavailable then.",
        make_graph(),
    )
    result = first_result(report)

    assert result.status == "supported_current"
    assert result.supporting_current_state_ids == ["S2"]
    assert report.recommended_response_policy == "accept_premise"


def test_location_premise_supported() -> None:
    """A Melbourne location premise should match current location."""
    report = PremiseChecker().check_query(
        "Book something in Melbourne on Friday afternoon.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.attribute == "location"
    assert result.status == "supported_current"
    assert result.supporting_current_state_ids == ["S3"]


def test_location_premise_contradicted() -> None:
    """A London location premise should conflict with current Melbourne state."""
    report = PremiseChecker().check_query(
        "Book something in London on Friday afternoon.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.value == "London"
    assert result.status == "contradicted_by_current"
    assert result.conflicting_current_state_ids == ["S3"]
    assert report.recommended_response_policy == "correct_stale_premise"


def test_preference_premise_supported() -> None:
    """A general Italian preference premise should be supported."""
    report = PremiseChecker().check_query(
        "Recommend Italian food because I like Italian.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.attribute == "preference"
    assert result.status == "supported_current"
    assert result.supporting_current_state_ids == ["S4"]


def test_temporary_preference_contradiction() -> None:
    """A tonight-specific Italian premise should conflict with tonight state."""
    report = PremiseChecker().check_query(
        "Recommend Italian food tonight because I want Italian tonight.",
        make_graph(),
    )
    result = first_result(report)

    assert result.status == "contradicted_by_current"
    assert "S5" in result.conflicting_current_state_ids
    assert report.recommended_response_policy == "correct_stale_premise"


def test_deadline_premise_contradicted() -> None:
    """A Monday deadline premise should conflict with current Wednesday."""
    report = PremiseChecker().check_query(
        "Remind me before the Monday deadline.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.entity == "project"
    assert result.premise.attribute == "deadline"
    assert result.status == "contradicted_by_current"
    assert result.conflicting_current_state_ids == ["S6"]


def test_task_status_premise_contradicted() -> None:
    """A completed-task premise should conflict with needs-review state."""
    report = PremiseChecker().check_query(
        "Since the task is completed, archive it.",
        make_graph(),
    )
    result = first_result(report)

    assert result.premise.attribute == "status"
    assert result.status == "contradicted_by_current"
    assert result.conflicting_current_state_ids == ["S7"]
    assert result.stale_support_state_ids == ["S8"]


def test_uncertain_matching() -> None:
    """A premise only matching uncertain state should ask for clarification."""
    graph = make_graph()
    graph.add_evidence(evidence("E7", "User may be free Saturday."))
    graph.add_state(
        state(
            "S9",
            "user",
            "availability",
            "maybe free",
            "uncertain",
            "E7",
            "Saturday",
        )
    )

    report = PremiseChecker().check_query(
        "Schedule it Saturday because I may be free.",
        graph,
    )
    result = first_result(report)

    assert result.status == "uncertain"
    assert result.uncertain_state_ids == ["S9"]
    assert report.has_uncertainty is True
    assert report.recommended_response_policy == "ask_clarification"


def test_unsupported_low_risk_premise_or_no_premise() -> None:
    """Unsupported low-risk queries should not trigger stale correction."""
    report = PremiseChecker().check_query("Find a cafe near the museum.", make_graph())

    assert report.recommended_response_policy != "correct_stale_premise"
    if report.results:
        assert first_result(report).status in {"unsupported", "no_premise_detected"}
    else:
        assert report.premises == []


def test_report_serialization() -> None:
    """Premise reports should serialize through Pydantic."""
    report = PremiseChecker().check_query(
        "Remind me before the Monday deadline.",
        make_graph(),
    )
    payload = report.model_dump()

    assert payload["query"] == "Remind me before the Monday deadline."
    assert payload["results"][0]["status"] == "contradicted_by_current"


def test_wrapper_function_returns_report() -> None:
    """Functional wrapper should return a PremiseCheckReport."""
    report = check_query_premises(
        "Book something in Melbourne on Friday afternoon.",
        make_graph(),
    )

    assert isinstance(report, PremiseCheckReport)
    assert report.results[0].status == "supported_current"


def test_no_mutation() -> None:
    """Premise checking should not mutate graph state statuses."""
    graph = make_graph()
    before = {state.state_id: state.status for state in graph.list_states()}

    PremiseChecker().check_query(
        "Schedule the meeting on Friday afternoon because I am free then.",
        graph,
    )

    after = {state.state_id: state.status for state in graph.list_states()}
    assert after == before


def test_deterministic_output() -> None:
    """Running the same query twice should produce identical reports."""
    query = "Recommend Italian food tonight because I want Italian tonight."
    first = PremiseChecker().check_query(query, make_graph()).model_dump()
    second = PremiseChecker().check_query(query, make_graph()).model_dump()

    assert first == second


def test_query_premise_schema_instantiates() -> None:
    """The premise schema should be directly constructible."""
    premise = QueryPremise(
        premise_id="P1",
        text="because I am free",
        entity="user",
        attribute="availability",
        value="free",
        time_scope="Friday afternoon",
        condition_scope=None,
        confidence=0.9,
    )

    assert premise.model_dump()["premise_id"] == "P1"
