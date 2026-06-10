"""Tests for deterministic conflict detection."""

from __future__ import annotations

from stategraph.core.conflict_detection import (
    ConflictDetector,
    detect_conflict,
    detect_conflicts,
)
from stategraph.schemas import ConflictDecision, StateLink, StateNode


def state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    time_scope: str | None = None,
    condition_scope: str | None = None,
    status: str = "current",
) -> StateNode:
    """Create a test state."""
    return StateNode(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope=time_scope,
        condition_scope=condition_scope,
        status=status,  # type: ignore[arg-type]
        evidence_id=None,
        confidence=1.0,
    )


def availability_free() -> StateNode:
    """Existing free availability state."""
    return state(
        "S1",
        "user",
        "availability",
        "free",
        time_scope="Friday afternoon",
    )


def assert_confidence_range(decision: ConflictDecision) -> None:
    """Assert decision confidence is in range."""
    assert decision.confidence is not None
    assert 0.0 <= decision.confidence <= 1.0


def test_duplicate() -> None:
    """Equivalent states should be duplicates."""
    existing = availability_free()
    candidate = state(
        "C1",
        "user",
        "availability",
        "free",
        time_scope="Friday afternoon",
    )

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "duplicate"
    assert_confidence_range(decision)


def test_consistent_different_attributes() -> None:
    """Different non-conflicting attributes for same entity should coexist."""
    existing = state("S1", "user", "location", "London", time_scope="today")
    candidate = state("C1", "user", "availability", "busy", time_scope="today")

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "consistent"
    assert_confidence_range(decision)


def test_update_deadline() -> None:
    """Same entity and attribute with a non-opposite new value is an update."""
    existing = state("S1", "project", "deadline", "Monday")
    candidate = state("C1", "project", "deadline", "Wednesday")

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "update"
    assert_confidence_range(decision)


def test_explicit_conflict_availability() -> None:
    """Opposite availability values in the same time scope should conflict."""
    existing = availability_free()
    candidate = state(
        "C1",
        "user",
        "availability",
        "unavailable",
        time_scope="Friday afternoon",
    )

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "explicit_conflict"
    assert_confidence_range(decision)


def test_explicit_conflict_validity() -> None:
    """Opposite feasibility values in the same time scope should conflict."""
    existing = state(
        "S1",
        "meeting_plan",
        "feasibility",
        "valid",
        time_scope="Friday afternoon",
    )
    candidate = state(
        "C1",
        "meeting_plan",
        "feasibility",
        "invalid",
        time_scope="Friday afternoon",
    )

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "explicit_conflict"
    assert_confidence_range(decision)


def test_implicit_invalidation_through_availability() -> None:
    """Unavailable user should implicitly invalidate a dependent meeting plan."""
    existing = state(
        "S1",
        "meeting_plan",
        "feasibility",
        "valid",
        time_scope="Friday afternoon",
    )
    candidate = state(
        "C1",
        "user",
        "availability",
        "unavailable",
        time_scope="Friday afternoon",
    )

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "implicit_invalidation"
    assert_confidence_range(decision)


def test_implicit_invalidation_through_location() -> None:
    """Changed user location should invalidate a local plan location."""
    existing = state("S1", "local_plan", "location", "London", time_scope="today")
    candidate = state("C1", "user", "location", "Manchester", time_scope="today")

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "implicit_invalidation"
    assert_confidence_range(decision)


def test_temporary_exception() -> None:
    """A narrow preference exception should be temporary_exception."""
    existing = state("S1", "user", "preference", "likes Italian food")
    candidate = state("C1", "user", "preference", "does not want Italian tonight")

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "temporary_exception"
    assert_confidence_range(decision)


def test_uncertain() -> None:
    """Unrelated states should be uncertain."""
    existing = state("S1", "project", "priority", "medium")
    candidate = state("C1", "user", "location", "Manchester")

    decision = ConflictDetector().detect_conflict(candidate, existing)

    assert decision.label == "uncertain"
    assert_confidence_range(decision)


def test_wrapper_function_returns_decision() -> None:
    """Functional wrapper should return a ConflictDecision."""
    decision = detect_conflict(
        state("C1", "project", "deadline", "Wednesday"),
        state("S1", "project", "deadline", "Monday"),
    )

    assert isinstance(decision, ConflictDecision)
    assert decision.label == "update"


def test_batch_detection() -> None:
    """Batch detection should return one decision per existing state."""
    candidate = state(
        "C1",
        "user",
        "availability",
        "unavailable",
        time_scope="Friday afternoon",
    )
    existing_states = [
        availability_free(),
        state("S2", "project", "priority", "medium"),
    ]

    decisions = detect_conflicts(candidate, existing_states)

    assert [decision.label for decision in decisions] == [
        "explicit_conflict",
        "uncertain",
    ]


def test_no_mutation() -> None:
    """Detection should not mutate candidate or existing state status."""
    existing = availability_free()
    candidate = state(
        "C1",
        "user",
        "availability",
        "unavailable",
        time_scope="Friday afternoon",
    )

    ConflictDetector().detect_conflict(candidate, existing)

    assert existing.status == "current"
    assert candidate.status == "current"


def test_link_aware_detection_not_uncertain() -> None:
    """Strong links should help same-slot differences avoid uncertainty."""
    existing = state("S1", "project", "deadline", "Monday")
    candidate = state("C1", "project", "deadline", "Wednesday")
    link = StateLink(
        candidate_state_id="C1",
        matched_state_id="S1",
        match_type="entity_attribute_time",
        score=0.95,
        reason="same slot",
        features={},
    )

    decision = ConflictDetector().detect_conflict(candidate, existing, link)

    assert decision.label in {"update", "explicit_conflict"}
    assert decision.label != "uncertain"


def test_strict_time_matching_prevents_cross_time_explicit_conflict() -> None:
    """Strict time matching should avoid explicit conflict for non-overlap."""
    existing = state(
        "S1",
        "user",
        "availability",
        "free",
        time_scope="Friday afternoon",
    )
    candidate = state(
        "C1",
        "user",
        "availability",
        "unavailable",
        time_scope="Saturday morning",
    )

    decision = ConflictDetector(strict_time_matching=True).detect_conflict(
        candidate,
        existing,
    )

    assert decision.label != "explicit_conflict"
    assert_confidence_range(decision)

