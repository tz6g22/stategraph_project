"""Deterministic query premise checking.

This module extracts lightweight query premises and compares them with the
stored state graph. It does not retrieve final evidence, generate answers,
mutate graph state, revise states, or propagate invalidations.
"""

from __future__ import annotations

import re

from stategraph.core.conflict_detection import (
    ConflictDetector,
    are_opposite_values,
    compatible_time_scope,
    normalize_text as normalize_conflict_text,
    token_overlap as conflict_token_overlap,
    tokenize as conflict_tokenize,
)
from stategraph.core.graph_store import GraphStore
from stategraph.schemas import (
    PremiseCheckReport,
    PremiseCheckResult,
    QueryPremise,
    ResponsePolicy,
    StateNode,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
KNOWN_LOCATIONS = {
    "london": "London",
    "melbourne": "Melbourne",
    "manchester": "Manchester",
    "paris": "Paris",
    "rome": "Rome",
}
DAY_NAMES = {
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
    "saturday": "Saturday",
    "sunday": "Sunday",
}
TIME_PHRASES = [
    "friday afternoon",
    "saturday afternoon",
    "monday morning",
    "monday afternoon",
    "tuesday morning",
    "tuesday afternoon",
    "wednesday morning",
    "wednesday afternoon",
    "thursday morning",
    "thursday afternoon",
    "today",
    "tonight",
    "tomorrow",
    "saturday",
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
]
FOOD_TERMS = {
    "italian": "Italian",
    "chinese": "Chinese",
    "indian": "Indian",
    "thai": "Thai",
    "mexican": "Mexican",
    "japanese": "Japanese",
}
LOW_CONFIDENCE = 0.35
MEDIUM_CONFIDENCE = 0.70
HIGH_CONFIDENCE = 0.85


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic lexical matching."""
    return normalize_conflict_text(text)


def tokenize(text: str | None) -> set[str]:
    """Return normalized tokens."""
    return conflict_tokenize(text)


def token_overlap(a: str | None, b: str | None) -> float:
    """Return Jaccard token overlap for two strings."""
    return conflict_token_overlap(a, b)


def detect_time_scope(query: str) -> str | None:
    """Return a simple time scope from a query, if one is explicit."""
    normalized = normalize_text(query)
    for phrase in TIME_PHRASES:
        if normalize_text(phrase) in normalized:
            return _canonical_time_scope(phrase)
    return None


def detect_availability_premise(query: str) -> QueryPremise | None:
    """Detect user availability premises from transparent lexical cues."""
    normalized = normalize_text(query)
    value: str | None = None
    confidence = MEDIUM_CONFIDENCE

    if _contains_any(normalized, {"unavailable", "not free", "busy"}):
        value = "unavailable" if "unavailable" in normalized or "not free" in normalized else "busy"
        confidence = HIGH_CONFIDENCE
    elif _contains_any(normalized, {"may be free", "might be free", "maybe free"}):
        value = "maybe free"
        confidence = 0.55
    elif _contains_any(normalized, {"i am free", "i'm free", "im free", "free then", "available"}):
        value = "free" if "available" not in normalized else "available"
        confidence = HIGH_CONFIDENCE

    if value is None:
        return None
    if "availability" in normalized and not _contains_any(
        normalized,
        {"free", "available", "unavailable", "busy"},
    ):
        return None
    return QueryPremise(
        premise_id="P0",
        text=_premise_text(query),
        entity="user",
        attribute="availability",
        value=value,
        time_scope=detect_time_scope(query),
        condition_scope=None,
        confidence=confidence,
    )


def detect_location_premise(query: str) -> QueryPremise | None:
    """Detect location premises for known city-like references."""
    normalized = normalize_text(query)
    location = None
    for token, canonical in sorted(KNOWN_LOCATIONS.items()):
        if token in normalized.split():
            location = canonical
            break
    if location is None:
        return None
    if not _contains_any(normalized, {"in", "near", "at", "book", "restaurant", "something"}):
        return None
    return QueryPremise(
        premise_id="P0",
        text=_premise_text(query),
        entity="user",
        attribute="location",
        value=location,
        time_scope=detect_time_scope(query),
        condition_scope=None,
        confidence=MEDIUM_CONFIDENCE,
    )


def detect_preference_premise(query: str) -> QueryPremise | None:
    """Detect simple food preference premises."""
    normalized = normalize_text(query)
    food = None
    for token, canonical in sorted(FOOD_TERMS.items()):
        if token in normalized.split():
            food = canonical
            break
    if food is None:
        return None

    negative = _contains_any(
        normalized,
        {
            "do not want",
            "does not want",
            "dont want",
            "doesnt want",
            "not want",
            "dislike",
            "no italian",
        },
    )
    positive = _contains_any(
        normalized,
        {"i like", "like italian", "want italian", "recommend italian"},
    )
    if not negative and not positive:
        return None

    value = f"does not want {food}" if negative else f"likes {food}"
    if "want" in normalized and not negative:
        value = f"wants {food}"
    return QueryPremise(
        premise_id="P0",
        text=_premise_text(query),
        entity="user",
        attribute="preference",
        value=value,
        time_scope=detect_time_scope(query),
        condition_scope=None,
        confidence=HIGH_CONFIDENCE if "because" in normalized else MEDIUM_CONFIDENCE,
    )


def detect_deadline_premise(query: str) -> QueryPremise | None:
    """Detect a simple project/task deadline premise."""
    normalized = normalize_text(query)
    if "deadline" not in normalized:
        return None
    for token, canonical in DAY_NAMES.items():
        if token in normalized.split():
            return QueryPremise(
                premise_id="P0",
                text=_premise_text(query),
                entity="project",
                attribute="deadline",
                value=canonical,
                time_scope=None,
                condition_scope=None,
                confidence=HIGH_CONFIDENCE,
            )
    return None


def detect_task_status_premise(query: str) -> QueryPremise | None:
    """Detect a simple task status premise."""
    normalized = normalize_text(query)
    if "task" not in normalized:
        return None
    value: str | None = None
    if _contains_any(normalized, {"completed", "complete", "done"}):
        value = "completed"
    elif _contains_any(normalized, {"needs review", "needs-review", "bug", "buggy"}):
        value = "needs-review"
    if value is None:
        return None
    return QueryPremise(
        premise_id="P0",
        text=_premise_text(query),
        entity="task",
        attribute="status",
        value=value,
        time_scope=detect_time_scope(query),
        condition_scope=None,
        confidence=HIGH_CONFIDENCE,
    )


def make_premise_id(index: int) -> str:
    """Return a deterministic one-based premise id."""
    return f"P{index + 1}"


class PremiseChecker:
    """Check whether a query relies on stale or contradicted state."""

    def __init__(
        self,
        conflict_detector: ConflictDetector | None = None,
        strict_time_matching: bool = False,
        min_premise_confidence: float = 0.2,
    ) -> None:
        """Configure deterministic premise checking."""
        if not 0.0 <= min_premise_confidence <= 1.0:
            raise ValueError("min_premise_confidence must be between 0.0 and 1.0")
        self.strict_time_matching = strict_time_matching
        self.min_premise_confidence = min_premise_confidence
        self.conflict_detector = conflict_detector or ConflictDetector(
            strict_time_matching=strict_time_matching
        )

    def extract_premises(self, query: str) -> list[QueryPremise]:
        """Extract lightweight premise candidates from a query."""
        if not query.strip():
            return []

        detectors = [
            detect_availability_premise,
            detect_location_premise,
            detect_preference_premise,
            detect_deadline_premise,
            detect_task_status_premise,
        ]
        premises: list[QueryPremise] = []
        seen_slots: set[tuple[str | None, str | None, str | None, str | None]] = set()
        for detector in detectors:
            premise = detector(query)
            if premise is None or premise.confidence < self.min_premise_confidence:
                continue
            slot = (
                premise.entity,
                premise.attribute,
                normalize_text(premise.value),
                normalize_text(premise.time_scope),
            )
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            premises.append(
                premise.model_copy(
                    update={"premise_id": make_premise_id(len(premises))}
                )
            )
        return premises

    def check_premise(
        self,
        premise: QueryPremise,
        graph: GraphStore,
    ) -> PremiseCheckResult:
        """Check one premise against graph states without mutating graph."""
        if not premise.entity or not premise.attribute or not premise.value:
            return PremiseCheckResult(
                premise=premise,
                status="no_premise_detected",
                supporting_current_state_ids=[],
                conflicting_current_state_ids=[],
                stale_support_state_ids=[],
                historical_support_state_ids=[],
                uncertain_state_ids=[],
                reason="premise is missing entity, attribute, or value",
                recommended_response_policy="proceed_without_premise",
            )

        current_states = self._matching_slot_states(graph.list_current_states(), premise)
        stale_states = self._matching_slot_states(graph.list_stale_states(), premise)
        historical_states = self._matching_slot_states(
            graph.list_historical_states(),
            premise,
        )
        uncertain_states = self._matching_slot_states(
            graph.list_uncertain_states(),
            premise,
        )

        supporting_current = [
            state.state_id
            for state in current_states
            if self._state_supports_premise(state, premise)
        ]
        conflicting_current = [
            state.state_id
            for state in current_states
            if self._state_conflicts_with_premise(state, premise)
        ]
        stale_support = [
            state.state_id
            for state in stale_states
            if self._state_supports_premise(state, premise)
        ]
        historical_support = [
            state.state_id
            for state in historical_states
            if self._state_supports_premise(state, premise)
        ]
        uncertain_ids = [
            state.state_id
            for state in uncertain_states
            if self._state_supports_premise(state, premise)
            or self._state_conflicts_with_premise(state, premise)
            or token_overlap(state.value, premise.value) > 0.0
        ]

        if conflicting_current:
            return self._result(
                premise=premise,
                status="contradicted_by_current",
                supporting_current_state_ids=supporting_current,
                conflicting_current_state_ids=conflicting_current,
                stale_support_state_ids=stale_support,
                historical_support_state_ids=historical_support,
                uncertain_state_ids=uncertain_ids,
                reason="premise conflicts with one or more current states",
                policy="correct_stale_premise",
            )
        if supporting_current:
            return self._result(
                premise=premise,
                status="supported_current",
                supporting_current_state_ids=supporting_current,
                conflicting_current_state_ids=[],
                stale_support_state_ids=stale_support,
                historical_support_state_ids=historical_support,
                uncertain_state_ids=uncertain_ids,
                reason="premise is supported by current state",
                policy="accept_premise",
            )
        if stale_support:
            return self._result(
                premise=premise,
                status="supported_by_stale",
                supporting_current_state_ids=[],
                conflicting_current_state_ids=[],
                stale_support_state_ids=stale_support,
                historical_support_state_ids=historical_support,
                uncertain_state_ids=uncertain_ids,
                reason="premise is only supported by stale state",
                policy="correct_stale_premise",
            )
        if historical_support:
            return self._result(
                premise=premise,
                status="supported_by_historical",
                supporting_current_state_ids=[],
                conflicting_current_state_ids=[],
                stale_support_state_ids=[],
                historical_support_state_ids=historical_support,
                uncertain_state_ids=uncertain_ids,
                reason="premise is only supported by historical state",
                policy="correct_stale_premise",
            )
        if uncertain_ids:
            return self._result(
                premise=premise,
                status="uncertain",
                supporting_current_state_ids=[],
                conflicting_current_state_ids=[],
                stale_support_state_ids=[],
                historical_support_state_ids=[],
                uncertain_state_ids=uncertain_ids,
                reason="premise only matches uncertain state",
                policy="ask_clarification",
            )
        return self._result(
            premise=premise,
            status="unsupported",
            supporting_current_state_ids=[],
            conflicting_current_state_ids=[],
            stale_support_state_ids=[],
            historical_support_state_ids=[],
            uncertain_state_ids=[],
            reason="no matching graph state supports or contradicts the premise",
            policy="proceed_without_premise",
        )

    def check_query(
        self,
        query: str,
        graph: GraphStore,
    ) -> PremiseCheckReport:
        """Check all extracted query premises and aggregate response policy."""
        premises = self.extract_premises(query)
        results = [self.check_premise(premise, graph) for premise in premises]
        notes: list[str] = []
        if not premises:
            notes.append("no explicit premise detected")

        has_contradiction = any(
            result.status == "contradicted_by_current" for result in results
        )
        has_stale_premise = any(
            result.status in {"supported_by_stale", "supported_by_historical"}
            or bool(result.stale_support_state_ids)
            or bool(result.historical_support_state_ids)
            for result in results
        )
        has_uncertainty = any(result.status == "uncertain" for result in results)
        policy = self._aggregate_policy(results)

        return PremiseCheckReport(
            query=query,
            premises=premises,
            results=results,
            has_stale_premise=has_stale_premise,
            has_contradiction=has_contradiction,
            has_uncertainty=has_uncertainty,
            recommended_response_policy=policy,
            notes=notes,
        )

    def check(self, query: str, graph: GraphStore) -> PremiseCheckReport:
        """Compatibility wrapper for the previous checker interface."""
        return self.check_query(query, graph)

    def _matching_slot_states(
        self,
        states: list[StateNode],
        premise: QueryPremise,
    ) -> list[StateNode]:
        """Return states with compatible entity, attribute, and time scope."""
        return [
            state
            for state in states
            if state.entity == premise.entity
            and state.attribute == premise.attribute
            and compatible_time_scope(
                premise.time_scope,
                state.time_scope,
                strict=self.strict_time_matching,
            )
        ]

    def _state_supports_premise(
        self,
        state: StateNode,
        premise: QueryPremise,
    ) -> bool:
        """Return whether a state supports a premise value."""
        if premise.value is None:
            return False
        normalized_state = normalize_text(state.value)
        normalized_premise = normalize_text(premise.value)
        if not normalized_state or not normalized_premise:
            return False
        if normalized_state == normalized_premise:
            return True
        if normalized_premise in normalized_state or normalized_state in normalized_premise:
            return not are_opposite_values(state.value, premise.value)
        overlap = token_overlap(state.value, premise.value)
        if state.attribute == "preference":
            return overlap >= 0.30 and not are_opposite_values(state.value, premise.value)
        return overlap >= 0.60 and not are_opposite_values(state.value, premise.value)

    def _state_conflicts_with_premise(
        self,
        state: StateNode,
        premise: QueryPremise,
    ) -> bool:
        """Return whether a current state contradicts a premise."""
        if premise.value is None:
            return False
        if state.attribute == "preference" and not premise.time_scope and state.time_scope:
            return False
        if are_opposite_values(state.value, premise.value):
            return True

        premise_state = StateNode(
            state_id=f"premise:{premise.premise_id}",
            entity=premise.entity or "unknown",
            attribute=premise.attribute or "unknown",
            value=premise.value,
            time_scope=premise.time_scope,
            condition_scope=premise.condition_scope,
            status="uncertain",
            evidence_id=None,
            confidence=premise.confidence,
        )
        decision = self.conflict_detector.detect_conflict(
            candidate=premise_state,
            existing=state,
        )
        if decision.label == "explicit_conflict":
            return True

        normalized_state = normalize_text(state.value)
        normalized_premise = normalize_text(premise.value)
        if not normalized_state or not normalized_premise:
            return False
        if normalized_state == normalized_premise:
            return False
        if state.attribute in {"deadline", "location", "status"}:
            return True
        if state.attribute == "availability":
            return token_overlap(state.value, premise.value) == 0.0
        return False

    @staticmethod
    def _aggregate_policy(results: list[PremiseCheckResult]) -> ResponsePolicy:
        """Aggregate result policies with deterministic priority."""
        if any(result.status == "contradicted_by_current" for result in results):
            return "correct_stale_premise"
        if any(
            result.status in {"supported_by_stale", "supported_by_historical"}
            for result in results
        ):
            return "correct_stale_premise"
        if any(result.status == "uncertain" for result in results):
            return "ask_clarification"
        if results and all(result.status == "supported_current" for result in results):
            return "accept_premise"
        return "proceed_without_premise"

    @staticmethod
    def _result(
        premise: QueryPremise,
        status: str,
        supporting_current_state_ids: list[str],
        conflicting_current_state_ids: list[str],
        stale_support_state_ids: list[str],
        historical_support_state_ids: list[str],
        uncertain_state_ids: list[str],
        reason: str,
        policy: str,
    ) -> PremiseCheckResult:
        """Build a result with deterministic id ordering."""
        return PremiseCheckResult(
            premise=premise,
            status=status,  # type: ignore[arg-type]
            supporting_current_state_ids=sorted(set(supporting_current_state_ids)),
            conflicting_current_state_ids=sorted(set(conflicting_current_state_ids)),
            stale_support_state_ids=sorted(set(stale_support_state_ids)),
            historical_support_state_ids=sorted(set(historical_support_state_ids)),
            uncertain_state_ids=sorted(set(uncertain_state_ids)),
            reason=reason,
            recommended_response_policy=policy,  # type: ignore[arg-type]
        )


def extract_premises(query: str) -> list[QueryPremise]:
    """Functional wrapper for PremiseChecker.extract_premises."""
    return PremiseChecker().extract_premises(query)


def check_query_premises(query: str, graph: GraphStore) -> PremiseCheckReport:
    """Functional wrapper for PremiseChecker.check_query."""
    return PremiseChecker().check_query(query, graph)


def _contains_any(normalized_text: str, phrases: set[str]) -> bool:
    """Return whether normalized text contains any normalized phrase."""
    return any(normalize_text(phrase) in normalized_text for phrase in phrases)


def _canonical_time_scope(phrase: str) -> str:
    """Return a readable time-scope string."""
    tokens = phrase.split()
    if len(tokens) == 1:
        return DAY_NAMES.get(tokens[0], tokens[0])
    return " ".join(DAY_NAMES.get(token, token) for token in tokens)


def _premise_text(query: str) -> str:
    """Return compact premise text for reports."""
    return " ".join(query.strip().split())


PremiseCheckingEngine = PremiseChecker
check_premises = check_query_premises
