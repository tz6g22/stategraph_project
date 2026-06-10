"""Rule-based conflict and update relation detection.

This module classifies the relation between a candidate state and an existing
state. It does not mutate graph state, add edges, revise status, or propagate
invalidations.
"""

from __future__ import annotations

import re

from stategraph.schemas import ConflictDecision, StateLink, StateNode


DUPLICATE_CONFIDENCE = 0.95
EXPLICIT_CONFLICT_CONFIDENCE = 0.90
UPDATE_CONFIDENCE = 0.80
TEMPORARY_EXCEPTION_CONFIDENCE = 0.75
IMPLICIT_INVALIDATION_CONFIDENCE = 0.78
CONSISTENT_CONFIDENCE = 0.70
UNCERTAIN_CONFIDENCE = 0.35

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TEMPORARY_SIGNALS = {
    "tonight",
    "today only",
    "this time",
    "temporarily",
    "for now",
    "just today",
    "exception",
}
NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "cannot",
    "cant",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "unavailable",
    "invalid",
    "disabled",
    "closed",
    "failed",
    "bug",
    "buggy",
}
PLAN_TOKENS = {
    "plan",
    "meeting",
    "restaurant",
    "reservation",
    "recommendation",
    "reminder",
    "action",
    "task",
}
DEPENDENCY_VALUE_TOKENS = {
    "valid",
    "feasible",
    "scheduled",
    "recommended",
    "completed",
}
OPPOSITE_VALUE_GROUPS = [
    ({"free", "available"}, {"unavailable", "busy", "not free"}),
    ({"valid", "feasible"}, {"invalid", "infeasible"}),
    ({"completed", "complete", "done"}, {"needs review", "needs-review", "buggy", "bug", "failed"}),
    ({"open"}, {"closed"}),
    ({"enabled"}, {"disabled"}),
    ({"true", "yes"}, {"false", "no"}),
    ({"likes", "like", "wants", "want"}, {"dislikes", "does not want", "do not want", "dont want", "no"}),
]


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic lexical matching."""
    if text is None:
        return ""
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def tokenize(text: str | None) -> set[str]:
    """Return normalized tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def same_text(a: str | None, b: str | None) -> bool:
    """Return whether two strings normalize to the same non-empty text."""
    normalized_a = normalize_text(a)
    normalized_b = normalize_text(b)
    return bool(normalized_a and normalized_a == normalized_b)


def token_overlap(a: str | None, b: str | None) -> float:
    """Return Jaccard token overlap for two strings."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compatible_time_scope(
    a: str | None,
    b: str | None,
    strict: bool = False,
) -> bool:
    """Return whether two time scopes are compatible."""
    normalized_a = normalize_text(a)
    normalized_b = normalize_text(b)
    if not normalized_a and not normalized_b:
        return True
    if strict and (not normalized_a or not normalized_b):
        return False
    if not strict and (not normalized_a or not normalized_b):
        return True
    if normalized_a == normalized_b:
        return True
    return token_overlap(normalized_a, normalized_b) > 0.0


def is_narrow_time_scope(
    time_scope: str | None,
    condition_scope: str | None = None,
) -> bool:
    """Return whether a time or condition scope appears temporary/narrow."""
    text = f"{normalize_text(time_scope)} {normalize_text(condition_scope)}".strip()
    if not text:
        return False
    return any(normalize_text(signal) in text for signal in TEMPORARY_SIGNALS)


def has_negation(text: str | None) -> bool:
    """Return whether text contains a simple negation or negative value."""
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    if tokens & {normalize_text(term) for term in NEGATION_TERMS if " " not in term}:
        return True
    return any(normalize_text(term) in normalized for term in NEGATION_TERMS if " " in term)


def are_opposite_values(a: str | None, b: str | None) -> bool:
    """Return whether two values are transparent lexical opposites."""
    normalized_a = normalize_text(a)
    normalized_b = normalize_text(b)
    if not normalized_a or not normalized_b or normalized_a == normalized_b:
        return False

    if has_negation(normalized_a) and token_overlap(normalized_a, normalized_b) > 0.0:
        return True
    if has_negation(normalized_b) and token_overlap(normalized_a, normalized_b) > 0.0:
        return True

    for positive_values, negative_values in OPPOSITE_VALUE_GROUPS:
        a_positive = any(normalize_text(value) in normalized_a for value in positive_values)
        a_negative = any(normalize_text(value) in normalized_a for value in negative_values)
        b_positive = any(normalize_text(value) in normalized_b for value in positive_values)
        b_negative = any(normalize_text(value) in normalized_b for value in negative_values)
        if (a_positive and b_negative) or (a_negative and b_positive):
            return True
    return False


def is_temporary_exception(candidate: StateNode, existing: StateNode) -> bool:
    """Return whether candidate is a narrow exception to an existing state."""
    if candidate.entity != existing.entity or candidate.attribute != existing.attribute:
        return False
    candidate_scope_text = " ".join(
        part
        for part in [
            candidate.value,
            candidate.time_scope,
            candidate.condition_scope,
        ]
        if part
    )
    existing_scope_text = " ".join(
        part
        for part in [
            existing.value,
            existing.time_scope,
            existing.condition_scope,
        ]
        if part
    )
    candidate_is_narrow = is_narrow_time_scope(
        candidate.time_scope,
        candidate.condition_scope,
    ) or any(normalize_text(signal) in normalize_text(candidate_scope_text) for signal in TEMPORARY_SIGNALS)
    existing_is_narrow = is_narrow_time_scope(
        existing.time_scope,
        existing.condition_scope,
    ) or any(normalize_text(signal) in normalize_text(existing_scope_text) for signal in TEMPORARY_SIGNALS)
    if not candidate_is_narrow or existing_is_narrow:
        return False
    return candidate.value != existing.value and (
        are_opposite_values(candidate.value, existing.value)
        or has_negation(candidate.value)
        or token_overlap(candidate.value, existing.value) > 0.0
    )


def is_implicit_invalidation(candidate: StateNode, existing: StateNode) -> bool:
    """Return whether candidate makes an existing state/action no longer valid."""
    if not compatible_time_scope(candidate.time_scope, existing.time_scope):
        return False

    candidate_tokens = tokenize(
        " ".join(
            part
            for part in [
                candidate.entity,
                candidate.attribute,
                candidate.value,
                candidate.condition_scope,
            ]
            if part
        )
    )
    existing_tokens = tokenize(
        " ".join(
            part
            for part in [
                existing.entity,
                existing.attribute,
                existing.value,
                existing.condition_scope,
            ]
            if part
        )
    )

    existing_is_action_or_plan = bool(existing_tokens & PLAN_TOKENS) or (
        existing.attribute in {"feasibility", "status"} and bool(existing_tokens & DEPENDENCY_VALUE_TOKENS)
    )
    candidate_blocks_availability = (
        candidate.attribute == "availability"
        and bool(candidate_tokens & {"unavailable", "busy"})
    )
    if existing_is_action_or_plan and candidate_blocks_availability:
        return True

    candidate_rejects_preference = (
        candidate.attribute in {"preference", "food", "cuisine"}
        and has_negation(candidate.value)
    )
    if existing_is_action_or_plan and candidate_rejects_preference:
        overlap = token_overlap(candidate.value, existing.value) + token_overlap(
            candidate.value,
            existing.condition_scope,
        )
        if overlap > 0.0 or bool(existing_tokens & {"restaurant", "italian", "food"}):
            return True

    existing_is_local_plan = "local" in existing_tokens or "plan" in existing_tokens
    location_changed = (
        candidate.attribute == "location"
        and existing.attribute == "location"
        and candidate.value != existing.value
    )
    if location_changed and existing_is_local_plan:
        return True

    candidate_marks_rework = bool(candidate_tokens & {"bug", "buggy", "failed", "review"})
    existing_completed = bool(existing_tokens & {"completed", "complete", "done", "valid"})
    if existing_is_action_or_plan and candidate_marks_rework and existing_completed:
        return True

    return False


class ConflictDetector:
    """Deterministic rule-based relation classifier for linked states."""

    def __init__(
        self,
        strict_time_matching: bool = False,
        use_implicit_rules: bool = True,
    ) -> None:
        """Configure time matching and implicit invalidation rules."""
        self.strict_time_matching = strict_time_matching
        self.use_implicit_rules = use_implicit_rules

    def detect_conflict(
        self,
        candidate: StateNode,
        existing: StateNode,
        link: StateLink | None = None,
    ) -> ConflictDecision:
        """Classify one candidate-existing relation."""
        same_entity = candidate.entity == existing.entity
        same_attribute = candidate.attribute == existing.attribute
        same_value = same_text(candidate.value, existing.value)
        compatible_time = compatible_time_scope(
            candidate.time_scope,
            existing.time_scope,
            strict=self.strict_time_matching,
        )
        compatible_condition = self._compatible_condition_scope(
            candidate.condition_scope,
            existing.condition_scope,
        )

        if same_entity and same_attribute and same_value and compatible_time and compatible_condition:
            return ConflictDecision(
                label="duplicate",
                reason="same entity, attribute, value, time, and condition",
                confidence=DUPLICATE_CONFIDENCE,
            )

        if is_temporary_exception(candidate, existing):
            return ConflictDecision(
                label="temporary_exception",
                reason="candidate is a narrow temporary exception to an existing state",
                confidence=TEMPORARY_EXCEPTION_CONFIDENCE,
            )

        if (
            same_entity
            and same_attribute
            and compatible_time
            and are_opposite_values(candidate.value, existing.value)
        ):
            confidence = EXPLICIT_CONFLICT_CONFIDENCE
            if link is not None and link.match_type == "entity_attribute_time":
                confidence = min(1.0, confidence + 0.03)
            return ConflictDecision(
                label="explicit_conflict",
                reason="same entity and attribute with incompatible values in compatible time scope",
                confidence=confidence,
            )

        if same_entity and same_attribute and not same_value:
            confidence = UPDATE_CONFIDENCE
            if link is not None and link.match_type == "entity_attribute_time":
                confidence = min(1.0, confidence + 0.03)
            return ConflictDecision(
                label="update",
                reason="same entity and attribute with a different non-duplicate value",
                confidence=confidence,
            )

        if self.use_implicit_rules and is_implicit_invalidation(candidate, existing):
            return ConflictDecision(
                label="implicit_invalidation",
                reason="candidate makes an existing state or action no longer valid",
                confidence=IMPLICIT_INVALIDATION_CONFIDENCE,
            )

        if self._can_coexist(candidate, existing, link):
            return ConflictDecision(
                label="consistent",
                reason="states can coexist under transparent rules",
                confidence=CONSISTENT_CONFIDENCE,
            )

        confidence = UNCERTAIN_CONFIDENCE
        if link is not None and link.score >= 0.7:
            confidence = 0.45
        return ConflictDecision(
            label="uncertain",
            reason="no deterministic conflict or consistency rule fired",
            confidence=confidence,
        )

    def detect_conflicts(
        self,
        candidate: StateNode,
        existing_states: list[StateNode],
        links: list[StateLink] | None = None,
    ) -> list[ConflictDecision]:
        """Classify candidate relations against multiple existing states."""
        link_by_state_id = {link.matched_state_id: link for link in links or []}
        return [
            self.detect_conflict(
                candidate=candidate,
                existing=existing,
                link=link_by_state_id.get(existing.state_id),
            )
            for existing in existing_states
        ]

    def detect(
        self,
        existing_state: StateNode | None,
        candidate_state: StateNode,
    ) -> ConflictDecision:
        """Compatibility wrapper for the previous detector interface."""
        if existing_state is None:
            return ConflictDecision(
                label="uncertain",
                reason="no existing state provided for comparison",
                confidence=UNCERTAIN_CONFIDENCE,
            )
        return self.detect_conflict(candidate=candidate_state, existing=existing_state)

    def link_candidates(
        self,
        existing_states: list[StateNode],
        candidate_states: list[StateNode],
    ) -> list[ConflictDecision]:
        """Compatibility wrapper returning decisions for candidate-state pairs."""
        decisions: list[ConflictDecision] = []
        for candidate in candidate_states:
            if not existing_states:
                decisions.append(
                    ConflictDecision(
                        label="uncertain",
                        reason="no existing states available for comparison",
                        confidence=UNCERTAIN_CONFIDENCE,
                    )
                )
                continue
            decisions.extend(self.detect_conflicts(candidate, existing_states))
        return decisions

    @staticmethod
    def _compatible_condition_scope(a: str | None, b: str | None) -> bool:
        """Return whether two condition scopes are equivalent for duplicates."""
        normalized_a = normalize_text(a)
        normalized_b = normalize_text(b)
        if not normalized_a and not normalized_b:
            return True
        return bool(normalized_a and normalized_a == normalized_b)

    def _can_coexist(
        self,
        candidate: StateNode,
        existing: StateNode,
        link: StateLink | None,
    ) -> bool:
        """Return whether states are transparently consistent."""
        if candidate.entity == existing.entity and candidate.attribute != existing.attribute:
            return True
        if same_text(candidate.value, existing.value) and compatible_time_scope(
            candidate.time_scope,
            existing.time_scope,
            strict=self.strict_time_matching,
        ):
            return True
        if link is not None and link.match_type in {"same_entity", "same_attribute"}:
            return True
        return False


def detect_conflict(
    candidate: StateNode,
    existing: StateNode,
    link: StateLink | None = None,
) -> ConflictDecision:
    """Functional wrapper for ConflictDetector.detect_conflict."""
    return ConflictDetector().detect_conflict(candidate, existing, link)


def detect_conflicts(
    candidate: StateNode,
    existing_states: list[StateNode],
    links: list[StateLink] | None = None,
) -> list[ConflictDecision]:
    """Functional wrapper for ConflictDetector.detect_conflicts."""
    return ConflictDetector().detect_conflicts(candidate, existing_states, links)
