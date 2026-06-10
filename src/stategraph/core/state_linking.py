"""Rule-based state linking.

This module identifies existing states that may be relevant to a candidate
state. It does not classify conflicts, revise state status, or propagate
invalidation.
"""

from __future__ import annotations

import re
from typing import get_args

from stategraph.core.graph_store import GraphStore
from stategraph.schemas import StateLink, StateNode, StateStatus


ENTITY_ATTRIBUTE_TIME_SCORE = 0.95
SAME_ENTITY_ATTRIBUTE_SCORE = 0.85
SAME_ENTITY_SCORE = 0.55
SAME_ATTRIBUTE_SCORE = 0.45
SAME_TIME_SCOPE_BONUS = 0.15
SHARED_EVIDENCE_BONUS = 0.10
GRAPH_NEIGHBOR_BONUS = 0.10
WEAK_TEXT_OVERLAP_BONUS = 0.05

DEFAULT_INCLUDE_STATUSES = {"current", "uncertain"}
DEPENDENCY_EDGE_TYPES = {
    "depends-on",
    "derived-from",
    "affects-action",
    "updates",
    "invalidates",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic rule matching."""
    if text is None:
        return ""
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def tokenize(text: str | None) -> set[str]:
    """Return normalized tokens from text."""
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def token_overlap(a: str | None, b: str | None) -> float:
    """Return Jaccard overlap between two token sets."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def same_normalized_text(a: str | None, b: str | None) -> bool:
    """Return whether two strings normalize to the same non-empty text."""
    normalized_a = normalize_text(a)
    normalized_b = normalize_text(b)
    return bool(normalized_a and normalized_a == normalized_b)


def time_scope_similarity(a: str | None, b: str | None) -> float:
    """Return a lightweight similarity score for time scopes."""
    if same_normalized_text(a, b):
        return 1.0
    return token_overlap(a, b)


def state_text_overlap(candidate: StateNode, existing: StateNode) -> float:
    """Return token overlap across value, time, and condition fields."""
    candidate_text = " ".join(
        part
        for part in [
            candidate.value,
            candidate.time_scope,
            candidate.condition_scope,
        ]
        if part
    )
    existing_text = " ".join(
        part
        for part in [
            existing.value,
            existing.time_scope,
            existing.condition_scope,
        ]
        if part
    )
    return token_overlap(candidate_text, existing_text)


def is_allowed_status(state: StateNode, include_statuses: set[str]) -> bool:
    """Return whether a state status is included by the linker."""
    return state.status in include_statuses


class StateLinker:
    """Deterministic rule-based linker for candidate states."""

    def __init__(
        self,
        include_statuses: set[str] | None = None,
        max_links: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Configure status filtering, link limit, and score threshold."""
        self.include_statuses = set(include_statuses or DEFAULT_INCLUDE_STATUSES)
        self.max_links = max_links
        self.min_score = min_score
        self._validate_include_statuses(self.include_statuses)
        if max_links is not None and max_links < 0:
            raise ValueError("max_links must be non-negative or None")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")

    def find_linked_states(
        self,
        candidate: StateNode,
        graph: GraphStore,
    ) -> list[StateLink]:
        """Find existing graph states related to a candidate state."""
        links: list[StateLink] = []
        for existing in graph.list_states():
            if existing.state_id == candidate.state_id:
                continue
            if not is_allowed_status(existing, self.include_statuses):
                continue
            link = self.score_pair(candidate, existing, graph=graph)
            if link is None:
                continue
            if link.score < self.min_score:
                continue
            links.append(link)

        links.sort(key=lambda link: (-link.score, link.matched_state_id))
        if self.max_links is not None:
            return links[: self.max_links]
        return links

    def score_pair(
        self,
        candidate: StateNode,
        existing: StateNode,
        graph: GraphStore | None = None,
    ) -> StateLink | None:
        """Score one candidate-existing state pair."""
        same_entity = candidate.entity == existing.entity
        same_attribute = candidate.attribute == existing.attribute
        time_similarity = time_scope_similarity(
            candidate.time_scope,
            existing.time_scope,
        )
        shared_evidence = (
            candidate.evidence_id is not None
            and candidate.evidence_id == existing.evidence_id
        )
        graph_neighbor = self._has_graph_neighbor_signal(candidate, existing, graph)
        text_overlap = state_text_overlap(candidate, existing)

        base_score = 0.0
        match_type = ""
        reason = ""

        if same_entity and same_attribute and time_similarity > 0.0:
            base_score = ENTITY_ATTRIBUTE_TIME_SCORE
            match_type = "entity_attribute_time"
            reason = "same entity, attribute, and overlapping time scope"
        elif same_entity and same_attribute:
            base_score = SAME_ENTITY_ATTRIBUTE_SCORE
            match_type = "same_entity_attribute"
            reason = "same entity and attribute"
        elif same_entity:
            base_score = SAME_ENTITY_SCORE
            match_type = "same_entity"
            reason = "same entity"
        elif same_attribute:
            base_score = SAME_ATTRIBUTE_SCORE
            match_type = "same_attribute"
            reason = "same attribute"
        elif time_similarity > 0.0:
            base_score = SAME_TIME_SCOPE_BONUS * time_similarity
            match_type = "same_time_scope"
            reason = "same or overlapping time scope"
        elif shared_evidence:
            base_score = SHARED_EVIDENCE_BONUS
            match_type = "evidence_neighbor"
            reason = "shared evidence_id"
        elif graph_neighbor:
            base_score = GRAPH_NEIGHBOR_BONUS
            match_type = "dependency_neighbor"
            reason = "dependency-related graph neighbor"
        elif text_overlap > 0.0:
            base_score = WEAK_TEXT_OVERLAP_BONUS * text_overlap
            match_type = "weak_text_overlap"
            reason = "weak text overlap"

        if base_score <= 0.0:
            return None

        score = base_score
        if time_similarity > 0.0 and match_type not in {
            "entity_attribute_time",
            "same_time_scope",
        }:
            score += SAME_TIME_SCOPE_BONUS * time_similarity
        if shared_evidence and match_type != "evidence_neighbor":
            score += SHARED_EVIDENCE_BONUS
        if graph_neighbor and match_type != "dependency_neighbor":
            score += GRAPH_NEIGHBOR_BONUS
        if text_overlap > 0.0 and match_type != "weak_text_overlap":
            score += WEAK_TEXT_OVERLAP_BONUS * text_overlap

        score = min(1.0, round(score, 6))
        features: dict[str, bool | float | str | None] = {
            "same_entity": same_entity,
            "same_attribute": same_attribute,
            "time_scope_similarity": round(time_similarity, 6),
            "shared_evidence": shared_evidence,
            "graph_neighbor": graph_neighbor,
            "text_overlap": round(text_overlap, 6),
            "candidate_status": candidate.status,
            "existing_status": existing.status,
        }

        return StateLink(
            candidate_state_id=candidate.state_id,
            matched_state_id=existing.state_id,
            match_type=match_type,
            score=score,
            reason=reason,
            features=features,
        )

    def link(
        self,
        existing_states: list[StateNode],
        candidate_states: list[StateNode],
    ) -> list[tuple[str | None, str]]:
        """Compatibility wrapper for the initial placeholder interface."""
        _ = existing_states
        return [(None, candidate.state_id) for candidate in candidate_states]

    @staticmethod
    def _validate_include_statuses(include_statuses: set[str]) -> None:
        """Validate linker status filters."""
        allowed_statuses = set(get_args(StateStatus))
        unsupported = include_statuses - allowed_statuses
        if unsupported:
            joined = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported include_statuses: {joined}")

    def _has_graph_neighbor_signal(
        self,
        candidate: StateNode,
        existing: StateNode,
        graph: GraphStore | None,
    ) -> bool:
        """Return whether graph topology offers a weak linking signal."""
        if graph is None:
            return False

        if graph.has_state(candidate.state_id) and graph.has_state(existing.state_id):
            if graph.has_edge(candidate.state_id, existing.state_id) or graph.has_edge(
                existing.state_id,
                candidate.state_id,
            ):
                return True

        if not graph.has_state(existing.state_id):
            return False

        neighbor_edges = [
            *graph.outgoing_edges(existing.state_id),
            *graph.incoming_edges(existing.state_id),
        ]
        for edge in neighbor_edges:
            if edge.edge_type not in DEPENDENCY_EDGE_TYPES:
                continue
            neighbor_id = edge.target if edge.source == existing.state_id else edge.source
            if not graph.has_state(neighbor_id):
                continue
            neighbor = graph.get_state(neighbor_id)
            if (
                candidate.entity == neighbor.entity
                or candidate.attribute == neighbor.attribute
                or (
                    candidate.evidence_id is not None
                    and candidate.evidence_id == neighbor.evidence_id
                )
            ):
                return True
        return False


def find_linked_states(
    candidate: StateNode,
    graph: GraphStore,
    include_statuses: set[str] | None = None,
    max_links: int | None = None,
    min_score: float = 0.0,
) -> list[StateLink]:
    """Functional wrapper for StateLinker.find_linked_states."""
    linker = StateLinker(
        include_statuses=include_statuses,
        max_links=max_links,
        min_score=min_score,
    )
    return linker.find_linked_states(candidate, graph)
