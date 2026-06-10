"""Retrieval stubs for current states and supporting evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph_store import InMemoryStateGraph
from .schemas import EvidenceNode, StateNode


@dataclass(slots=True)
class RetrievalResult:
    """Retrieved state and evidence context for answer generation."""

    states: list[StateNode] = field(default_factory=list)
    evidence: list[EvidenceNode] = field(default_factory=list)


class StateRetriever:
    """Retrieve current states and their supporting evidence."""

    def retrieve(
        self,
        query: str,
        graph: InMemoryStateGraph,
        limit: int = 10,
    ) -> RetrievalResult:
        """Return current states with any directly linked evidence."""
        _ = query
        states = graph.get_current_states()[:limit]
        evidence_ids = {state.evidence_id for state in states if state.evidence_id}
        evidence = [
            graph.evidence[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in graph.evidence
        ]
        return RetrievalResult(states=states, evidence=evidence)
