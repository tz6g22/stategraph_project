"""Candidate state extraction stubs."""

from __future__ import annotations

from .schemas import EvidenceNode, StateNode


class CandidateStateExtractor:
    """Extract candidate state nodes from new observations.

    TODO: Add an LLM-free heuristic extractor or an optional model-backed
    extractor in a later stage. The skeleton must not call external APIs.
    """

    def extract(
        self,
        observation: str,
        evidence: EvidenceNode | None = None,
    ) -> list[StateNode]:
        """Return candidate state nodes for an observation.

        The placeholder returns no states because extraction logic is out of
        scope for the initial framework.
        """
        _ = observation
        _ = evidence
        return []
