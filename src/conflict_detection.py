"""Conflict and update relation detection stubs."""

from __future__ import annotations

from .schemas import ConflictDecision, StateNode


class ConflictDetector:
    """Detect conflicts, duplicates, and update relations between states."""

    def detect(
        self,
        existing_state: StateNode | None,
        candidate_state: StateNode,
    ) -> ConflictDecision:
        """Return a placeholder conflict label for a candidate state.

        TODO: Implement entity/attribute matching and relation classification.
        """
        _ = existing_state
        _ = candidate_state
        return ConflictDecision(
            label="uncertain",
            reason="stub: conflict detection is not implemented",
            confidence=0.0,
        )

    def link_candidates(
        self,
        existing_states: list[StateNode],
        candidate_states: list[StateNode],
    ) -> list[ConflictDecision]:
        """Return placeholder links between candidates and existing states."""
        _ = existing_states
        return [
            self.detect(existing_state=None, candidate_state=candidate)
            for candidate in candidate_states
        ]
