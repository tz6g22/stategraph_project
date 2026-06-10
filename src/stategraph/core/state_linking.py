"""State linking stub.

This module will eventually connect candidate states to existing graph states.
It intentionally contains no linking algorithm yet.
"""

from __future__ import annotations

from stategraph.schemas import StateNode


class StateLinker:
    """Placeholder linker for candidate and existing states."""

    def link(
        self,
        existing_states: list[StateNode],
        candidate_states: list[StateNode],
    ) -> list[tuple[str | None, str]]:
        """Return placeholder links for candidates.

        TODO: Implement entity, attribute, temporal, and conditional linking.
        """
        _ = existing_states
        return [(None, candidate.state_id) for candidate in candidate_states]

