"""State status revision stubs."""

from __future__ import annotations

from .graph_store import InMemoryStateGraph
from stategraph.schemas import ConflictDecision, StateNode


class StateRevisionEngine:
    """Apply conflict results to graph state statuses.

    TODO: Implement rules for current, stale, historical, and uncertain states.
    """

    def revise(
        self,
        graph: InMemoryStateGraph,
        candidate_states: list[StateNode],
        conflict_results: list[ConflictDecision],
    ) -> list[str]:
        """Insert candidates and return state ids directly invalidated.

        Placeholder logic stores candidate states without changing existing
        statuses and returns an empty invalidation list.
        """
        _ = conflict_results
        for candidate in candidate_states:
            graph.add_state(candidate)
        return []
