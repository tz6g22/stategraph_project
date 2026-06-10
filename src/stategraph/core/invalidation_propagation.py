"""Invalidation propagation stubs."""

from __future__ import annotations

from .graph_store import InMemoryStateGraph


class InvalidationPropagator:
    """Propagate invalidation across typed state edges."""

    def propagate(
        self,
        graph: InMemoryStateGraph,
        starting_state_ids: list[str],
    ) -> set[str]:
        """Return additional invalidated states.

        TODO: Implement propagation behavior for depends-on, derived-from, and
        affects-action edges. The current skeleton performs no propagation.
        """
        _ = graph
        _ = starting_state_ids
        return set()

