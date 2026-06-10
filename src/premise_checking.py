"""Query premise checking stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .graph_store import InMemoryStateGraph


PremiseStatus = Literal["valid", "stale", "uncertain"]


@dataclass(slots=True)
class PremiseCheckResult:
    """Result of checking whether a query premise is valid."""

    status: PremiseStatus
    reason: str | None = None
    related_state_ids: list[str] | None = None


class PremiseChecker:
    """Check whether a query relies on stale or invalidated state."""

    def check(self, query: str, graph: InMemoryStateGraph) -> PremiseCheckResult:
        """Return an uncertain placeholder premise result."""
        _ = query
        _ = graph
        return PremiseCheckResult(
            status="uncertain",
            reason="stub: premise checking is not implemented",
            related_state_ids=[],
        )

