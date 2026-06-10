"""Invalidation propagation over typed dependency edges.

This module propagates already-established invalidations through graph
dependencies. It does not infer conflicts, create candidate states, add
revision edges, check premises, retrieve context, or generate answers.
"""

from __future__ import annotations

from collections import deque
from typing import get_args

from stategraph.core.graph_store import GraphStore
from stategraph.schemas import (
    EdgeType,
    PropagationReport,
    PropagationStep,
    StateEdge,
    StateNode,
    StateStatus,
)


DEFAULT_PROPAGATION_EDGE_TYPES = {"depends-on", "derived-from", "affects-action"}
TERMINAL_SKIP_STATUSES = {"historical"}
UNCERTAIN_NOTE = (
    "uncertain state left unchanged during invalidation propagation"
)


class InvalidationPropagator:
    """Propagate stale status through dependency-like state edges."""

    def __init__(
        self,
        propagation_edge_types: set[str] | None = None,
        stale_status: str = "stale",
        uncertain_status: str = "uncertain",
        max_depth: int | None = None,
        dry_run: bool = False,
    ) -> None:
        """Configure propagation policy.

        Edges are interpreted as ``dependent -> prerequisite``. If a
        prerequisite becomes invalid, propagation walks incoming edges and
        marks the dependent state stale when policy allows it.
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        edge_types = (
            DEFAULT_PROPAGATION_EDGE_TYPES
            if propagation_edge_types is None
            else propagation_edge_types
        )
        self.propagation_edge_types = set(edge_types)
        self.stale_status = stale_status
        self.uncertain_status = uncertain_status
        self.max_depth = max_depth
        self.dry_run = dry_run

        self._validate_edge_types(self.propagation_edge_types)
        self._validate_status(self.stale_status)
        self._validate_status(self.uncertain_status)

    def propagate(
        self,
        graph: GraphStore,
        seed_state_ids: list[str] | None = None,
        *,
        starting_state_ids: list[str] | None = None,
    ) -> PropagationReport:
        """Propagate invalidation from seed states to dependent states.

        ``starting_state_ids`` is accepted as a compatibility keyword for the
        previous stub runner; new code should pass ``seed_state_ids``.
        """
        if seed_state_ids is None:
            if starting_state_ids is None:
                raise TypeError("seed_state_ids is required")
            seed_state_ids = starting_state_ids
        elif starting_state_ids is not None:
            raise ValueError("Pass either seed_state_ids or starting_state_ids, not both")

        seeds = sorted(set(seed_state_ids))
        for state_id in seeds:
            graph.get_state(state_id)

        report = PropagationReport(seed_state_ids=seeds, dry_run=self.dry_run)
        if not seeds:
            report.notes.append("no seed states provided")
            return report

        queue: deque[tuple[str, int]] = deque((state_id, 0) for state_id in seeds)
        visited = set(seeds)

        while queue:
            current_state_id, depth = queue.popleft()
            incoming_edges = self._incoming_propagation_edges(graph, current_state_id)
            if self.max_depth is not None and depth >= self.max_depth:
                if incoming_edges:
                    report.max_depth_reached = True
                continue

            for edge in incoming_edges:
                affected_state_id = edge.source
                next_depth = depth + 1
                if self.max_depth is not None and next_depth > self.max_depth:
                    report.max_depth_reached = True
                    continue

                if affected_state_id in visited:
                    report.cycle_detected = True
                    report.notes.append(
                        "cycle detected while following "
                        f"{edge.edge_type} edge {affected_state_id} -> "
                        f"{current_state_id}"
                    )
                    continue

                visited.add(affected_state_id)
                step, action = self._apply_propagation_step(
                    graph=graph,
                    invalidated_state_id=current_state_id,
                    affected_state_id=affected_state_id,
                    edge=edge,
                    depth=next_depth,
                )
                report.propagation_steps.append(step)
                if action == "propagated":
                    report.propagated_state_ids.append(affected_state_id)
                    queue.append((affected_state_id, next_depth))
                elif action == "unchanged_continue":
                    report.unchanged_state_ids.append(affected_state_id)
                    queue.append((affected_state_id, next_depth))
                elif action == "unchanged_stop":
                    report.unchanged_state_ids.append(affected_state_id)
                    if step.old_status == "uncertain":
                        report.notes.append(
                            f"{UNCERTAIN_NOTE}: {affected_state_id}"
                        )
                elif action == "skipped":
                    report.skipped_state_ids.append(affected_state_id)
                else:
                    raise ValueError(f"Unknown propagation action: {action}")

        return self._finalize_report(report)

    def _apply_propagation_step(
        self,
        graph: GraphStore,
        invalidated_state_id: str,
        affected_state_id: str,
        edge: StateEdge,
        depth: int,
    ) -> tuple[PropagationStep, str]:
        """Apply or simulate one propagation step."""
        affected = graph.get_state(affected_state_id)
        self._validate_status(affected.status)

        old_status = affected.status
        new_status = self._new_status_for(affected)
        reason = self._step_reason(
            invalidated_state_id=invalidated_state_id,
            affected_state_id=affected_state_id,
            edge=edge,
            old_status=old_status,
            new_status=new_status,
        )
        step = PropagationStep(
            source_state_id=invalidated_state_id,
            target_state_id=affected_state_id,
            edge_type=edge.edge_type,
            old_status=old_status,
            new_status=new_status,
            depth=depth,
            reason=reason,
        )

        if old_status == "current":
            if not self.dry_run:
                graph.update_state_status(affected_state_id, new_status)
            return step, "propagated"

        if old_status == "stale":
            return step, "unchanged_continue"

        if old_status == "uncertain":
            return step, "unchanged_stop"

        if old_status in TERMINAL_SKIP_STATUSES:
            return step, "skipped"

        raise ValueError(f"Unsupported state status: {old_status}")

    def _incoming_propagation_edges(
        self,
        graph: GraphStore,
        state_id: str,
    ) -> list[StateEdge]:
        """Return deterministic incoming edges that can propagate invalidation."""
        return [
            edge
            for edge in graph.incoming_edges(state_id)
            if edge.edge_type in self.propagation_edge_types
        ]

    def _new_status_for(self, state: StateNode) -> str:
        """Return the status that propagation would assign to a state."""
        if state.status == "current":
            return self.stale_status
        if state.status == "uncertain":
            return state.status
        return state.status

    @staticmethod
    def _step_reason(
        invalidated_state_id: str,
        affected_state_id: str,
        edge: StateEdge,
        old_status: str,
        new_status: str,
    ) -> str:
        """Build a concise explanation for one propagation step."""
        if old_status == "current" and new_status == "stale":
            return (
                f"{affected_state_id} depends on invalidated state "
                f"{invalidated_state_id} via {edge.edge_type}"
            )
        if old_status == "stale":
            return (
                f"{affected_state_id} already stale; continuing propagation "
                f"through {edge.edge_type}"
            )
        if old_status == "uncertain":
            return (
                f"{affected_state_id} is uncertain; leaving unchanged during "
                f"{edge.edge_type} propagation"
            )
        if old_status == "historical":
            return (
                f"{affected_state_id} is historical; skipping propagation "
                f"through {edge.edge_type}"
            )
        return (
            f"{affected_state_id} reached from {invalidated_state_id} "
            f"through {edge.edge_type}"
        )

    @staticmethod
    def _validate_edge_types(edge_types: set[str]) -> None:
        """Validate configured propagation edge types."""
        allowed_edge_types = set(get_args(EdgeType))
        invalid = sorted(edge_types - allowed_edge_types)
        if invalid:
            raise ValueError(f"Unsupported propagation edge type(s): {invalid}")

    @staticmethod
    def _validate_status(status: str) -> None:
        """Validate a state status literal."""
        if status not in get_args(StateStatus):
            raise ValueError(f"Unsupported state status: {status}")

    @staticmethod
    def _finalize_report(report: PropagationReport) -> PropagationReport:
        """Return a report with deterministic unique lists."""
        update = {
            "seed_state_ids": sorted(set(report.seed_state_ids)),
            "propagated_state_ids": sorted(set(report.propagated_state_ids)),
            "unchanged_state_ids": sorted(set(report.unchanged_state_ids)),
            "skipped_state_ids": sorted(set(report.skipped_state_ids)),
            "propagation_steps": sorted(
                report.propagation_steps,
                key=InvalidationPropagator._step_sort_key,
            ),
            "notes": list(dict.fromkeys(report.notes)),
        }
        return report.model_copy(update=update)

    @staticmethod
    def _step_sort_key(step: PropagationStep) -> tuple[int, str, str, str, str]:
        """Return a deterministic sort key for propagation steps."""
        return (
            step.depth,
            step.source_state_id,
            step.target_state_id,
            step.edge_type,
            step.reason,
        )


def propagate_invalidation(
    graph: GraphStore,
    seed_state_ids: list[str],
    propagation_edge_types: set[str] | None = None,
    max_depth: int | None = None,
    dry_run: bool = False,
) -> PropagationReport:
    """Functional wrapper for InvalidationPropagator.propagate."""
    return InvalidationPropagator(
        propagation_edge_types=propagation_edge_types,
        max_depth=max_depth,
        dry_run=dry_run,
    ).propagate(graph, seed_state_ids)


InvalidationPropagationEngine = InvalidationPropagator
propagate = propagate_invalidation
