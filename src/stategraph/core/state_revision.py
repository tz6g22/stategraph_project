"""State revision policies.

This module applies already-computed conflict decisions to a GraphStore. It
does not detect conflicts, propagate invalidations, check premises, retrieve
context, or generate answers.
"""

from __future__ import annotations

from stategraph.core.graph_store import GraphStore
from stategraph.schemas import (
    ConflictDecision,
    RevisionItem,
    RevisionReport,
    StateEdge,
    StateNode,
)


MUTATING_CURRENT_LABELS = {
    "consistent",
    "update",
    "explicit_conflict",
    "implicit_invalidation",
    "temporary_exception",
}
KNOWN_LABELS = {
    "consistent",
    "duplicate",
    "update",
    "explicit_conflict",
    "implicit_invalidation",
    "temporary_exception",
    "uncertain",
}


class StateReviser:
    """Apply revision decisions to a GraphStore."""

    def __init__(
        self,
        add_duplicate_candidates: bool = False,
        mark_updates_as_historical: bool = True,
        mark_conflicts_as_stale: bool = True,
        add_support_edges_for_consistent: bool = False,
    ) -> None:
        """Configure deterministic revision policies."""
        self.add_duplicate_candidates = add_duplicate_candidates
        self.mark_updates_as_historical = mark_updates_as_historical
        self.mark_conflicts_as_stale = mark_conflicts_as_stale
        self.add_support_edges_for_consistent = add_support_edges_for_consistent

    def revise(
        self,
        graph: GraphStore,
        candidate: StateNode,
        revision_items: list[RevisionItem],
    ) -> RevisionReport:
        """Apply revision items for a candidate state."""
        for item in revision_items:
            if item.decision.label not in KNOWN_LABELS:
                raise ValueError(f"Unknown decision label: {item.decision.label}")
            graph.get_state(item.existing_state_id)

        report = RevisionReport(
            candidate_state_id=candidate.state_id,
            candidate_added=False,
            candidate_final_status=None,
            decisions=revision_items,
        )
        final_status = self._candidate_final_status(revision_items)

        if not revision_items:
            final_status = "current"
            self._ensure_candidate(graph, candidate, final_status, report)
            report.notes.append("no linked states; candidate added as current")
            return self._dedupe_report(report)

        all_duplicates = all(
            item.decision.label == "duplicate" for item in revision_items
        )
        if all_duplicates and not self.add_duplicate_candidates:
            report.notes.append("candidate skipped because all decisions are duplicate")
        else:
            self._ensure_candidate(graph, candidate, final_status, report)

        for item in revision_items:
            created_edges, skipped_edges = self._apply_decision(graph, candidate, item)
            report.created_edges.extend(created_edges)
            report.skipped_edges.extend(skipped_edges)
            self._record_item(report, item)

        return self._dedupe_report(report)

    def apply_decision(
        self,
        graph: GraphStore,
        candidate: StateNode,
        item: RevisionItem,
    ) -> list[StateEdge]:
        """Apply one revision item and return newly created edges."""
        created_edges, _ = self._apply_decision(graph, candidate, item)
        return created_edges

    def _apply_decision(
        self,
        graph: GraphStore,
        candidate: StateNode,
        item: RevisionItem,
    ) -> tuple[list[StateEdge], list[dict[str, str]]]:
        """Apply one revision item and return created and skipped edges."""
        existing = graph.get_state(item.existing_state_id)
        label = item.decision.label
        if label not in KNOWN_LABELS:
            raise ValueError(f"Unknown decision label: {label}")

        if label == "duplicate":
            if self.add_duplicate_candidates and graph.has_state(candidate.state_id):
                edge = self._make_edge(candidate, existing, "supports", item.decision)
                return self._add_edge_if_absent(graph, edge)
            return [], []

        if label == "consistent":
            if self.add_support_edges_for_consistent and graph.has_state(
                candidate.state_id
            ):
                edge = self._make_edge(candidate, existing, "supports", item.decision)
                return self._add_edge_if_absent(graph, edge)
            return [], []

        if label == "update":
            status = "historical" if self.mark_updates_as_historical else "stale"
            graph.update_state_status(existing.state_id, status)
            edge = self._make_edge(candidate, existing, "updates", item.decision)
            return self._add_edge_if_absent(graph, edge)

        if label == "explicit_conflict":
            if self.mark_conflicts_as_stale:
                graph.update_state_status(existing.state_id, "stale")
            edge = self._make_edge(candidate, existing, "invalidates", item.decision)
            return self._add_edge_if_absent(graph, edge)

        if label == "implicit_invalidation":
            graph.update_state_status(existing.state_id, "stale")
            edge = self._make_edge(candidate, existing, "invalidates", item.decision)
            return self._add_edge_if_absent(graph, edge)

        if label == "temporary_exception":
            edge = self._make_edge(candidate, existing, "affects-action", item.decision)
            return self._add_edge_if_absent(graph, edge)

        if label == "uncertain":
            return [], []

        raise ValueError(f"Unknown decision label: {label}")

    def revise_from_decisions(
        self,
        graph: GraphStore,
        candidate: StateNode,
        decisions_by_state: dict[str, ConflictDecision],
    ) -> RevisionReport:
        """Convert a decision mapping into RevisionItem objects and revise."""
        items = [
            RevisionItem(existing_state_id=state_id, decision=decision)
            for state_id, decision in sorted(decisions_by_state.items())
        ]
        return self.revise(graph, candidate, items)

    def _ensure_candidate(
        self,
        graph: GraphStore,
        candidate: StateNode,
        status: str,
        report: RevisionReport,
    ) -> None:
        """Add candidate once, or record an existing-candidate note."""
        if graph.has_state(candidate.state_id):
            report.candidate_added = False
            report.candidate_final_status = graph.get_state(candidate.state_id).status
            report.notes.append("candidate already exists in graph; not added again")
            return

        revised_candidate = self._copy_state_with_status(candidate, status)
        graph.add_state(revised_candidate)
        report.candidate_added = True
        report.candidate_final_status = revised_candidate.status

    def _candidate_final_status(self, revision_items: list[RevisionItem]) -> str:
        """Determine candidate status from revision decisions."""
        labels = {item.decision.label for item in revision_items}
        if labels & MUTATING_CURRENT_LABELS:
            return "current"
        if labels == {"uncertain"}:
            return "uncertain"
        if labels == {"duplicate"} and self.add_duplicate_candidates:
            return "historical"
        if labels == {"duplicate"}:
            return "current"
        return "uncertain"

    @staticmethod
    def _copy_state_with_status(state: StateNode, status: str) -> StateNode:
        """Return a copy of a state with updated status."""
        if hasattr(state, "model_copy"):
            return state.model_copy(update={"status": status})
        payload = state.model_dump() if hasattr(state, "model_dump") else dict(state)
        payload["status"] = status
        return StateNode(**payload)

    @staticmethod
    def _make_edge(
        candidate: StateNode,
        existing: StateNode,
        edge_type: str,
        decision: ConflictDecision,
    ) -> StateEdge:
        """Build a typed revision edge."""
        return StateEdge(
            source=candidate.state_id,
            target=existing.state_id,
            edge_type=edge_type,  # type: ignore[arg-type]
            reason=decision.reason,
        )

    @staticmethod
    def _add_edge_if_absent(
        graph: GraphStore,
        edge: StateEdge,
    ) -> tuple[list[StateEdge], list[dict[str, str]]]:
        """Add an edge unless the same typed edge already exists."""
        if graph.has_edge(edge.source, edge.target, edge.edge_type):
            return [], [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "reason": "duplicate edge already exists",
                }
            ]
        graph.add_edge(edge)
        return [edge], []

    def _record_item(self, report: RevisionReport, item: RevisionItem) -> None:
        """Record report fields for one decision."""
        state_id = item.existing_state_id
        label = item.decision.label
        if label == "duplicate":
            report.duplicate_state_ids.append(state_id)
            if not self.add_duplicate_candidates:
                report.notes.append(
                    f"candidate skipped as duplicate of existing state {state_id}"
                )
        elif label == "consistent":
            report.unchanged_state_ids.append(state_id)
        elif label == "update":
            report.updated_state_ids.append(state_id)
            if self.mark_updates_as_historical:
                report.historical_state_ids.append(state_id)
            else:
                report.invalidated_state_ids.append(state_id)
        elif label == "explicit_conflict":
            if self.mark_conflicts_as_stale:
                report.invalidated_state_ids.append(state_id)
            else:
                report.unchanged_state_ids.append(state_id)
        elif label == "implicit_invalidation":
            report.invalidated_state_ids.append(state_id)
        elif label == "temporary_exception":
            report.unchanged_state_ids.append(state_id)
        elif label == "uncertain":
            report.uncertain_state_ids.append(state_id)
        else:
            raise ValueError(f"Unknown decision label: {label}")

    @staticmethod
    def _dedupe_report(report: RevisionReport) -> RevisionReport:
        """Return a report with deterministic unique id lists and skipped edges."""
        update = {
            "updated_state_ids": sorted(set(report.updated_state_ids)),
            "invalidated_state_ids": sorted(set(report.invalidated_state_ids)),
            "historical_state_ids": sorted(set(report.historical_state_ids)),
            "unchanged_state_ids": sorted(set(report.unchanged_state_ids)),
            "uncertain_state_ids": sorted(set(report.uncertain_state_ids)),
            "duplicate_state_ids": sorted(set(report.duplicate_state_ids)),
            "skipped_edges": StateReviser._dedupe_dicts(report.skipped_edges),
            "notes": list(dict.fromkeys(report.notes)),
        }
        return report.model_copy(update=update)

    @staticmethod
    def _dedupe_dicts(items: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return deterministic unique dictionaries."""
        by_key: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for item in items:
            key = tuple(sorted(item.items()))
            by_key.setdefault(key, item)
        return [by_key[key] for key in sorted(by_key)]


def revise_state(
    graph: GraphStore,
    candidate: StateNode,
    revision_items: list[RevisionItem],
) -> RevisionReport:
    """Functional wrapper for StateReviser.revise."""
    return StateReviser().revise(graph, candidate, revision_items)


def revise_from_decisions(
    graph: GraphStore,
    candidate: StateNode,
    decisions_by_state: dict[str, ConflictDecision],
) -> RevisionReport:
    """Functional wrapper for StateReviser.revise_from_decisions."""
    return StateReviser().revise_from_decisions(graph, candidate, decisions_by_state)


class StateRevisionEngine(StateReviser):
    """Backward-compatible revision engine name."""

    def revise(  # type: ignore[override]
        self,
        graph: GraphStore,
        candidate_states: list[StateNode] | StateNode,
        conflict_results: list[ConflictDecision] | list[RevisionItem],
    ) -> list[str] | RevisionReport:
        """Support both legacy batch insert and new revision semantics."""
        if isinstance(candidate_states, StateNode):
            return super().revise(
                graph,
                candidate_states,
                conflict_results,  # type: ignore[arg-type]
            )

        for candidate in candidate_states:
            if not graph.has_state(candidate.state_id):
                graph.add_state(candidate)
        _ = conflict_results
        return []
