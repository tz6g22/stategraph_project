"""Minimal replaceable graph store and JSONL parsing helpers.

The graph store is intentionally simple. It provides enough structure for
import tests and future pipeline wiring without implementing full StateGraph
revision or propagation algorithms.
"""

from __future__ import annotations

from typing import Any

from stategraph.schemas import DatasetExample, EvidenceNode, StateEdge, StateNode


class InMemoryStateGraph:
    """Minimal in-memory state graph abstraction.

    TODO: Replace or adapt this store when the project needs NetworkX, a graph
    database, or another persistence layer.
    """

    def __init__(self) -> None:
        """Create empty state, evidence, and edge collections."""
        self.states: dict[str, StateNode] = {}
        self.evidence: dict[str, EvidenceNode] = {}
        self.edges: list[StateEdge] = []

    def add_state(self, state: StateNode) -> None:
        """Insert or replace a state node by id."""
        self.states[state.state_id] = state

    def add_evidence(self, evidence: EvidenceNode) -> None:
        """Insert or replace an evidence node by id."""
        self.evidence[evidence.evidence_id] = evidence

    def add_edge(self, edge: StateEdge) -> None:
        """Append a typed state edge."""
        self.edges.append(edge)

    def get_state(self, state_id: str) -> StateNode | None:
        """Return a state node by id, if present."""
        return self.states.get(state_id)

    def get_current_states(self) -> list[StateNode]:
        """Return states currently marked as current."""
        return [state for state in self.states.values() if state.status == "current"]

    def edges_from(self, state_id: str) -> list[StateEdge]:
        """Return outgoing edges for a state id."""
        return [edge for edge in self.edges if edge.source == state_id]

    def edges_to(self, state_id: str) -> list[StateEdge]:
        """Return incoming edges for a state id."""
        return [edge for edge in self.edges if edge.target == state_id]


def evidence_from_dict(payload: dict[str, Any]) -> EvidenceNode:
    """Create an evidence node from a JSON-like dictionary."""
    return EvidenceNode(
        evidence_id=str(payload.get("evidence_id") or payload.get("id") or ""),
        text=str(payload.get("text") or payload.get("content") or ""),
        source=payload.get("source"),
        timestamp=payload.get("timestamp"),
    )


def state_from_dict(payload: dict[str, Any]) -> StateNode:
    """Create a state node from a JSON-like dictionary."""
    return StateNode(
        state_id=str(payload.get("state_id", "")),
        entity=str(payload.get("entity", "")),
        attribute=str(payload.get("attribute", "")),
        value=str(payload.get("value", "")),
        time_scope=payload.get("time_scope"),
        condition_scope=payload.get("condition_scope"),
        status=payload.get("status", "uncertain"),
        evidence_id=payload.get("evidence_id"),
        confidence=float(payload.get("confidence", 0.0)),
    )


def dataset_example_from_dict(payload: dict[str, Any]) -> DatasetExample:
    """Create a dataset example from a JSON-like dictionary.

    TODO: Expand this parser once dataset-specific adapters are implemented.
    """
    case_id = str(payload.get("case_id", ""))
    history: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("history", [])):
        if isinstance(item, dict):
            history.append(dict(item))
        else:
            history.append(
                {
                    "evidence_id": f"{case_id or 'case'}_history_{index}",
                    "text": str(item),
                    "source": "history",
                }
            )

    new_observation = payload.get("new_observation")
    if isinstance(new_observation, str):
        new_observation_payload: dict[str, Any] | None = {
            "text": new_observation,
            "source": "new_observation",
        }
    elif isinstance(new_observation, dict):
        new_observation_payload = dict(new_observation)
    else:
        new_observation_payload = None

    gold_current_states: list[str] = []
    for item in payload.get("gold_current_states", []):
        if isinstance(item, dict):
            gold_current_states.append(str(item.get("state_id", "")))
        else:
            gold_current_states.append(str(item))

    return DatasetExample(
        case_id=case_id,
        history=history,
        new_observation=new_observation_payload,
        query=str(payload.get("query", "")),
        gold_current_states=gold_current_states,
        gold_invalidated_states=[
            str(item) for item in payload.get("gold_invalidated_states", [])
        ],
        gold_keep_states=[str(item) for item in payload.get("gold_keep_states", [])],
        gold_answer=payload.get("gold_answer"),
        expected_behavior=payload.get("expected_behavior"),
    )
