"""Minimal replaceable graph store and JSONL parsing helpers.

The graph store is intentionally simple. It provides enough structure for
import tests and future pipeline wiring without implementing full StateGraph
revision or propagation algorithms.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, get_args

import networkx as nx

from stategraph.schemas import (
    DatasetExample,
    EdgeType,
    EvidenceNode,
    StateEdge,
    StateNode,
    StateStatus,
)


class GraphStore:
    """In-memory graph storage abstraction for StateGraph experiments.

    The schema dictionaries are the source of truth. The NetworkX graph mirrors
    state, evidence, and relation metadata for future graph operations.
    """

    def __init__(self) -> None:
        """Create empty state, evidence, and edge collections."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.states: dict[str, StateNode] = {}
        self.evidence: dict[str, EvidenceNode] = {}
        self.edges: list[StateEdge] = []
        self._entity_index: dict[str, set[str]] = {}
        self._attribute_index: dict[str, set[str]] = {}
        self._entity_attribute_index: dict[tuple[str, str], set[str]] = {}

    def add_state(self, state: StateNode) -> None:
        """Insert a state node by id."""
        if state.state_id in self.states:
            raise ValueError(f"Duplicate state_id: {state.state_id}")
        self.states[state.state_id] = state
        self._index_state(state)
        self._sync_state_node(state)
        if state.evidence_id in self.evidence:
            self._sync_evidence_state_link(state.evidence_id, state.state_id)

    def add_evidence(self, evidence: EvidenceNode) -> None:
        """Insert an evidence node by id."""
        if evidence.evidence_id in self.evidence:
            raise ValueError(f"Duplicate evidence_id: {evidence.evidence_id}")
        self.evidence[evidence.evidence_id] = evidence
        self._sync_evidence_node(evidence)
        for state in self.states.values():
            if state.evidence_id == evidence.evidence_id:
                self._sync_evidence_state_link(evidence.evidence_id, state.state_id)

    def add_edge(self, edge: StateEdge) -> None:
        """Append a typed state edge between existing states."""
        self._validate_edge_type(edge.edge_type)
        if edge.source not in self.states:
            raise KeyError(f"Unknown source state_id: {edge.source}")
        if edge.target not in self.states:
            raise KeyError(f"Unknown target state_id: {edge.target}")
        self.edges.append(edge)
        self._sync_state_edge(edge)

    def get_state(self, state_id: str) -> StateNode:
        """Return a state node by id."""
        if state_id not in self.states:
            raise KeyError(f"Unknown state_id: {state_id}")
        return self.states[state_id]

    def get_evidence(self, evidence_id: str) -> EvidenceNode:
        """Return an evidence node by id."""
        if evidence_id not in self.evidence:
            raise KeyError(f"Unknown evidence_id: {evidence_id}")
        return self.evidence[evidence_id]

    def has_state(self, state_id: str) -> bool:
        """Return whether a state id exists."""
        return state_id in self.states

    def has_evidence(self, evidence_id: str) -> bool:
        """Return whether an evidence id exists."""
        return evidence_id in self.evidence

    def list_evidence(self) -> list[EvidenceNode]:
        """List evidence nodes in deterministic order."""
        return [self.evidence[evidence_id] for evidence_id in sorted(self.evidence)]

    def update_state_status(self, state_id: str, status: str) -> None:
        """Update a state status after validating it."""
        self._validate_status(status)
        state = self.get_state(state_id)
        payload = self._model_to_dict(state)
        payload["status"] = status
        updated = self._coerce_model(StateNode, payload)
        self.states[state_id] = updated
        self._sync_state_node(updated)

    def list_states(
        self,
        status: str | None = None,
        entity: str | None = None,
        attribute: str | None = None,
        time_scope: str | None = None,
        evidence_id: str | None = None,
    ) -> list[StateNode]:
        """List states, optionally filtered by status."""
        if status is not None:
            self._validate_status(status)
        states = self._sorted_states(self.states.values())
        if status is not None:
            states = [state for state in states if state.status == status]
        if entity is not None:
            states = [state for state in states if state.entity == entity]
        if attribute is not None:
            states = [state for state in states if state.attribute == attribute]
        if time_scope is not None:
            states = [state for state in states if state.time_scope == time_scope]
        if evidence_id is not None:
            states = [state for state in states if state.evidence_id == evidence_id]
        return states

    def list_current_states(self) -> list[StateNode]:
        """Return states currently marked as current."""
        return self.list_states(status="current")

    def list_stale_states(self) -> list[StateNode]:
        """Return states currently marked as stale."""
        return self.list_states(status="stale")

    def list_uncertain_states(self) -> list[StateNode]:
        """Return states currently marked as uncertain."""
        return self.list_states(status="uncertain")

    def list_historical_states(self) -> list[StateNode]:
        """Return states currently marked as historical."""
        return self.list_states(status="historical")

    def get_current_states(self) -> list[StateNode]:
        """Compatibility alias for list_current_states."""
        return self.list_current_states()

    def list_entities(self) -> list[str]:
        """List indexed entity names in deterministic order."""
        return sorted(self._entity_index)

    def list_attributes(self, entity: str | None = None) -> list[str]:
        """List attributes, optionally constrained to one entity."""
        if entity is None:
            return sorted(self._attribute_index)
        return sorted(
            attribute
            for indexed_entity, attribute in self._entity_attribute_index
            if indexed_entity == entity
        )

    def find_states_by_entity(self, entity: str) -> list[StateNode]:
        """Return states for an entity."""
        return self._states_for_ids(self._entity_index.get(entity, set()))

    def find_states_by_attribute(self, attribute: str) -> list[StateNode]:
        """Return states for an attribute."""
        return self._states_for_ids(self._attribute_index.get(attribute, set()))

    def find_states_by_entity_attribute(
        self,
        entity: str,
        attribute: str,
    ) -> list[StateNode]:
        """Return states matching an entity and attribute."""
        return self._states_for_ids(
            self._entity_attribute_index.get((entity, attribute), set())
        )

    def list_edges(
        self,
        edge_type: str | None = None,
        source: str | None = None,
        target: str | None = None,
    ) -> list[StateEdge]:
        """List edges, optionally filtered by edge type."""
        if edge_type is not None:
            self._validate_edge_type(edge_type)
        edges = self._sorted_edges(self.edges)
        if edge_type is not None:
            edges = [edge for edge in edges if edge.edge_type == edge_type]
        if source is not None:
            edges = [edge for edge in edges if edge.source == source]
        if target is not None:
            edges = [edge for edge in edges if edge.target == target]
        return edges

    def has_edge(
        self,
        source: str,
        target: str,
        edge_type: str | None = None,
    ) -> bool:
        """Return whether a matching state-state edge exists."""
        if edge_type is not None:
            self._validate_edge_type(edge_type)
        return any(
            edge.source == source
            and edge.target == target
            and (edge_type is None or edge.edge_type == edge_type)
            for edge in self.edges
        )

    def successors(
        self,
        state_id: str,
        edge_type: str | None = None,
    ) -> list[StateNode]:
        """Return successor states, optionally filtered by edge type."""
        self.get_state(state_id)
        target_ids = {edge.target for edge in self.outgoing_edges(state_id, edge_type)}
        return self._states_for_ids(target_ids)

    def predecessors(
        self,
        state_id: str,
        edge_type: str | None = None,
    ) -> list[StateNode]:
        """Return predecessor states, optionally filtered by edge type."""
        self.get_state(state_id)
        source_ids = {edge.source for edge in self.incoming_edges(state_id, edge_type)}
        return self._states_for_ids(source_ids)

    def outgoing_edges(
        self,
        state_id: str,
        edge_type: str | None = None,
    ) -> list[StateEdge]:
        """Return outgoing edges for a state id."""
        self.get_state(state_id)
        return self.list_edges(edge_type=edge_type, source=state_id)

    def incoming_edges(
        self,
        state_id: str,
        edge_type: str | None = None,
    ) -> list[StateEdge]:
        """Return incoming edges for a state id."""
        self.get_state(state_id)
        return self.list_edges(edge_type=edge_type, target=state_id)

    def edges_from(self, state_id: str) -> list[StateEdge]:
        """Compatibility alias for outgoing_edges."""
        return self.outgoing_edges(state_id)

    def edges_to(self, state_id: str) -> list[StateEdge]:
        """Compatibility alias for incoming_edges."""
        return self.incoming_edges(state_id)

    def states_supported_by(self, evidence_id: str) -> list[StateNode]:
        """Return states whose evidence_id matches an evidence node."""
        self.get_evidence(evidence_id)
        return self.list_states(evidence_id=evidence_id)

    def evidence_for_state(self, state_id: str) -> EvidenceNode | None:
        """Return evidence for a state, if available."""
        state = self.get_state(state_id)
        if state.evidence_id is None:
            return None
        return self.evidence.get(state.evidence_id)

    def validate_integrity(self) -> None:
        """Validate internal graph-store consistency."""
        if len(self.states) != len(set(self.states)):
            raise ValueError("Duplicate state IDs detected")
        if len(self.evidence) != len(set(self.evidence)):
            raise ValueError("Duplicate evidence IDs detected")

        for state_id, state in self.states.items():
            if state.state_id != state_id:
                raise ValueError(f"State key mismatch for {state_id}")
            self._validate_status(state.status)
            if state.evidence_id is not None and state.evidence_id not in self.evidence:
                raise ValueError(
                    f"State {state_id} references missing evidence_id: "
                    f"{state.evidence_id}"
                )
            if not self.graph.has_node(self._state_node_key(state_id)):
                raise ValueError(f"NetworkX graph missing state node: {state_id}")

        for evidence_id, evidence in self.evidence.items():
            if evidence.evidence_id != evidence_id:
                raise ValueError(f"Evidence key mismatch for {evidence_id}")
            if not self.graph.has_node(self._evidence_node_key(evidence_id)):
                raise ValueError(f"NetworkX graph missing evidence node: {evidence_id}")

        for edge in self.edges:
            self._validate_edge_type(edge.edge_type)
            if edge.source not in self.states:
                raise ValueError(f"Edge references missing source state: {edge.source}")
            if edge.target not in self.states:
                raise ValueError(f"Edge references missing target state: {edge.target}")
            if not self.graph.has_edge(
                self._state_node_key(edge.source),
                self._state_node_key(edge.target),
            ):
                raise ValueError(
                    "NetworkX graph missing edge: "
                    f"{edge.source} -> {edge.target}"
                )

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-serializable graph representation."""
        self.validate_integrity()
        return {
            "states": [self._model_to_dict(state) for state in self.list_states()],
            "evidence": [
                self._model_to_dict(evidence) for evidence in self.list_evidence()
            ],
            "edges": [self._model_to_dict(edge) for edge in self.list_edges()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphStore":
        """Reconstruct a graph store from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("GraphStore data must be a dictionary")
        store = cls()
        for evidence_payload in data.get("evidence", []):
            store.add_evidence(store._coerce_model(EvidenceNode, evidence_payload))
        for state_payload in data.get("states", []):
            store.add_state(store._coerce_model(StateNode, state_payload))
        for edge_payload in data.get("edges", []):
            store.add_edge(store._coerce_model(StateEdge, edge_payload))
        store.validate_integrity()
        return store

    def save_json(self, path: str | Path) -> None:
        """Save the graph store to a pretty JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load_json(cls, path: str | Path) -> "GraphStore":
        """Load a graph store from a JSON file."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def clear(self) -> None:
        """Remove all states, evidence, edges, indexes, and graph metadata."""
        self.graph.clear()
        self.states.clear()
        self.evidence.clear()
        self.edges.clear()
        self._entity_index.clear()
        self._attribute_index.clear()
        self._entity_attribute_index.clear()

    def copy(self) -> "GraphStore":
        """Return an independent copy of this graph store."""
        return self.from_dict(self.to_dict())

    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        """Convert a Pydantic model or dataclass to a plain dictionary."""
        if hasattr(model, "model_dump"):
            return model.model_dump(mode="json")
        if is_dataclass(model):
            return asdict(model)
        if isinstance(model, dict):
            return model
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

    @staticmethod
    def _coerce_model(model_type: type[Any], payload: Any) -> Any:
        """Create a schema model from a raw payload."""
        if hasattr(model_type, "model_validate"):
            return model_type.model_validate(payload)
        return model_type(**payload)

    @staticmethod
    def _validate_status(status: str) -> None:
        """Validate a state status literal."""
        if status not in get_args(StateStatus):
            raise ValueError(f"Unsupported state status: {status}")

    @staticmethod
    def _validate_edge_type(edge_type: str) -> None:
        """Validate an edge type literal."""
        if edge_type not in get_args(EdgeType):
            raise ValueError(f"Unsupported edge type: {edge_type}")

    @staticmethod
    def _state_node_key(state_id: str) -> str:
        """Return the internal NetworkX node key for a state."""
        return f"state:{state_id}"

    @staticmethod
    def _evidence_node_key(evidence_id: str) -> str:
        """Return the internal NetworkX node key for evidence."""
        return f"evidence:{evidence_id}"

    @staticmethod
    def _edge_sort_key(edge: StateEdge) -> tuple[str, str, str, str]:
        """Return a deterministic sort key for state-state edges."""
        return (edge.source, edge.target, edge.edge_type, edge.reason or "")

    @staticmethod
    def _state_sort_key(state: StateNode) -> str:
        """Return a deterministic sort key for states."""
        return state.state_id

    def _sorted_states(self, states: Any) -> list[StateNode]:
        """Return states sorted by state id."""
        return sorted(states, key=self._state_sort_key)

    def _sorted_edges(self, edges: Any) -> list[StateEdge]:
        """Return edges sorted by source, target, type, and reason."""
        return sorted(edges, key=self._edge_sort_key)

    def _states_for_ids(self, state_ids: set[str]) -> list[StateNode]:
        """Return states for ids in deterministic order."""
        return [self.states[state_id] for state_id in sorted(state_ids)]

    def _index_state(self, state: StateNode) -> None:
        """Update direct lookup indexes for a state."""
        self._entity_index.setdefault(state.entity, set()).add(state.state_id)
        self._attribute_index.setdefault(state.attribute, set()).add(state.state_id)
        self._entity_attribute_index.setdefault(
            (state.entity, state.attribute),
            set(),
        ).add(state.state_id)

    def _sync_state_node(self, state: StateNode) -> None:
        """Synchronize a state node into the NetworkX mirror graph."""
        self.graph.add_node(
            self._state_node_key(state.state_id),
            node_type="state",
            state_id=state.state_id,
            state=state,
        )

    def _sync_evidence_node(self, evidence: EvidenceNode) -> None:
        """Synchronize an evidence node into the NetworkX mirror graph."""
        self.graph.add_node(
            self._evidence_node_key(evidence.evidence_id),
            node_type="evidence",
            evidence_id=evidence.evidence_id,
            evidence=evidence,
        )

    def _sync_state_edge(self, edge: StateEdge) -> None:
        """Synchronize a state-state edge into the NetworkX mirror graph."""
        source_key = self._state_node_key(edge.source)
        target_key = self._state_node_key(edge.target)
        existing_edges = self.graph.get_edge_data(source_key, target_key, {}).get(
            "edges",
            [],
        )
        self.graph.add_edge(
            source_key,
            target_key,
            edge_type=edge.edge_type,
            reason=edge.reason,
            edges=[*existing_edges, edge],
        )

    def _sync_evidence_state_link(self, evidence_id: str, state_id: str) -> None:
        """Synchronize an evidence-state support link into NetworkX."""
        self.graph.add_edge(
            self._evidence_node_key(evidence_id),
            self._state_node_key(state_id),
            edge_type="supports",
            reason="state.evidence_id",
        )


InMemoryStateGraph = GraphStore


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
