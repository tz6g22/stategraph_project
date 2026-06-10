"""Central schema definitions for StateGraph experiments."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


NonEmptyStr = Annotated[str, Field(min_length=1)]

StateStatus = Literal["current", "stale", "historical", "uncertain"]
EdgeType = Literal[
    "supports",
    "updates",
    "invalidates",
    "depends-on",
    "derived-from",
    "affects-action",
]
ConflictLabel = Literal[
    "consistent",
    "duplicate",
    "update",
    "explicit_conflict",
    "implicit_invalidation",
    "temporary_exception",
    "uncertain",
]


class StateGraphBaseModel(BaseModel):
    """Base schema settings for project data models."""

    model_config = ConfigDict(str_strip_whitespace=True)


class StateNode(StateGraphBaseModel):
    """A structured claim about an entity at a time or condition scope."""

    state_id: NonEmptyStr
    entity: NonEmptyStr
    attribute: NonEmptyStr
    value: NonEmptyStr
    time_scope: str | None = None
    condition_scope: str | None = None
    status: StateStatus = "uncertain"
    evidence_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceNode(StateGraphBaseModel):
    """Raw text evidence associated with one or more states."""

    evidence_id: NonEmptyStr
    text: NonEmptyStr
    source: str | None = None
    timestamp: str | None = None


class StateEdge(StateGraphBaseModel):
    """Typed relation between two state nodes."""

    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: EdgeType
    reason: str | None = None


class DatasetExample(StateGraphBaseModel):
    """A single JSONL-compatible evaluation case."""

    case_id: NonEmptyStr
    history: list[dict[str, Any]] = Field(default_factory=list)
    new_observation: dict[str, Any] | None = None
    query: NonEmptyStr
    gold_current_states: list[str] = Field(default_factory=list)
    gold_invalidated_states: list[str] = Field(default_factory=list)
    gold_keep_states: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    expected_behavior: str | None = None


class ConflictDecision(StateGraphBaseModel):
    """A labeled consistency decision for candidate state handling."""

    label: ConflictLabel
    reason: NonEmptyStr
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

