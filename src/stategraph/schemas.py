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
StateLinkMatchType = Literal[
    "entity_attribute_time",
    "same_entity_attribute",
    "same_entity",
    "same_attribute",
    "same_time_scope",
    "dependency_neighbor",
    "evidence_neighbor",
    "weak_text_overlap",
]
PremiseStatus = Literal[
    "supported_current",
    "contradicted_by_current",
    "supported_by_stale",
    "supported_by_historical",
    "uncertain",
    "unsupported",
    "no_premise_detected",
]
ResponsePolicy = Literal[
    "accept_premise",
    "correct_stale_premise",
    "reject_premise",
    "ask_clarification",
    "answer_with_current_state",
    "proceed_without_premise",
]
RetrievalWarningType = Literal[
    "stale_premise",
    "contradicted_premise",
    "historical_premise",
    "uncertain_premise",
    "excluded_stale_state",
    "excluded_historical_state",
    "missing_evidence",
    "token_budget_truncated",
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


class StateLink(StateGraphBaseModel):
    """A rule-based link between a candidate state and an existing state."""

    candidate_state_id: NonEmptyStr
    matched_state_id: NonEmptyStr
    match_type: StateLinkMatchType
    score: float = Field(ge=0.0, le=1.0)
    reason: NonEmptyStr
    features: dict[str, bool | float | str | None] = Field(default_factory=dict)


class RevisionItem(StateGraphBaseModel):
    """One revision decision for a candidate against an existing state."""

    existing_state_id: NonEmptyStr
    decision: ConflictDecision
    link: StateLink | None = None


class RevisionReport(StateGraphBaseModel):
    """Structured report describing graph mutations from state revision."""

    candidate_state_id: NonEmptyStr
    candidate_added: bool
    candidate_final_status: StateStatus | None = None
    updated_state_ids: list[str] = Field(default_factory=list)
    invalidated_state_ids: list[str] = Field(default_factory=list)
    historical_state_ids: list[str] = Field(default_factory=list)
    unchanged_state_ids: list[str] = Field(default_factory=list)
    uncertain_state_ids: list[str] = Field(default_factory=list)
    duplicate_state_ids: list[str] = Field(default_factory=list)
    created_edges: list[StateEdge] = Field(default_factory=list)
    skipped_edges: list[dict[str, str]] = Field(default_factory=list)
    decisions: list[RevisionItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PropagationStep(StateGraphBaseModel):
    """One invalidation propagation action along a dependency edge."""

    source_state_id: NonEmptyStr
    target_state_id: NonEmptyStr
    edge_type: EdgeType
    old_status: StateStatus
    new_status: StateStatus
    depth: int = Field(ge=0)
    reason: NonEmptyStr


class PropagationReport(StateGraphBaseModel):
    """Structured report for dependency-based invalidation propagation."""

    seed_state_ids: list[str] = Field(default_factory=list)
    propagated_state_ids: list[str] = Field(default_factory=list)
    unchanged_state_ids: list[str] = Field(default_factory=list)
    skipped_state_ids: list[str] = Field(default_factory=list)
    propagation_steps: list[PropagationStep] = Field(default_factory=list)
    max_depth_reached: bool = False
    cycle_detected: bool = False
    dry_run: bool = False
    notes: list[str] = Field(default_factory=list)


class QueryPremise(StateGraphBaseModel):
    """A lightweight premise extracted from a user query."""

    premise_id: NonEmptyStr
    text: NonEmptyStr
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    time_scope: str | None = None
    condition_scope: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class PremiseCheckResult(StateGraphBaseModel):
    """Result of checking one extracted query premise against graph state."""

    premise: QueryPremise
    status: PremiseStatus
    supporting_current_state_ids: list[str] = Field(default_factory=list)
    conflicting_current_state_ids: list[str] = Field(default_factory=list)
    stale_support_state_ids: list[str] = Field(default_factory=list)
    historical_support_state_ids: list[str] = Field(default_factory=list)
    uncertain_state_ids: list[str] = Field(default_factory=list)
    reason: NonEmptyStr
    recommended_response_policy: ResponsePolicy


class PremiseCheckReport(StateGraphBaseModel):
    """Query-level premise checking report."""

    query: str
    premises: list[QueryPremise] = Field(default_factory=list)
    results: list[PremiseCheckResult] = Field(default_factory=list)
    has_stale_premise: bool = False
    has_contradiction: bool = False
    has_uncertainty: bool = False
    recommended_response_policy: ResponsePolicy = "proceed_without_premise"
    notes: list[str] = Field(default_factory=list)


class RetrievedState(StateGraphBaseModel):
    """A state selected for retrieval context."""

    state_id: NonEmptyStr
    entity: NonEmptyStr
    attribute: NonEmptyStr
    value: NonEmptyStr
    time_scope: str | None = None
    condition_scope: str | None = None
    status: StateStatus
    evidence_id: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieval_reason: NonEmptyStr


class RetrievedEvidence(StateGraphBaseModel):
    """Evidence selected to support retrieved states."""

    evidence_id: NonEmptyStr
    text: NonEmptyStr
    source: str | None = None
    timestamp: str | None = None
    supporting_state_ids: list[str] = Field(default_factory=list)


class RetrievalWarning(StateGraphBaseModel):
    """Warning emitted while retrieving premise-aware context."""

    warning_type: RetrievalWarningType
    state_ids: list[str] = Field(default_factory=list)
    message: NonEmptyStr


class RetrievalResult(StateGraphBaseModel):
    """Structured context returned by premise-aware retrieval."""

    query: str
    current_states: list[RetrievedState] = Field(default_factory=list)
    supporting_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    uncertain_states: list[RetrievedState] = Field(default_factory=list)
    correction_states: list[RetrievedState] = Field(default_factory=list)
    excluded_stale_state_ids: list[str] = Field(default_factory=list)
    excluded_historical_state_ids: list[str] = Field(default_factory=list)
    warnings: list[RetrievalWarning] = Field(default_factory=list)
    premise_policy: ResponsePolicy | None = None
    token_budget: int | None = None
    estimated_tokens: int = Field(default=0, ge=0)
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def states(self) -> list[RetrievedState]:
        """Compatibility alias for previously retrieved states."""
        return self.current_states

    @property
    def evidence(self) -> list[RetrievedEvidence]:
        """Compatibility alias for previously retrieved evidence."""
        return self.supporting_evidence


class AnswerWarning(StateGraphBaseModel):
    """Warning emitted during deterministic answer generation."""

    code: NonEmptyStr
    message: NonEmptyStr
    state_id: str | None = None
    evidence_id: str | None = None


class AnswerCitation(StateGraphBaseModel):
    """Mapping from a used state to supporting evidence ids."""

    state_id: NonEmptyStr
    evidence_ids: list[str] = Field(default_factory=list)


class AnswerGenerationConfig(StateGraphBaseModel):
    """Configuration for deterministic answer generation."""

    abstain_on_empty_context: bool = True
    include_uncertain_notes: bool = True
    include_warning_notes: bool = True
    max_answer_states: int | None = None


class GeneratedAnswer(StateGraphBaseModel):
    """Structured answer generated from retrieved StateGraph context."""

    answer: str
    used_state_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    corrected_premises: list[str] = Field(default_factory=list)
    warnings: list[AnswerWarning] = Field(default_factory=list)
    abstained: bool = False

    def __str__(self) -> str:
        """Return the answer text for compatibility with string contexts."""
        return self.answer
