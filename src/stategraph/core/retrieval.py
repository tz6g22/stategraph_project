"""Premise-aware retrieval of safe current-state context.

This module ranks graph states and evidence for later answer generation. It is
read-only: it does not mutate graph state, revise statuses, propagate
invalidations, generate answers, call LLMs, or use embedding models.
"""

from __future__ import annotations

from collections import defaultdict

from stategraph.core.conflict_detection import (
    normalize_text as normalize_conflict_text,
    token_overlap as conflict_token_overlap,
    tokenize as conflict_tokenize,
)
from stategraph.core.graph_store import GraphStore
from stategraph.schemas import (
    EvidenceNode,
    PremiseCheckReport,
    RetrievedEvidence,
    RetrievedState,
    RetrievalResult,
    RetrievalWarning,
    StateNode,
)


PREMISE_STATE_BOOST = 0.35
PREMISE_SLOT_BOOST = 0.15
BASE_EMPTY_QUERY_SCORE = 0.05


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic lexical matching."""
    return normalize_conflict_text(text)


def tokenize(text: str | None) -> set[str]:
    """Return normalized tokens."""
    tokens = conflict_tokenize(text)
    if tokens & {"i", "me", "my", "mine"}:
        tokens.add("user")
    return tokens


def token_overlap(a: str | None, b: str | None) -> float:
    """Return Jaccard token overlap for two strings."""
    return conflict_token_overlap(a, b)


def state_to_text(state: StateNode | RetrievedState) -> str:
    """Render a state into compact retrieval text."""
    return " ".join(
        part
        for part in [
            state.entity,
            state.attribute,
            state.value,
            state.time_scope,
            state.condition_scope,
            state.status,
        ]
        if part
    )


def evidence_to_text(evidence: EvidenceNode | RetrievedEvidence) -> str:
    """Render evidence into compact retrieval text."""
    return " ".join(
        part
        for part in [
            evidence.evidence_id,
            evidence.text,
            evidence.source,
            evidence.timestamp,
        ]
        if part
    )


def estimate_tokens(text: str | None) -> int:
    """Estimate token count using whitespace tokens."""
    if not text:
        return 0
    return len(text.split())


def score_state_relevance(
    query: str,
    state: StateNode,
    premise_report: PremiseCheckReport | None = None,
) -> tuple[float, str]:
    """Score state relevance with deterministic lexical and premise signals."""
    query_tokens = tokenize(query)
    if not query_tokens:
        base_score = BASE_EMPTY_QUERY_SCORE
    else:
        entity_score = _field_overlap(query_tokens, state.entity)
        attribute_score = _field_overlap(query_tokens, state.attribute)
        value_score = _field_overlap(query_tokens, state.value)
        time_score = _field_overlap(query_tokens, state.time_scope)
        condition_score = _field_overlap(query_tokens, state.condition_scope)
        base_score = (
            0.22 * entity_score
            + 0.28 * attribute_score
            + 0.28 * value_score
            + 0.17 * time_score
            + 0.05 * condition_score
        )

    boost = 0.0
    reasons: list[str] = []
    if base_score > 0:
        reasons.append("query overlap")
    if premise_report is not None:
        premise_state_ids = _premise_referenced_state_ids(premise_report)
        if state.state_id in premise_state_ids:
            boost += PREMISE_STATE_BOOST
            reasons.append("referenced by premise report")
        elif _matches_any_premise_slot(state, premise_report):
            boost += PREMISE_SLOT_BOOST
            reasons.append("matches premise slot")

    score = max(0.0, min(1.0, base_score + boost))
    if not reasons:
        reasons.append("low lexical relevance")
    return score, "; ".join(reasons)


def score_evidence_relevance(
    query: str,
    evidence: EvidenceNode,
    supporting_state_ids: list[str],
) -> tuple[float, str]:
    """Score evidence relevance by query overlap and support count."""
    overlap = _field_overlap(tokenize(query), evidence.text)
    support_bonus = min(0.4, 0.12 * len(set(supporting_state_ids)))
    score = max(0.0, min(1.0, overlap + support_bonus))
    reason = "supports retrieved states"
    if overlap > 0:
        reason = "query overlap and supports retrieved states"
    return score, reason


class StateRetriever:
    """Retrieve safe current context plus optional correction context."""

    def __init__(
        self,
        include_uncertain: bool = True,
        include_correction_context: bool = True,
        max_states: int | None = None,
        max_evidence: int | None = None,
        token_budget: int | None = None,
    ) -> None:
        """Configure deterministic retrieval limits and context policy."""
        for name, value in {
            "max_states": max_states,
            "max_evidence": max_evidence,
            "token_budget": token_budget,
        }.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        self.include_uncertain = include_uncertain
        self.include_correction_context = include_correction_context
        self.max_states = max_states
        self.max_evidence = max_evidence
        self.token_budget = token_budget

    def retrieve(
        self,
        query: str,
        graph: GraphStore,
        premise_report: PremiseCheckReport | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        """Retrieve premise-aware state and evidence context."""
        effective_max_states = self.max_states
        if limit is not None and effective_max_states is None:
            effective_max_states = limit

        notes: list[str] = []
        warnings = self._warnings_from_premise_report(premise_report)
        excluded_stale_state_ids = [state.state_id for state in graph.list_stale_states()]
        excluded_historical_state_ids = [
            state.state_id for state in graph.list_historical_states()
        ]
        if excluded_stale_state_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="excluded_stale_state",
                    state_ids=excluded_stale_state_ids,
                    message="stale states excluded from current retrieval context",
                )
            )
        if excluded_historical_state_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="excluded_historical_state",
                    state_ids=excluded_historical_state_ids,
                    message="historical states excluded from current retrieval context",
                )
            )

        current_states = self._rank_states(
            query=query,
            states=graph.list_current_states(),
            premise_report=premise_report,
        )
        if effective_max_states is not None and len(current_states) > effective_max_states:
            current_states = current_states[:effective_max_states]
            notes.append(f"current states truncated by max_states={effective_max_states}")

        uncertain_states: list[RetrievedState] = []
        if self.include_uncertain:
            uncertain_states = self._rank_states(
                query=query,
                states=graph.list_uncertain_states(),
                premise_report=premise_report,
            )
            if effective_max_states is not None and len(uncertain_states) > effective_max_states:
                uncertain_states = uncertain_states[:effective_max_states]
                notes.append(
                    f"uncertain states truncated by max_states={effective_max_states}"
                )

        correction_states: list[RetrievedState] = []
        if self.include_correction_context and premise_report is not None:
            correction_states = self._correction_states(
                query=query,
                graph=graph,
                premise_report=premise_report,
                notes=notes,
            )

        supporting_evidence, evidence_warnings = self._collect_evidence(
            query=query,
            graph=graph,
            states=[*current_states, *uncertain_states, *correction_states],
        )
        warnings.extend(evidence_warnings)
        if self.max_evidence is not None and len(supporting_evidence) > self.max_evidence:
            supporting_evidence = supporting_evidence[: self.max_evidence]
            notes.append(f"supporting evidence truncated by max_evidence={self.max_evidence}")

        result = RetrievalResult(
            query=query,
            current_states=current_states,
            supporting_evidence=supporting_evidence,
            uncertain_states=uncertain_states,
            correction_states=correction_states,
            excluded_stale_state_ids=excluded_stale_state_ids,
            excluded_historical_state_ids=excluded_historical_state_ids,
            warnings=self._dedupe_warnings(warnings),
            premise_policy=(
                premise_report.recommended_response_policy
                if premise_report is not None
                else None
            ),
            token_budget=self.token_budget,
            estimated_tokens=self._estimate_result_tokens(
                current_states,
                supporting_evidence,
                uncertain_states,
                correction_states,
                warnings,
                notes,
            ),
            truncated=False,
            notes=list(dict.fromkeys(notes)),
        )
        if self.token_budget is not None:
            result = self._apply_token_budget(result)
        return result

    def _rank_states(
        self,
        query: str,
        states: list[StateNode],
        premise_report: PremiseCheckReport | None,
    ) -> list[RetrievedState]:
        """Convert and rank states deterministically."""
        retrieved: list[RetrievedState] = []
        for state in states:
            score, reason = score_state_relevance(query, state, premise_report)
            retrieved.append(self._to_retrieved_state(state, score, reason))
        return sorted(retrieved, key=self._state_sort_key)

    def _correction_states(
        self,
        query: str,
        graph: GraphStore,
        premise_report: PremiseCheckReport,
        notes: list[str],
    ) -> list[RetrievedState]:
        """Return stale/historical states relevant to premise correction."""
        correction_ids: set[str] = set()
        for result in premise_report.results:
            correction_ids.update(result.stale_support_state_ids)
            correction_ids.update(result.historical_support_state_ids)

        correction_states: list[RetrievedState] = []
        for state_id in sorted(correction_ids):
            if not graph.has_state(state_id):
                notes.append(f"premise report referenced missing state {state_id}")
                continue
            state = graph.get_state(state_id)
            score, reason = score_state_relevance(query, state, premise_report)
            correction_states.append(
                self._to_retrieved_state(
                    state,
                    max(score, 0.5),
                    f"premise correction context; {reason}",
                )
            )
        return sorted(correction_states, key=self._state_sort_key)

    def _collect_evidence(
        self,
        query: str,
        graph: GraphStore,
        states: list[RetrievedState],
    ) -> tuple[list[RetrievedEvidence], list[RetrievalWarning]]:
        """Collect and group evidence for retrieved states."""
        supporting_state_ids_by_evidence: dict[str, set[str]] = defaultdict(set)
        warnings: list[RetrievalWarning] = []
        for state in states:
            if not state.evidence_id:
                continue
            if not graph.has_evidence(state.evidence_id):
                warnings.append(
                    RetrievalWarning(
                        warning_type="missing_evidence",
                        state_ids=[state.state_id],
                        message=f"state {state.state_id} references missing evidence {state.evidence_id}",
                    )
                )
                continue
            supporting_state_ids_by_evidence[state.evidence_id].add(state.state_id)

        evidence_entries: list[RetrievedEvidence] = []
        for evidence_id, supporting_state_ids in supporting_state_ids_by_evidence.items():
            evidence = graph.get_evidence(evidence_id)
            score_evidence_relevance(query, evidence, sorted(supporting_state_ids))
            evidence_entries.append(
                RetrievedEvidence(
                    evidence_id=evidence.evidence_id,
                    text=evidence.text,
                    source=evidence.source,
                    timestamp=evidence.timestamp,
                    supporting_state_ids=sorted(supporting_state_ids),
                )
            )
        evidence_entries.sort(
            key=lambda item: (-len(item.supporting_state_ids), item.evidence_id)
        )
        return evidence_entries, warnings

    def _warnings_from_premise_report(
        self,
        premise_report: PremiseCheckReport | None,
    ) -> list[RetrievalWarning]:
        """Create retrieval warnings from premise checking output."""
        if premise_report is None:
            return []
        warnings: list[RetrievalWarning] = []
        stale_ids: set[str] = set()
        historical_ids: set[str] = set()
        contradicted_ids: set[str] = set()
        uncertain_ids: set[str] = set()
        for result in premise_report.results:
            stale_ids.update(result.stale_support_state_ids)
            historical_ids.update(result.historical_support_state_ids)
            contradicted_ids.update(result.conflicting_current_state_ids)
            uncertain_ids.update(result.uncertain_state_ids)

        if stale_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="stale_premise",
                    state_ids=sorted(stale_ids),
                    message="query premise is supported by stale state",
                )
            )
        if historical_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="historical_premise",
                    state_ids=sorted(historical_ids),
                    message="query premise is supported by historical state",
                )
            )
        if premise_report.has_contradiction or contradicted_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="contradicted_premise",
                    state_ids=sorted(contradicted_ids),
                    message="query premise conflicts with current state",
                )
            )
        if premise_report.has_uncertainty or uncertain_ids:
            warnings.append(
                RetrievalWarning(
                    warning_type="uncertain_premise",
                    state_ids=sorted(uncertain_ids),
                    message="query premise only matches uncertain state",
                )
            )
        return warnings

    def _apply_token_budget(self, result: RetrievalResult) -> RetrievalResult:
        """Trim context greedily to fit an approximate token budget."""
        assert self.token_budget is not None
        current_states: list[RetrievedState] = []
        uncertain_states: list[RetrievedState] = []
        correction_states: list[RetrievedState] = []
        supporting_evidence: list[RetrievedEvidence] = []
        warnings = list(result.warnings)
        notes = list(result.notes)
        used_tokens = estimate_tokens(result.query)
        truncated = False

        for source, target in [
            (result.current_states, current_states),
            (result.uncertain_states, uncertain_states),
            (result.correction_states, correction_states),
        ]:
            for state in source:
                cost = estimate_tokens(state_to_text(state))
                if used_tokens + cost > self.token_budget:
                    truncated = True
                    continue
                target.append(state)
                used_tokens += cost

        kept_state_ids = {
            state.state_id
            for state in [*current_states, *uncertain_states, *correction_states]
        }
        for evidence in result.supporting_evidence:
            supporting_ids = [
                state_id
                for state_id in evidence.supporting_state_ids
                if state_id in kept_state_ids
            ]
            if not supporting_ids:
                truncated = True
                continue
            candidate = evidence.model_copy(update={"supporting_state_ids": supporting_ids})
            cost = estimate_tokens(evidence_to_text(candidate))
            if used_tokens + cost > self.token_budget:
                truncated = True
                continue
            supporting_evidence.append(candidate)
            used_tokens += cost

        if truncated:
            warnings.append(
                RetrievalWarning(
                    warning_type="token_budget_truncated",
                    state_ids=[],
                    message="retrieval context was truncated by token budget",
                )
            )
        return result.model_copy(
            update={
                "current_states": current_states,
                "uncertain_states": uncertain_states,
                "correction_states": correction_states,
                "supporting_evidence": supporting_evidence,
                "warnings": self._dedupe_warnings(warnings),
                "estimated_tokens": used_tokens,
                "truncated": truncated,
                "notes": list(dict.fromkeys(notes)),
            }
        )

    @staticmethod
    def _to_retrieved_state(
        state: StateNode,
        score: float,
        reason: str,
    ) -> RetrievedState:
        """Convert a StateNode into a retrieval schema."""
        return RetrievedState(
            state_id=state.state_id,
            entity=state.entity,
            attribute=state.attribute,
            value=state.value,
            time_scope=state.time_scope,
            condition_scope=state.condition_scope,
            status=state.status,
            evidence_id=state.evidence_id,
            relevance_score=max(0.0, min(1.0, score)),
            retrieval_reason=reason,
        )

    @staticmethod
    def _state_sort_key(state: RetrievedState) -> tuple[float, str, str, str]:
        """Return deterministic sort key for retrieved states."""
        return (-state.relevance_score, state.entity, state.attribute, state.state_id)

    @staticmethod
    def _estimate_result_tokens(
        current_states: list[RetrievedState],
        supporting_evidence: list[RetrievedEvidence],
        uncertain_states: list[RetrievedState],
        correction_states: list[RetrievedState],
        warnings: list[RetrievalWarning],
        notes: list[str],
    ) -> int:
        """Estimate total retrieval payload tokens."""
        total = 0
        for state in [*current_states, *uncertain_states, *correction_states]:
            total += estimate_tokens(state_to_text(state))
        for evidence in supporting_evidence:
            total += estimate_tokens(evidence_to_text(evidence))
        for warning in warnings:
            total += estimate_tokens(warning.message)
        for note in notes:
            total += estimate_tokens(note)
        return total

    @staticmethod
    def _dedupe_warnings(warnings: list[RetrievalWarning]) -> list[RetrievalWarning]:
        """Return warnings in deterministic unique order."""
        unique: dict[tuple[str, tuple[str, ...], str], RetrievalWarning] = {}
        for warning in warnings:
            normalized = warning.model_copy(
                update={"state_ids": sorted(set(warning.state_ids))}
            )
            key = (
                normalized.warning_type,
                tuple(normalized.state_ids),
                normalized.message,
            )
            unique.setdefault(key, normalized)
        return [unique[key] for key in sorted(unique)]


def retrieve_context(
    query: str,
    graph: GraphStore,
    premise_report: PremiseCheckReport | None = None,
    include_uncertain: bool = True,
    include_correction_context: bool = True,
    max_states: int | None = None,
    max_evidence: int | None = None,
    token_budget: int | None = None,
) -> RetrievalResult:
    """Functional wrapper for StateRetriever.retrieve."""
    return StateRetriever(
        include_uncertain=include_uncertain,
        include_correction_context=include_correction_context,
        max_states=max_states,
        max_evidence=max_evidence,
        token_budget=token_budget,
    ).retrieve(query, graph, premise_report)


def _field_overlap(query_tokens: set[str], field_value: str | None) -> float:
    """Return overlap between query tokens and one state/evidence field."""
    field_tokens = tokenize(field_value)
    if not query_tokens or not field_tokens:
        return 0.0
    return len(query_tokens & field_tokens) / len(field_tokens)


def _premise_referenced_state_ids(premise_report: PremiseCheckReport) -> set[str]:
    """Return state ids explicitly referenced by premise results."""
    state_ids: set[str] = set()
    for result in premise_report.results:
        state_ids.update(result.supporting_current_state_ids)
        state_ids.update(result.conflicting_current_state_ids)
        state_ids.update(result.stale_support_state_ids)
        state_ids.update(result.historical_support_state_ids)
        state_ids.update(result.uncertain_state_ids)
    return state_ids


def _matches_any_premise_slot(
    state: StateNode,
    premise_report: PremiseCheckReport,
) -> bool:
    """Return whether a state matches any extracted premise entity/attribute."""
    return any(
        premise.entity == state.entity and premise.attribute == state.attribute
        for premise in premise_report.premises
        if premise.entity and premise.attribute
    )


RetrievalEngine = StateRetriever
retrieve = retrieve_context
