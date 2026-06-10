"""Deterministic state-grounded answer generation.

This module turns retrieval output into a stable template-based answer. It does
not call LLM APIs, generate nondeterministically, retrieve additional context,
revise graph state, propagate invalidations, or evaluate metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stategraph.schemas import (
    AnswerCitation,
    AnswerGenerationConfig,
    AnswerWarning,
    GeneratedAnswer,
    RetrievalResult,
)


STALE_OR_CONFLICT_TERMS = {
    "stale",
    "premise",
    "contradict",
    "conflict",
    "invalid",
    "invalidated",
}
STALE_OR_CONFLICT_STATUSES = {
    "stale",
    "conflict",
    "conflicting",
    "contradicted",
    "contradicted_by_current",
    "invalid",
    "invalidated",
    "supported_by_stale",
    "supported_by_historical",
}


class AnswerGenerator:
    """Generate deterministic answers from retrieved current-state context."""

    def __init__(self, config: AnswerGenerationConfig | None = None) -> None:
        """Configure deterministic answer generation."""
        self.config = config or AnswerGenerationConfig()

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult | None = None,
        premise_reports: Sequence[Any] | Any | None = None,
        **legacy_kwargs: Any,
    ) -> GeneratedAnswer:
        """Generate a state-grounded answer from retrieval output.

        ``retrieval`` and ``premise_check`` are accepted as legacy keyword
        aliases used by the skeleton runner before this module returned a
        structured answer object.
        """
        if retrieval_result is None:
            retrieval_result = legacy_kwargs.get("retrieval")
        if premise_reports is None and "premise_check" in legacy_kwargs:
            premise_reports = legacy_kwargs["premise_check"]
        if retrieval_result is None:
            raise TypeError("retrieval_result is required")

        warnings = self._warnings_from_retrieval(retrieval_result)
        warnings.extend(self._warnings_from_premise_reports(premise_reports))
        corrected_premises = self._corrected_premises(
            retrieval_result=retrieval_result,
            premise_reports=premise_reports,
        )

        current_states = [
            state
            for state in list(_get(retrieval_result, "current_states", []))
            if _get(state, "status") == "current"
        ]
        current_states = sorted(
            current_states,
            key=lambda state: (
                _get(state, "entity", ""),
                _get(state, "attribute", ""),
                _state_id(state),
            ),
        )
        if self.config.max_answer_states is not None:
            current_states = current_states[: self.config.max_answer_states]

        uncertain_states = list(_get(retrieval_result, "uncertain_states", []))
        if uncertain_states and self.config.include_uncertain_notes:
            for state in uncertain_states:
                warnings.append(
                    AnswerWarning(
                        code="uncertain_state_available",
                        message=(
                            "Uncertain state was retrieved but not used as a "
                            "current fact."
                        ),
                        state_id=_state_id(state),
                    )
                )

        if not current_states and self.config.abstain_on_empty_context:
            answer = self._build_abstention_answer(corrected_premises)
            return GeneratedAnswer(
                answer=answer,
                used_state_ids=[],
                used_evidence_ids=[],
                citations=[],
                corrected_premises=corrected_premises,
                warnings=self._dedupe_warnings(warnings),
                abstained=True,
            )

        used_state_ids: list[str] = []
        used_evidence_ids: list[str] = []
        citations: list[AnswerCitation] = []
        answer_parts: list[str] = []
        if corrected_premises:
            answer_parts.append(self._correction_sentence())

        for state in current_states:
            state_id = _state_id(state)
            evidence_ids = _evidence_ids_for_state(state, retrieval_result)
            if not evidence_ids:
                warnings.append(
                    AnswerWarning(
                        code="missing_evidence",
                        message="Used current state has no supporting evidence id.",
                        state_id=state_id,
                    )
                )
            used_state_ids.append(state_id)
            used_evidence_ids.extend(evidence_ids)
            citations.append(
                AnswerCitation(
                    state_id=state_id,
                    evidence_ids=sorted(set(evidence_ids)),
                )
            )
            answer_parts.append(_state_to_sentence(state))

        if uncertain_states and self.config.include_uncertain_notes:
            answer_parts.append(
                "Some related state information is uncertain and was not used "
                "as a current fact."
            )
        if self.config.include_warning_notes and self._has_warning_code(
            warnings,
            "token_budget_truncated",
        ):
            answer_parts.append("The retrieved context was truncated by a token budget.")

        return GeneratedAnswer(
            answer=" ".join(part for part in answer_parts if part).strip(),
            used_state_ids=sorted(set(used_state_ids)),
            used_evidence_ids=sorted(set(used_evidence_ids)),
            citations=sorted(citations, key=lambda citation: citation.state_id),
            corrected_premises=corrected_premises,
            warnings=self._dedupe_warnings(warnings),
            abstained=False,
        )

    @staticmethod
    def _warnings_from_retrieval(retrieval_result: Any) -> list[AnswerWarning]:
        """Convert retrieval warnings into answer warnings."""
        answer_warnings: list[AnswerWarning] = []
        for warning in _get(retrieval_result, "warnings", []) or []:
            code = str(
                _get(
                    warning,
                    "code",
                    _get(warning, "warning_type", "retrieval_warning"),
                )
            )
            message = str(_get(warning, "message", code))
            state_ids = list(_get(warning, "state_ids", []) or [])
            if state_ids:
                for state_id in state_ids:
                    answer_warnings.append(
                        AnswerWarning(
                            code=code,
                            message=message,
                            state_id=str(state_id),
                        )
                    )
            else:
                answer_warnings.append(AnswerWarning(code=code, message=message))
        return answer_warnings

    @staticmethod
    def _warnings_from_premise_reports(
        premise_reports: Sequence[Any] | Any | None,
    ) -> list[AnswerWarning]:
        """Convert premise report/status objects into answer warnings."""
        warnings: list[AnswerWarning] = []
        for report in _as_sequence(premise_reports):
            status = _normalized_status(_get(report, "status", ""))
            policy = _normalized_status(
                _get(report, "recommended_response_policy", "")
            )
            if _status_requires_correction(status) or "correct_stale_premise" in policy:
                warnings.append(
                    AnswerWarning(
                        code="stale_or_contradicted_premise",
                        message=_premise_message(report),
                    )
                )
            for result in _get(report, "results", []) or []:
                result_status = _normalized_status(_get(result, "status", ""))
                if _status_requires_correction(result_status):
                    warnings.append(
                        AnswerWarning(
                            code="stale_or_contradicted_premise",
                            message=_premise_message(result),
                        )
                    )
        return warnings

    def _corrected_premises(
        self,
        retrieval_result: Any,
        premise_reports: Sequence[Any] | Any | None,
    ) -> list[str]:
        """Return stale or conflicted premise messages from retrieval/reports."""
        corrected: list[str] = []
        for warning in _get(retrieval_result, "warnings", []) or []:
            code = str(
                _get(
                    warning,
                    "code",
                    _get(warning, "warning_type", ""),
                )
            )
            message = str(_get(warning, "message", code))
            if _text_indicates_correction(f"{code} {message}"):
                corrected.append(message)

        for report in _as_sequence(premise_reports):
            status = _normalized_status(_get(report, "status", ""))
            policy = _normalized_status(
                _get(report, "recommended_response_policy", "")
            )
            if _status_requires_correction(status) or "correct_stale_premise" in policy:
                corrected.append(_premise_message(report))
            for result in _get(report, "results", []) or []:
                result_status = _normalized_status(_get(result, "status", ""))
                if _status_requires_correction(result_status):
                    corrected.append(_premise_message(result))

        return list(dict.fromkeys(item for item in corrected if item))

    @staticmethod
    def _build_abstention_answer(corrected_premises: list[str]) -> str:
        """Build deterministic abstention text."""
        parts: list[str] = []
        if corrected_premises:
            parts.append(
                "The query appears to rely on outdated or contradicted information."
            )
        parts.append(
            "I have insufficient current information to answer safely."
        )
        return " ".join(parts)

    @staticmethod
    def _correction_sentence() -> str:
        """Return deterministic correction lead-in text."""
        return "The query appears to rely on outdated or contradicted information."

    @staticmethod
    def _dedupe_warnings(warnings: list[AnswerWarning]) -> list[AnswerWarning]:
        """Return deterministic unique answer warnings."""
        unique: dict[tuple[str, str, str | None, str | None], AnswerWarning] = {}
        for warning in warnings:
            key = (
                warning.code,
                warning.message,
                warning.state_id,
                warning.evidence_id,
            )
            unique.setdefault(key, warning)
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _has_warning_code(warnings: list[AnswerWarning], code: str) -> bool:
        """Return whether warning code exists."""
        return any(warning.code == code for warning in warnings)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Safely read an attribute or mapping value."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _state_id(state: Any) -> str:
    """Return a state id from a state-like object."""
    return str(_get(state, "state_id", _get(state, "id", "")))


def _state_to_sentence(state: Any) -> str:
    """Render a current state as deterministic answer text."""
    entity = _get(state, "entity", "the entity")
    attribute = _get(state, "attribute", "state")
    value = _get(state, "value", "unknown")
    time_scope = _get(state, "time_scope")
    condition_scope = _get(state, "condition_scope")
    scope_parts: list[str] = []
    if time_scope:
        scope_parts.append(f"for {time_scope}")
    if condition_scope:
        scope_parts.append(f"under {condition_scope}")
    scope = f" {' '.join(scope_parts)}" if scope_parts else ""
    return f"The current information indicates that {entity} {attribute} is {value}{scope}."


def _evidence_ids_for_state(state: Any, retrieval_result: Any) -> list[str]:
    """Return supporting evidence ids for a state-like object."""
    state_id = _state_id(state)
    evidence_ids: list[str] = []
    direct_evidence_id = _get(state, "evidence_id")
    if direct_evidence_id:
        evidence_ids.append(str(direct_evidence_id))
    direct_evidence_ids = _get(state, "evidence_ids", None)
    if direct_evidence_ids:
        evidence_ids.extend(str(evidence_id) for evidence_id in direct_evidence_ids)

    evidence_entries = list(_get(retrieval_result, "supporting_evidence", []) or [])
    if not evidence_entries:
        evidence_entries = list(_get(retrieval_result, "evidence", []) or [])
    for evidence in evidence_entries:
        evidence_id = _get(evidence, "evidence_id", _get(evidence, "id", None))
        if not evidence_id:
            continue
        supporting_state_ids = list(_get(evidence, "supporting_state_ids", []) or [])
        evidence_state_id = _get(evidence, "state_id", None)
        if state_id in {str(item) for item in supporting_state_ids} or evidence_state_id == state_id:
            evidence_ids.append(str(evidence_id))
    return sorted(set(evidence_ids))


def _as_sequence(items: Sequence[Any] | Any | None) -> list[Any]:
    """Return premise report input as a list without treating dict as sequence."""
    if items is None:
        return []
    if isinstance(items, dict):
        return [items]
    if isinstance(items, (str, bytes)):
        return [items]
    if isinstance(items, Sequence):
        return list(items)
    return [items]


def _normalized_status(status: Any) -> str:
    """Normalize status-like text."""
    return str(status or "").strip().lower().replace("-", "_")


def _status_requires_correction(status: str) -> bool:
    """Return whether a status indicates a stale or conflicted premise."""
    if status in STALE_OR_CONFLICT_STATUSES:
        return True
    return any(term in status for term in STALE_OR_CONFLICT_TERMS)


def _text_indicates_correction(text: str) -> bool:
    """Return whether warning text indicates stale/conflicted premise risk."""
    normalized = _normalized_status(text)
    return any(term in normalized for term in STALE_OR_CONFLICT_TERMS)


def _premise_message(report_or_result: Any) -> str:
    """Build a correction message from a premise report-like object."""
    message = _get(report_or_result, "message", None)
    if message:
        return str(message)
    premise = _get(report_or_result, "premise", None)
    if premise is not None:
        premise_text = _get(premise, "text", None)
        if premise_text:
            return f"Premise requires correction: {premise_text}"
        value = _get(premise, "value", None)
        attribute = _get(premise, "attribute", None)
        if attribute or value:
            return f"Premise requires correction: {attribute or 'state'}={value or 'unknown'}"
    status = _get(report_or_result, "status", None)
    if status:
        return f"Premise requires correction due to status: {status}"
    return "Premise requires correction."
