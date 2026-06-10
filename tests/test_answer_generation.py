"""Tests for deterministic state-grounded answer generation."""

from __future__ import annotations

from stategraph.core.answer_generation import AnswerGenerator
from stategraph.runners.run_method import StateGraphPipeline
from stategraph.schemas import (
    GeneratedAnswer,
    RetrievedEvidence,
    RetrievedState,
    RetrievalResult,
    RetrievalWarning,
)


def retrieved_state(
    state_id: str,
    entity: str = "user",
    attribute: str = "availability",
    value: str = "unavailable",
    status: str = "current",
    evidence_id: str | None = "E1",
    time_scope: str | None = "Friday afternoon",
) -> RetrievedState:
    """Create a retrieved state fixture."""
    return RetrievedState(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        time_scope=time_scope,
        condition_scope=None,
        status=status,  # type: ignore[arg-type]
        evidence_id=evidence_id,
        relevance_score=0.9,
        retrieval_reason="test fixture",
    )


def retrieved_evidence(
    evidence_id: str,
    supporting_state_ids: list[str],
) -> RetrievedEvidence:
    """Create retrieved evidence fixture."""
    return RetrievedEvidence(
        evidence_id=evidence_id,
        text=f"Evidence {evidence_id}",
        source="test",
        timestamp=None,
        supporting_state_ids=supporting_state_ids,
    )


def retrieval_result(
    *,
    current_states: list[RetrievedState] | None = None,
    supporting_evidence: list[RetrievedEvidence] | None = None,
    uncertain_states: list[RetrievedState] | None = None,
    correction_states: list[RetrievedState] | None = None,
    warnings: list[RetrievalWarning] | None = None,
) -> RetrievalResult:
    """Create a retrieval result fixture."""
    return RetrievalResult(
        query="What is my Friday availability?",
        current_states=current_states or [],
        supporting_evidence=supporting_evidence or [],
        uncertain_states=uncertain_states or [],
        correction_states=correction_states or [],
        excluded_stale_state_ids=[],
        excluded_historical_state_ids=[],
        warnings=warnings or [],
        premise_policy=None,
        token_budget=None,
        estimated_tokens=0,
        truncated=False,
        notes=[],
    )


def warning_codes(answer: GeneratedAnswer) -> set[str]:
    """Return warning codes from generated answer."""
    return {warning.code for warning in answer.warnings}


def test_generates_answer_from_current_states() -> None:
    """Current states should produce a non-abstaining answer."""
    result = retrieval_result(
        current_states=[retrieved_state("S1")],
        supporting_evidence=[retrieved_evidence("E1", ["S1"])],
    )

    answer = AnswerGenerator().generate(
        "What is my Friday availability?",
        result,
    )

    assert answer.abstained is False
    assert answer.used_state_ids == ["S1"]
    assert answer.used_evidence_ids == ["E1"]
    assert "unavailable" in answer.answer
    assert "user availability" in answer.answer


def test_does_not_use_stale_or_historical_states() -> None:
    """Correction context should not be promoted to current answer facts."""
    result = retrieval_result(
        current_states=[retrieved_state("S_current")],
        supporting_evidence=[retrieved_evidence("E1", ["S_current"])],
        correction_states=[
            retrieved_state("S_stale", status="stale", value="free", evidence_id="E2"),
            retrieved_state(
                "S_historical",
                status="historical",
                value="busy",
                evidence_id="E3",
            ),
        ],
    )

    answer = AnswerGenerator().generate("What is my availability?", result)

    assert answer.used_state_ids == ["S_current"]
    assert "S_stale" not in answer.used_state_ids
    assert "S_historical" not in answer.used_state_ids
    assert "free" not in answer.answer


def test_corrects_stale_premise_before_answering() -> None:
    """Stale-premise warnings should add a correction sentence."""
    result = retrieval_result(
        current_states=[retrieved_state("S_current")],
        supporting_evidence=[retrieved_evidence("E1", ["S_current"])],
        warnings=[
            RetrievalWarning(
                warning_type="stale_premise",
                state_ids=["S_stale"],
                message="query premise is supported by stale state",
            )
        ],
    )

    answer = AnswerGenerator().generate("Schedule because I am free.", result)

    assert answer.abstained is False
    assert answer.corrected_premises
    assert answer.answer.startswith(
        "The query appears to rely on outdated or contradicted information."
    )
    assert "query premise is supported by stale state" in answer.corrected_premises


def test_uncertain_states_are_not_promoted_to_current_facts() -> None:
    """Uncertain states should not be used as current facts."""
    result = retrieval_result(
        uncertain_states=[
            retrieved_state(
                "S_uncertain",
                value="maybe free",
                status="uncertain",
                evidence_id="E2",
                time_scope="Saturday",
            )
        ]
    )

    answer = AnswerGenerator().generate("Can I schedule Saturday?", result)

    assert answer.abstained is True
    assert answer.used_state_ids == []
    assert "S_uncertain" not in answer.used_state_ids
    assert "uncertain_state_available" in warning_codes(answer)


def test_missing_evidence_warns_but_does_not_crash() -> None:
    """A current state without evidence should still be answerable."""
    result = retrieval_result(
        current_states=[retrieved_state("S1", evidence_id=None)],
        supporting_evidence=[],
    )

    answer = AnswerGenerator().generate("What is my availability?", result)

    assert answer.abstained is False
    assert answer.used_state_ids == ["S1"]
    assert answer.used_evidence_ids == []
    assert "missing_evidence" in warning_codes(answer)


def test_empty_current_states_abstains() -> None:
    """No current context should produce a safe abstention."""
    answer = AnswerGenerator().generate(
        "What is my availability?",
        retrieval_result(),
    )

    assert answer.abstained is True
    assert answer.used_state_ids == []
    assert answer.used_evidence_ids == []
    assert "insufficient current information" in answer.answer


def test_evidence_ids_are_grouped_into_citations() -> None:
    """Citations should map used states to supporting evidence ids."""
    state = retrieved_state("S1", evidence_id=None)
    result = retrieval_result(
        current_states=[state],
        supporting_evidence=[
            retrieved_evidence("E1", ["S1"]),
            retrieved_evidence("E2", ["S1"]),
        ],
    )

    answer = AnswerGenerator().generate("What is my availability?", result)

    assert answer.citations == [
        {"state_id": "S1", "evidence_ids": ["E1", "E2"]}
    ] or answer.citations[0].model_dump() == {
        "state_id": "S1",
        "evidence_ids": ["E1", "E2"],
    }
    assert answer.used_evidence_ids == ["E1", "E2"]


def test_generator_accepts_dict_like_premise_report() -> None:
    """Dict-like premise reports should trigger correction."""
    result = retrieval_result(
        current_states=[retrieved_state("S1")],
        supporting_evidence=[retrieved_evidence("E1", ["S1"])],
    )
    premise_report = {
        "status": "stale",
        "premise": {"text": "because I am free"},
        "recommended_response_policy": "correct_stale_premise",
    }

    answer = AnswerGenerator().generate(
        "Schedule because I am free.",
        result,
        premise_reports=[premise_report],
    )

    assert answer.corrected_premises
    assert "Premise requires correction" in answer.corrected_premises[0]


def test_runner_includes_generated_answer() -> None:
    """The skeleton runner should expose structured generated answer output."""
    output = StateGraphPipeline().run_example(
        {
            "case_id": "case-answer",
            "history": [],
            "new_observation": {"text": "No structured state extraction yet."},
            "query": "What is my current availability?",
            "gold_current_states": [],
            "gold_invalidated_states": [],
            "gold_keep_states": [],
            "gold_answer": None,
        }
    )

    assert "prediction" in output
    assert "answer" in output
    assert "generated_answer" in output
    assert output["prediction"] == output["answer"]
    assert output["generated_answer"]["answer"] == output["answer"]
    assert "retrieved_state_ids" in output
