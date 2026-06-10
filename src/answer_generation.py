"""Answer generation stubs."""

from __future__ import annotations

from .premise_checking import PremiseCheckResult
from .retrieval import RetrievalResult


class AnswerGenerator:
    """Generate final answers from retrieved state context.

    TODO: Add deterministic templates or optional model-backed generation in a
    later stage. The skeleton must not call LLM APIs.
    """

    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
        premise_check: PremiseCheckResult | None = None,
    ) -> str:
        """Return a placeholder final answer."""
        _ = query
        _ = retrieval
        _ = premise_check
        return "TODO: answer generation is not implemented."

