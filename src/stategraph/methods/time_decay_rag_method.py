"""Time-decay RAG baseline stub."""

from __future__ import annotations

from stategraph.schemas import DatasetExample


class TimeDecayRagBaseline:
    """Placeholder time-decay retrieval baseline.

    TODO: Add timestamp parsing and decay-weighted retrieval later.
    """

    name = "time_decay_rag"

    def predict(self, example: DatasetExample) -> str:
        """Return a placeholder prediction."""
        _ = example
        return "TODO: time_decay_rag prediction is not implemented."
