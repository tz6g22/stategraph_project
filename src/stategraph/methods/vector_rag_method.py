"""Vector RAG baseline stub."""

from __future__ import annotations

from stategraph.schemas import DatasetExample


class VectorRagBaseline:
    """Placeholder vector retrieval baseline.

    TODO: Add local embeddings or a deterministic retrieval adapter later.
    """

    name = "vector_rag"

    def predict(self, example: DatasetExample) -> str:
        """Return a placeholder prediction."""
        _ = example
        return "TODO: vector_rag prediction is not implemented."
