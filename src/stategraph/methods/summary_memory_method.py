"""Summary memory baseline stub."""

from __future__ import annotations

from stategraph.schemas import DatasetExample


class SummaryMemoryBaseline:
    """Placeholder summary-memory baseline.

    TODO: Add summary construction and query answering in a later stage.
    """

    name = "summary_memory"

    def predict(self, example: DatasetExample) -> str:
        """Return a placeholder prediction."""
        _ = example
        return "TODO: summary_memory prediction is not implemented."
