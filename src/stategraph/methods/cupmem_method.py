"""CupMem reimplementation baseline stub."""

from __future__ import annotations

from stategraph.schemas import DatasetExample


class CupMemBaseline:
    """Placeholder CupMem-style baseline.

    TODO: Revisit the paper details and implement only after defining the MVP
    evaluation contract.
    """

    name = "cupmem_reimpl"

    def predict(self, example: DatasetExample) -> str:
        """Return a placeholder prediction."""
        _ = example
        return "TODO: cupmem_reimpl prediction is not implemented."
