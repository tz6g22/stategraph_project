"""StateGraph method wrapper stub."""

from __future__ import annotations

from stategraph.schemas import DatasetExample


class StateGraphMethod:
    """Placeholder method wrapper for the StateGraph pipeline."""

    name = "stategraph"

    def predict(self, example: DatasetExample) -> str:
        """Return a placeholder prediction.

        TODO: Wire this wrapper to the StateGraph runner when algorithms exist.
        """
        _ = example
        return "TODO: stategraph prediction is not implemented."

