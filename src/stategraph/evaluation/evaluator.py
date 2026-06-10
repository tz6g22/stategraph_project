"""Evaluation orchestration stubs."""

from __future__ import annotations

from stategraph.evaluation.metrics import METRIC_REGISTRY, MetricResult


class Evaluator:
    """Placeholder evaluator for future experiment outputs."""

    def evaluate(self) -> list[MetricResult]:
        """Return placeholder metric results.

        TODO: Load predictions and gold examples, then compute metrics.
        """
        return [metric() for metric in METRIC_REGISTRY.values()]

