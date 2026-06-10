"""Evaluation metric stubs for StateGraph experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class MetricResult:
    """A scalar metric result with optional details."""

    name: str
    value: float | None
    details: dict[str, Any] = field(default_factory=dict)


def _stub_metric(name: str) -> MetricResult:
    """Return a placeholder metric result."""
    return MetricResult(
        name=name,
        value=None,
        details={"todo": "metric implementation is pending"},
    )


def final_answer_accuracy(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Final Answer Accuracy."""
    _ = args
    _ = kwargs
    return _stub_metric("Final Answer Accuracy")


def state_resolution_accuracy(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub State Resolution Accuracy."""
    _ = args
    _ = kwargs
    return _stub_metric("State Resolution Accuracy")


def stale_premise_rejection_rate(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Stale Premise Rejection Rate."""
    _ = args
    _ = kwargs
    return _stub_metric("Stale Premise Rejection Rate")


def propagation_precision(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Propagation Precision."""
    _ = args
    _ = kwargs
    return _stub_metric("Propagation Precision")


def propagation_recall(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Propagation Recall."""
    _ = args
    _ = kwargs
    return _stub_metric("Propagation Recall")


def propagation_f1(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Propagation F1."""
    _ = args
    _ = kwargs
    return _stub_metric("Propagation F1")


def action_accuracy(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Action Accuracy."""
    _ = args
    _ = kwargs
    return _stub_metric("Action Accuracy")


def evidence_faithfulness(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Evidence Faithfulness."""
    _ = args
    _ = kwargs
    return _stub_metric("Evidence Faithfulness")


def token_cost(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Token Cost."""
    _ = args
    _ = kwargs
    return _stub_metric("Token Cost")


def latency(*args: Any, **kwargs: Any) -> MetricResult:
    """Stub Latency."""
    _ = args
    _ = kwargs
    return _stub_metric("Latency")


METRIC_REGISTRY: dict[str, Callable[..., MetricResult]] = {
    "final_answer_accuracy": final_answer_accuracy,
    "state_resolution_accuracy": state_resolution_accuracy,
    "stale_premise_rejection_rate": stale_premise_rejection_rate,
    "propagation_precision": propagation_precision,
    "propagation_recall": propagation_recall,
    "propagation_f1": propagation_f1,
    "action_accuracy": action_accuracy,
    "evidence_faithfulness": evidence_faithfulness,
    "token_cost": token_cost,
    "latency": latency,
}

