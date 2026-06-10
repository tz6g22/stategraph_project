"""StateGraph pipeline runner skeleton."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from stategraph.core.answer_generation import AnswerGenerator
from stategraph.core.conflict_detection import ConflictDetector
from stategraph.core.graph_store import (
    InMemoryStateGraph,
    dataset_example_from_dict,
    evidence_from_dict,
)
from stategraph.core.invalidation_propagation import InvalidationPropagator
from stategraph.core.premise_checking import PremiseChecker
from stategraph.core.retrieval import StateRetriever
from stategraph.core.state_extraction import CandidateStateExtractor
from stategraph.core.state_revision import StateRevisionEngine
from stategraph.schemas import EvidenceNode


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    """Yield dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class StateGraphPipeline:
    """Wires the eight-step StateGraph pipeline with placeholder components."""

    def __init__(self) -> None:
        """Initialize pipeline components."""
        self.extractor = CandidateStateExtractor()
        self.conflict_detector = ConflictDetector()
        self.revision_engine = StateRevisionEngine()
        self.propagator = InvalidationPropagator()
        self.premise_checker = PremiseChecker()
        self.retriever = StateRetriever()
        self.answer_generator = AnswerGenerator()

    def run_example(self, payload: dict[str, object]) -> dict[str, object]:
        """Run the skeleton pipeline on one JSON-like example."""
        example = dataset_example_from_dict(payload)
        graph = InMemoryStateGraph()
        for index, evidence_payload in enumerate(example.history):
            if "evidence_id" not in evidence_payload:
                evidence_payload["evidence_id"] = f"{example.case_id}_history_{index}"
            graph.add_evidence(evidence_from_dict(evidence_payload))

        observation_payload = dict(example.new_observation or {})
        observation_text = str(
            observation_payload.get("text") or observation_payload.get("content") or ""
        )

        observation_evidence = EvidenceNode(
            evidence_id=str(
                observation_payload.get("evidence_id")
                or f"{example.case_id}_new_observation"
            ),
            text=observation_text or "No new observation provided.",
            source=observation_payload.get("source") or "new_observation",
            timestamp=observation_payload.get("timestamp"),
        )
        graph.add_evidence(observation_evidence)

        # 1. extract candidate states from observation
        candidate_states = self.extractor.extract(
            observation_text,
            evidence=observation_evidence,
        )

        # 2. link candidate states to existing states
        # 3. detect conflict or update relation
        conflict_results = self.conflict_detector.link_candidates(
            existing_states=list(graph.states.values()),
            candidate_states=candidate_states,
        )

        # 4. revise state status
        directly_invalidated = self.revision_engine.revise(
            graph=graph,
            candidate_states=candidate_states,
            conflict_results=conflict_results,
        )

        # 5. propagate invalidation through typed edges
        propagated_invalidations = self.propagator.propagate(
            graph=graph,
            starting_state_ids=directly_invalidated,
        )

        # 6. check query premise
        premise_result = self.premise_checker.check(example.query, graph)

        # 7. retrieve current states and supporting evidence
        retrieval_result = self.retriever.retrieve(example.query, graph)

        # 8. generate final answer
        answer = self.answer_generator.generate(
            query=example.query,
            retrieval=retrieval_result,
            premise_check=premise_result,
        )

        return {
            "case_id": example.case_id,
            "prediction": answer,
            "premise_status": premise_result.status,
            "directly_invalidated_state_ids": directly_invalidated,
            "propagated_invalidated_state_ids": sorted(propagated_invalidations),
            "retrieved_state_ids": [state.state_id for state in retrieval_result.states],
        }


def run_stategraph(input_path: Path, output_path: Path) -> None:
    """Run the StateGraph skeleton over JSONL examples."""
    pipeline = StateGraphPipeline()
    rows = [pipeline.run_example(payload) for payload in read_jsonl(input_path)]
    write_jsonl(output_path, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    run_stategraph(args.input, args.output)


if __name__ == "__main__":
    main()
