# StateGraph Project

Initial framework for **Graph-Based State Revision and Invalidation Propagation for Long-Term Memory Agents**.

This repository is intentionally a skeleton. It defines schemas, small module boundaries, runner entry points, baseline stubs, evaluation stubs, and output directories. It does not implement the full StateGraph algorithms yet.

## MVP Scope

The MVP compares the following methods once data adapters and algorithms are implemented:

- `vector_rag`
- `summary_memory`
- `time_decay_rag`
- `cupmem_reimpl`
- `stategraph`

External memory systems are not integrated in this scaffold. `Mem0`, `Graphiti`, `A-MEM`, and `Letta` only have runner stubs that raise `NotImplementedError`.

## Directory Structure

```text
stategraph_project/
|-- README.md
|-- __init__.py
|-- pyproject.toml
|-- baselines/
|   |-- __init__.py
|   |-- amem_runner.py
|   |-- cupmem_reimpl.py
|   |-- graphiti_runner.py
|   |-- letta_runner.py
|   |-- mem0_runner.py
|   |-- summary_memory.py
|   |-- time_decay_rag.py
|   `-- vector_rag.py
|-- configs/
|   |-- baseline.yaml
|   |-- dataset.yaml
|   `-- model.yaml
|-- data/
|   |-- longmemeval/
|   |-- longmemeval_v2/
|   |-- memora/
|   |-- stale/
|   `-- statechangebench/
|-- outputs/
|   |-- error_analysis/
|   |-- graphs/
|   |-- metrics/
|   `-- predictions/
|-- scripts/
|   |-- __init__.py
|   |-- build_statechangebench.py
|   |-- evaluate_all.py
|   `-- make_tables.py
|-- src/
|   |-- __init__.py
|   |-- answer_generation.py
|   |-- conflict_detection.py
|   |-- graph_store.py
|   |-- invalidation_propagation.py
|   |-- metrics.py
|   |-- premise_checking.py
|   |-- retrieval.py
|   |-- run_baseline.py
|   |-- run_stategraph.py
|   |-- state_extraction.py
|   `-- state_revision.py
`-- tests/
    |-- test_imports.py
    `-- test_schema.py
```

Key modules:

- `src/graph_store.py` defines `StateNode`, `EvidenceNode`, `StateEdge`, `DatasetExample`, and a replaceable in-memory graph store.
- `src/run_stategraph.py` wires the eight-step StateGraph pipeline skeleton.
- `src/run_baseline.py` provides a JSONL runner for MVP baseline stubs.
- `src/metrics.py` lists metric stubs for future evaluation.

## StateGraph Pipeline Skeleton

1. Extract candidate states from observation.
2. Link candidate states to existing states.
3. Detect conflict or update relation.
4. Revise state status.
5. Propagate invalidation through typed edges.
6. Check query premise.
7. Retrieve current states and supporting evidence.
8. Generate final answer.

## Data Format

Future experiments should use JSONL examples compatible with `DatasetExample`:

```json
{"case_id":"example-1","history":[],"new_observation":"","query":"","gold_current_states":[],"gold_invalidated_states":[],"gold_keep_states":[],"gold_answer":""}
```

Datasets are not downloaded by this project skeleton.

## Running Future Stubs

Run import-safe tests:

```bash
python3 -m unittest discover -s stategraph_project/tests
```

Run a baseline stub on a JSONL file:

```bash
python -m stategraph_project.src.run_baseline --baseline vector_rag --input data/examples.jsonl --output outputs/predictions/vector_rag.jsonl
```

Run the StateGraph skeleton:

```bash
python -m stategraph_project.src.run_stategraph --input data/examples.jsonl --output outputs/predictions/stategraph.jsonl
```

These commands are lightweight placeholders. They do not call LLM APIs, install external systems, or run expensive experiments.

## Next Coding Stage TODOs

- Add dataset adapters and validation for each benchmark.
- Implement deterministic candidate state extraction.
- Implement state linking and conflict classification.
- Define state revision rules and edge-specific invalidation propagation.
- Add premise checking, retrieval, and answer generation behavior.
- Implement metrics and error analysis reports.
