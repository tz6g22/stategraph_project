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
|-- pyproject.toml
|-- configs/
|   |-- dataset/
|   |-- experiment/
|   |-- method/
|   `-- model/
|-- data/
|   |-- README.md
|   |-- processed/
|   |-- raw/
|   `-- statechangebench/
|       |-- dev.jsonl
|       `-- test.jsonl
|-- experiments/
|   |-- ablation/
|   `-- mvp/
|-- outputs/
|   |-- error_analysis/
|   |-- graphs/
|   |-- metrics/
|   |-- predictions/
|   `-- tables/
|-- scripts/
|   |-- __init__.py
|   |-- build_statechangebench.py
|   |-- evaluate_all.py
|   `-- make_tables.py
|-- src/
|   `-- stategraph/
|       |-- baselines/
|       |-- core/
|       |-- evaluation/
|       |-- llm/
|       |-- methods/
|       |-- runners/
|       |-- utils/
|       |-- dataset_io.py
|       `-- schemas.py
`-- tests/
    |-- test_dataset_io.py
    |-- test_imports.py
    `-- test_schema.py
```

Key modules:

- `src/stategraph/schemas.py` defines the core schemas.
- `src/stategraph/dataset_io.py` provides JSONL dataset I/O.
- `src/stategraph/core/graph_store.py` defines the replaceable in-memory graph store.
- `src/stategraph/runners/run_method.py` wires the eight-step StateGraph pipeline skeleton.
- `src/stategraph/runners/run_experiment.py` provides a JSONL runner for MVP method stubs.
- `src/stategraph/evaluation/metrics.py` lists metric stubs for future evaluation.

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
{"case_id":"example-1","history":[],"new_observation":{"text":""},"query":"Example query?","gold_current_states":[],"gold_invalidated_states":[],"gold_keep_states":[],"gold_answer":"","expected_behavior":""}
```

Datasets are not downloaded by this project skeleton.

## Running Future Stubs

Run import-safe tests:

```bash
pytest -q
```

Run a baseline stub on a JSONL file:

```bash
PYTHONPATH=src python -m stategraph.runners.run_experiment --baseline vector_rag --input data/examples.jsonl --output outputs/predictions/vector_rag.jsonl
```

Run the StateGraph skeleton:

```bash
PYTHONPATH=src python -m stategraph.runners.run_method --input data/examples.jsonl --output outputs/predictions/stategraph.jsonl
```

These commands are lightweight placeholders. They do not call LLM APIs, install external systems, or run expensive experiments.

## Next Coding Stage TODOs

- Add dataset adapters and validation for each benchmark.
- Implement deterministic candidate state extraction.
- Implement state linking and conflict classification.
- Define state revision rules and edge-specific invalidation propagation.
- Add premise checking, retrieval, and answer generation behavior.
- Implement metrics and error analysis reports.
