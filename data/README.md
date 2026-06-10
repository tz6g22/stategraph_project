# Data Placement Guide

This framework stage does not download any datasets.

- `data/stale/`: STALE/CUPMem-style experiments.
- `data/statechangebench/`: self-built StateChangeBench diagnostic set.
- `data/longmemeval/`: optional extension dataset.
- `data/longmemeval_v2/`: optional dynamic state tracking extension dataset.
- `data/memora/`: optional stale/obsolete memory evaluation.

Raw downloaded files should not be committed if they are large. Prefer storing
processed examples as JSONL, with each line following the `DatasetExample`
schema from `stategraph_project.src.schemas`.

