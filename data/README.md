# Data Placement Guide

This framework stage does not download any datasets.

- `data/raw/stale/`: raw STALE/CUPMem-style files.
- `data/raw/longmemeval/`: raw optional LongMemEval files.
- `data/raw/longmemeval_v2/`: raw optional dynamic state tracking files.
- `data/raw/memora/`: raw optional stale/obsolete memory files.
- `data/processed/stale/`: processed STALE/CUPMem-style JSONL examples.
- `data/processed/statechangebench/`: processed self-built diagnostic examples.
- `data/processed/longmemeval/`: processed optional LongMemEval examples.
- `data/processed/memora/`: processed optional stale/obsolete memory examples.
- `data/statechangebench/`: small hand-written StateChangeBench splits.

Raw downloaded files should not be committed if they are large. Prefer storing
processed examples as JSONL, with each line following the `DatasetExample`
schema from `stategraph.schemas`.
