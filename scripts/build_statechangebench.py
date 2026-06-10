"""Build a tiny StateChangeBench dev split.

This script is a placeholder and does not download datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stategraph.dataset_io import write_jsonl
from stategraph.schemas import DatasetExample


def build_dev_examples() -> list[DatasetExample]:
    """Create a small hand-written StateChangeBench dev split."""
    return [
        DatasetExample(
            case_id="SCB_001",
            history=[
                {
                    "evidence_id": "SCB_001_E1",
                    "text": "The user is free on Friday afternoon.",
                    "source": "calendar_note",
                },
                {
                    "evidence_id": "SCB_001_E2",
                    "text": "Plan a Friday afternoon meeting with Sam.",
                    "source": "assistant_plan",
                },
            ],
            new_observation={
                "evidence_id": "SCB_001_E3",
                "text": "I now have a flight to Melbourne on Friday afternoon.",
                "source": "user_update",
            },
            query="Can we still schedule the meeting with Sam on Friday afternoon?",
            gold_current_states=["SCB_001_state_flight_melbourne"],
            gold_invalidated_states=[
                "SCB_001_state_free_friday_afternoon",
                "SCB_001_state_meeting_plan_friday",
            ],
            gold_keep_states=[],
            gold_answer=(
                "No. The Friday afternoon meeting plan is no longer valid because "
                "the user has a flight to Melbourne then."
            ),
            expected_behavior=(
                "Reject the stale availability premise and invalidate the meeting plan."
            ),
        ),
        DatasetExample(
            case_id="SCB_002",
            history=[
                {
                    "evidence_id": "SCB_002_E1",
                    "text": "The user wanted Italian food for dinner.",
                    "source": "preference_memory",
                },
                {
                    "evidence_id": "SCB_002_E2",
                    "text": "Recommend an Italian restaurant tonight.",
                    "source": "assistant_plan",
                },
            ],
            new_observation={
                "evidence_id": "SCB_002_E3",
                "text": "I do not want Italian tonight.",
                "source": "user_update",
            },
            query="Should I recommend the Italian restaurant tonight?",
            gold_current_states=["SCB_002_state_no_italian_tonight"],
            gold_invalidated_states=[
                "SCB_002_state_wants_italian",
                "SCB_002_state_italian_recommendation",
            ],
            gold_keep_states=[],
            gold_answer=(
                "No. The Italian recommendation is stale because the user does "
                "not want Italian tonight."
            ),
            expected_behavior="Invalidate the old dinner preference and recommendation.",
        ),
        DatasetExample(
            case_id="SCB_003",
            history=[
                {
                    "evidence_id": "SCB_003_E1",
                    "text": "The user is in London.",
                    "source": "location_memory",
                },
                {
                    "evidence_id": "SCB_003_E2",
                    "text": "Suggest a London local plan for today.",
                    "source": "assistant_plan",
                },
            ],
            new_observation={
                "evidence_id": "SCB_003_E3",
                "text": "I am in Manchester today.",
                "source": "user_update",
            },
            query="Is the London local plan still appropriate today?",
            gold_current_states=["SCB_003_state_in_manchester_today"],
            gold_invalidated_states=[
                "SCB_003_state_in_london",
                "SCB_003_state_london_plan",
            ],
            gold_keep_states=[],
            gold_answer=(
                "No. The London plan is stale because the user is in Manchester today."
            ),
            expected_behavior="Invalidate location-dependent London plans.",
        ),
        DatasetExample(
            case_id="SCB_004",
            history=[
                {
                    "evidence_id": "SCB_004_E1",
                    "text": "The report deadline is Monday.",
                    "source": "task_memory",
                },
                {
                    "evidence_id": "SCB_004_E2",
                    "text": "Remind the user about the report on Monday.",
                    "source": "assistant_plan",
                },
            ],
            new_observation={
                "evidence_id": "SCB_004_E3",
                "text": "The report deadline changed to Wednesday.",
                "source": "user_update",
            },
            query="When should the report reminder be scheduled?",
            gold_current_states=["SCB_004_state_deadline_wednesday"],
            gold_invalidated_states=["SCB_004_state_monday_reminder"],
            gold_keep_states=["SCB_004_state_report_task"],
            gold_answer=(
                "Schedule it for Wednesday. The Monday reminder is stale, but "
                "the report task should remain."
            ),
            expected_behavior=(
                "Invalidate the old reminder while preserving the underlying task."
            ),
        ),
        DatasetExample(
            case_id="SCB_005",
            history=[
                {
                    "evidence_id": "SCB_005_E1",
                    "text": "The data export task was marked completed.",
                    "source": "task_tracker",
                }
            ],
            new_observation={
                "evidence_id": "SCB_005_E2",
                "text": "The export result has a bug and needs review.",
                "source": "user_update",
            },
            query="Is the data export task still completed?",
            gold_current_states=["SCB_005_state_needs_review"],
            gold_invalidated_states=["SCB_005_state_completed"],
            gold_keep_states=["SCB_005_state_export_task"],
            gold_answer=(
                "No. The completed state is invalidated because the result has "
                "a bug; the task now needs review."
            ),
            expected_behavior=(
                "Invalidate completed status and mark the task as needs-review."
            ),
        ),
    ]


def build_statechangebench(output_dir: Path) -> Path:
    """Write the tiny dev split and return its output path."""
    output_path = output_dir / "dev.jsonl"
    examples = build_dev_examples()
    write_jsonl(output_path, examples)
    print(f"Wrote {len(examples)} examples to {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/statechangebench"))
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    build_statechangebench(args.output_dir)


if __name__ == "__main__":
    main()
