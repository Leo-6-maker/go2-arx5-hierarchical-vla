"""Locked Stage22 target split used before any model selection or test run."""

from __future__ import annotations

from typing import Iterable


STAGE22_SPLIT_LOCK_SCHEMA = "STAGE22_TARGET_SPLIT_LOCK_V1"
TRAIN_TARGET_IDS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
VALIDATION_TARGET_IDS = ("red_target_holdout_e",)
FINAL_TEST_TARGET_IDS = ("red_target_holdout_f",)


def locked_target_ids(split: str) -> tuple[str, ...]:
    """Return the frozen target IDs for one Stage22 split."""

    values = {
        "train": TRAIN_TARGET_IDS,
        "validation": VALIDATION_TARGET_IDS,
        "final_test": FINAL_TEST_TARGET_IDS,
    }
    if split not in values:
        raise ValueError(f"unknown locked Stage22 split: {split}")
    return values[split]


def validate_locked_target_universe(target_ids: Iterable[str]) -> tuple[str, ...]:
    """Require the exact six-target universe used by the Stage22 fixture."""

    observed = tuple(dict.fromkeys(str(value) for value in target_ids))
    expected = TRAIN_TARGET_IDS + VALIDATION_TARGET_IDS + FINAL_TEST_TARGET_IDS
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError(
            "Stage22 target universe differs from the locked train/validation/final-test split"
        )
    if set(TRAIN_TARGET_IDS) & set(VALIDATION_TARGET_IDS + FINAL_TEST_TARGET_IDS):
        raise AssertionError("Stage22 locked split has overlapping target IDs")
    return expected


def split_lock_payload() -> dict[str, object]:
    """Return the canonical, Git-trackable split-lock payload."""

    return {
        "schema": STAGE22_SPLIT_LOCK_SCHEMA,
        "status": "LOCKED_BEFORE_GPU_INFERENCE",
        "train_target_ids": list(TRAIN_TARGET_IDS),
        "validation_target_ids": list(VALIDATION_TARGET_IDS),
        "final_test_target_ids": list(FINAL_TEST_TARGET_IDS),
        "model_selection_split": "validation",
        "final_test_policy": "single_locked_read_after_model_selection",
        "target_actor_in_model_input": False,
        "selection_leakage_guard": "final_test_f_is_not_used_for_training_or_checkpoint_selection",
    }
