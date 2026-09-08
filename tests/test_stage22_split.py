from __future__ import annotations

import pytest

from go2arx5_vla.data import (
    FINAL_TEST_TARGET_IDS,
    TRAIN_TARGET_IDS,
    VALIDATION_TARGET_IDS,
    locked_target_ids,
    validate_locked_target_universe,
)


def test_stage22_split_is_disjoint_and_exact() -> None:
    universe = TRAIN_TARGET_IDS + VALIDATION_TARGET_IDS + FINAL_TEST_TARGET_IDS
    assert locked_target_ids("train") == TRAIN_TARGET_IDS
    assert locked_target_ids("validation") == VALIDATION_TARGET_IDS
    assert locked_target_ids("final_test") == FINAL_TEST_TARGET_IDS
    assert validate_locked_target_universe(universe) == universe
    assert len(set(universe)) == len(universe)


def test_stage22_split_rejects_leakage_or_missing_target() -> None:
    with pytest.raises(ValueError, match="target universe"):
        validate_locked_target_universe(TRAIN_TARGET_IDS + VALIDATION_TARGET_IDS)
    with pytest.raises(ValueError, match="unknown locked"):
        locked_target_ids("holdout")
