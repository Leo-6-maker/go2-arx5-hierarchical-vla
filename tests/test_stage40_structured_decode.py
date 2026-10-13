"""CPU-only Stage40 StructuredActionV1 live-decode contract checks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.stage40_structured_decode import (
    EXPECTED_STRUCTURED_STATS_SHA256,
    decode_raw_structured_action,
    load_structured_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
D0_ROOT = Path("/scratch/e1192548/leon-storage/runs/stage40_d0_decode_20260907")
RUN_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage40_c_20260907_r1/stage40_live_604652.hopper-m-02"
)
STATS = Path(os.environ.get(
    "STAGE40_STRUCTURED_STATS",
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/"
    "experiment_cfg/structured_action_statistics.json",
))
TS = np.arange(40, dtype=np.float64) * 0.05


def test_structured_statistics_are_frozen() -> None:
    if not STATS.is_file() and "STAGE40_STRUCTURED_STATS" not in os.environ:
        pytest.skip("historical private statistics are not present on this host")
    lower, upper, mask, digest = load_structured_statistics(STATS)
    assert digest == EXPECTED_STRUCTURED_STATS_SHA256
    assert lower.shape == (132,)
    assert upper.shape == (132,)
    assert mask.shape == (40, 132)


def test_saved_d0_outputs_redecode_exactly() -> None:
    if not STATS.is_file() or not RUN_ROOT.is_dir() or not D0_ROOT.is_dir():
        pytest.skip("historical private decode evidence is not present on this host")
    for seed in (2223, 2228, 2233, 2238):
        with np.load(RUN_ROOT / "mailbox" / "responses" / f"stage40-s{seed}-q0.npz", allow_pickle=False) as data:
            raw = np.asarray(data["raw_universal_action"])
            state = np.asarray(data["input_state"], dtype=np.float32)
        with np.load(D0_ROOT / f"seed_{seed}_derived_paths.npz", allow_pickle=False) as data:
            expected = np.asarray(data["structured_physical"], dtype=np.float64)
        _, physical, _ = decode_raw_structured_action(
            raw,
            state,
            stats_path=STATS,
            source_timestamps=TS,
            observation_timestamp=0.0,
            chunk_id=f"test-{seed}",
        )
        assert np.array_equal(physical, expected)


def test_stage40_path_does_not_use_dense_decode_fallback() -> None:
    live = (ROOT / "scripts" / "stage40_live_one_handoff.py").read_text(encoding="utf-8")
    worker = (ROOT / "scripts" / "stage21_same_scene_groot_worker.py").read_text(encoding="utf-8")
    assert "decode_raw_structured_action" in live
    assert "_model_chunk" not in live
    assert "--action-contract" in worker
    assert "inference._run_one_raw" in worker
    assert "STRUCTURED_ACTION_V1_DEFERRED_TO_NATIVE" in worker
