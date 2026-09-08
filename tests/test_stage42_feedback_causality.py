"""CPU-only checks for the Stage42 causal audit contract."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - login-node fallback
    np = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

if np is not None:
    import stage42_feedback_causality as stage42  # noqa: E402
else:  # pragma: no cover - login-node fallback
    stage42 = None


def _snapshot() -> dict:
    records = {}
    for index, request_id in enumerate(stage42.Q_IDS):
        body = np.full((96, 96, 3), index, dtype=np.uint8)
        wrist = np.full((96, 96, 3), index + 1, dtype=np.uint8)
        state = np.linspace(0.0, 1.0, 10, dtype=np.float32) + index
        raw = np.zeros((1, 40, 132), dtype=np.float32)
        raw += index
        decoded = np.zeros((40, 10), dtype=np.float64)
        decoded += index
        records[request_id] = {
            "observation_timestamp": float(index),
            "body": body,
            "wrist": wrist,
            "state": state,
            "raw": raw,
            "decoded": decoded,
            "hashes": {
                "body_sha256": stage42.sha256_array(body),
                "wrist_sha256": stage42.sha256_array(wrist),
                "state_v1_sha256": stage42.sha256_array(state),
                "raw_action_sha256": stage42.sha256_array(raw),
                "decoded_action_sha256": stage42.sha256_array(decoded),
            },
            "request_npz_sha256": "a" * 64,
            "safety_gate": {"decision": "ACCEPT", "issues": []},
        }
    return {"records": records}


def test_within_run_profile_records_exact_hash_matrix_and_diversity() -> None:
    if np is None:
        return
    profile = stage42._within_run_profile(_snapshot())
    assert profile["classification"] == "WITHIN_RUN_ACTION_DIVERSITY_PRESENT"
    assert profile["state_v1_distinct"] is True
    assert profile["raw_action_distinct"] is True
    assert profile["decoded_action_distinct"] is True
    assert profile["hash_equality_matrices"]["raw_action_d132"][0][0] is True
    assert profile["hash_equality_matrices"]["raw_action_d132"][0][1] is False
    assert len(profile["pairwise"]) == 55
    assert len(profile["adjacent"]) == 10


def test_within_run_byte_identical_classification_is_exact() -> None:
    if np is None:
        return
    snapshot = _snapshot()
    first = snapshot["records"][stage42.Q_IDS[0]]
    for request_id in stage42.Q_IDS[1:]:
        current = snapshot["records"][request_id]
        for field in ("raw", "decoded"):
            current[field] = first[field].copy()
            current["hashes"]["%s_action_sha256" % ("raw" if field == "raw" else "decoded")] = first["hashes"]["%s_action_sha256" % ("raw" if field == "raw" else "decoded")]
    profile = stage42._within_run_profile(snapshot)
    assert profile["classification"] == "WITHIN_RUN_ACTIONS_BYTE_IDENTICAL"


def test_stage42_bundle_budget_is_fail_closed() -> None:
    if np is None:
        return
    bundle = {
        "schema": "CORELAB_STAGE42_EXECUTION_BUNDLE_V1",
        "stage_id": stage42.STAGE_ID,
        "pi_gate": stage42.PI_GATE,
        "dev_seed": stage42.DEV_SEED,
        "predecessor": {"stage41_head": stage42.STAGE41_HEAD, "stage41_bundle_sha256": stage42.STAGE41_BUNDLE_SHA256},
        "budgets": {"replay_forwards": 3, "counterfactual_forwards": 6, "total_gr00t_forwards": 9, "wbc_steps": 0, "reference_updates": 0},
        "counterfactual_plan": {"ids": list(stage42.CF_IDS)},
        "forbidden_operations": stage42.expected_forbidden_operations(),
    }
    stage42._verify_bundle(bundle)
    bundle["budgets"]["total_gr00t_forwards"] = 10
    try:
        stage42._verify_bundle(bundle)
    except stage42.Stage42EvidenceError:
        pass
    else:
        raise AssertionError("drifted Stage42 budget was accepted")
