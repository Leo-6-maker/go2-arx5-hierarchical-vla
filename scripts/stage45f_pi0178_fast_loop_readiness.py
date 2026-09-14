#!/usr/bin/env python3
"""CPU-only preparation for the PI0178 fast experiment loop.

This module binds already accepted artifacts and future execution contracts. It
does not construct a model, run inference, train, submit PBS/H200 work, or
invoke WBC/simulator code. The only payload reader is the explicitly supplied
PI0163 C1-q0 bundle, which is a permitted visited-state source and is never
replaced by a Validation-E or Holdout-F sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Sequence


PI_GATE = 5663122737
PR_POINTER = 5663124714
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"
SUPPORT_BRANCH = "stage45f/corelab-fast-loop-readiness"
PI0177_PR = 74
PI0177_HEAD = "da47605ac16c4d4906bae0c8f6eaaf86d0fb3e5e"
PI0177_BRANCH = "stage45f/localpc-validation-e-hopper-resume"

Q0_NPZ_SHA = "e0376e138bb1610a557164d3afffa3da1d7d0446a3b28406374360bbe7c441ad"
Q0_JSON_SHA = "a4d53a962af98b074b8a9f4a88a2b238e15254db362b2be06b2134d454c67e06"
Q0_EXPERT_CHUNK_SHA = "36deefb15ec152aad579ab0f1ec6f8a730446fd3eca0bccab833d794feab1c25"
Q0_CONDITION = "C1_X_PLUS_5CM"
Q0_TARGET = "red_target_train_a"

FULL24_TREE_SHA = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
N2_CHECKPOINT_TREE_SHA = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
BASE_TREE_SHA = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
TRAINING_CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
Q0_SCHEDULE_SHA = "cc14ca8ab624ef312df7dc510930876c29fb521ca907505f54c9ed3af0d4f3f8"

TRAIN_SEED = 6164
UPDATE_COUNT = 256
FULL24_SLOT_COUNT = 24
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
ACTIVE_SCALARS = 190

TRAIN_TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)

STAGE43_SOURCE_FILES = {
    "scripts/stage43_corelab_evidence.py": "4d85ebfb3c69ceb59bd8a7e22ea0c24f8a92f5af96ce1d3e5d78ecc8829edf73",
    "scripts/stage43_corelab_initial_condition.py": "f7318f2478cc295d405444cc6691d3522e10f49cfdb7c661ad102558406e9f7e",
}
PROMOTION_SOURCE_FILES = {
    "scripts/stage20s_offline_interface_seal.py": "98f1ff53bc14555083de6924d9d1b1c39992566657bcf8508138fe84ec1b6f87",
    "scripts/stage20s_native_closed_loop.py": "d0d921ce55667d4b0db5f7ac6ce4aa939b4d58dd20a60789ad4adfcf8ac61c45",
    "scripts/stage20s_evaluate_closed_loop.py": "f78222a73ec415463e45465143dce582019ed8c74df9f2294fb42e70d4ffe659",
    "src/go2arx5_vla/safety/trajectory_gate.py": "cd1d2667b5e8f67a99e08baca28ffc1abc492a34dd0ef585f3739259b359d1c4",
    "src/go2arx5_vla/providers/groot.py": "94a1c74c480249f952f4ea27ecac8d0871c88c1d2625fb9b2351ca37641409f0",
    "src/go2arx5_vla/adapters/groot_to_wbc.py": "f5e657964b00b2b1ee4ed0d6bdb84c516b8fc31e045d765a289dd002c479358c",
    "src/go2arx5_vla/evaluation/checkpoint_eval.py": "f65c38e3b7e03e3e0d788c4f91d101a1d243e1eccc2df9bd330bd5f124801536",
    "src/go2arx5_vla/evaluation/trajectory_metrics.py": "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4",
}

DEV_CASES = (
    ("C0_NOMINAL", (0.0, 0.0, 0.0)),
    ("C1_X_PLUS_5CM", (0.05, 0.0, 0.0)),
    ("C2_X_MINUS_5CM", (-0.05, 0.0, 0.0)),
    ("C3_Y_PLUS_5CM", (0.0, 0.05, 0.0)),
    ("C4_Y_MINUS_5CM", (0.0, -0.05, 0.0)),
)

DEV_ARTIFACT_IDENTITIES = {
    "stage43_branch": "stage43-dev/corelab-controlled-cohort",
    "stage43_head": "b2a70dcec0765cf69819bb76c180e5c92ae3eac3",
    "dev_seed": 4242,
    "language": "move the end effector to the red target",
    "target_world_m": [0.30, 0.10, 0.58],
    "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
    "tcp_semantic_id": "x5_link6_gripper_center",
    "wbc_tree_sha256": "9d653c39c73d5755a29d46f4dff91b69120244f660f41a977db471d58ba8b193",
    "wbc_arm_sha256": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "wbc_dog_sha256": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "trajectory_sha256": "e09d03e4f837234a1ed7504f6d16f202a40d7fab2c3cbbc0f2e2fa17b0b33722",
}

MODEL_SIDE_METRICS = (
    "target_progress_m",
    "final_target_distance_m",
    "path_length_m",
    "roughness_m",
    "safety_gate_decision",
    "all_finite",
    "action_shape",
)
CLOSED_LOOP_METRICS = (
    "initial_target_distance_m",
    "final_target_distance_m",
    "target_progress_m",
    "safety_gate_decision",
    "safety_gate_issues",
    "wbc_tracking_status",
    "wbc_tracking_abort_phase",
    "tracking_rms_m",
    "eef_path_length_m",
    "eef_roughness_m",
)


class ReadinessError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} must be a regular file: {path}")
    return path.resolve()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def full24_sequence(seed: int = TRAIN_SEED, update_count: int = UPDATE_COUNT) -> list[int]:
    rng = random.Random(int(seed))
    sequence: list[int] = []
    while len(sequence) < int(update_count):
        cycle = list(range(FULL24_SLOT_COUNT))
        rng.shuffle(cycle)
        sequence.extend(cycle)
    return sequence[: int(update_count)]


def exposure_schedule(seed: int = TRAIN_SEED, update_count: int = UPDATE_COUNT) -> dict[str, Any]:
    sequence = full24_sequence(seed, update_count)
    e_slots = [
        {"position": index, "source": "FULL24", "full24_slot": int(slot)}
        for index, slot in enumerate(sequence)
    ]
    s_slots = [
        {
            "position": index,
            "source": "Q0" if index % 2 == 1 else "FULL24",
            "full24_slot": None if index % 2 == 1 else int(slot),
            "q0_label": Q0_CONDITION if index % 2 == 1 else None,
        }
        for index, slot in enumerate(sequence)
    ]
    return {
        "seed": int(seed),
        "update_count": int(update_count),
        "full24_sequence": sequence,
        "sequence_sha256": canonical_sha(sequence),
        "E": e_slots,
        "S": s_slots,
        "E_full24_slots": len(e_slots),
        "S_full24_slots": sum(item["source"] == "FULL24" for item in s_slots),
        "S_q0_slots": sum(item["source"] == "Q0" for item in s_slots),
        "paired_positions_identical": all(e["position"] == s["position"] for e, s in zip(e_slots, s_slots)),
    }


def validate_exposure_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    require(schedule.get("seed") == TRAIN_SEED, "schedule seed drift")
    require(schedule.get("update_count") == UPDATE_COUNT, "schedule update count drift")
    sequence = schedule.get("full24_sequence")
    require(isinstance(sequence, list) and len(sequence) == UPDATE_COUNT, "FULL24 sequence length drift")
    require(canonical_sha(sequence) == Q0_SCHEDULE_SHA, "deterministic schedule hash drift")
    arm_e = schedule.get("E")
    arm_s = schedule.get("S")
    require(isinstance(arm_e, list) and isinstance(arm_s, list), "E/S schedule missing")
    require(len(arm_e) == len(arm_s) == UPDATE_COUNT, "E/S schedule length drift")
    require(all(item.get("source") == "FULL24" for item in arm_e), "E schedule is not FULL24-only")
    require(sum(item.get("source") == "FULL24" for item in arm_s) == 128, "S FULL24 exposure drift")
    require(sum(item.get("source") == "Q0" for item in arm_s) == 128, "S Q0 exposure drift")
    require(all(item.get("q0_label") == Q0_CONDITION for item in arm_s if item.get("source") == "Q0"), "S q0 identity drift")
    require(schedule.get("paired_positions_identical") is True, "E/S positions are not paired")
    return {
        "status": "PASS",
        "sequence_sha256": Q0_SCHEDULE_SHA,
        "E_full24_slots": 256,
        "S_full24_slots": 128,
        "S_q0_slots": 128,
        "optimizer_updates_per_arm": 256,
    }


def audit_q0_bundle(npz_path: Path, json_path: Path) -> dict[str, Any]:
    """Audit only the permitted PI0163 C1-q0 payload and its ABI."""

    import numpy as np

    npz_path = regular_file(npz_path, "PI0163 q0 NPZ")
    json_path = regular_file(json_path, "PI0163 q0 JSON")
    npz_sha = sha256_file(npz_path)
    json_sha = sha256_file(json_path)
    require(npz_sha == Q0_NPZ_SHA, f"q0 NPZ SHA drift: {npz_sha}")
    require(json_sha == Q0_JSON_SHA, f"q0 JSON SHA drift: {json_sha}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    require(metadata.get("schema") == "CORELAB_PI0162_SYNCHRONIZED_SNAPSHOT_V1", "q0 schema drift")
    require(metadata.get("condition_id") == Q0_CONDITION, "q0 condition drift")
    require(metadata.get("target_id") == Q0_TARGET, "q0 target drift")
    require(metadata.get("expert_chunk_source") == "STAGE22_SAME_SCENE_EXPERT", "q0 expert source drift")
    require(metadata.get("expert_chunk_sha256") == Q0_EXPERT_CHUNK_SHA, "q0 expert chunk identity drift")
    require(metadata.get("target_excluded_from_model_input") is True, "q0 target exclusion is not explicit")
    model_inputs = set(metadata.get("model_input_fields", []))
    model_excludes = set(metadata.get("model_input_excludes", []))
    require({"body_rgb", "wrist_rgb", "state_v1", "language"}.issubset(model_inputs), "q0 model input keys drift")
    require("target_position_world_m" in model_excludes and "expert_physical_local" in model_excludes, "q0 leakage exclusion drift")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"body_rgb", "wrist_rgb", "state_v1", "expert_physical_local", "expert_timestamps", "eef_local_m", "anchor_world_m", "target_world_m"}
        require(required.issubset(set(archive.files)), f"q0 NPZ keys drift: {archive.files}")
        body = np.asarray(archive["body_rgb"])
        wrist = np.asarray(archive["wrist_rgb"])
        state = np.asarray(archive["state_v1"], dtype=np.float32).reshape(-1)
        expert = np.asarray(archive["expert_physical_local"], dtype=np.float64)
        timestamps = np.asarray(archive["expert_timestamps"], dtype=np.float64).reshape(-1)
        eef_local = np.asarray(archive["eef_local_m"], dtype=np.float64).reshape(-1)
        target = np.asarray(archive["target_world_m"], dtype=np.float64).reshape(-1)
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, "q0 body ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, "q0 wrist ABI drift")
    require(state.shape == (STATE_DIM,) and expert.shape == (HORIZON, 10) and timestamps.shape == (HORIZON,), "q0 state/action ABI drift")
    require(eef_local.shape == target.shape == (3,), "q0 pose ABI drift")
    require(np.isfinite(state).all() and np.isfinite(expert).all() and np.isfinite(timestamps).all(), "q0 contains non-finite values")
    require(np.all(np.diff(timestamps) > 0.0), "q0 timestamps are not strictly increasing")
    require(np.allclose(eef_local, state[:3], atol=1e-5, rtol=0.0), "q0 eef_local != StateV1[:3]")
    require(np.allclose(np.asarray(metadata["state_v1"], dtype=np.float32), state, atol=1e-6, rtol=0.0), "q0 JSON/NPZ state mismatch")
    require(np.allclose(np.asarray(metadata["expert_physical_local"], dtype=np.float64), expert, atol=1e-12, rtol=0.0), "q0 JSON/NPZ expert mismatch")
    return {
        "status": "PASS",
        "condition_id": Q0_CONDITION,
        "target_id": Q0_TARGET,
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "npz_sha256": npz_sha,
        "json_sha256": json_sha,
        "expert_chunk_sha256": Q0_EXPERT_CHUNK_SHA,
        "body_shape": list(body.shape),
        "wrist_shape": list(wrist.shape),
        "state_v1_shape": list(state.shape),
        "expert_action_shape": list(expert.shape),
        "timestamp_count": int(timestamps.size),
        "target_excluded_from_model_input": True,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def training_runspec() -> dict[str, Any]:
    schedule = exposure_schedule()
    validate_exposure_schedule(schedule)
    return {
        "schema": "STAGE45F_PI0178_VISITED_STATE_TRAINING_RUNSPEC_V1",
        "status": "PREPARED_NOT_EXECUTED",
        "base_checkpoint": {
            "selection": "PI_LATE_BOUND",
            "identity": None,
            "reference_only_identity": "N2(seed2223)@2000",
            "reference_tree_sha256": N2_CHECKPOINT_TREE_SHA,
            "scientific_selection_executed": False,
        },
        "arms": {
            "E": {"name": "FULL24_ONLY", "optimizer_updates": 256, "full24_slots": 256, "q0_slots": 0},
            "S": {"name": "FIFTY_PERCENT_FULL24_PLUS_FIXED_Q0", "optimizer_updates": 256, "full24_slots": 128, "q0_slots": 128},
        },
        "schedule": {"seed": TRAIN_SEED, "sequence_sha256": Q0_SCHEDULE_SHA, "paired_positions_identical": True, "q0_replacement_rule": "zero_based_position % 2 == 1"},
        "recipe": {
            "optimizer": "adamw_torch",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": "cosine",
            "cosine_horizon_updates": 256,
            "warmup_steps": 10,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "workers": 0,
            "trainable_modules": ["projector", "diffusion"],
            "frozen_modules": ["visual", "llm", "vlln"],
            "loss": "masked StructuredActionV1 flow matching",
            "timing": "T16",
            "active_scalars": ACTIVE_SCALARS,
            "horizon": HORIZON,
            "action_dim": ACTION_DIM,
        },
        "checkpoint_policy": {
            "save_steps": [64, 128, 192, 256],
            "required_state_files": ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"],
            "fresh_output_root": True,
            "resume_from_checkpoint": False,
        },
        "execution_counters": {"training_executed": False, "backward": 0, "optimizer_steps": 0, "checkpoint_writes": 0, "h200_submissions": 0, "pbs_submissions": 0},
    }


def h200_ledger_template() -> dict[str, Any]:
    return {
        "schema": "STAGE45F_PI0178_H200_TRAINING_EXECUTION_LEDGER_TEMPLATE_V1",
        "status": "TEMPLATE_ONLY",
        "append_only": True,
        "fresh_child_per_arm": True,
        "resume_requires": ["same_base_checkpoint_identity", "same_schedule_hash", "same_arm", "same_output_root", "sealed_optimizer_scheduler_rng_state"],
        "phases": [
            "PRE_SUBMIT",
            "ARTIFACT_ADMISSION",
            "PROCESSOR_BUILD",
            "MODEL_LOAD",
            "TRAINING_STEP_BEGIN",
            "BACKWARD_BEGIN",
            "OPTIMIZER_STEP_BEGIN",
            "CHECKPOINT_SEAL",
            "MODEL_RELEASE",
            "FINAL_SEAL",
        ],
        "per_event_fields": ["sequence", "timestamp_utc", "phase", "arm", "step", "elapsed_s", "rss_kib", "gpu_uuid", "free_vram_mib", "torch_max_memory_allocated", "torch_max_memory_reserved", "status", "error"],
        "execution_counters": {"training_executed": False, "optimizer_steps": 0, "checkpoint_writes": 0, "h200_submissions": 0},
    }


def _hash_bindings(repo_root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    records = []
    for relative, expected_sha in expected.items():
        path = regular_file(repo_root / relative, relative)
        observed = sha256_file(path)
        require(observed == expected_sha, f"source hash drift: {relative}: {observed}")
        records.append({"path": relative, "sha256": observed})
    return {"status": "PASS", "records": records}


def promotion_harness_plan(repo_root: Path) -> dict[str, Any]:
    stage43 = _hash_bindings(repo_root, STAGE43_SOURCE_FILES)
    promotion = _hash_bindings(repo_root, PROMOTION_SOURCE_FILES)
    return {
        "schema": "STAGE45F_PI0178_CANDIDATE_PROMOTION_HARNESS_V1",
        "status": "READY_NOT_EXECUTED",
        "fixed_dev_initial_conditions": [{"condition_id": name, "base_delta_xy_m": list(delta)} for name, delta in DEV_CASES],
        "dev_artifact_identities": DEV_ARTIFACT_IDENTITIES,
        "source_bindings": {"stage43": stage43, "promotion_chain": promotion},
        "model_side_scorecard": {"metrics": list(MODEL_SIDE_METRICS), "execution": False, "candidate_inference": False},
        "closed_loop_scorecard": {"metrics": list(CLOSED_LOOP_METRICS), "execution": False, "wbc_executed": False, "simulator_executed": False, "case_selection_after_result": False},
        "future_consumer_rule": "candidate checkpoint identity is supplied later; no best-checkpoint selection is performed here",
    }


def readiness_plan(repo_root: Path | None = None) -> dict[str, Any]:
    schedule = exposure_schedule()
    schedule_summary = validate_exposure_schedule(schedule)
    plan: dict[str, Any] = {
        "schema": "STAGE45F_PI0178_FAST_LOOP_READINESS_V1",
        "status": "READY_NOT_EXECUTED",
        "authority": {"issue_11_comment": PI_GATE, "pr_73_pointer": PR_POINTER, "source_head": SOURCE_HEAD, "support_branch": SUPPORT_BRANCH},
        "pi0177_runner_review": {
            "pr": PI0177_PR,
            "head": PI0177_HEAD,
            "branch": PI0177_BRANCH,
            "cpu_static": "PASS",
            "cpu_regression": "3/3 PASS",
            "plan": "PASS_READY_NOT_EXECUTED",
            "canary_contract": {"cohort": "FULL24", "window": "first frozen TRAIN24 window", "model_loads": 1, "conditioning": 1, "action_head": 4, "validation_e_model_calls": 0},
        },
        "visited_state": {"q0": {"condition": Q0_CONDITION, "target": Q0_TARGET, "npz_sha256": Q0_NPZ_SHA, "json_sha256": Q0_JSON_SHA, "expert_chunk_sha256": Q0_EXPERT_CHUNK_SHA, "payload_read_scope": "PI0163 C1-q0 only"}, "schedule": schedule_summary, "runspec": training_runspec(), "h200_ledger": h200_ledger_template()},
        "candidate_promotion": {"status": "LATE_BOUND_DEV_BINDING", "fixed_cases": [name for name, _ in DEV_CASES], "execution": False},
        "hard_zeroes": {"validation_e_model_calls": 0, "validation_e_new_payload_read": 0, "holdout_f_read": 0, "training": 0, "backward": 0, "optimizer_steps": 0, "checkpoint_writes": 0, "h200_submissions": 0, "pbs_submissions": 0, "wbc_steps": 0, "simulator_steps": 0, "hardware_steps": 0, "hf_checkpoint_transfer": 0, "scientific_winner_selection": 0},
    }
    if repo_root is not None:
        plan["candidate_promotion"] = promotion_harness_plan(Path(repo_root))
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--q0-audit", action="store_true")
    modes.add_argument("--promotion-audit", action="store_true")
    parser.add_argument("--q0-npz", type=Path)
    parser.add_argument("--q0-json", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(readiness_plan(args.repo), indent=2, sort_keys=True))
        return 0
    if args.q0_audit:
        require(args.q0_npz is not None and args.q0_json is not None, "q0 audit requires --q0-npz and --q0-json")
        print(json.dumps(audit_q0_bundle(args.q0_npz, args.q0_json), indent=2, sort_keys=True))
        return 0
    print(json.dumps(promotion_harness_plan(args.repo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
