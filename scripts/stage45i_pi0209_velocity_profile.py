#!/usr/bin/env python3
"""PI0209 velocity-profile closure on the accepted Stage45H training lane.

The Stage45H schedule, dataset factory, runtime bootstrap, and fresh child
process are reused.  This file only renames the three supervised arms and
installs the one frozen expert-relative XYZ velocity loss before model build.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45h_pi0208_medium as medium  # noqa: E402
from go2arx5_vla import stage45i_velocity_loss as velocity  # noqa: E402


SOURCE_SCHEDULE_SHA256 = "0c3a57ded84884d9f9c0eabb455bf7d277eb786bb7603eaaa1d2a638e6379663"
BASE_S25_TREE_SHA256 = medium.BASE_S25_TREE_SHA256
BASE_S25_TREE_PREFIX = medium.BASE_S25_TREE_PREFIX
BASE_S25_GLOBAL_STEP = medium.BASE_S25_GLOBAL_STEP
BASE_S25_LABEL = medium.BASE_S25_LABEL
FULL24_TREE_SHA256 = medium.FULL24_TREE_SHA256
STRUCTURED_SOURCE_STATS_SHA256 = medium.STRUCTURED_SOURCE_STATS_SHA256
UPDATE_COUNT = medium.UPDATE_COUNT
SUPPORT_UPDATES = medium.SUPPORT_UPDATES
SUPPORT_FRACTION = medium.SUPPORT_FRACTION
SEEDS = medium.SEEDS
ARM_ORDER = tuple(f"MSUPV_{seed}" for seed in SEEDS)
SCREEN_SURFACES = ("FULL24", "MEDIUM_SUPPORT_EVAL")
FULL24_COUNT = 24
SUPPORT_EVAL_COUNT = medium.EVAL_ROW_COUNT


class Stage45IError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage45IError(message)


def _source_arm(seed: int) -> str:
    return f"MSUP_{seed}"


def prepare_schedule(source_path: Path, output_path: Path) -> int:
    require(medium.sha256_file(source_path) == SOURCE_SCHEDULE_SHA256, "Stage45H schedule SHA drift")
    source = medium.read_json(source_path, canonical=True)
    require(source.get("schema") == "LOCALPC_PI0208_STAGE45H_MATCHED_SCHEDULES_V1", "Stage45H schedule schema drift")
    require(source.get("status") == "SEALED", "Stage45H schedule is not sealed")
    require(int(source.get("update_count", -1)) == UPDATE_COUNT, "update count drift")
    require(int(source.get("support_updates", -1)) == SUPPORT_UPDATES, "support update count drift")
    require(float(source.get("support_fraction", -1.0)) == SUPPORT_FRACTION, "support fraction drift")

    arms: dict[str, list[dict[str, Any]]] = {}
    source_arms: dict[str, str] = {}
    sequence_sha: dict[str, str] = {}
    for seed in SEEDS:
        old_arm = _source_arm(seed)
        new_arm = f"MSUPV_{seed}"
        slots = copy.deepcopy(source.get("arms", {}).get(old_arm))
        require(isinstance(slots, list) and len(slots) == UPDATE_COUNT, f"missing sealed arm: {old_arm}")
        require(sum(item.get("source") == "FULL24" for item in slots) == UPDATE_COUNT - SUPPORT_UPDATES, f"FULL24 count drift: {old_arm}")
        require(sum(item.get("source") == "STATE_SUPPORT_MEDIUM" for item in slots) == SUPPORT_UPDATES, f"support count drift: {old_arm}")
        for position, slot in enumerate(slots):
            require(int(slot.get("position", -1)) == position, f"schedule position drift: {old_arm}:{position}")
        arms[new_arm] = slots
        source_arms[new_arm] = old_arm
        sequence_sha[new_arm] = medium.canonical_sha(slots)

    schedule = {
        "schema": "LOCALPC_PI0209_STAGE45I_VELOCITY_PROFILE_SCHEDULE_V1",
        "status": "SEALED",
        "source_schedule_sha256": SOURCE_SCHEDULE_SHA256,
        "source_schedule_schema": source["schema"],
        "source_arm_mapping": source_arms,
        "arm_order": list(ARM_ORDER),
        "arm_sequence_sha256": sequence_sha,
        "seeds": list(SEEDS),
        "update_count": UPDATE_COUNT,
        "support_updates": SUPPORT_UPDATES,
        "support_fraction": SUPPORT_FRACTION,
        "row_count": int(source.get("row_count", -1)),
        "velocity_loss_contract": velocity.contract(),
        "validation_e_read": False,
        "holdout_f_read": False,
        "scientific_semantics_changed": ["expert_relative_physical_xyz_adjacent_step_velocity_auxiliary_loss"],
        "no_other_changes": True,
        "arms": arms,
    }
    medium.write_json(output_path, schedule)
    print(json.dumps({"status": "PASS", "output": str(output_path), "sha256": medium.sha256_file(output_path), "arms": list(ARM_ORDER)}, sort_keys=True))
    return 0


def _load_bounds(stats_path: Path) -> tuple[Any, Any, str]:
    document = medium.read_json(stats_path)
    stats = document.get("statistics", document)
    lower = stats.get("selected_lower")
    upper = stats.get("selected_upper")
    require(lower is not None and upper is not None, "structured normalization bounds are missing")
    import numpy as np

    lower_array = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper_array = np.asarray(upper, dtype=np.float32).reshape(-1)
    require(lower_array.shape == (132,) and upper_array.shape == (132,), "structured normalization bounds shape drift")
    return lower_array, upper_array, medium.sha256_file(stats_path)


def materialize_expert(label: Any) -> Any:
    """Materialize a frozen StructuredActionV1 label through the project codec."""

    import numpy as np
    from go2arx5_vla.structured_action import decode_structured_action, materialize_structured_action
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows

    intent = decode_structured_action(
        label,
        anchor_state_v1=label.anchor_state_v1,
        source_timestamps=label.source_timestamps,
        source_duration_s=label.source_duration_s,
        timing_mode=label.timing_mode,
    )
    chunk = materialize_structured_action(label, intent, source_timestamps=label.source_timestamps)
    rotations = rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation))
    physical = np.concatenate(
        [np.asarray(chunk.eef_position, dtype=np.float32), rotations.astype(np.float32), np.asarray(chunk.gripper, dtype=np.float32).reshape(40, 1)],
        axis=1,
    )
    require(physical.shape == (40, 10) and bool(np.isfinite(physical).all()), "materialized expert label is invalid")
    return physical


def _adapt_schedule(schedule: Mapping[str, Any], arm: str) -> dict[str, list[dict[str, Any]]]:
    require(arm in schedule.get("arms", {}), f"schedule arm missing: {arm}")
    return {arm: medium._stage45f_schedule(schedule["arms"][arm])}


def train_one(args: argparse.Namespace, schedule: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], arm: str, seed: int) -> dict[str, Any]:
    import stage45f_pi0183_matched_training as training

    output = args.output_root / arm
    if output.exists():
        result_path = output / "training_result.json"
        if result_path.is_file():
            result = medium.read_json(result_path)
            if result.get("status") == "PASS" and result.get("arm") == arm and int(result.get("global_step", -1)) == UPDATE_COUNT and result.get("velocity_loss_contract", {}).get("schema") == velocity.contract()["schema"]:
                return {"arm": arm, "status": "SKIPPED_EXISTING_PASS", "result_sha256": medium.sha256_file(result_path), "checkpoint_tree_sha256": result.get("checkpoint_tree_sha256")}
        raise Stage45IError(f"training output exists but is not a completed PASS: {output}")

    env = training.prepare_runtime(output, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=False)
    os_env = {
        "STAGE45F_FROZEN_STRUCTURED_STATS_PATH": str(args.stats),
        "STAGE45F_FROZEN_STRUCTURED_STATS_SHA256": STRUCTURED_SOURCE_STATS_SHA256,
        "STAGE45I_ARM": arm,
        "STAGE45I_SEED": str(seed),
        "STAGE45I_SUPPORT_ROOT": str(args.support_root),
        "STAGE45I_SCHEDULE_SHA256": medium.sha256_file(args.schedule),
    }
    os.environ.update(os_env)
    training.bind_cosmos_processor(args.cosmos_root)
    lower, upper, stats_sha = _load_bounds(args.stats)
    train_args = argparse.Namespace(**vars(args))
    train_args.arm = arm
    config = training.load_config(train_args, output, UPDATE_COUNT, seed=seed)
    training.install_scheduled_factory(
        config,
        train_args,
        _adapt_schedule(schedule, arm),
        rows,
        training.PROCESSOR_CONFIG_SHA256,
        velocity_context=velocity.set_current_expert_label,
        velocity_materializer=materialize_expert,
    )
    patch_record = velocity.install_velocity_loss_patch(lower=lower, upper=upper)
    velocity.reset_runtime_counters()
    sequence = list(schedule["arms"][arm])
    invocation = {
        "schema": "LOCALPC_PI0209_STAGE45I_VELOCITY_TRAINING_INVOCATION_V1",
        "status": "STARTED",
        "arm": arm,
        "seed": seed,
        "process_pid": os.getpid(),
        "updates": UPDATE_COUNT,
        "schedule_sha256": medium.sha256_file(args.schedule),
        "arm_sequence_sha256": medium.canonical_sha(sequence),
        "base_checkpoint_label": BASE_S25_LABEL,
        "base_checkpoint_global_step": BASE_S25_GLOBAL_STEP,
        "base_checkpoint_tree_sha256": BASE_S25_TREE_SHA256,
        "support_fraction": SUPPORT_FRACTION,
        "support_updates": SUPPORT_UPDATES,
        "runtime_environment": env,
        "velocity_loss_contract": velocity.contract(),
        "velocity_patch": patch_record,
        "normalization_stats_sha256": stats_sha,
        "science_contract": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "scheduler": "cosine",
            "warmup_steps": 10,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "precision": "bf16",
            "trainable_modules": ["projector", "diffusion"],
            "frozen_modules": ["visual", "llm", "vlln"],
            "fresh_process_optimizer_scheduler_rng": True,
        },
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    medium.write_json(output / "training_invocation.json", invocation)
    from gr00t.experiment.experiment import run as gr00t_run

    gr00t_run(config)
    checkpoint = output / f"checkpoint-{UPDATE_COUNT}"
    state = medium.read_json(checkpoint / "trainer_state.json")
    require(int(state.get("global_step", -1)) == UPDATE_COUNT, f"{arm} global step drift")
    counters = velocity.runtime_counters()
    require(counters == {"action_head_forwards": UPDATE_COUNT, "velocity_terms": UPDATE_COUNT}, f"{arm} velocity counters drift: {counters}")
    prefix = f"stage45i/{arm}/checkpoint-{UPDATE_COUNT}"
    checkpoint_tree, entries = training.tree_digest(checkpoint, prefix)
    result = {
        "schema": "LOCALPC_PI0209_STAGE45I_VELOCITY_TRAINING_RESULT_V1",
        "status": "PASS",
        "arm": arm,
        "seed": seed,
        "output_root": str(output),
        "final_checkpoint": str(checkpoint),
        "global_step": UPDATE_COUNT,
        "updates": UPDATE_COUNT,
        "schedule_sha256": medium.sha256_file(args.schedule),
        "arm_sequence_sha256": medium.canonical_sha(sequence),
        "base_checkpoint_label": BASE_S25_LABEL,
        "base_checkpoint_global_step": BASE_S25_GLOBAL_STEP,
        "base_checkpoint_tree_sha256": BASE_S25_TREE_SHA256,
        "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_tree_prefix": prefix,
        "checkpoint_file_count": len(entries),
        "checkpoint_bytes": sum(int(item["bytes"]) for item in entries),
        "model_forward": UPDATE_COUNT,
        "backward": UPDATE_COUNT,
        "optimizer_steps": UPDATE_COUNT,
        "support_updates": SUPPORT_UPDATES,
        "support_fraction": SUPPORT_FRACTION,
        "velocity_loss_contract": velocity.contract(),
        "velocity_patch": patch_record,
        "runtime_counters": counters,
        "identities": {"base_checkpoint": {"label": BASE_S25_LABEL, "global_step": BASE_S25_GLOBAL_STEP, "tree_sha256": BASE_S25_TREE_SHA256}, "structured_stats_sha256": STRUCTURED_SOURCE_STATS_SHA256, "full24_tree_sha256": FULL24_TREE_SHA256},
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    medium.write_json(output / "training_result.json", result)
    return {"arm": arm, "status": "PASS", "checkpoint_tree_sha256": checkpoint_tree, "result_sha256": medium.sha256_file(output / "training_result.json")}


def _training_command(args: argparse.Namespace, arm: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "train-arm",
        "--arm", arm,
        "--schedule", str(args.schedule),
        "--rows", str(args.rows),
        "--support-root", str(args.support_root),
        "--output-root", str(args.output_root),
        "--dataset", str(args.dataset),
        "--base-model", str(args.base_model),
        "--groot-source", str(args.groot_source),
        "--groot-manifest", str(args.groot_manifest),
        "--cosmos-root", str(args.cosmos_root),
        "--modality", str(args.modality),
        "--config", str(args.config),
        "--stats", str(args.stats),
        "--ffmpeg-root", str(args.ffmpeg_root),
    ]


def train_all(args: argparse.Namespace) -> int:
    schedule = medium.read_json(args.schedule, canonical=True)
    require(schedule.get("schema") == "LOCALPC_PI0209_STAGE45I_VELOCITY_PROFILE_SCHEDULE_V1", "Stage45I schedule schema drift")
    rows = medium.load_rows(args.rows, args.support_root)
    identities = medium._validate_training_inputs(args, rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in ARM_ORDER:
        process = subprocess.run(_training_command(args, arm), cwd=str(REPO), env=os.environ.copy(), check=False)
        require(process.returncode == 0, f"{arm} fresh training process failed: rc={process.returncode}")
        result_path = args.output_root / arm / "training_result.json"
        result = medium.read_json(result_path)
        require(result.get("status") == "PASS" and result.get("arm") == arm and int(result.get("global_step", -1)) == UPDATE_COUNT, f"{arm} child result is not PASS")
        records.append({"arm": arm, "status": "PASS", "checkpoint_tree_sha256": result.get("checkpoint_tree_sha256"), "result_sha256": medium.sha256_file(result_path)})
    result = {
        "schema": "LOCALPC_PI0209_STAGE45I_VELOCITY_TRAINING_RESULT_SET_V1",
        "status": "PASS",
        "arm_order": list(ARM_ORDER),
        "records": records,
        "identities": identities,
        "schedule_sha256": medium.sha256_file(args.schedule),
        "updates_per_arm": UPDATE_COUNT,
        "total_forward": len(ARM_ORDER) * UPDATE_COUNT,
        "total_backward": len(ARM_ORDER) * UPDATE_COUNT,
        "total_optimizer_steps": len(ARM_ORDER) * UPDATE_COUNT,
        "velocity_loss_contract": velocity.contract(),
        "Validation-E": 0,
        "Holdout-F": 0,
    }
    medium.write_json(args.output_root / "training_result_set.json", result, overwrite=True)
    print(json.dumps({"status": "PASS", "output": str(args.output_root / "training_result_set.json"), "arms": len(records), "updates_per_arm": UPDATE_COUNT}, sort_keys=True))
    return 0


def train_arm(args: argparse.Namespace) -> int:
    require(args.arm in ARM_ORDER, f"unknown Stage45I arm: {args.arm}")
    schedule = medium.read_json(args.schedule, canonical=True)
    rows = medium.load_rows(args.rows, args.support_root)
    result = train_one(args, schedule, rows, args.arm, int(args.arm.split("_")[-1]))
    print(json.dumps({"status": "PASS", "arm": args.arm, "result": result}, sort_keys=True))
    return 0


def preflight(args: argparse.Namespace) -> int:
    schedule = medium.read_json(args.schedule, canonical=True)
    require(schedule.get("schema") == "LOCALPC_PI0209_STAGE45I_VELOCITY_PROFILE_SCHEDULE_V1", "Stage45I schedule schema drift")
    require(schedule.get("arm_order") == list(ARM_ORDER), "Stage45I arm order drift")
    rows = medium.load_rows(args.rows, args.support_root)
    identities = medium._validate_training_inputs(args, rows)
    lower, upper, stats_sha = _load_bounds(args.stats)
    require(stats_sha == STRUCTURED_SOURCE_STATS_SHA256, "structured stats SHA drift")
    record = {
        "schema": "LOCALPC_PI0209_STAGE45I_STATIC_PREFLIGHT_V1",
        "status": "PASS",
        "schedule_sha256": medium.sha256_file(args.schedule),
        "source_schedule_sha256": schedule.get("source_schedule_sha256"),
        "arm_order": list(ARM_ORDER),
        "arm_sequence_sha256": schedule.get("arm_sequence_sha256"),
        "identities": identities,
        "velocity_loss_contract": velocity.contract(),
        "normalization_bounds_sha256": {"stats": stats_sha, "lower": velocity._array_sha256(lower), "upper": velocity._array_sha256(upper)},
        "model_forward": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    medium.write_json(args.output, record)
    print(json.dumps({"status": "PASS", "output": str(args.output), "schedule_sha256": record["schedule_sha256"], "rows": len(rows)}, sort_keys=True))
    return 0


def _expert_gate_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Check frozen expert labels only; this never constructs a model."""

    import numpy as np
    import stage45c_h200_eval_runtime_adapter as stage45c
    import stage45g_pi0208_hopper_model_screen as screen_impl

    records: list[tuple[str, Mapping[str, Any]]] = []
    for snapshot in stage45c._verify_snapshots(args.snapshot_root):
        item = dict(snapshot)
        item["surface"] = "FULL24"
        records.append((str(snapshot["window_id"]), item))
    rows = medium.load_rows(args.rows, args.support_root, minimum=1)
    for row in rows.values():
        row_id = str(row["row_id"])
        visited_path = medium._row_file(args.support_root, row, "visited_row_npz", f"expert audit visited {row_id}")
        expert_path = medium._row_file(args.support_root, row, "expert_label_npz", f"expert audit label {row_id}")
        with np.load(visited_path, allow_pickle=False) as visited:
            state = np.asarray(visited["state_v1"], dtype=np.float64).reshape(-1)
            anchor = np.asarray(visited["anchor_world_m"], dtype=np.float64).reshape(-1)
            target = np.asarray(visited["target_world_m"], dtype=np.float64).reshape(-1)
        with np.load(expert_path, allow_pickle=False) as expert_archive:
            expert = np.asarray(expert_archive["expert_action_structured_v1"], dtype=np.float64)
            timestamps = np.asarray(expert_archive["source_timestamps"], dtype=np.float64)
        require(state.shape == (10,) and anchor.shape == (3,) and target.shape == (3,), f"expert audit state drift: {row_id}")
        require(expert.shape == (40, 10) and timestamps.shape == (40,), f"expert audit label drift: {row_id}")
        records.append((row_id, {"surface": "MEDIUM_SUPPORT_EVAL", "window_id": row_id, "state_v1": state.tolist(), "anchor_world_m": anchor.tolist(), "target_world_m": target.tolist(), "source_timestamps": timestamps.tolist(), "observation_timestamp": float(row["observation_timestamp_s"]), "expert_physical_action": expert.tolist()}))
    require(len(records) == FULL24_COUNT + SUPPORT_EVAL_COUNT, f"expert audit count drift: {len(records)}")
    decisions: Counter[str] = Counter()
    issue_codes: Counter[str] = Counter()
    for chunk_id, snapshot in records:
        _profile, gate = screen_impl._profile_gate(np.asarray(snapshot["expert_physical_action"], dtype=np.float64), snapshot, f"expert:{chunk_id}")
        decision = str(gate["decision"])
        decisions[decision] += 1
        for issue in gate.get("issues", []):
            issue_codes[str(issue.get("code"))] += 1
    return {"schema": "LOCALPC_PI0209_EXPERT_LABEL_SAFETY_AUDIT_V1", "count": len(records), "decision_counts": dict(sorted(decisions.items())), "issue_code_counts": dict(sorted(issue_codes.items())), "all_accept": decisions == Counter({"ACCEPT": len(records)}), "model_forward": 0, "WBC": 0, "simulator": 0}


def screen_arm(args: argparse.Namespace) -> int:
    """Run one candidate through the accepted Stage45H screen implementation."""

    import stage45h_pi0208_medium as medium_impl

    require(args.arm in ARM_ORDER, f"unknown Stage45I screen arm: {args.arm}")
    original_order = medium_impl.ARM_ORDER
    original_rule = medium_impl._medium_screen_rule
    medium_impl.ARM_ORDER = (args.arm,)
    medium_impl._medium_screen_rule = lambda summaries: {"classification": "CANDIDATE_ONLY", "arm": args.arm}
    try:
        rc = medium_impl.screen(args)
    finally:
        medium_impl.ARM_ORDER = original_order
        medium_impl._medium_screen_rule = original_rule
    require(rc == 0, "reused Stage45H screen failed")
    output = args.output_root.resolve()
    result_path = output / "screen_result.json"
    result = medium.read_json(result_path, canonical=True)
    expert_audit = _expert_gate_audit(args)
    result.update({"schema": "LOCALPC_PI0209_STAGE45I_HOPPER_SCREEN_ARM_V1", "candidate_arm": args.arm, "expert_gate_audit": expert_audit, "execution_head": args.expected_head})
    medium.write_json(result_path, result, overwrite=True)
    seal_path = output / "screen_seal.json"
    seal = dict(result)
    seal["terminal_token"] = "LOCALPC_PI0209_HOPPER_SCREEN_ARM — PASS"
    medium.write_json(seal_path, seal, overwrite=True)
    print(json.dumps({"status": "PASS", "arm": args.arm, "expert_all_accept": expert_audit["all_accept"], "screen_result": str(result_path)}, sort_keys=True))
    return 0


def _row_map(root: Path, arm: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows_root = root / "rows" / arm
    medium.regular_dir(rows_root, f"screen rows {arm}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(rows_root.glob("*.json")):
        row = medium.read_json(path, canonical=True)
        key = (str(row.get("surface")), str(row.get("window_id")))
        require(key not in result, f"duplicate screen row: {arm}:{key}")
        result[key] = row
    require(len(result) == FULL24_COUNT + SUPPORT_EVAL_COUNT, f"screen row count drift: {arm}={len(result)}")
    return result


def _reject_codes(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> set[str]:
    return {str(issue.get("code")) for row in rows.values() for issue in row.get("safety_gate", {}).get("issues", []) if str(issue.get("severity")) == "REJECT"}


def aggregate_screen(args: argparse.Namespace) -> int:
    import numpy as np

    control_path = args.control_screen.resolve()
    control = medium.read_json(control_path, canonical=True)
    require(control.get("status") == "PASS", "historical control screen is not PASS")
    control_root = control_path.parent
    old_sha = medium.sha256_file(control_path)
    per_seed: dict[str, Any] = {}
    pooled = {"MCTRL": 0, "MSUPV": 0}
    expert_audits: dict[str, Any] = {}
    candidate_records: dict[str, Any] = {}
    for seed in SEEDS:
        control_arm = f"MCTRL_{seed}"
        accepted_arm = f"MSUP_{seed}"
        candidate_arm = f"MSUPV_{seed}"
        candidate_root = args.candidate_root.resolve() / candidate_arm
        candidate_result_path = candidate_root / "screen_result.json"
        candidate = medium.read_json(candidate_result_path, canonical=True)
        require(candidate.get("status") == "PASS" and candidate.get("candidate_arm") == candidate_arm, f"candidate screen result drift: {candidate_arm}")
        candidate_summary = candidate.get("summaries", {}).get(candidate_arm)
        control_summaries = control.get("summaries", {})
        mctrl_summary = control_summaries.get(control_arm)
        accepted_summary = control_summaries.get(accepted_arm)
        require(isinstance(candidate_summary, Mapping) and isinstance(mctrl_summary, Mapping) and isinstance(accepted_summary, Mapping), f"screen summary missing: {seed}")
        candidate_rows = _row_map(candidate_root, candidate_arm)
        control_rows = _row_map(control_root, control_arm)
        candidate_codes = _reject_codes(candidate_rows)
        control_codes = _reject_codes(control_rows)
        new_codes = sorted(candidate_codes - control_codes)
        expert = candidate.get("expert_gate_audit", {})
        require(int(expert.get("count", -1)) == FULL24_COUNT + SUPPORT_EVAL_COUNT, f"expert audit count drift: {candidate_arm}")
        expert_audits[candidate_arm] = expert
        support = candidate_summary["MEDIUM_SUPPORT_EVAL"]
        mctrl_support = mctrl_summary["MEDIUM_SUPPORT_EVAL"]
        accepted_support = accepted_summary["MEDIUM_SUPPORT_EVAL"]
        full = candidate_summary["FULL24"]
        mctrl_full = mctrl_summary["FULL24"]
        candidate_rejects = sum(int(candidate_summary[surface]["safety_gate_counts"]["REJECT"]) for surface in SCREEN_SURFACES)
        mctrl_rejects = sum(int(mctrl_summary[surface]["safety_gate_counts"]["REJECT"]) for surface in SCREEN_SURFACES)
        pooled["MSUPV"] += candidate_rejects
        pooled["MCTRL"] += mctrl_rejects
        checks = {
            "support_trajectory_lower_than_mctrl": float(support["mean_trajectory_error_m"]) < float(mctrl_support["mean_trajectory_error_m"]),
            "support_endpoint_lower_than_mctrl": float(support["mean_endpoint_error_m"]) < float(mctrl_support["mean_endpoint_error_m"]),
            "support_trajectory_within_1_02x_accepted_msup": float(support["mean_trajectory_error_m"]) <= 1.02 * float(accepted_support["mean_trajectory_error_m"]),
            "support_endpoint_within_1_02x_accepted_msup": float(support["mean_endpoint_error_m"]) <= 1.02 * float(accepted_support["mean_endpoint_error_m"]),
            "full24_trajectory_within_1_10x_mctrl": float(full["mean_trajectory_error_m"]) <= 1.10 * float(mctrl_full["mean_trajectory_error_m"]),
            "full24_endpoint_within_1_10x_mctrl": float(full["mean_endpoint_error_m"]) <= 1.10 * float(mctrl_full["mean_endpoint_error_m"]),
            "finite_and_abi_valid": int(support["finite_count"]) == SUPPORT_EVAL_COUNT and int(full["finite_count"]) == FULL24_COUNT and int(support["abi_valid_count"]) == SUPPORT_EVAL_COUNT and int(full["abi_valid_count"]) == FULL24_COUNT,
            "matched_rejects_not_above_mctrl": candidate_rejects <= mctrl_rejects,
            "no_new_reject_mechanism": not new_codes,
            "expert_88_of_88_accept": bool(expert.get("all_accept")),
        }
        per_seed[str(seed)] = {"candidate_arm": candidate_arm, "candidate_tree_sha256": candidate.get("candidate_records", {}).get(candidate_arm, {}).get("tree_sha256"), "checks": checks, "positive": all(checks.values()), "candidate_rejects": candidate_rejects, "mctrl_rejects": mctrl_rejects, "new_reject_issue_codes": new_codes, "candidate_summary": candidate_summary, "mctrl_summary": mctrl_summary, "accepted_msup_summary": accepted_summary}
        candidate_records[candidate_arm] = candidate.get("candidate_records", {}).get(candidate_arm, {})
    positive = sum(bool(item["positive"]) for item in per_seed.values())
    pooled_checks = {"at_least_2_of_3_seeds": positive >= 2, "pooled_rejects_not_above_mctrl": pooled["MSUPV"] <= pooled["MCTRL"], "expert_labels_88_of_88_accept": all(bool(item.get("all_accept")) for item in expert_audits.values())}
    classification = "STAGE45I_HOPPER_SCREEN_QUALIFIED" if all(pooled_checks.values()) else "STAGE45I_HOPPER_SCREEN_NOT_QUALIFIED"
    result = {"schema": "LOCALPC_PI0209_STAGE45I_HOPPER_SCREEN_V1", "status": "PASS", "classification": classification, "execution_head": args.expected_head, "control_screen_sha256": old_sha, "control_screen_path": str(control_path), "candidate_records": candidate_records, "per_seed": per_seed, "pooled": pooled, "pooled_checks": pooled_checks, "expert_audits": expert_audits, "screen_rule": {"support": "candidate must improve trajectory and endpoint against matched MCTRL and remain within 1.02x accepted MSUP", "full24": "candidate must remain within 1.10x matched MCTRL", "safety": "matched reject count must not increase and no new reject mechanism", "qualification": "at least 2/3 seeds plus pooled reject and 88/88 expert-label acceptance"}, "scientific_zeroes": {"training": 0, "backward": 0, "optimizer_steps": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}}
    output = args.output_root.resolve()
    require(not output.exists() and not output.is_symlink(), f"aggregate output must be fresh: {output}")
    medium.write_json(output / "screen_result.json", result)
    result["terminal_token"] = "LOCALPC_PI0209_HOPPER_SCREEN — " + ("QUALIFIED" if classification.endswith("QUALIFIED") else "NOT_QUALIFIED")
    medium.write_json(output / "screen_seal.json", result)
    print(json.dumps({"status": "PASS", "classification": classification, "positive_seed_count": positive, "pooled": pooled}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("schedule", "preflight", "train", "train-arm", "screen-arm", "aggregate-screen"))
    parser.add_argument("--source-schedule", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--groot-manifest", type=Path)
    parser.add_argument("--cosmos-root", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--ffmpeg-root", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--control-screen", type=Path)
    parser.add_argument("--expected-head")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "schedule":
            require(args.source_schedule is not None and args.output is not None, "--source-schedule and --output required")
            return prepare_schedule(args.source_schedule, args.output)
        if args.mode == "preflight":
            for name in ("output", "schedule", "rows", "support_root", "dataset", "base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
                require(getattr(args, name) is not None, f"--{name.replace('_', '-')} required")
            return preflight(args)
        if args.mode == "train-arm":
            for name in ("arm", "schedule", "rows", "support_root", "output_root", "dataset", "base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
                require(getattr(args, name) is not None, f"--{name.replace('_', '-')} required")
            return train_arm(args)
        if args.mode == "screen-arm":
            for name in ("arm", "output_root", "training_root", "dataset", "snapshot_root", "rows", "support_root", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root", "expected_head"):
                require(getattr(args, name) is not None, f"--{name.replace('_', '-')} required")
            return screen_arm(args)
        if args.mode == "aggregate-screen":
            for name in ("candidate_root", "control_screen", "output_root", "expected_head"):
                require(getattr(args, name) is not None, f"--{name.replace('_', '-')} required")
            return aggregate_screen(args)
        for name in ("schedule", "rows", "support_root", "output_root", "dataset", "base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
            require(getattr(args, name) is not None, f"--{name.replace('_', '-')} required")
        return train_all(args)
    except Exception as exc:
        print(json.dumps({"schema": "LOCALPC_PI0209_STAGE45I_FAILURE_V1", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
