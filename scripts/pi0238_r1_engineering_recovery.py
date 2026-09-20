#!/usr/bin/env python3
"""PI0238-R1 bounded engineering recovery runner.

The wrapper only audits sealed artifacts and runtime contracts.  It never
loads GR00T/VLM weights, calls get_action, trains, runs WBC, or starts a sim.
"""

from __future__ import annotations

import argparse
import csv
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import pi0238_audit_to_sim as legacy


R2_PROTOCOL_COMMIT = "9f766201420680956967c6ffc5ffea71cb4990d4"
R2_AUDITED_PARENT = "e4817bbae47919687b3748eea5e536b41e04c972"
R1_PROTOCOL_COMMIT = "b940fc9c1186ce01e14c13638f251f1e5e051b76"
R1_AUDITED_PARENT = "d553ca3a9f1145f72e131f2d66aefd5c32200f1e"
ORIGINAL_PROTOCOL_COMMIT = "044e4d5ef8b4c1effb8236190c0eb083fa49979b"
PI0237_PROTOCOL_COMMIT = "6e83c5494772190f5c8082f730f06723d9c9bfda"
FRESH_JOB = "623593.hopper-m-02"
SEED = 22301
HORIZON = 40
DT = 0.05
UNIVERSAL_DIM = 132
PHYSICAL_DIM = 10
PROFILE_STEPS = (1000, 2500, 5000, 7500, 10000)
P237_MODALITY_SHA = "244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25"
P237_TRAIN_TREE = "66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81"
P237_CALIB_TREE = "f53e85e8245735ec0f9a3ee9d1008d20385a4cfd8100eb66a15c015d7ecaf371"
PHYSICAL_TOL = {"xyz_m": 1e-5, "so3_rad": 1e-4, "gripper": 1e-5, "normalized": 1e-5}
QSTAT = "/cm/shared/apps/pbspro/current/bin/qstat"


class R1Blocker(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R1Blocker(message)


sha256_file = legacy.sha256_file
sha256_array = legacy.sha256_array
json_load = legacy.json_load
write_json = legacy.write_json
stats = legacy.stats
source_rollout = legacy.source_rollout
full_metadata = legacy.full_metadata
episode_rows = legacy.episode_rows
load_window = legacy.load_window
tree_sha256 = legacy.tree_sha256


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = subprocess.run(
        argv, cwd=cwd, env=None if env is None else dict(env),
        capture_output=True, text=True, check=False
    )
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def pbs_job_lines(text: str, own_job: str | None = None) -> list[str]:
    own_prefix = own_job.split(".", 1)[0] if own_job else None
    return [
        line for line in text.splitlines()
        if re.match(r"^\s*\d+\.[A-Za-z0-9_.-]+\b", line)
        and (not own_prefix or not re.match(rf"^\s*{re.escape(own_prefix)}(?:[.\s-])", line))
    ]


def path_no_symlink(path: Path, *, allow_missing: bool = False) -> Path:
    path = Path(path)
    require(path.is_absolute(), f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise R1Blocker(f"symlink path component is not allowed: {current}")
        if not current.exists() and not allow_missing:
            raise R1Blocker(f"path component is missing: {current}")
        if not current.exists():
            break
    return path.resolve(strict=not allow_missing)


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def cfg_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in cfg["paths"].items() if isinstance(value, str)}


def load_config(path: Path) -> dict[str, Any]:
    payload = json_load(path)
    require(payload.get("schema") == "PI0238_R2_RESOLVED_LAUNCH_V1", "resolved_launch schema drift")
    require(payload.get("protocol_commit") == R2_PROTOCOL_COMMIT, "R2 protocol commit drift")
    require(payload.get("audited_parent") == R2_AUDITED_PARENT, "R2 audited parent drift")
    return payload


def config_namespace(cfg: Mapping[str, Any], phase: str) -> argparse.Namespace:
    paths = cfg_paths(cfg)
    values = {
        key: paths[key]
        for key in (
            "output_root", "cache_root", "tmp_root", "p237_root", "run_root",
            "train_dataset", "calib_dataset", "prep_root", "vla_source",
            "stage10_root", "groot_source", "baseline", "fresh_root",
            "cosmos_snapshot", "cosmos_cache_root", "modality_config",
            "protocol", "original_protocol", "script", "pbs", "launch_config",
        )
        if key in paths
    }
    values.update({"phase": phase, "config": cfg})
    return argparse.Namespace(**values)


def paths_for_make(args: argparse.Namespace) -> dict[str, Path]:
    names = (
        "output_root", "cache_root", "tmp_root", "p237_root", "run_root",
        "train_dataset", "calib_dataset", "prep_root", "vla_source",
        "stage10_root", "groot_source", "baseline", "fresh_root",
        "cosmos_snapshot", "cosmos_cache_root", "modality_config",
        "protocol", "original_protocol", "script", "pbs", "launch_config",
    )
    result = {}
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            result[name] = Path(value).resolve()
    require(
        len(result) == len(names),
        f"resolved_launch missing path fields: {sorted(set(names) - set(result))}",
    )
    return result
def raw_rot6d_quality(action: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(action, dtype=np.float64).reshape(-1, PHYSICAL_DIM)
    a, b = raw[:, 3:6], raw[:, 6:9]
    cross = np.cross(a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cross_norm = np.linalg.norm(cross, axis=1)
    return {
        "rows": int(len(raw)),
        "norm_a": stats(norm_a),
        "norm_b": stats(norm_b),
        "unit_norm_deviation_count": int(np.count_nonzero(
            (np.abs(norm_a - 1.0) > 1e-3) | (np.abs(norm_b - 1.0) > 1e-3)
        )),
        "abs_dot": stats(np.abs(np.sum(a * b, axis=1))),
        "cross_norm": stats(cross_norm),
        "raw_degenerate_count": int(np.count_nonzero(
            (norm_a < 1e-8) | (norm_b < 1e-8) | (cross_norm < 1e-8)
        )),
        "measured_before_orthogonalization": True,
    }


def provenance_index(path: Path, key: str = "dataset_episode_index") -> dict[int, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row[key])] = row
    return result


def runtime_modules(args: argparse.Namespace) -> dict[str, Any]:
    sys.path[:0] = [
        str(args.prep_root / "scripts"),
        str(args.vla_source / "src"),
        str(args.stage10_root / "src" / "isaac-gr00t"),
    ]
    import stage21_same_scene_groot_inference as stage21
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import (
        geodesic_angle, matrix_to_quat, rot6d_rows_to_matrix,
    )
    selector_path = args.prep_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py"
    spec = importlib.util.spec_from_file_location("pi0238_r1_selector", selector_path)
    require(spec is not None and spec.loader is not None, "selector import failed")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    return {
        "stage21": stage21, "selector": selector,
        "gate": TrajectorySafetyGateV1, "TrajectoryChunkV1": TrajectoryChunkV1,
        "matrix_to_quat": matrix_to_quat,
        "rot6d_rows_to_matrix": rot6d_rows_to_matrix,
        "geodesic_angle": geodesic_angle,
    }


def shifted_chunk(chunk: Any, shift: float, chunk_type: Any) -> Any:
    return chunk_type(
        chunk_id=chunk.chunk_id, source=chunk.source,
        observation_timestamp=float(chunk.observation_timestamp) + shift,
        point_timestamps=np.asarray(chunk.point_timestamps) + shift,
        eef_position=np.asarray(chunk.eef_position).copy(),
        eef_orientation=np.asarray(chunk.eef_orientation).copy(),
        source_frame=chunk.source_frame, tcp_semantic_id=chunk.tcp_semantic_id,
        validity_start=float(chunk.validity_start) + shift,
        validity_end=float(chunk.validity_end) + shift,
        gripper=None if chunk.gripper is None else np.asarray(chunk.gripper).copy(),
        eef_rot6d=None if chunk.eef_rot6d is None else np.asarray(chunk.eef_rot6d).copy(),
        provenance=dict(chunk.provenance),
    )


def g2(args: argparse.Namespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    modules = runtime_modules(args)
    gate = modules["gate"]()
    selector = modules["selector"]
    expert_path = args.p237_root / "evidence" / "pi0237_p2_expert_gate.json"
    expert = json_load(expert_path)
    calib_meta = full_metadata(args.calib_dataset, "CALIB", 68)
    train_meta = full_metadata(args.train_dataset, "TRAIN", 612)
    prov_calib = provenance_index(
        args.calib_dataset / "meta" / "stage19_episode_provenance.jsonl"
    )
    time_calib = provenance_index(
        args.calib_dataset / "meta" / "stage19_action_timestamps.jsonl"
    )
    casebook = []
    all_rows = []
    counts = Counter()
    for row in expert["rows"]:
        if row.get("split") != "CALIB" or row.get("decision") != "REJECT":
            continue
        source_id = str(row["source_episode_id"])
        state, action, timestamps = load_window(args.calib_dataset, row)
        state_matrix = modules["rot6d_rows_to_matrix"](state[3:9].reshape(1, 6))[0]
        action_matrix = modules["rot6d_rows_to_matrix"](action[:, 3:9])
        chunk = selector._build_chunk(
            action[None, ...], chunk_id=f"pi0238-r1-case-{int(row['episode_index']):04d}"
        )
        original = gate.evaluate(
            chunk, now=0.0, achieved_position=state[:3],
            achieved_orientation=modules["matrix_to_quat"](state_matrix),
        )
        shifted = shifted_chunk(chunk, 17.0, modules["TrajectoryChunkV1"])
        shifted_result = gate.evaluate(
            shifted, now=17.0, achieved_position=state[:3],
            achieved_orientation=modules["matrix_to_quat"](state_matrix),
        )
        invariant = all(
            np.isclose(
                float(original.metrics.get(key)),
                float(shifted_result.metrics.get(key)),
                rtol=0.0, atol=1e-10,
            )
            for key in (
                "linear_speed_mps", "angular_speed_radps",
                "initial_rotation_discontinuity_rad",
            )
        )
        provenance = prov_calib.get(int(row["episode_index"]), {})
        timing = time_calib.get(int(row["episode_index"]), {})
        casebook.append(
            {
                "episode_index": int(row["episode_index"]),
                "source_episode_id": source_id,
                "source_rollout_id": source_rollout(source_id),
                "metadata": calib_meta[source_id],
                "source_state_rot6d": state[3:9].tolist(),
                "source_action0_rot6d": action[0, 3:9].tolist(),
                "state_rotation_matrix": state_matrix.tolist(),
                "action0_rotation_matrix": action_matrix[0].tolist(),
                "state_to_action0_angle_rad": float(
                    modules["geodesic_angle"](state_matrix, action_matrix[0])
                ),
                "raw_rot6d_quality": raw_rot6d_quality(action),
                "source_frame": provenance.get(
                    "source_frame", "NOT_OBSERVED_IN_SEALED_ARTIFACT"
                ),
                "tcp_semantic_id": provenance.get(
                    "tcp_semantic_id", "NOT_OBSERVED_IN_SEALED_ARTIFACT"
                ),
                "desired_label": provenance.get(
                    "desired_label", "NOT_OBSERVED_IN_SEALED_ARTIFACT"
                ),
                "source_native_indices": provenance.get(
                    "source_native_indices", "NOT_OBSERVED_IN_SEALED_ARTIFACT"
                ),
                "synchronization_metrics": provenance.get(
                    "synchronization_metrics", "NOT_OBSERVED_IN_SEALED_ARTIFACT"
                ),
                "action_timestamp_record": {
                    key: timing.get(key, "NOT_OBSERVED_IN_SEALED_ARTIFACT")
                    for key in (
                        "observation_timestamp", "future_action_point_timestamps",
                        "future_source_desired_indices", "label_source",
                        "achieved_role", "timestamp_authority",
                    )
                },
                "gate_original": {
                    "decision": original.decision.value,
                    "issues": [issue.code for issue in original.issues],
                    "metrics": original.metrics,
                },
                "clock_origin_shift": {
                    "shift_s": 17.0,
                    "decision": shifted_result.decision.value,
                    "linear_speed_mps": shifted_result.metrics.get("linear_speed_mps"),
                    "angular_speed_radps": shifted_result.metrics.get("angular_speed_radps"),
                    "initial_rotation_discontinuity_rad": shifted_result.metrics.get(
                        "initial_rotation_discontinuity_rad"
                    ),
                    "invariant_speed_rotation": invariant,
                    "age_shifted_only": True,
                },
            }
        )
    require(len(casebook) == 12, f"rotation reject casebook count drift: {len(casebook)}")
    for split, dataset, rows, metadata in (
        ("CALIB", args.calib_dataset, episode_rows(args.calib_dataset), calib_meta),
        ("TRAIN", args.train_dataset, episode_rows(args.train_dataset), train_meta),
    ):
        for row in rows:
            state, action, timestamps = load_window(dataset, row)
            state_matrix = modules["rot6d_rows_to_matrix"](state[3:9].reshape(1, 6))[0]
            result = gate.evaluate(
                selector._build_chunk(
                    action[None, ...],
                    chunk_id=f"pi0238-r1-{split.lower()}-{int(row['episode_index']):05d}",
                ),
                now=0.0,
                achieved_position=state[:3],
                achieved_orientation=modules["matrix_to_quat"](state_matrix),
            )
            counts[result.decision.value] += 1
            all_rows.append(
                {
                    "split": split,
                    "episode_index": int(row["episode_index"]),
                    "source_episode_id": row["source_episode_id"],
                    "source_rollout_id": source_rollout(str(row["source_episode_id"])),
                    "target_family": metadata[row["source_episode_id"]]["target_family"],
                    "jitter_family": metadata[row["source_episode_id"]]["jitter_family"],
                    "phase": metadata[row["source_episode_id"]]["phase"],
                    "decision": result.decision.value,
                    "issue_codes": [issue.code for issue in result.issues],
                    "state_sha256": sha256_array(state),
                    "action_sha256": sha256_array(action),
                    "timestamp_sha256": sha256_array(timestamps),
                    "raw_rot6d_quality": raw_rot6d_quality(action),
                    "initial_rotation_rad": result.metrics.get(
                        "initial_rotation_discontinuity_rad"
                    ),
                    "linear_speed_mps": result.metrics.get("linear_speed_mps"),
                    "angular_speed_radps": result.metrics.get("angular_speed_radps"),
                }
            )
    complete = (
        len(all_rows) == len(episode_rows(args.train_dataset)) + len(episode_rows(args.calib_dataset))
        and len(casebook) == 12
        and all(item["clock_origin_shift"]["invariant_speed_rotation"] for item in casebook)
    )
    census = {
        "schema": "PI0238_R1_G2_EXPERT_GATE_CENSUS_V2",
        "execution_status": "COMPLETE" if complete else "BLOCKED",
        "scientific_status": "DESCRIPTIVE_ONLY",
        "counts": dict(counts),
        "total_rows": len(all_rows),
        "train_rows": len(episode_rows(args.train_dataset)),
        "calib_rows": len(episode_rows(args.calib_dataset)),
        "casebook_reject_count": len(casebook),
        "clock_origin_control": (
            "PASS"
            if all(item["clock_origin_shift"]["invariant_speed_rotation"] for item in casebook)
            else "FAIL"
        ),
        "raw_rot6d_measured_before_orthogonalization": True,
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
    }
    write_json(
        paths["output_root"] / "evidence" / "g2_rotation_casebook.json",
        {
            "schema": "PI0238_R1_G2_ROTATION_CASEBOOK_V2",
            "execution_status": census["execution_status"],
            "rows": casebook,
            "missingness": "NOT_OBSERVED_IN_SEALED_ARTIFACT where explicitly marked",
        },
    )
    write_json(paths["output_root"] / "evidence" / "g2_full_label_gate_census.json", census)
    with (paths["output_root"] / "evidence" / "g2_full_label_gate_census.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in all_rows for key in row if key != "issue_codes"})
        fields += ["issue_codes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            output = dict(row)
            output["issue_codes"] = ";".join(row["issue_codes"])
            output["raw_rot6d_quality"] = json.dumps(
                row["raw_rot6d_quality"], sort_keys=True
            )
            writer.writerow(output)
    return census
def describe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first": describe(value[0]) if value else None,
        }
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__, "value": str(value)[:100]}


def batch_inputs(batch: Any) -> Mapping[str, Any]:
    if isinstance(batch, Mapping) and "inputs" in batch:
        inputs = batch["inputs"]
    else:
        inputs = getattr(batch, "data", {}).get("inputs")
    require(isinstance(inputs, Mapping), "collator did not expose inputs mapping")
    return inputs


def as_array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def physical_errors(decoded: np.ndarray, expert_action: np.ndarray, modules: Mapping[str, Any]) -> dict[str, Any]:
    decoded = np.asarray(decoded, dtype=np.float64)
    expert_action = np.asarray(expert_action, dtype=np.float64)
    xyz = np.linalg.norm(decoded[:, :3] - expert_action[:, :3], axis=1)
    so3 = modules["geodesic_angle"](
        modules["rot6d_rows_to_matrix"](decoded[:, 3:9]),
        modules["rot6d_rows_to_matrix"](expert_action[:, 3:9]),
    )
    grip = np.abs(decoded[:, 9] - expert_action[:, 9])
    return {
        "xyz_max_m": float(np.max(xyz)),
        "so3_max_rad": float(np.max(so3)),
        "gripper_max": float(np.max(grip)),
        "xyz_pass": bool(np.max(xyz) <= PHYSICAL_TOL["xyz_m"]),
        "so3_pass": bool(np.max(so3) <= PHYSICAL_TOL["so3_rad"]),
        "gripper_pass": bool(np.max(grip) <= PHYSICAL_TOL["gripper"]),
    }


def active_slots(mask: np.ndarray) -> tuple[list[int], int, list[int]]:
    values = np.asarray(mask)
    require(values.ndim == 3 and values.shape[1:] == (HORIZON, UNIVERSAL_DIM), "mask shape is not [B,40,132]")
    slots = np.flatnonzero(np.any(values != 0, axis=(0, 1))).tolist()
    times = np.flatnonzero(np.any(values != 0, axis=(0, 2))).tolist()
    return slots, int(np.count_nonzero(values)), times


def validate_mask_payload(action: np.ndarray, mask: np.ndarray, expected_slots: list[int], expected_count: int) -> bool:
    slots, count, _ = active_slots(mask)
    inactive = sorted(set(range(UNIVERSAL_DIM)) - set(slots))
    payload_in_inactive = bool(np.any(np.abs(action[:, :, inactive]) > 1e-12)) if inactive else False
    return slots == expected_slots and count == expected_count and not payload_in_inactive


def g3(args: argparse.Namespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    })
    modules = runtime_modules(args)
    stage21 = modules["stage21"]
    stage28_path = args.prep_root / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"
    spec = importlib.util.spec_from_file_location("pi0238_r1_stage28", stage28_path)
    require(spec is not None and spec.loader is not None, "Stage28 source import failed")
    stage28 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage28)
    training_args = argparse.Namespace(
        stage10_root=args.stage10_root,
        base_model_path=args.baseline,
        dataset_dir=args.train_dataset,
        modality_config=args.modality_config,
        groot_source=args.groot_source,
        training_output_dir=paths["output_root"] / "not_training_output",
    )
    config = stage28._build_training_config(training_args)
    config_summary = stage28._config_summary(config)

    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    import go2_arx5_modality

    modality = getattr(
        go2_arx5_modality,
        "GO2_ARX5_MODALITY_CONFIG",
        getattr(go2_arx5_modality, "SIM_VISUAL_REACH_MODALITY_CONFIG", None),
    )
    require(modality is not None, "modality configuration symbol missing")
    train_processor, embodiment = stage21._build_processor(
        args.stage10_root, args.baseline, args.train_dataset / "meta", args.modality_config
    )
    train_processor.train()
    eval_processor, eval_embodiment = stage21._build_processor(
        args.stage10_root, args.baseline, args.train_dataset / "meta", args.modality_config
    )
    eval_processor.eval()
    datasets = {
        "TRAIN": ShardedSingleStepDataset(
            args.train_dataset, EmbodimentTag.NEW_EMBODIMENT, copy.deepcopy(modality),
            shard_size=128, episode_sampling_rate=1.0, seed=SEED, allow_padding=False,
        ),
        "CALIB": ShardedSingleStepDataset(
            args.calib_dataset, EmbodimentTag.NEW_EMBODIMENT, copy.deepcopy(modality),
            shard_size=128, episode_sampling_rate=1.0, seed=SEED, allow_padding=False,
        ),
    }
    for dataset in datasets.values():
        dataset.processor = train_processor

    cohort = json_load(args.p237_root / "evidence" / "pi0237_cohort.json")
    selected = (
        [("TRAIN", args.train_dataset, row) for row in cohort["train"]]
        + [("CALIB", args.calib_dataset, row) for row in cohort["calib"]]
    )
    selected_keys = {(str(root), int(row["episode_index"])) for _, root, row in selected}
    p3 = json_load(args.p237_root / "evidence" / "pi0237_p3_codec_roundtrip.json")
    for row in p3.get("rows", []):
        if float(row.get("normalized_roundtrip_max_abs", 0.0)) > PHYSICAL_TOL["normalized"]:
            root = args.calib_dataset if row.get("split") == "CALIB" else args.train_dataset
            key = (str(root), int(row["episode_index"]))
            if key not in selected_keys:
                selected.append((str(row.get("split", "CALIB")), root, row))
                selected_keys.add(key)
    for row in json_load(args.p237_root / "evidence" / "pi0237_p2_expert_gate.json")["rows"]:
        if row.get("decision") == "REJECT":
            root = args.calib_dataset if row.get("split") == "CALIB" else args.train_dataset
            key = (str(root), int(row["episode_index"]))
            if key not in selected_keys:
                selected.append((str(row.get("split", "CALIB")), root, row))
                selected_keys.add(key)
    require(len(selected) <= 142, f"G3 union exceeds 142: {len(selected)}")

    samples = []
    expected_slots = None
    expected_count = None
    for split, root, item in selected:
        dataset = datasets[split]
        datapoint = dataset.get_datapoint(
            dataset.episode_loader[int(item["episode_index"])], 0
        )
        batch = train_processor.collator([datapoint])
        inputs = batch_inputs(batch)
        require(
            {"action", "action_mask", "embodiment_id"}.issubset(inputs),
            f"actual training fields incomplete: {sorted(inputs)}",
        )
        action = as_array(inputs["action"])
        mask = as_array(inputs["action_mask"])
        embodiment_id = as_array(inputs["embodiment_id"])
        require(action.shape == (1, HORIZON, UNIVERSAL_DIM), f"action shape drift: {action.shape}")
        require(mask.shape == action.shape, f"action mask shape drift: {mask.shape}")
        require(np.isfinite(action).all() and np.isfinite(mask).all(), "nonfinite action/mask")
        slots, scalar_count, times = active_slots(mask)
        if expected_slots is None:
            expected_slots, expected_count = slots, scalar_count
        require(slots == expected_slots and scalar_count == expected_count, "active slot/mask drift")
        state, expert_action, _ = load_window(root, item)
        state_for_decode = {
            "eef_9d": state[None, None, :9],
            "gripper_position": state[None, None, 9:],
        }
        decoded_groups = eval_processor.decode_action(
            action.astype(np.float32), eval_embodiment, state_for_decode
        )
        decoded = np.concatenate(
            [
                np.asarray(decoded_groups["eef_9d"]),
                np.asarray(decoded_groups["gripper_position"]),
            ],
            axis=-1,
        )[0]
        raw_action = {
            "eef_9d": decoded[:, :9].astype(np.float32),
            "gripper_position": decoded[:, 9:].astype(np.float32),
        }
        raw_state = {
            "eef_9d": state[None, :9].astype(np.float32),
            "gripper_position": state[None, 9:].astype(np.float32),
        }
        sap = train_processor.state_action_processor
        actual_norm = sap.apply_action(raw_action, embodiment.value, state=raw_state)
        actual_norm10 = np.concatenate(
            [actual_norm["eef_9d"], actual_norm["gripper_position"]], axis=-1
        )
        no_clip = copy.deepcopy(sap)
        no_clip.clip_outliers = False
        unclipped = no_clip.apply_action(raw_action, embodiment.value, state=raw_state)
        unclipped10 = np.concatenate(
            [unclipped["eef_9d"], unclipped["gripper_position"]], axis=-1
        )
        no_clip64 = copy.deepcopy(sap)
        no_clip64.clip_outliers = False
        raw_action64 = {key: value.astype(np.float64) for key, value in raw_action.items()}
        raw_state64 = {key: value.astype(np.float64) for key, value in raw_state.items()}
        unclipped64 = no_clip64.apply_action(
            raw_action64, embodiment.value, state=raw_state64
        )
        unclipped64_10 = np.concatenate(
            [unclipped64["eef_9d"], unclipped64["gripper_position"]], axis=-1
        )
        original_norm = action[0, :, :PHYSICAL_DIM]
        samples.append(
            {
                "split": split, "root": root, "item": item,
                "datapoint": datapoint, "batch": batch, "inputs": inputs,
                "action": action, "mask": mask, "embodiment_id": embodiment_id,
                "state": state, "expert_action": expert_action, "decoded": decoded,
                "physical": physical_errors(decoded, expert_action, modules),
                "normalized": {
                    "original": original_norm,
                    "actual": actual_norm10,
                    "unclipped": unclipped10,
                    "unclipped64": unclipped64_10,
                    "max_abs": float(np.max(np.abs(actual_norm10 - original_norm))),
                    "float64_cast_control": float(
                        np.max(np.abs(unclipped64_10 - unclipped10))
                    ),
                    "preclip_out_of_range_count": int(
                        np.count_nonzero(np.abs(unclipped10) > 1.0)
                    ),
                    "clip_changed_count": int(
                        np.count_nonzero(np.abs(unclipped10 - actual_norm10) > 0.0)
                    ),
                },
                "trace": {
                    "split": split, "episode_index": int(item["episode_index"]),
                    "source_episode_id": item["source_episode_id"],
                    "datapoint": describe(datapoint), "batch": describe(batch),
                    "inputs_keys": sorted(str(key) for key in inputs),
                    "action_key": "inputs.action", "mask_key": "inputs.action_mask",
                    "embodiment_key": "inputs.embodiment_id",
                    "action_shape": list(action.shape), "mask_shape": list(mask.shape),
                    "embodiment_id": embodiment_id.tolist(),
                    "active_slots": slots, "active_scalar_count": scalar_count,
                    "time_active": times,
                },
            }
        )

    require(expected_slots is not None and expected_count is not None, "no actual batch observed")
    errors = []
    numerical = []
    for sample in samples:
        physical = dict(sample["physical"])
        physical.update(
            {
                "split": sample["split"],
                "episode_index": int(sample["item"]["episode_index"]),
                "source_episode_id": sample["item"]["source_episode_id"],
                "action_tensor_sha256": sha256_array(sample["action"]),
                "expert_action_sha256": sha256_array(sample["expert_action"]),
                "decoded_sha256": sha256_array(sample["decoded"]),
                "normalized_dtype": str(sample["action"].dtype),
                "action_mask_scalar_count": int(expected_count),
                "active_slots": expected_slots,
                "finite": bool(np.isfinite(sample["decoded"]).all()),
            }
        )
        errors.append(physical)
        normalized = sample["normalized"]
        numerical.append(
            {
                "split": sample["split"],
                "episode_index": int(sample["item"]["episode_index"]),
                "source_episode_id": sample["item"]["source_episode_id"],
                "normalized_roundtrip_max_abs": normalized["max_abs"],
                "float64_cast_control_max_abs": normalized["float64_cast_control"],
                "preclip_out_of_range_count": normalized["preclip_out_of_range_count"],
                "clip_changed_count": normalized["clip_changed_count"],
                "clip_outliers": bool(train_processor.state_action_processor.clip_outliers),
                "normalization_params_source": "real StateActionProcessor.norm_params",
                "physical_tolerances": PHYSICAL_TOL,
            }
        )
    numerical_excess = [
        row for row in numerical
        if row["normalized_roundtrip_max_abs"] > PHYSICAL_TOL["normalized"]
    ]
    train_contract = {
        "class": f"{type(train_processor).__module__}.{type(train_processor).__name__}",
        "stats_sha256": sha256_file(args.train_dataset / "meta" / "stats.json"),
        "relative_stats_sha256": sha256_file(args.train_dataset / "meta" / "relative_stats.json"),
        "modality_sha256": sha256_file(args.modality_config),
        "embodiment": embodiment.value,
        "clip_outliers": bool(train_processor.state_action_processor.clip_outliers),
        "use_relative_action": bool(train_processor.state_action_processor.use_relative_action),
        "max_action_horizon": int(train_processor.max_action_horizon),
    }
    eval_contract = {
        "class": f"{type(eval_processor).__module__}.{type(eval_processor).__name__}",
        "stats_sha256": sha256_file(args.train_dataset / "meta" / "stats.json"),
        "relative_stats_sha256": sha256_file(args.train_dataset / "meta" / "relative_stats.json"),
        "modality_sha256": sha256_file(args.modality_config),
        "embodiment": eval_embodiment.value,
        "clip_outliers": bool(eval_processor.state_action_processor.clip_outliers),
        "use_relative_action": bool(eval_processor.state_action_processor.use_relative_action),
        "max_action_horizon": int(eval_processor.max_action_horizon),
    }
    negative = {
        "wrong_slot": {
            "executed": True,
            "rejected": not validate_mask_payload(
                np.where(
                    np.arange(UNIVERSAL_DIM)[None, None, :] == (expected_slots[-1] + 1),
                    1.0, samples[0]["action"]
                ),
                samples[0]["mask"], expected_slots, expected_count
            ) if expected_slots[-1] + 1 < UNIVERSAL_DIM else False,
        },
        "wrong_mask": {
            "executed": True,
            "rejected": (
                active_slots(
                    np.where(
                        np.arange(UNIVERSAL_DIM)[None, None, :] == expected_slots[0],
                        0.0, samples[0]["mask"]
                    )
                )[0] != expected_slots
            ),
        },
        "wrong_anchor": {
            "executed": True,
            "rejected": False,
            "reason": "checked by state-shift decode below",
        },
        "nonunit_rot6d": {
            "executed": True,
            "rejected_by_raw_quality": False,
            "reason": "independent raw rot6d quality predicate; production decoder unchanged",
        },
    }
    nonunit = samples[0]["decoded"].copy()
    nonunit[:, 3:9] *= 2.0
    negative["nonunit_rot6d"]["raw_quality"] = raw_rot6d_quality(nonunit)
    negative["nonunit_rot6d"]["rejected_by_raw_quality"] = bool(
        negative["nonunit_rot6d"]["raw_quality"]["unit_norm_deviation_count"] > 0
    )
    shifted_state = samples[0]["state"].copy()
    shifted_state[0] += 0.01
    shifted_groups = eval_processor.decode_action(
        samples[0]["action"].astype(np.float32), eval_embodiment,
        {"eef_9d": shifted_state[None, None, :9], "gripper_position": shifted_state[None, None, 9:]},
    )
    shifted_decoded = np.concatenate(
        [np.asarray(shifted_groups["eef_9d"]), np.asarray(shifted_groups["gripper_position"])],
        axis=-1,
    )[0]
    negative["wrong_anchor"]["rejected"] = bool(
        np.max(np.abs(shifted_decoded - samples[0]["decoded"])) > PHYSICAL_TOL["xyz_m"]
    )
    parity_ok = (
        len(samples) <= 142
        and all(row["finite"] for row in errors)
        and all(
            row["xyz_pass"] and row["so3_pass"] and row["gripper_pass"]
            for row in errors
        )
        and expected_slots
        and len(numerical) == len(samples)
    )
    result = {
        "schema": "PI0238_R1_G3_ACTUAL_BATCH_PARITY_V2",
        "execution_status": "COMPLETE",
        "parity_status": "PASS" if parity_ok else "FAIL",
        "numerical_status": (
            "NORMALIZED_TOLERANCE_EXCESS_REQUIRES_REVIEW"
            if numerical_excess else "PASS"
        ),
        "fixed_sample_count": len(samples), "base_cohort_count": 128,
        "actual_training_config": config_summary,
        "training_source_sha256": sha256_file(stage28_path),
        "processor_contract": {
            "train": train_contract, "eval": eval_contract,
            "resolved_equal": train_contract == eval_contract,
            "processor_config_sha256": sha256_file(args.baseline / "processor_config.json"),
        },
        "action_fields": {
            "action_key": "inputs.action", "mask_key": "inputs.action_mask",
            "embodiment_key": "inputs.embodiment_id",
            "action_shape": [1, HORIZON, UNIVERSAL_DIM],
            "mask_shape": [1, HORIZON, UNIVERSAL_DIM],
            "active_slots_observed": expected_slots,
            "active_scalar_count": expected_count,
        },
        "batch_traces": [sample["trace"] for sample in samples],
        "rows": errors, "normalization_rows": numerical,
        "numerical_excess_rows": numerical_excess,
        "negative_controls": negative,
        "target_xyz_in_model_input": False,
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
        "training_backward_calls": 0, "optimizer_steps": 0,
    }
    write_json(paths["output_root"] / "evidence" / "g3_actual_batch_trace.json", result)
    write_json(
        paths["output_root"] / "evidence" / "g3_effective_train_eval_config.json",
        {
            "actual_training_config": config_summary,
            "processor_contract": result["processor_contract"],
            "physical_tolerances": PHYSICAL_TOL,
        },
    )
    write_json(
        paths["output_root"] / "evidence" / "g3_132_slot_mask_tests.json",
        {
            "schema": "PI0238_R1_G3_SLOT_MASK_TESTS_V2",
            "execution_status": "COMPLETE",
            "action_fields": result["action_fields"],
            "negative_controls": negative,
        },
    )
    write_json(
        paths["output_root"] / "evidence" / "g3_numerical_casebook.json",
        {
            "schema": "PI0238_R1_G3_NUMERICAL_CASEBOOK_V2",
            "execution_status": "COMPLETE",
            "tolerances": PHYSICAL_TOL,
            "rows": numerical, "excess_rows": numerical_excess,
            "explanations": {
                "float32_cast": "measured actual float32 versus float64 no-clip StateActionProcessor",
                "clip_predicate": "measured unclipped versus clipped actual StateActionProcessor",
            },
        },
    )
    return {
        "execution_status": result["execution_status"],
        "parity_status": result["parity_status"],
        "numerical_status": result["numerical_status"],
        "fixed_sample_count": len(samples),
    }


def g4(args: argparse.Namespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    pre = json_load(paths["output_root"] / "evidence" / "r2_preflight.json")
    evidence = paths["output_root"] / "evidence"
    phase_files = {
        "g0": evidence / "g0_identity_and_access.json",
        "g1": evidence / "g1_failure_decomposition.json",
        "g2": evidence / "g2_full_label_gate_census.json",
        "g3": evidence / "g3_actual_batch_trace.json",
    }
    blocker_records: list[dict[str, Any]] = [
        item if isinstance(item, dict) else {"code": str(item), "type": "UNCLASSIFIED", "stage": "G4"}
        for item in pre["g4_admission"]["blockers"]
    ]

    def add_blocker(code: str, kind: str, stage: str = "G4") -> None:
        if not any(item.get("code") == code for item in blocker_records):
            blocker_records.append({"code": code, "type": kind, "stage": stage})

    phase = {
        name: json_load(path) if path.is_file() else {"execution_status": "NOT_RUN"}
        for name, path in phase_files.items()
    }
    phase_status = {}
    for name, result in phase.items():
        status = result.get("execution_status", result.get("status"))
        phase_status[name] = status
        if status not in {
            "COMPLETE",
            "PASS_READ_ONLY",
            "PASS_WITH_QUOTA_UNOBSERVED",
            "PASS_WITH_CAPACITY_UNPROVEN",
        }:
            add_blocker(f"{name.upper()}_NOT_COMPLETE", "EVIDENCE", name.upper())
    g3_result = phase["g3"]
    if g3_result.get("parity_status") != "PASS":
        add_blocker("G3_PARITY_NOT_PASS", "SCIENCE", "G3")
    if g3_result.get("numerical_status") != "PASS":
        add_blocker("G3_NUMERICAL_STATUS_NOT_PASS", "SCIENCE", "G3")
    processor_contract = g3_result.get("processor_contract", {})
    resolved_equal = processor_contract.get(
        "resolved_equal",
        g3_result.get("parity", {}).get("processor_train_eval_resolved_equal", False),
    )
    if not resolved_equal:
        add_blocker("TRAIN_EVAL_PROCESSOR_CONTRACT_MISMATCH", "SCIENCE", "G3")
    if not pre["quota"].get("admission_capacity_pass", False):
        add_blocker("FILESYSTEM_QUOTA_CAPACITY_NOT_PROVEN", "RESOURCE", "G4")
    admitted = not blocker_records
    result = {
        "schema": "PI0238_R2_G4_DIAGNOSTIC_ADMISSION_V3",
        "execution_status": "ADMISSION_READY_H200_REQUIRED" if admitted else "NOT_ADMITTED",
        "g4_admitted": admitted,
        "blockers": [item["code"] for item in blocker_records],
        "blocker_records": blocker_records,
        "admission_checks": {
            "preflight": pre["status"],
            "phase_execution": phase_status,
            "g3_parity": g3_result.get("parity_status"),
            "g3_numerical": g3_result.get("numerical_status"),
            "processor_contract_equal": resolved_equal,
            "quota_capacity_pass": pre["quota"].get("admission_capacity_pass", False),
            "cuda_disabled_cpu_phase": True,
            "h200_execution": "NOT_STARTED",
        },
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
        "attempted_queries": 0, "completed_queries": 0,
        "selected_checkpoint": "NONE",
        "reason": (
            "CPU phase records independent admission only; H200 G4 is not run here"
            if admitted else "one or more typed admission predicates are not proven"
        ),
    }
    write_json(paths["output_root"] / "evidence" / "g4_diagnostic_admission.json", result)
    return result


def g5(args: argparse.Namespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    evidence = paths["output_root"] / "evidence"
    statuses = {}
    for phase, name in (
        ("g0", "g0_identity_and_access.json"),
        ("g1", "g1_failure_decomposition.json"),
        ("g2", "g2_full_label_gate_census.json"),
        ("g3", "g3_actual_batch_trace.json"),
        ("g4", "g4_diagnostic_admission.json"),
    ):
        path = evidence / name
        statuses[phase] = (
            json_load(path) if path.is_file()
            else {"execution_status": "NOT_RUN", "reason": "missing evidence"}
        )
    blocker_records = statuses["g4"].get("blocker_records", [])
    if not blocker_records:
        blocker_records = [
            {"code": str(item), "type": "UNCLASSIFIED", "stage": "G4"}
            for item in statuses["g4"].get("blockers", [])
        ]
    phase_status = {
        phase: value.get("execution_status", value.get("status"))
        for phase, value in statuses.items()
    }
    if any(item.get("type") in {"RESOURCE", "ENGINEERING"} for item in blocker_records):
        classification = "BLOCKED_ENGINEERING_OR_RESOURCE"
    elif any(item.get("type") == "SCIENCE" for item in blocker_records):
        classification = "STOP_FOR_PI_RECIPE_DECISION"
    elif any(value in {None, "NOT_RUN", "BLOCKED", "FAIL_READ_ONLY"} for value in phase_status.values()):
        classification = "DIAGNOSTIC_NOT_ESTABLISHED"
    else:
        classification = "STOP_FOR_PI_RECIPE_DECISION"
    result = {
        "schema": "PI0238_R2_G5_DECISION_PACKAGE_V3",
        "status": "STOP_FOR_PI_RECIPE_DECISION",
        "classification": classification,
        "selected_checkpoint": "NONE",
        "phase_status": phase_status,
        "g4_blockers": [item["code"] for item in blocker_records],
        "g4_blocker_records": blocker_records,
        "classification_basis": "typed blocker records and explicit phase statuses; not blocker-list emptiness",
        "scientific_conclusion": (
            "This R1 package makes no training-recipe conclusion; "
            "it separates engineering/resource state from scientific evidence."
        ),
        "next_run_card": {
            "status": "PI_REVIEW_REQUIRED",
            "training": 0, "backward": 0, "optimizer": 0,
            "wbc": 0, "sim": 0, "dagger": 0,
            "dev_e_reads": 0, "final_f_reads": 0,
        },
        "evidence_files": sorted(
            str(path.relative_to(evidence))
            for path in evidence.iterdir() if path.is_file()
        ),
    }
    write_json(evidence / "g5_decision_package.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-launch", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--phase", choices=("g0", "g1", "g2", "g3", "g4", "g5"))
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    for name in (
        "output_root", "cache_root", "tmp_root", "p237_root", "run_root",
        "train_dataset", "calib_dataset", "prep_root", "vla_source",
        "stage10_root", "groot_source", "baseline", "fresh_root",
        "cosmos_snapshot", "cosmos_cache_root", "modality_config",
        "protocol", "original_protocol", "script", "pbs",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    args = parser.parse_args()
    if args.make_launch:
        require(args.python is not None, "--python is required for --make-launch")
        print(json.dumps(make_launch(args), indent=2))
        return 0

    cfg = load_config(args.launch_config)
    namespace = config_namespace(cfg, args.phase or "preflight")
    if args.preflight_only:
        print(json.dumps(preflight(namespace, initialize=False), indent=2))
        return 0
    if args.initialize:
        print(json.dumps(preflight(namespace, initialize=True), indent=2))
        return 0
    require(args.phase is not None, "--phase is required")
    try:
        # Reuse the repository's mature G0-G3 audit implementation.  R1 owns
        # only launch/preflight and the closed G4/G5 admission/reporting shell.
        phase_handlers = {
            "g0": legacy.g0, "g1": legacy.g1, "g2": legacy.g2,
            "g3": legacy.g3, "g4": g4, "g5": g5,
        }
        result = phase_handlers[args.phase](namespace)
    except Exception as exc:
        root = cfg_paths(cfg)["output_root"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "evidence").mkdir(exist_ok=True)
        write_json(
            root / "evidence" / f"{args.phase}_failure.json",
            {
                "schema": f"PI0238_R1_{args.phase.upper()}_FAILURE_V2",
                "execution_status": "BLOCKED",
                "error_type": type(exc).__name__, "error": str(exc),
                "argv": sys.argv, "model_weight_loads": 0,
                "model_forward_calls": 0, "get_action_calls": 0,
            },
        )
        raise
    print(json.dumps({"phase": args.phase, "result": result}, indent=2, default=str))
    return 0







def make_launch(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for_make(args)
    require(not paths["output_root"].exists(), f"new output root exists: {paths['output_root']}")
    require(not paths["cache_root"].exists(), f"new cache root exists: {paths['cache_root']}")
    require(not paths["tmp_root"].exists(), f"new tmp root exists: {paths['tmp_root']}")
    require(paths["script"].is_file() and paths["pbs"].is_file(), "staged runner/PBS missing")
    mature_script = paths["script"].with_name("pi0238_audit_to_sim.py")
    require(mature_script.is_file(), f"staged mature runner missing: {mature_script}")
    phase_argv = {
        phase: [
            str(args.python), str(paths["script"]), "--phase", phase,
            "--launch-config", str(paths["launch_config"])
        ]
        for phase in ("g0", "g1", "g2", "g3", "g4", "g5")
    }
    payload = {
        "schema": "PI0238_R2_RESOLVED_LAUNCH_V1",
        "protocol_commit": R2_PROTOCOL_COMMIT,
        "original_protocol_commit": ORIGINAL_PROTOCOL_COMMIT,
        "pi0237_protocol_commit": PI0237_PROTOCOL_COMMIT,
        "audited_parent": R2_AUDITED_PARENT,
        "r1_protocol_commit": R1_PROTOCOL_COMMIT,
        "r1_audited_parent": R1_AUDITED_PARENT,
        "branch": "pi0238/manifest-first-recovery-r2",
        "paths": {key: str(value) for key, value in paths.items()},
        "implementation": {
            "script_sha256": sha256_file(paths["script"]),
            "mature_script_sha256": sha256_file(mature_script),
            "pbs_sha256": sha256_file(paths["pbs"]),
            "protocol_sha256": sha256_file(paths["protocol"]),
            "original_protocol_sha256": sha256_file(paths["original_protocol"]),
        },
        "python": str(args.python),
        "phase_argv": phase_argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "resource_request": {
            "project": "CFP03-CF-004", "queue": "auto",
            "select": "select=1:ncpus=12:mem=225GB", "walltime": "02:00:00",
            "cuda_disabled": True, "model_activity": 0,
        },
        "contract": {
            "seed": SEED, "dt_s": DT, "horizon": HORIZON,
            "physical_dim": PHYSICAL_DIM, "universal_dim": UNIVERSAL_DIM,
            "action_contract": "official_dense_relative_H40",
            "target_xyz_in_model_input": False,
            "dev_e_source_read_count": 0, "final_f_source_read_count": 0,
            "g4_query_budget": 256,
            "r2_cpu_storage_limit_bytes": 1 << 30,
        },
        "read_boundary": {
            "preflight_full_tree": False, "dev_e_source_read_count": 0,
            "final_f_source_read_count": 0, "locked_data_mutation": False,
        },
        "old_jobs": ["624206.hopper-m-02", "624209.hopper-m-02", "624229.hopper-m-02"],
        "fresh_family": {"job_id": FRESH_JOB, "source": "PI0237_S2_ADMISSION"},
    }
    write_json(paths["launch_config"], payload)
    return payload


def quota_parse(raw: Mapping[str, Any]) -> dict[str, Any]:
    text = "\n".join(str(raw.get(key, "")) for key in ("stdout", "stderr"))
    numbers = re.findall(
        r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:K|M|G|T|KiB|MiB|GiB|TiB)?(?![A-Za-z0-9])",
        text, flags=re.I
    )
    lower = text.lower()
    has_limit_terms = any(
        term in lower for term in ("quota", "limit", "grace", "blocks", "inode", "files")
    )
    return {
        "nonempty": bool(text.strip()),
        "parseable": bool(text.strip() and numbers and has_limit_terms),
        "numeric_tokens": numbers[:40],
    }


def disk_usage(path: Path) -> dict[str, int]:
    import shutil
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def quota_snapshot(path: Path) -> dict[str, Any]:
    commands = {
        name: run_capture(argv)
        for name, argv in (
            ("quota_s", ["quota", "-s"]),
            ("quota_v", ["quota", "-v"]),
            ("quota_user", ["quota", "-u", "e1192548"]),
        )
    }
    parsed = {name: quota_parse(item) for name, item in commands.items()}
    observed = any(item["parseable"] for item in parsed.values())
    observation_path = path if path.exists() else path.parent
    usage = disk_usage(observation_path)
    statvfs = os.statvfs(observation_path)
    free_inodes = int(statvfs.f_bavail)
    cpu_budget = 1 << 30
    cpu_floor = max(2 << 30, 3 * cpu_budget)
    cpu_storage_ok = usage["free_bytes"] >= cpu_floor and free_inodes >= 10_000
    # A quota command can be syntactically parseable without exposing the
    # byte/inode limit for this exact fileset.  Do not turn that into G4 PASS.
    quota_limit_bytes = None
    quota_limit_inodes = None
    capacity_pass = bool(
        observed
        and quota_limit_bytes is not None
        and quota_limit_inodes is not None
        and usage["free_bytes"] >= quota_limit_bytes
        and free_inodes >= quota_limit_inodes
    )
    report_hashes = {
        name: sha256_bytes((item["stdout"] + item["stderr"]).encode())
        for name, item in commands.items()
    }
    return {
        "commands": commands, "parsed": parsed,
        "status": "OBSERVED_PARSEABLE" if observed else "FILESYSTEM_QUOTA_UNOBSERVED",
        "admission_capacity_pass": capacity_pass,
        "capacity_evidence": {
            "fileset_observation_path": str(observation_path),
            "quota_limit_bytes": quota_limit_bytes,
            "quota_limit_inodes": quota_limit_inodes,
            "df_observation": usage,
            "free_inodes": free_inodes,
            "cpu_output_budget_bytes": cpu_budget,
            "cpu_min_free_bytes": cpu_floor,
            "cpu_storage_limited_diagnostic_only": cpu_storage_ok,
            "report_sha256": report_hashes,
            "calculation": "G4 requires parsed byte and inode limits for the measured fileset; df/free only gates the R2 CPU-limited continuation.",
        },
        "interpretation": "quota reports without exact byte/inode limits remain FILESYSTEM_QUOTA_UNOBSERVED; CPU continuation is separately bounded by statvfs and 1 GiB output/cache budget",
    }


def modality_probe(args: argparse.Namespace) -> dict[str, Any]:
    paths = args.config["paths"]
    code = r'''
import hashlib, importlib, inspect, json, os, sys
from pathlib import Path
stage10 = Path(sys.argv[1])
modality = Path(sys.argv[2])
vla = Path(sys.argv[3])
prep = Path(sys.argv[4])
original_sys_path = list(sys.path)
sys.path[:] = [
    str(stage10 / "src" / "isaac-gr00t"),
    str(modality.parent), str(prep / "scripts"), str(vla / "src")
] + [p for p in original_sys_path if p]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.modules.pop(modality.stem, None)
module = importlib.import_module(modality.stem)
def digest(path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
out = {
    "module_file": str(Path(module.__file__).resolve()),
    "module_sha256": digest(Path(module.__file__)),
    "public_symbols": sorted(k for k in dir(module) if "MODALITY_CONFIG" in k),
}
import stage21_same_scene_groot_inference as stage21
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from go2arx5_vla.safety import TrajectorySafetyGateV1
out["imports"] = {
    "stage21_file": str(Path(stage21.__file__).resolve()),
    "stage21_build_processor_signature": str(inspect.signature(stage21._build_processor)),
    "stage21_run_one_signature": str(inspect.signature(stage21._run_one)),
    "loader_module": LeRobotEpisodeLoader.__module__,
    "loader_getitem_signature": str(inspect.signature(LeRobotEpisodeLoader.__getitem__)),
    "sharded_module": ShardedSingleStepDataset.__module__,
    "sharded_get_datapoint_signature": str(inspect.signature(ShardedSingleStepDataset.get_datapoint)),
    "gate_class": str(TrajectorySafetyGateV1),
}
print(json.dumps(out))
'''
    env = dict(os.environ)
    env.update({"PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"})
    result = run_capture(
        [
            str(args.config["python"]), "-c", code,
            str(paths["stage10_root"]), str(paths["modality_config"]),
            str(paths["vla_source"]), str(paths["prep_root"]),
        ],
        # Verify-only must not create or depend on the attempt's tmp directory.
        cwd=Path(paths["script"]).parent, env=env
    )
    require(
        result["returncode"] == 0,
        f"clean modality/runtime import failed: {result['stderr'][-2000:]}",
    )
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    data = json.loads(lines[-1])
    actual = Path(data["module_file"])
    require(actual.resolve() == Path(paths["modality_config"]).resolve(), f"modality origin drift: {actual}")
    require(data["module_sha256"] == P237_MODALITY_SHA, "actual modality SHA drift")
    stage21_file = Path(data["imports"]["stage21_file"])
    require(
        stage21_file.resolve()
        == Path(paths["prep_root"]) / "scripts" / "stage21_same_scene_groot_inference.py",
        "Stage21 import origin drift",
    )
    return {"argv": result["argv"], "probe": data}


def inventory_parse(path: Path, base_dir: Path, allowed_roots: list[Path]) -> dict[str, Any]:
    require(path.is_file(), f"inventory missing: {path}")
    rows = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.*)$", line)
        require(match is not None, f"malformed inventory line {line_no}")
        expected, raw = match.groups()
        candidate = Path(raw)
        resolved = (candidate if candidate.is_absolute() else base_dir / candidate).resolve()
        require(
            any(under(resolved, root) for root in allowed_roots),
            f"inventory path outside allowlist: {raw}",
        )
        require(resolved not in seen, f"duplicate inventory path: {resolved}")
        seen.add(resolved)
        require(resolved.is_file() and not resolved.is_symlink(), f"inventory unsafe/missing: {resolved}")
        actual = sha256_file(resolved)
        rows.append({
            "line": line_no, "path": str(resolved),
            "expected_sha256": expected.lower(), "actual_sha256": actual,
            "match": expected.lower() == actual,
        })
    require(rows, "empty evidence inventory cannot pass")
    require(all(row["match"] for row in rows), "evidence inventory hash mismatch")
    return {"path": str(path), "base_dir": str(base_dir), "count": len(rows), "rows": rows, "all_match": True}


def write_probe(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    marker = root / "preflight_probe.bin"
    payload = b"PI0238-R1-PREFLIGHT\0" * 65536
    with marker.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    result = {"path": str(marker), "bytes": marker.stat().st_size, "sha256": sha256_file(marker), "fsync": True}
    marker.unlink()
    return result


def preflight(args: argparse.Namespace, *, initialize: bool = False) -> dict[str, Any]:
    cfg = args.config
    paths = cfg_paths(cfg)
    scratch = Path("/scratch/e1192548")
    for name, path in paths.items():
        if name in {"output_root", "cache_root", "tmp_root", "launch_config"}:
            continue
        path_no_symlink(path, allow_missing=False)
        require(under(path, scratch), f"path outside approved scratch: {name}={path}")
    path_no_symlink(paths["launch_config"], allow_missing=False)
    for name in ("output_root", "cache_root", "tmp_root"):
        path = paths[name]
        require(not path.exists(), f"new R2 path already exists: {name}={path}")
        require(under(path, scratch), f"new path outside approved scratch: {path}")
    require(paths["script"].resolve() == Path(__file__).resolve(), "runner identity is not staged script")
    require(sha256_file(paths["script"]) == cfg["implementation"]["script_sha256"], "runner SHA drift")
    mature_script = paths["script"].with_name("pi0238_audit_to_sim.py")
    require(mature_script.is_file(), f"mature runner missing: {mature_script}")
    require(sha256_file(mature_script) == cfg["implementation"]["mature_script_sha256"], "mature runner SHA drift")
    require(sha256_file(paths["pbs"]) == cfg["implementation"]["pbs_sha256"], "PBS SHA drift")
    require(sha256_file(paths["protocol"]) == cfg["implementation"]["protocol_sha256"], "R2 protocol SHA drift")
    require(sha256_file(paths["original_protocol"]) == cfg["implementation"]["original_protocol_sha256"], "original protocol SHA drift")
    require(sha256_file(paths["modality_config"]) == P237_MODALITY_SHA, "modality file SHA drift")

    s2_path = paths["p237_root"] / "evidence" / "pi0236_s2_evaluator_admission.json"
    require(s2_path.is_file(), f"sealed S2 admission missing: {s2_path}")
    fresh_resolution = legacy.resolve_sealed_checkpoints(
        legacy.json_load(s2_path)["fresh_family"],
        expected_root=paths["fresh_root"],
        require_exists=True,
    )
    inventory = legacy.inventory_parse(
        paths["p237_root"] / "evidence" / "pi0237_evidence_inventory.sha256",
        paths["p237_root"] / "evidence",
        [paths["p237_root"] / "evidence", paths["prep_root"] / "evidence"],
    )
    quota = quota_snapshot(paths["output_root"])
    import_root = modality_probe(args)
    qstat = (
        run_capture([QSTAT, "-u", "e1192548"])
        if Path(QSTAT).is_file()
        else {"argv": [QSTAT, "-u", "e1192548"], "returncode": 127, "stdout": "", "stderr": "qstat unavailable"}
    )
    qstat_text = qstat["stdout"] + qstat["stderr"]
    own_job = os.environ.get("PBS_JOBID")
    active_other_jobs = pbs_job_lines(qstat_text, own_job)
    blockers: list[dict[str, Any]] = []
    if quota["status"] != "OBSERVED_PARSEABLE":
        blockers.append({"code": "FILESYSTEM_QUOTA_UNOBSERVED", "type": "RESOURCE", "stage": "G4"})
    if qstat["returncode"] != 0:
        blockers.append({"code": "QSTAT_UNOBSERVED", "type": "RESOURCE", "stage": "PREFLIGHT"})
    if active_other_jobs:
        blockers.append({"code": "OTHER_ACTIVE_JOBS_PRESENT", "type": "RESOURCE", "stage": "PREFLIGHT", "jobs": active_other_jobs})
    result = {
        "schema": "PI0238_R2_PREFLIGHT_V2",
        "mode": "INITIALIZE" if initialize else "VERIFY_ONLY",
        "status": "CPU_STORAGE_LIMITED_DIAGNOSTIC_ONLY" if not blockers or all(item["code"] == "FILESYSTEM_QUOTA_UNOBSERVED" for item in blockers) else "BLOCKED_ENGINEERING_OR_RESOURCE",
        "argv": list(sys.argv), "cwd": str(Path.cwd()), "python": sys.executable, "python_version": platform.python_version(),
        "paths": {key: str(value) for key, value in paths.items()},
        "implementation": {**cfg["implementation"], "mature_script": {"path": str(mature_script), "sha256": sha256_file(mature_script)}},
        "runtime_import": import_root,
        "sealed_checkpoint_resolution": {"manifest": str(s2_path), **fresh_resolution},
        "inventory": inventory, "quota": quota,
        "write_probe": {"status": "PENDING_INITIALIZE" if not initialize else "NOT_RUN"}, "qstat": qstat,
        "active_other_jobs": active_other_jobs, "own_job": own_job,
        "read_boundary": cfg["read_boundary"], "full_tree_hashes": "NOT_RUN_IN_PREFLIGHT",
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
        "g4_admission": {
            "allowed": False, "status": "NOT_ADMITTED",
            "blockers": blockers or [{"code": "G4_REQUIRES_H200_AND_CAPACITY_PROOF", "type": "ADMISSION", "stage": "G4"}],
        },
    }
    if initialize:
        paths["cache_root"].mkdir(parents=True, exist_ok=False)
        paths["tmp_root"].mkdir(parents=True, exist_ok=False)
        paths["output_root"].mkdir(parents=True, exist_ok=False)
        result["write_probe"] = write_probe(paths["output_root"] / "preflight_probe")
        write_json(paths["output_root"] / "evidence" / "r2_preflight.json", result)
    return result


def old_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Path]]:
    p237 = args.p237_root / "evidence"
    run_path = args.run_root / "pi0236_train_calib_evaluation_run.json"
    run = json_load(run_path)
    profiles = legacy.profile_records(args)
    required = {
        "s2": p237 / "pi0236_s2_evaluator_admission.json",
        "s3": p237 / "pi0236_s3_provenance_lock.json",
        "s4": p237 / "pi0236_s4_selection_freeze.json",
        "s5": p237 / "pi0236_s5_diagnostics.json",
        "state": p237 / "pi0236_execution_state.json",
        "cohort": p237 / "pi0237_cohort.json",
        "cohort_csv": p237 / "pi0237_cohort.csv",
        "seal": p237 / "pi0228_train_calib_seal.json",
        "authority": p237 / "pi0228_train_calib_target_authority.json",
    }
    for path in required.values():
        require(path.is_file(), f"missing sealed input: {path}")
    return run, profiles, required
def g0(args: argparse.Namespace) -> dict[str, Any]:
    cfg = args.config
    paths = cfg_paths(cfg)
    pre = json_load(paths["output_root"] / "evidence" / "r1_preflight.json")
    run, profiles, required = old_inputs(args)
    require(run.get("execution_status") == "COMPLETE", "PI0236 run is not COMPLETE")
    cohort = json_load(required["cohort"])
    require(
        len(cohort.get("train", [])) == 64 and len(cohort.get("calib", [])) == 64,
        "fixed cohort cardinality drift",
    )
    train_rollouts = {item["source_rollout_id"] for item in cohort["train"]}
    calib_rollouts = {item["source_rollout_id"] for item in cohort["calib"]}
    require(train_rollouts.isdisjoint(calib_rollouts), "fixed cohort source-rollout leakage")
    seal = json_load(required["seal"])
    require(seal.get("status") == "PASS", "TRAIN/CALIB seal not PASS")
    train_meta = full_metadata(args.train_dataset, "TRAIN", 612)
    calib_meta = full_metadata(args.calib_dataset, "CALIB", 68)
    require(
        {x["source_rollout_id"] for x in train_meta.values()}.isdisjoint(
            {x["source_rollout_id"] for x in calib_meta.values()}
        ),
        "dataset source-rollout leakage",
    )
    trees = {}
    for name, root, prefix, expected in (
        ("train", args.train_dataset, "train", P237_TRAIN_TREE),
        ("calib", args.calib_dataset, "train_calib", P237_CALIB_TREE),
    ):
        actual, count, total = tree_sha256(root, prefix)
        require(actual == expected, f"{name} tree hash drift: {actual}")
        trees[name] = {
            "root": str(root), "prefix": prefix, "tree_sha256": actual,
            "file_count": count, "total_bytes": total, "expected": expected,
        }
    locks = {
        name: json_load(root / "meta" / "pi0227_train_only_split_lock.json")
        for name, root in (("train", args.train_dataset), ("calib", args.calib_dataset))
    }
    require(
        all(
            item.get("e_source_read_count") == 0
            and item.get("f_source_read_count") == 0
            and item.get("target_xyz_in_model_input") is False
            for item in locks.values()
        ),
        "split lock drift",
    )
    fresh_admission = json_load(required["s2"])["fresh_family"]
    require(fresh_admission.get("job_id") == FRESH_JOB, "fresh family is not 623593")
    fresh_root = Path(fresh_admission["root"])
    fresh_checkpoints = {
        str(step): {
            "path": str(fresh_root / "training" / f"checkpoint-{step}"),
            "exists": (fresh_root / "training" / f"checkpoint-{step}").is_dir(),
        }
        for step in (2500, 5000, 7500, 10000)
    }
    require(all(item["exists"] for item in fresh_checkpoints.values()), "fresh checkpoint family incomplete")
    inventory = inventory_parse(
        args.p237_root / "evidence" / "pi0237_evidence_inventory.sha256",
        args.p237_root / "evidence",
        [args.p237_root / "evidence", args.prep_root / "evidence"],
    )
    runtime_identity = {}
    for name in (
        "protocol", "original_protocol", "stage10_root", "groot_source", "prep_root",
        "vla_source", "baseline", "modality_config", "cosmos_snapshot",
    ):
        path = paths[name]
        runtime_identity[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    quota_status = pre["quota"]["status"]
    result = {
        "schema": "PI0238_R1_G0_IDENTITY_AND_ACCESS_V2",
        "execution_status": "COMPLETE",
        "scientific_status": "DESCRIPTIVE_ONLY",
        "quota_status": quota_status,
        "g4_admitted": False,
        "g4_admission_blockers": (
            ["FILESYSTEM_QUOTA_UNOBSERVED"]
            if quota_status != "OBSERVED_PARSEABLE"
            else ["H200_G4_NOT_SUBMITTED_IN_CPU_PHASE"]
        ),
        "protocol_chain": {
            "r1": R1_PROTOCOL_COMMIT,
            "original": ORIGINAL_PROTOCOL_COMMIT,
            "pi0237": PI0237_PROTOCOL_COMMIT,
        },
        "old_jobs_preserved": cfg["old_jobs"],
        "fresh_family": {
            "job_id": FRESH_JOB, "root": str(fresh_root), "checkpoints": fresh_checkpoints,
        },
        "profiles": {
            str(step): {
                "sha256": legacy.EXPECTED_PROFILES[step],
                "forward_count": profiles[step]["forward_count"],
            }
            for step in PROFILE_STEPS
        },
        "cohort": {
            "train_count": 64, "calib_count": 64,
            "train_rollouts": len(train_rollouts),
            "calib_rollouts": len(calib_rollouts),
            "rollout_disjoint": True,
        },
        "dataset": {
            "trees": trees, "split_locks": locks,
            "train_rollouts": 612, "calib_rollouts": 68,
        },
        "runtime_identity": runtime_identity,
        "inventory": inventory,
        "preflight_sha256": sha256_file(paths["output_root"] / "evidence" / "r1_preflight.json"),
        "read_boundary": cfg["read_boundary"],
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
    }
    write_json(paths["output_root"] / "evidence" / "g0_identity_and_access.json", result)
    write_json(
        paths["output_root"] / "evidence" / "cohort_binding.json",
        {
            "schema": "PI0238_FIXED_COHORT_BINDING_V2",
            "execution_status": "COMPLETE",
            "source": str(required["cohort"]),
            "sha256": sha256_file(required["cohort"]),
            "membership_changed": False,
            "train": cohort["train"], "calib": cohort["calib"],
        },
    )
    return result


def issue_codes(record: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("code"))
        for item in record.get("safety_gate", {}).get("issues", [])
        if item.get("code")
    ]


def g1(args: argparse.Namespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    _, profiles, required = old_inputs(args)
    metadata = full_metadata(args.calib_dataset, "CALIB", 68)
    expert = json_load(required["s2"].parent / "pi0237_p2_expert_gate.json")
    expert_map = {
        (str(row["source_episode_id"]), int(row["episode_index"])): row
        for row in expert["rows"]
        if row.get("split") == "CALIB"
    }
    require(len(expert_map) == 1088, "sealed expert CALIB join is incomplete")
    by_step: dict[int, list[dict[str, Any]]] = {}
    for step, profile in profiles.items():
        rows = []
        for record in profile["records"]:
            key = (str(record["source_episode_id"]), int(record["episode_index"]))
            meta = metadata.get(key[0])
            require(meta and int(meta["episode_index"]) == key[1], f"profile metadata mismatch {key}")
            metrics = record.get("safety_gate", {}).get("metrics", {})
            rows.append(
                {
                    "checkpoint_step": step,
                    "key": f"{key[0]}|{key[1]}",
                    "source_episode_id": key[0],
                    "episode_index": key[1],
                    **{
                        field: meta.get(field)
                        for field in (
                            "source_rollout_id", "target_family", "jitter_family",
                            "initial_state", "realization", "phase",
                        )
                    },
                    "decision": record.get("safety_gate", {}).get("decision"),
                    "issue_codes": issue_codes(record),
                    "reference_min_target_distance_m": record.get(
                        "reference_min_target_distance_m"
                    ),
                    **{
                        metric: metrics.get(metric)
                        for metric in (
                            "linear_speed_mps", "angular_speed_radps",
                            "translation_path_length_m", "se3_arc_length_m",
                            "initial_position_discontinuity_m",
                            "initial_rotation_discontinuity_rad", "chunk_age_s",
                        )
                    },
                }
            )
        by_step[step] = rows
    joins = {}
    for step, rows in by_step.items():
        confusion = Counter()
        extra_speed = Counter()
        expert_speeds = defaultdict(list)
        for row in rows:
            expert_row = expert_map[(row["source_episode_id"], row["episode_index"])]
            model_codes = set(row["issue_codes"])
            expert_codes = {str(x.get("code")) for x in expert_row.get("issues", [])}
            confusion[f"expert_{expert_row.get('decision')}__model_{row.get('decision')}"] += 1
            if (
                "LINEAR_SPEED_REJECT" in model_codes
                and "LINEAR_SPEED_REJECT" not in expert_codes
            ):
                extra_speed["model_extra_linear_speed_reject"] += 1
            expert_speeds[str(expert_row.get("decision"))].append(
                next(
                    (
                        issue.get("observed")
                        for issue in expert_row.get("issues", [])
                        if issue.get("code") == "LINEAR_SPEED_REJECT"
                    ),
                    None,
                )
            )
        joins[str(step)] = {
            "joined_rows": len(rows),
            "confusion": dict(confusion),
            "model_extra_speed": dict(extra_speed),
            "expert_speed_by_decision": {
                key: stats(value) for key, value in expert_speeds.items()
            },
        }
    result = {
        "schema": "PI0238_R1_G1_FAILURE_DECOMPOSITION_V2",
        "execution_status": (
            "COMPLETE" if all(len(rows) == 1088 for rows in by_step.values()) else "BLOCKED"
        ),
        "scientific_status": "DESCRIPTIVE_ONLY",
        "profile_row_counts": {str(key): len(value) for key, value in by_step.items()},
        "expert_join_by_step": joins,
        "all_profile_expert_joined": all(
            item["joined_rows"] == 1088 for item in joins.values()
        ),
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
        "missing_or_unobserved": [
            "per-tick spike index", "achieved TCP", "WBC tracking", "native termination",
        ],
    }
    write_json(paths["output_root"] / "evidence" / "g1_failure_decomposition.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
