#!/usr/bin/env python3
"""PI0160 matched-reset independent-expert reference campaign.

This native-runtime runner is deliberately separate from the Stage43 learned
policy runner.  It creates one fresh IsaacGym/WBC instance per fixed initial
condition, applies exactly one pre-S0 robot-base XY delta, and then executes
only the deterministic same-scene expert reference through the unchanged
SafetyGate/WBC stack.  GR00T is never constructed or called here.  The q0
observations and q0--q3 expert chunks are saved for the separately launched
fixed-N2 model diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

PI_GATE = "PI_GATE_STATE_0160"
PI_COMMENT = 5651703057
PR_COMMENT = 5651704759
RETURN_TOKEN = "CORELAB_PI0160_MATCHED_RESET_EXPERT_FOR_PI_REVIEW"
ADMISSION_TOKEN = "CORELAB_PI0160_MATCHED_EXPERT_ADMISSION_RECORDED"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
STARTING_HEAD = "f2d115811051ce0b1cd3c177e5b6c32bf1efce64"
TARGET_ID = "red_target_train_a"
TARGET_WORLD = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP = "x5_link6_gripper_center"
SEED = 4242
CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
DELTAS = {
    "C0_NOMINAL": (0.0, 0.0, 0.0),
    "C1_X_PLUS_5CM": (0.05, 0.0, 0.0),
    "C2_X_MINUS_5CM": (-0.05, 0.0, 0.0),
    "C3_Y_PLUS_5CM": (0.0, 0.05, 0.0),
    "C4_Y_MINUS_5CM": (0.0, -0.05, 0.0),
}
SCHEDULE = (49, 49, 49, 98)
HORIZON = 40
ACTION_PERIOD = 0.05
MOTION_DURATION = 1.95


class ExpertCampaignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpertCampaignError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value)).tobytes())


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
    return sha256_file(path)


def _nvidia_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    return {"hostname": platform.node(), "query_sha256": sha256_bytes(result.stdout.encode()), "query": result.stdout, "rows": rows}


def _select_gpu(snapshot: Mapping[str, Any], requested: str | None) -> dict[str, Any]:
    candidates = [row for row in snapshot["rows"] if "4080 SUPER" in str(row["name"]) and int(row["memory_total_mib"]) >= 30000 and int(row["memory_free_mib"]) >= 14 * 1024]
    if requested:
        candidates = [row for row in candidates if str(row["uuid"]) == str(requested)]
    require(candidates, f"no eligible 32GB RTX 4080 SUPER: {snapshot['rows']}")
    candidates.sort(key=lambda row: (-int(row["memory_free_mib"]), int(row["index"])))
    return dict(candidates[0])


def _prepare_native_environment(output: Path, args: argparse.Namespace, gpu_uuid: str) -> dict[str, Any]:
    """Use the already-qualified isolated native runtime layering."""

    native_env_root = Path(args.native_env_root).resolve(strict=True)
    isaacgym_python = Path(args.isaacgym_python).resolve(strict=True)
    native_compat = Path(args.native_compat_root).resolve(strict=True)
    wbc_source = Path(args.wbc_source_root).resolve(strict=True)
    for path, label in ((native_env_root, "native env"), (isaacgym_python, "IsaacGym Python"), (native_compat, "native compat"), (wbc_source, "WBC source")):
        require(path.is_dir() and not path.is_symlink(), f"{label} unavailable: {path}")
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    dirs = {name: runtime / name for name in ("home", "tmp", "cache", "config", "data", "torch_extensions")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    bind = isaacgym_python / "isaacgym" / "_bindings" / "linux-x86_64"
    public_env = native_env_root.parent / "public_native" / "python_env"
    pyroot = native_env_root / "lib" / "python3.8" / "site-packages"
    torch_pyroot = pyroot if (pyroot / "torch" / "lib").is_dir() else public_env / "lib" / "python3.8" / "site-packages"
    torch_lib = torch_pyroot / "torch" / "lib"
    env_lib = native_env_root / "lib" if (native_env_root / "lib" / "libpython3.8.so.1.0").is_file() else public_env / "lib"
    ninja = native_env_root / "bin" / "ninja"
    require(bind.is_dir() and torch_lib.is_dir() and env_lib.is_dir() and ninja.is_file(), "qualified native runtime layering is incomplete")
    for import_root in (SCRIPTS, REPO / "src", wbc_source, isaacgym_python, native_compat):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    current_path = os.environ.get("PATH", "")
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_uuid),
        "NVIDIA_VISIBLE_DEVICES": str(gpu_uuid),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(dirs["home"]),
        "TMPDIR": str(dirs["tmp"]),
        "TMP": str(dirs["tmp"]),
        "TEMP": str(dirs["tmp"]),
        "XDG_CACHE_HOME": str(dirs["cache"]),
        "XDG_CONFIG_HOME": str(dirs["config"]),
        "XDG_DATA_HOME": str(dirs["data"]),
        "TORCH_EXTENSIONS_DIR": str(dirs["torch_extensions"]),
        "MAX_JOBS": "1",
        "CXX": "/usr/bin/g++-10",
        "PATH": str(native_env_root / "bin") + os.pathsep + current_path,
        "PYTHONPATH": os.pathsep.join((str(SCRIPTS), str(REPO / "src"), str(wbc_source), str(isaacgym_python), str(native_compat))),
        "LD_LIBRARY_PATH": os.pathsep.join((str(bind), str(torch_lib), str(env_lib))),
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "CUDA_CACHE_PATH": str(dirs["cache"] / "cuda"),
        "TORCH_HOME": str(dirs["cache"] / "torch"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    return {"native_env_root": str(native_env_root), "isaacgym_python": str(isaacgym_python), "native_compat_root": str(native_compat), "wbc_source_root": str(wbc_source), "bind_dir": str(bind), "torch_lib_dir": str(torch_lib), "env_lib_dir": str(env_lib), "ninja": str(ninja), "gpu_uuid": str(gpu_uuid), "python": platform.python_version(), "isolated": True}


def _state_hashes(native: Any, env: Any, tensor_hash: Any) -> dict[str, Any]:
    return {"root": tensor_hash(native.root_states), "dof": tensor_hash(native.dof_state), "arm_history": tensor_hash(env.arm_obs_history), "dog_history": tensor_hash(env.dog_obs_history), "episode": int(native.episode_length_buf[0].detach().cpu().item()), "reset_buf": bool(native.reset_buf[0].detach().cpu().item()), "wbc_forward_count": int(native.stage21_wbc_forward_count), "reference_update_count": int(native.stage21_reference_update_count)}


def _capture_snapshot(native: Any, *, output: Path, condition: str, snapshot_id: str, sim_time: float, anchor: np.ndarray, target: np.ndarray, capture_scene: Any, current_eef: Any, current_state: Any, build_chunk: Any, expert_cfg: Any, write_ppm: Any, tensor_hash: Any) -> tuple[dict[str, Any], Any]:
    body, wrist = capture_scene(native)
    eef_world, quat = current_eef(native)
    eef_local = np.asarray(eef_world, dtype=np.float64) - anchor
    target_local = target - anchor
    state = current_state(native).as_array().astype(np.float32)
    chunk = build_chunk(chunk_id=f"{condition}-{snapshot_id}", observation_timestamp=float(sim_time), current_position=eef_local, target_position=target_local, current_quaternion=quat, config=expert_cfg, source_frame=SOURCE_FRAME, tcp_semantic_id=TCP)
    snap_dir = output / "conditions" / condition / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    body_path = snap_dir / f"{snapshot_id}_body.ppm"
    wrist_path = snap_dir / f"{snapshot_id}_wrist.ppm"
    body_sha = write_ppm(body_path, body)
    wrist_sha = write_ppm(wrist_path, wrist)
    physical = np.concatenate([chunk.eef_position, _rot6d_rows(chunk.eef_orientation), np.asarray(chunk.gripper, dtype=np.float64).reshape(HORIZON, 1)], axis=1)
    npz_path = snap_dir / f"{snapshot_id}.npz"
    np.savez_compressed(npz_path, body_rgb=np.asarray(body, dtype=np.uint8), wrist_rgb=np.asarray(wrist, dtype=np.uint8), state_v1=state, eef_world_m=np.asarray(eef_world, dtype=np.float64), eef_local_m=eef_local, anchor_world_m=anchor, target_world_m=target, expert_physical_local=physical, expert_timestamps=np.asarray(chunk.point_timestamps, dtype=np.float64))
    payload = {"schema": "CORELAB_PI0160_EXPERT_SNAPSHOT_V1", "condition_id": condition, "snapshot_id": snapshot_id, "target_id": TARGET_ID, "target_position_world_m": target.tolist(), "target_excluded_from_model_input": True, "observation_timestamp": float(sim_time), "simulator_instance_id": str(native.stage21_simulator_instance_id), "source_frame": SOURCE_FRAME, "tcp_semantic_id": TCP, "body_ppm": str(body_path), "wrist_ppm": str(wrist_path), "body_sha256": body_sha, "wrist_sha256": wrist_sha, "snapshot_npz": str(npz_path), "snapshot_npz_sha256": sha256_file(npz_path), "state_v1": state.tolist(), "state_v1_sha256": sha256_array(state), "eef_world_m": np.asarray(eef_world, dtype=np.float64).tolist(), "eef_local_m": eef_local.tolist(), "anchor_world_m": anchor.tolist(), "expert_physical_local": physical.tolist(), "expert_timestamps": np.asarray(chunk.point_timestamps, dtype=np.float64).tolist(), "expert_chunk_id": chunk.chunk_id, "expert_chunk_source": chunk.source, "expert_chunk_sha256": sha256_array(np.concatenate([chunk.point_timestamps[:, None], chunk.eef_position, chunk.eef_orientation, np.asarray(chunk.gripper).reshape(-1, 1)], axis=1)), "language": LANGUAGE, "native_control_dt_s": float(native.dt), "label_source": "independent_same_scene_smoothstep_to_hold", "model_input_fields": ["body_rgb", "wrist_rgb", "state_v1", "language"], "model_input_excludes": ["target_position_world_m", "target_id", "expert_physical_local", "condition_id"]}
    write_json(snap_dir / f"{snapshot_id}.json", payload)
    return payload, chunk


def _rot6d_rows(quaternions: Any) -> np.ndarray:
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows
    return rotation_matrix_to_rot6d_rows(quat_to_matrix(np.asarray(quaternions, dtype=np.float64)))


def _run_condition(args: argparse.Namespace, condition: str, trajectory: Mapping[str, np.ndarray], output: Path, native_helpers: Mapping[str, Any]) -> dict[str, Any]:
    build_native, capture_scene, current_eef, current_state, load_pose, set_target, policy_step, tensor_hash, initial_bind, update_reference = (native_helpers[key] for key in ("build_native", "capture_scene", "current_eef", "current_state", "load_pose", "set_target", "policy_step", "tensor_hash", "initial_bind", "update_reference"))
    cond_root = output / "conditions" / condition
    require(not cond_root.exists(), f"condition output exists: {cond_root}")
    cond_root.mkdir(parents=True)
    native = env = None
    rows: list[dict[str, Any]] = []
    segments: list[list[dict[str, Any]]] = []
    snapshots: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    state_events: list[dict[str, Any]] = []
    status = "BLOCKED"
    error: dict[str, Any] | None = None
    try:
        native_args = argparse.Namespace(mode="gt_receding", source_root=Path(args.wbc_source_root), trajectory=Path(args.trajectory), arm=Path(args.arm), dog=Path(args.dog), sim_device="cuda:0", seed=SEED)
        native, env, actors, loads, _ = build_native(native_args, trajectory)
        require(float(native.dt) > 0.0 and np.isfinite(float(native.dt)), "native dt invalid")
        observed_schedule = tuple(int(round(0.9750000000000001 / float(native.dt))) for _ in range(3)) + (int(round(2.0 * 0.9750000000000001 / float(native.dt))),)
        require(observed_schedule == SCHEDULE, f"native schedule drift: {observed_schedule}")
        set_target(native, TARGET_WORLD)
        anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3).astype(np.float64)
        # Importing the accepted helper keeps the one-time reset mutation byte-identical.
        import stage43_corelab_initial_condition as stage43
        setup = stage43._apply_initial_base_delta(native, env, condition, DELTAS[condition])
        state_events.append(setup)
        expert_cfg = native_helpers["expert_config"](horizon=HORIZON, action_sample_period_s=ACTION_PERIOD, motion_duration_s=MOTION_DURATION)
        gate = native_helpers["gate"]()
        sim_time = 0.0
        snapshot, chunk = _capture_snapshot(native, output=output, condition=condition, snapshot_id="q0", sim_time=sim_time, anchor=anchor, target=TARGET_WORLD, capture_scene=capture_scene, current_eef=current_eef, current_state=current_state, build_chunk=native_helpers["build_chunk"], expert_cfg=expert_cfg, write_ppm=native_helpers["write_ppm"], tensor_hash=tensor_hash)
        gate_result = gate.evaluate(chunk, now=sim_time, achieved_position=np.asarray(snapshot["eef_local_m"], dtype=np.float64), achieved_orientation=current_eef(native)[1])
        gates.append({"snapshot_id": "q0", "result": gate_result.as_dict()})
        require(gate_result.decision.value != "REJECT", f"SafetyGate rejected q0: {gate_result.as_dict()}")
        snapshots.append(snapshot)
        executor, bind_event, state_event = initial_bind(native, env, chunk, sim_time, {"stage": "Stage45F", "pi_gate": PI_GATE, "mode": "PI0160_INDEPENDENT_EXPERT_ONLY", "condition_id": condition, "gr00t_to_wbc": False})
        state_events.append(state_event)
        current_chunk = chunk
        update_count = 0
        for segment_index, step_count in enumerate(SCHEDULE):
            segment, sim_time, termination = native_helpers["step_rows"](native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=step_count, update_count=update_count)
            require(len(segment) == step_count and termination is None, f"{condition} segment {segment_index} terminated: {termination}, rows={len(segment)}")
            for row in segment:
                row["condition_id"] = condition
                row["desired_pos_world"] = (anchor + np.asarray(row["desired_pos"], dtype=np.float64)).tolist()
                row["achieved_pos_world"] = (anchor + np.asarray(row["achieved_pos"], dtype=np.float64)).tolist()
                row["target_position_world_m"] = TARGET_WORLD.tolist()
                row["reference_role"] = "independent_expert_desired_eef"
                row["gr00t_output_sent_to_wbc"] = False
            segments.append(segment)
            rows.extend(segment)
            if segment_index == len(SCHEDULE) - 1:
                break
            sim_time_before = sim_time
            next_snapshot, next_chunk = _capture_snapshot(native, output=output, condition=condition, snapshot_id=f"q{segment_index + 1}", sim_time=sim_time, anchor=anchor, target=TARGET_WORLD, capture_scene=capture_scene, current_eef=current_eef, current_state=current_state, build_chunk=native_helpers["build_chunk"], expert_cfg=expert_cfg, write_ppm=native_helpers["write_ppm"], tensor_hash=tensor_hash)
            gate_result = gate.evaluate(next_chunk, now=sim_time, achieved_position=np.asarray(next_snapshot["eef_local_m"], dtype=np.float64), achieved_orientation=current_eef(native)[1])
            gates.append({"snapshot_id": f"q{segment_index + 1}", "result": gate_result.as_dict()})
            require(gate_result.decision.value != "REJECT", f"SafetyGate rejected q{segment_index + 1}: {gate_result.as_dict()}")
            previous_position = np.asarray(current_chunk.eef_position[min(int(round(0.9750000000000001 / ACTION_PERIOD)), current_chunk.point_count - 1)], dtype=np.float64)
            previous_quat = np.asarray(current_chunk.eef_orientation[min(int(round(0.9750000000000001 / ACTION_PERIOD)), current_chunk.point_count - 1)], dtype=np.float64)
            update_event, update_state = update_reference(native, env, executor, current_chunk, next_chunk, sim_time, int(round(0.9750000000000001 / float(native.dt))), update_count + 1)
            # The accepted helper already measured achieved/reference continuity;
            # retain explicit expert-only and no-reset declarations in this layer.
            update_event = dict(update_event)
            update_event.update({"condition_id": condition, "snapshot_id": f"q{segment_index + 1}", "sim_time_s": sim_time, "previous_chunk_id": current_chunk.chunk_id, "new_chunk_id": next_chunk.chunk_id, "reference_only": True, "gr00t_output_sent_to_wbc": False, "physics_reset": False, "physics_teleport": False, "policy_history_reset": False, "sim_time_before_capture_s": sim_time_before})
            handoffs.append(update_event)
            state_events.append(update_state)
            snapshots.append(next_snapshot)
            current_chunk = next_chunk
            update_count += 1
        require(len(rows) == sum(SCHEDULE), f"{condition} row count drift: {len(rows)}")
        require(update_count == 3 and int(native.stage21_reference_update_count) == 3 and int(native.stage21_wbc_forward_count) == sum(SCHEDULE), f"{condition} native counter drift")
        require(all(item["result"]["decision"] in {"ACCEPT", "WARN"} for item in gates), f"{condition} inadmissible SafetyGate decision")
        positions = np.vstack((np.asarray(snapshot["eef_world_m"], dtype=np.float64), np.asarray([row["achieved_pos_world"] for row in rows], dtype=np.float64)))
        desired = np.asarray([row["desired_pos_world"] for row in rows], dtype=np.float64)
        dists = np.linalg.norm(positions - TARGET_WORLD[None, :], axis=1)
        track = np.linalg.norm(desired - positions[1:], axis=1)
        expert_ref = np.asarray([row["desired_pos_world"] for row in rows], dtype=np.float64)
        status = "PASS"
        metrics = {"frame": "world", "eef_point": TCP, "target_world_m": TARGET_WORLD.tolist(), "initial_target_error_m": float(dists[0]), "minimum_target_error_m": float(np.min(dists)), "minimum_error_sample_index": int(np.argmin(dists)), "final_target_error_m": float(dists[-1]), "absolute_progress_m": float(dists[0] - dists[-1]), "best_progress_m": float(dists[0] - np.min(dists)), "normalized_progress": float((dists[0] - dists[-1]) / dists[0]) if dists[0] > 0 else None, "tracking_rms_m": float(np.sqrt(np.mean(track * track))), "tracking_max_m": float(np.max(track)), "reference_path_length_m": float(np.sum(np.linalg.norm(np.diff(expert_ref, axis=0), axis=1))), "achieved_path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))), "progress_sign": "POSITIVE" if dists[0] > dists[-1] else ("ZERO" if np.isclose(dists[0], dists[-1]) else "NEGATIVE"), "sample_count_including_S0": int(len(dists)), "threshold_used": None}
        write_json(cond_root / "condition_setup.json", setup)
        write_json(cond_root / "expert_snapshots.json", {"schema": "CORELAB_PI0160_EXPERT_SNAPSHOTS_V1", "snapshots": snapshots})
        write_json(cond_root / "safety_gate.json", {"schema": "CORELAB_PI0160_SAFETY_GATE_V1", "decisions": gates})
        write_json(cond_root / "initial_bind.json", bind_event)
        write_json(cond_root / "handoffs.json", {"events": handoffs})
        write_json(cond_root / "state_events.json", {"events": state_events})
        write_json(cond_root / "rows.json", {"schema": "CORELAB_PI0160_EXPERT_ROWS_V1", "rows": rows, "segments": segments})
        write_json(cond_root / "metrics.json", metrics)
        write_json(cond_root / "condition_metadata.json", {"schema": "CORELAB_PI0160_EXPERT_CONDITION_V1", "condition_id": condition, "status": status, "target_id": TARGET_ID, "delta_base_xy_m": list(DELTAS[condition]), "target_world_m": TARGET_WORLD.tolist(), "schedule": list(SCHEDULE), "wbc_step_count": int(native.stage21_wbc_forward_count), "reference_update_count": int(native.stage21_reference_update_count), "gr00t_forward_count": 0, "model_forward_count": 0, "gr00t_output_sent_to_wbc": False, "native_control_dt_s": float(native.dt), "safety_gate_decisions": [item["result"]["decision"] for item in gates], "metrics": metrics, "loads": loads, "language": LANGUAGE})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[-4000:]}
        status = "BLOCKED" if not rows else "PARTIAL"
        write_json(cond_root / "failure.json", {"schema": "CORELAB_PI0160_EXPERT_FAILURE_V1", "status": status, "condition_id": condition, "error": error, "rows_saved": len(rows), "snapshots_saved": len(snapshots), "wbc_forward_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "gr00t_forward_count": 0, "no_retry_after_start": True})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {"condition_id": condition, "status": status, "error": error, "output_dir": str(cond_root), "rows": len(rows), "snapshots": len(snapshots), "wbc_steps": int(sum(SCHEDULE) if status == "PASS" else getattr(native, "stage21_wbc_forward_count", 0) if native is not None else 0), "reference_updates": int(3 if status == "PASS" else getattr(native, "stage21_reference_update_count", 0) if native is not None else 0), "gr00t_forward_count": 0, "model_forward_count": 0, "metrics_path": str(cond_root / "metrics.json") if (cond_root / "metrics.json").is_file() else None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    head = str(args.repository_head).lower()
    require(len(head) == 40 and all(char in "0123456789abcdef" for char in head), f"repository HEAD must be a concrete 40-hex SHA: {head}")
    snapshot = _nvidia_snapshot()
    gpu = _select_gpu(snapshot, args.gpu_uuid)
    runtime = _prepare_native_environment(output, args, str(gpu["uuid"]))
    admission = {"schema": "CORELAB_PI0160_MATCHED_EXPERT_ADMISSION_V1", "status": "ADMITTED_FOR_DIRECT_EXECUTION", "token": ADMISSION_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "head": head}, "target": {"target_id": TARGET_ID, "target_world_m": TARGET_WORLD.tolist(), "language": LANGUAGE, "target_xyz_in_model_input": False}, "conditions": [{"condition_id": c, "delta_base_xy_m": list(DELTAS[c])} for c in CONDITION_ORDER], "schedule_native_ticks": list(SCHEDULE), "expert_config": {"horizon": HORIZON, "action_sample_period_s": ACTION_PERIOD, "motion_duration_s": MOTION_DURATION, "orientation": "constant_current", "gripper": 0.5}, "gpu": {"selected": gpu, "snapshot": snapshot}, "runtime": runtime, "budget": {"technical_recovery_attempts_cap": 2, "simulator_instance_attempts_cap": 7, "native_ticks_cap": 1715, "binds_cap": 7, "reference_updates_cap": 21, "planned_conditions": 5, "planned_ticks": 1225, "planned_binds": 5, "planned_updates": 15}, "execution_boundary": {"expert_wbc_simulator": True, "gr00t_forward": 0, "model_forward": 0, "training": False, "icl": False, "pbs": False, "h200": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "gr00t_output_sent_to_wbc": False, "merge": False}}
    admission_sha = write_json(output / "admission.json", admission)
    # Native imports occur only after the admission artifact is sealed.
    from stage21_native_receding_horizon import _build_native, _capture_scene, _current_eef, _current_state_v1, _load_pose_file, _policy_step, _set_target
    from stage40_corelab_phase import _initial_bind, _step_rows, _tensor_hash, _update_reference
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.simulation import SameSceneExpertConfigV1, build_same_scene_expert_chunk, write_rgb_ppm
    trajectory = _load_pose_file(Path(args.trajectory))
    helpers = {"build_native": _build_native, "capture_scene": _capture_scene, "current_eef": _current_eef, "current_state": _current_state_v1, "load_pose": _load_pose_file, "set_target": _set_target, "policy_step": _policy_step, "tensor_hash": _tensor_hash, "initial_bind": _initial_bind, "update_reference": _update_reference, "step_rows": _step_rows, "gate": TrajectorySafetyGateV1, "expert_config": SameSceneExpertConfigV1, "build_chunk": build_same_scene_expert_chunk, "write_ppm": write_rgb_ppm}
    results = []
    for condition in CONDITION_ORDER:
        live = _select_gpu(_nvidia_snapshot(), str(gpu["uuid"]))
        write_json(output / f"{condition}_admission.json", {"condition_id": condition, "gpu": live, "model_attempt_started": False, "gr00t_forward_count": 0})
        result = _run_condition(args, condition, trajectory, output, helpers)
        results.append(result)
        if result["status"] != "PASS":
            break
    passed = sum(item["status"] == "PASS" for item in results)
    status = "COMPLETED" if passed == len(CONDITION_ORDER) else ("PARTIAL" if passed else "BLOCKED")
    summary = {"schema": "CORELAB_PI0160_MATCHED_EXPERT_RESULT_V1", "status": status, "token": RETURN_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "head": head}, "target": {"target_id": TARGET_ID, "target_world_m": TARGET_WORLD.tolist(), "language": LANGUAGE, "target_xyz_in_model_input": False}, "conditions": results, "counters": {"conditions_started": len(results), "conditions_passed": passed, "conditions_expected": len(CONDITION_ORDER), "expert_wbc_steps": int(sum(item["wbc_steps"] for item in results)), "reference_updates": int(sum(item["reference_updates"] for item in results)), "gr00t_forward_count": 0, "model_forward_count": 0, "model_construction_attempts": 0, "training_steps": 0, "safety_gate_rejects": int(sum(1 for item in results if item.get("error") and "SafetyGate rejected" in str(item["error"]))), "raw_model_rows": 0}, "planned_budget": {"native_ticks": 1225, "binds": 5, "reference_updates": 15, "gr00t_forward": 0, "model_calls_after_expert": {"conditioning": 10, "action_head": 35, "model_side": 45, "model_construction": 1}}, "gpu": {"selected": gpu, "initial_snapshot": snapshot}, "runtime": runtime, "admission_sha256": admission_sha, "execution_boundary": {"expert_wbc_simulator": True, "gr00t_forward": 0, "model_forward": 0, "training": False, "icl": False, "pbs": False, "h200": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "gr00t_output_sent_to_wbc": False, "merge": False}, "claim_boundary": "matched independent-expert reference and fixed-policy diagnostics; no model-to-WBC closed-loop or task-success claim", "recommendation": "Use matched visited-state teacher/local-field supervision as the single next intervention candidate; do not alter seed/noise or SafetyGate from this packet."}
    result_sha = write_json(output / "pi0160_matched_expert_result.json", summary)
    summary["result_sha256"] = result_sha
    write_json(output / "evidence_hashes.json", {"admission_sha256": admission_sha, "result_sha256": result_sha, "condition_result_sha256": {item["condition_id"]: sha256_file(Path(item["output_dir"]) / "condition_metadata.json") if Path(item["output_dir"], "condition_metadata.json").is_file() else None for item in results}})
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-head", required=True)
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--arm", required=True, type=Path)
    p.add_argument("--dog", required=True, type=Path)
    p.add_argument("--wbc-source-root", required=True, type=Path)
    p.add_argument("--native-env-root", required=True, type=Path)
    p.add_argument("--isaacgym-python", required=True, type=Path)
    p.add_argument("--native-compat-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--gpu-uuid", default=None)
    return p


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "token": result["token"], "conditions_passed": result["counters"]["conditions_passed"], "expert_wbc_steps": result["counters"]["expert_wbc_steps"], "reference_updates": result["counters"]["reference_updates"], "gr00t_forward_count": result["counters"]["gr00t_forward_count"], "output_root": result["conditions"][0]["output_dir"].split("/conditions/")[0] if result["conditions"] else None}, sort_keys=True), flush=True)
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
