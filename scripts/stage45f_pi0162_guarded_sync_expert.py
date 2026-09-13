#!/usr/bin/env python3
"""PI0162 guarded C0/C1 synchronized expert-only native campaign.

This runner is deliberately separate from the superseded PI0160 collector.
It applies one physical root-state delta, performs exactly one real native WBC
tick, and only then captures the synchronized S0 image/state/EEF.  No derived
rigid-body cache is edited by this runner.  GR00T is never constructed and its
output can never reach WBC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for path in (SCRIPTS, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage45f_pi0160_matched_reset_expert import (  # noqa: E402
    ACTION_PERIOD,
    BRANCH,
    DELTAS,
    HORIZON,
    LANGUAGE,
    MOTION_DURATION,
    SEED,
    SCHEDULE,
    SOURCE_FRAME,
    TARGET_ID,
    TARGET_WORLD,
    TCP,
    _nvidia_snapshot,
    _prepare_native_environment,
    _select_gpu,
)
from stage45f_pi0162_budget_guard import BudgetLedger, BudgetExceeded, BudgetGuardError  # noqa: E402


PI_GATE = "PI_GATE_STATE_0162"
PI_COMMENT = 5652576782
PR_COMMENT = 5652578721
RETURN_TOKEN = "CORELAB_PI0162_GUARDED_SYNC_DIAGNOSTIC_FOR_PI_REVIEW"
ADMISSION_TOKEN = "CORELAB_PI0162_GUARDED_CONTINUATION_ADMISSION_RECORDED"
CONDITION_ORDER = ("C0_NOMINAL", "C1_X_PLUS_5CM")


class CampaignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


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


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _state_hashes(native: Any, env: Any, tensor_hash: Any) -> dict[str, Any]:
    return {
        "root": tensor_hash(native.root_states),
        "dof": tensor_hash(native.dof_state),
        "arm_history": tensor_hash(env.arm_obs_history),
        "dog_history": tensor_hash(env.dog_obs_history),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "wbc_forward_count": int(native.stage21_wbc_forward_count),
        "reference_update_count": int(native.stage21_reference_update_count),
    }


def _apply_root_delta(native: Any, env: Any, condition: str, delta: Sequence[float], tensor_hash: Any) -> dict[str, Any]:
    """One official indexed root write; never mutate a derived cache."""
    import torch

    delta_np = np.asarray(delta, dtype=np.float64).reshape(3)
    require(np.isfinite(delta_np).all() and float(delta_np[2]) == 0.0, "initial root delta must be finite XY-only")
    require(native.stage21_wbc_forward_count == 0 and native.stage21_reference_update_count == 0, "root delta after execution")
    before = _state_hashes(native, env, tensor_hash)
    root_before = native.root_states.detach().cpu().numpy().copy()
    dof_before = native.dof_state.detach().cpu().numpy().copy()
    rigid_before = native.rigid_body_state.detach().cpu().numpy().copy()
    target_idx = int(native.stage21_target_root_index)
    target_before = root_before[target_idx].copy()
    native.root_states[0, :2] += torch.as_tensor(delta_np[:2], dtype=native.root_states.dtype, device=native.root_states.device)
    robot_index = torch.as_tensor([0], device=native.device, dtype=torch.int32)
    native.gym.set_actor_root_state_tensor_indexed(
        native.sim,
        native._stage21_gymtorch.unwrap_tensor(native.root_states),
        native._stage21_gymtorch.unwrap_tensor(robot_index),
        1,
    )
    native.gym.refresh_actor_root_state_tensor(native.sim)
    root_after = native.root_states.detach().cpu().numpy().copy()
    dof_after = native.dof_state.detach().cpu().numpy().copy()
    rigid_after = native.rigid_body_state.detach().cpu().numpy().copy()
    after = _state_hashes(native, env, tensor_hash)
    root_delta = root_after[0] - root_before[0]
    event = {
        "schema": "CORELAB_PI0162_ROOT_DELTA_V1",
        "condition_id": condition,
        "delta_base_xy_m": delta_np.tolist(),
        "mutation": "one official indexed robot root-state write before bootstrap tick",
        "physics_steps_before_bootstrap": 0,
        "settling_steps": 0,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "root_before_sha256": sha256_array(root_before),
        "root_after_sha256": sha256_array(root_after),
        "dof_before_sha256": sha256_array(dof_before),
        "dof_after_sha256": sha256_array(dof_after),
        "rigid_body_before_sha256": sha256_array(rigid_before),
        "rigid_body_after_sha256": sha256_array(rigid_after),
        "derived_cache_mutated_by_runner": False,
        "derived_cache_sync": "NONE; post-tick native physics refresh is required",
        "target_root_before": target_before.tolist(),
        "target_root_after": root_after[target_idx].tolist(),
        "root_delta_observed": root_delta.tolist(),
        "root_xy_delta_exact": bool(np.allclose(root_delta[:2], delta_np[:2], rtol=0.0, atol=1e-6)),
        "root_non_xy_unchanged": bool(np.array_equal(root_delta[2:], np.zeros_like(root_delta[2:]))),
        "target_root_unchanged": bool(np.array_equal(root_after[target_idx], target_before)),
        "other_actor_roots_unchanged": bool(np.array_equal(root_after[1:], root_before[1:])),
        "dof_state_unchanged": bool(np.array_equal(dof_after, dof_before)),
        "lifecycle_unchanged": bool(before["episode"] == after["episode"] and before["reset_buf"] == after["reset_buf"]),
        "policy_histories_unchanged": bool(before["arm_history"] == after["arm_history"] and before["dog_history"] == after["dog_history"]),
        "wbc_forward_count_after": int(native.stage21_wbc_forward_count),
        "reference_update_count_after": int(native.stage21_reference_update_count),
    }
    require(event["root_xy_delta_exact"] and event["root_non_xy_unchanged"], "root delta scope mismatch")
    require(event["target_root_unchanged"] and event["other_actor_roots_unchanged"], "root write changed target/other actor")
    require(event["dof_state_unchanged"] and event["lifecycle_unchanged"] and event["policy_histories_unchanged"], "root write changed lifecycle/DOF/history")
    require(event["wbc_forward_count_after"] == 0 and event["reference_update_count_after"] == 0, "root write consumed lifecycle counters")
    return event


def _rot6d_rows(quaternions: Any) -> np.ndarray:
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows
    return rotation_matrix_to_rot6d_rows(quat_to_matrix(np.asarray(quaternions, dtype=np.float64)))


def _capture_sync_snapshot(*, native: Any, env: Any, condition: str, snapshot_id: str, sim_time: float, anchor: np.ndarray, output: Path, capture_scene: Any, current_eef: Any, current_state: Any, tensor_hash: Any, build_chunk: Any, expert_cfg: Any, write_ppm: Any) -> tuple[dict[str, Any], Any]:
    """Capture one post-tick frame/state at a graphics-only boundary."""
    before = _state_hashes(native, env, tensor_hash)
    body, wrist = capture_scene(native)
    after = _state_hashes(native, env, tensor_hash)
    require(before == after, f"graphics refresh changed native state at {condition}/{snapshot_id}")
    eef_world, quat = current_eef(native)
    eef_world = np.asarray(eef_world, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    eef_local = eef_world - anchor
    state = current_state(native).as_array().astype(np.float32)
    target_root = native.root_states[int(native.stage21_target_root_index)].detach().cpu().numpy().astype(np.float64)
    red_counts = {"body": _red_pixel_count(body), "wrist": _red_pixel_count(wrist)}
    snap_dir = output / "conditions" / condition / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    body_path = snap_dir / f"{snapshot_id}_body.ppm"
    wrist_path = snap_dir / f"{snapshot_id}_wrist.ppm"
    body_sha = write_ppm(body_path, body)
    wrist_sha = write_ppm(wrist_path, wrist)
    geometry = {
        "body_camera_link": str(native.body_names[0]),
        "wrist_camera_link": str(native.body_names[native.ee_idx]),
        "body_camera_handle": str(native.stage21_camera_handles["body"]),
        "wrist_camera_handle": str(native.stage21_camera_handles["wrist"]),
        "resolution": [int(native.camera_props.width), int(native.camera_props.height)],
        "target_actor": "stage21_red_target",
        "target_root_index": int(native.stage21_target_root_index),
        "target_root_pose_world": target_root.tolist(),
    }
    payload = {
        "schema": "CORELAB_PI0162_SYNCHRONIZED_SNAPSHOT_V1",
        "condition_id": condition,
        "snapshot_id": snapshot_id,
        "target_id": TARGET_ID,
        "target_position_world_m": TARGET_WORLD.tolist(),
        "target_excluded_from_model_input": True,
        "observation_timestamp_s": float(sim_time),
        "wall_capture_time_ns": int(time.time_ns()),
        "simulator_instance_id": str(native.stage21_simulator_instance_id),
        "native_traj_sim_time_s": float(native.traj_sim_time[0].detach().cpu().item()),
        "native_wbc_forward_count": int(native.stage21_wbc_forward_count),
        "native_reference_update_count": int(native.stage21_reference_update_count),
        "graphics_refresh_only": True,
        "graphics_state_before": before,
        "graphics_state_after": after,
        "camera_geometry": geometry,
        "red_pixel_count": red_counts,
        "target_observable": bool(max(red_counts.values()) > 0),
        "body_ppm": str(body_path),
        "wrist_ppm": str(wrist_path),
        "body_sha256": body_sha,
        "wrist_sha256": wrist_sha,
        "body_shape": list(np.asarray(body).shape),
        "wrist_shape": list(np.asarray(wrist).shape),
        "state_v1": state.tolist(),
        "state_v1_sha256": sha256_array(state),
        "eef_world_m": eef_world.tolist(),
        "eef_world_sha256": sha256_array(eef_world),
        "eef_local_m": eef_local.tolist(),
        "anchor_world_m": anchor.tolist(),
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP,
        "model_input_fields": ["body_rgb", "wrist_rgb", "state_v1", "language"],
        "model_input_excludes": ["target_position_world_m", "target_id", "eef_world_m", "expert_physical_local", "condition_id"],
        "language": LANGUAGE,
    }
    npz_path = snap_dir / f"{snapshot_id}.npz"
    np.savez_compressed(npz_path, body_rgb=np.asarray(body, dtype=np.uint8), wrist_rgb=np.asarray(wrist, dtype=np.uint8), state_v1=state, eef_world_m=eef_world, eef_local_m=eef_local, anchor_world_m=anchor, target_world_m=TARGET_WORLD)
    payload["snapshot_npz"] = str(npz_path)
    payload["snapshot_npz_sha256"] = sha256_file(npz_path)
    write_json(snap_dir / f"{snapshot_id}.json", payload)
    require(payload["target_observable"], f"SYNC_OR_OBSERVABILITY_BLOCKED: target not observable at {condition}/{snapshot_id}")
    chunk = build_chunk(
        chunk_id=f"{condition}-{snapshot_id}",
        observation_timestamp=float(sim_time),
        current_position=eef_local,
        target_position=TARGET_WORLD - anchor,
        current_quaternion=quat,
        config=expert_cfg,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP,
    )
    physical = np.concatenate([chunk.eef_position, _rot6d_rows(chunk.eef_orientation), np.asarray(chunk.gripper, dtype=np.float64).reshape(HORIZON, 1)], axis=1)
    payload["expert_chunk_id"] = chunk.chunk_id
    payload["expert_chunk_source"] = chunk.source
    payload["expert_chunk_sha256"] = sha256_array(np.concatenate([chunk.point_timestamps[:, None], chunk.eef_position, chunk.eef_orientation, np.asarray(chunk.gripper).reshape(-1, 1)], axis=1))
    payload["expert_physical_local"] = physical.tolist()
    payload["expert_timestamps"] = np.asarray(chunk.point_timestamps, dtype=np.float64).tolist()
    payload["expert_physical_local_sha256"] = sha256_array(physical)
    write_json(snap_dir / f"{snapshot_id}.json", payload, overwrite=True)
    return payload, chunk


def _red_pixel_count(image: Any) -> int:
    rgb = np.asarray(image, dtype=np.uint8)
    red = (rgb[..., 0] > 100) & (rgb[..., 0] > rgb[..., 1] * 1.35) & (rgb[..., 0] > rgb[..., 2] * 1.35)
    return int(np.count_nonzero(red))


def _decorate_rows(rows: Sequence[Mapping[str, Any]], condition: str) -> list[dict[str, Any]]:
    decorated = []
    for row in rows:
        item = dict(row)
        # stage40_live_one_handoff already returns world-frame fields.  Preserve
        # the original names and bind explicit aliases without adding anchor a
        # second time (the PI0161 frame audit found that exact bug).
        item["condition_id"] = condition
        item["desired_pos_world"] = list(item["desired_pos"])
        item["achieved_pos_world"] = list(item["achieved_pos"])
        item["target_position_world_m"] = TARGET_WORLD.tolist()
        item["world_frame_source"] = "STAGE40_LIVE_ONE_HANDOFF_UPSTREAM_WORLD_FIELDS"
        item["reference_role"] = "independent_expert_desired_eef"
        item["gr00t_output_sent_to_wbc"] = False
        decorated.append(item)
    return decorated


def _pose_at(chunk: Any, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(chunk.point_timestamps, dtype=np.float64)
    index = int(np.clip(np.searchsorted(ts, float(timestamp), side="right") - 1, 0, len(ts) - 1))
    return np.asarray(chunk.eef_position[index], dtype=np.float64), np.asarray(chunk.eef_orientation[index], dtype=np.float64)


def _executor_prefix(chunk: Any) -> tuple[int, float]:
    timestamps = np.asarray(chunk.point_timestamps, dtype=np.float64)
    deltas = np.diff(timestamps)
    require(timestamps.shape == (HORIZON,) and np.allclose(deltas, ACTION_PERIOD, rtol=0.0, atol=1e-9), "expert chunk timestamp drift")
    prefix = int(round(0.9750000000000001 / float(deltas[0])))
    require(1 <= prefix <= HORIZON, "executor prefix outside H40")
    return prefix, float(deltas[0])


def _metrics(snapshots: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positions = np.vstack([np.asarray(snapshots[0]["eef_world_m"], dtype=np.float64), np.asarray([row["achieved_pos_world"] for row in rows], dtype=np.float64)])
    desired = np.asarray([row["desired_pos_world"] for row in rows], dtype=np.float64)
    distances = np.linalg.norm(positions - TARGET_WORLD[None, :], axis=1)
    track = np.linalg.norm(desired - positions[1:], axis=1)
    d0 = float(distances[0])
    dmin = float(np.min(distances))
    dfinal = float(distances[-1])
    return {
        "frame": "world",
        "eef_point": TCP,
        "target_world_m": TARGET_WORLD.tolist(),
        "initial_target_error_m": d0,
        "minimum_target_error_m": dmin,
        "final_target_error_m": dfinal,
        "absolute_progress_m": d0 - dfinal,
        "best_progress_m": d0 - dmin,
        "normalized_progress": (d0 - dfinal) / d0 if d0 > 0 else None,
        "progress_sign": "POSITIVE" if d0 > dfinal else ("ZERO" if np.isclose(d0, dfinal) else "NEGATIVE"),
        "tracking_rms_m": float(np.sqrt(np.mean(track * track))) if track.size else None,
        "tracking_max_m": float(np.max(track)) if track.size else None,
        "reference_path_length_m": float(np.sum(np.linalg.norm(np.diff(desired, axis=0), axis=1))) if len(desired) > 1 else 0.0,
        "achieved_path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0,
        "sample_count_including_S0": int(len(positions)),
        "threshold_used": None,
    }


def _run_condition(args: argparse.Namespace, guard: BudgetLedger, condition: str, trajectory: Mapping[str, np.ndarray], output: Path, helpers: Mapping[str, Any]) -> dict[str, Any]:
    build_native = helpers["build_native"]
    capture_scene = helpers["capture_scene"]
    current_eef = helpers["current_eef"]
    current_state = helpers["current_state"]
    policy_step = helpers["policy_step"]
    tensor_hash = helpers["tensor_hash"]
    set_target = helpers["set_target"]
    write_ppm = helpers["write_ppm"]
    build_chunk = helpers["build_chunk"]
    expert_config = helpers["expert_config"]
    gate_class = helpers["gate"]
    cond_root = output / "conditions" / condition
    require(not cond_root.exists(), f"condition output exists: {cond_root}")
    cond_root.mkdir(parents=True)
    native = env = None
    rows: list[dict[str, Any]] = []
    segments: list[list[dict[str, Any]]] = []
    snapshots: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    state_events: list[dict[str, Any]] = []
    status = "BLOCKED"
    error: dict[str, Any] | None = None
    try:
        guard.reserve("simulator_constructor", condition=condition, phase="native_constructor", head=args.repository_head)
        native_args = argparse.Namespace(mode="gt_receding", source_root=Path(args.wbc_source_root), trajectory=Path(args.trajectory), arm=Path(args.arm), dog=Path(args.dog), sim_device="cuda:0", seed=SEED)
        native, env, actors, loads, _ = build_native(native_args, trajectory)
        require(float(native.dt) > 0.0 and np.isfinite(float(native.dt)), "native dt invalid")
        require(int(round(0.9750000000000001 / float(native.dt))) == 49, f"native handoff dt drift: {native.dt}")
        set_target(native, TARGET_WORLD)
        anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3).astype(np.float64)
        setup = _apply_root_delta(native, env, condition, DELTAS[condition], tensor_hash)
        state_events.append(setup)
        write_json(cond_root / "root_delta.json", setup)

        guard.reserve("explicit_control_tick", condition=condition, phase="synchronized_bootstrap_tick", tick_index=0, head=args.repository_head)
        bootstrap_done, bootstrap_policy = policy_step(native, env, actors)
        bootstrap_termination = native.pop_stage16_termination()
        sim_time = float(native.traj_sim_time[0].detach().cpu().item())
        bootstrap = {
            "schema": "CORELAB_PI0162_BOOTSTRAP_TICK_V1",
            "condition_id": condition,
            "done": bool(bootstrap_done),
            "termination": bootstrap_termination,
            "policy": bootstrap_policy,
            "sim_time_s": sim_time,
            "native_dt_s": float(native.dt),
            "wbc_forward_count": int(native.stage21_wbc_forward_count),
            "reference_update_count": int(native.stage21_reference_update_count),
            "real_native_tick": True,
        }
        write_json(cond_root / "bootstrap_tick.json", bootstrap)
        require(sim_time > 0.0 and not bootstrap_done and bootstrap_termination is None, f"bootstrap tick terminated: {bootstrap}")

        expert_cfg = expert_config(horizon=HORIZON, action_sample_period_s=ACTION_PERIOD, motion_duration_s=MOTION_DURATION)
        snapshot, chunk = _capture_sync_snapshot(native=native, env=env, condition=condition, snapshot_id="q0", sim_time=sim_time, anchor=anchor, output=output, capture_scene=capture_scene, current_eef=current_eef, current_state=current_state, tensor_hash=tensor_hash, build_chunk=build_chunk, expert_cfg=expert_cfg, write_ppm=write_ppm)
        snapshots.append(snapshot)
        gate = gate_class()
        gate_result = gate.evaluate(chunk, now=sim_time, achieved_position=np.asarray(snapshot["eef_local_m"], dtype=np.float64), achieved_orientation=current_eef(native)[1])
        gates.append({"snapshot_id": "q0", "result": gate_result.as_dict()})
        require(gate_result.decision.value != "REJECT", f"SafetyGate rejected q0: {gate_result.as_dict()}")

        # The first WBC forward already happened.  q0 is therefore installed
        # with load_reference_chunk, never with initial_bind_reference.
        from go2arx5_vla.runtime import RecedingHorizonExecutorV1
        prefix_steps, action_dt = _executor_prefix(chunk)
        executor = RecedingHorizonExecutorV1(prefix_steps=prefix_steps, allowed_prefix_steps=(prefix_steps,), provenance={"stage": "Stage45F", "pi_gate": PI_GATE, "native_handoff_s": 0.9750000000000001, "native_handoff_steps": 49, "executor_prefix_steps": prefix_steps, "gr00t_to_wbc": False})
        guard.reserve("reference_update", condition=condition, phase="q0_load_reference_chunk", update_index=0, head=args.repository_head)
        before = _state_hashes(native, env, tensor_hash)
        executor_initial = executor.submit_initial(chunk, receive_timestamp=sim_time, execute_timestamp=sim_time, current_achieved_position=np.asarray(snapshot["eef_local_m"], dtype=np.float64), current_achieved_orientation=current_eef(native)[1])
        native_initial = native.load_reference_chunk(chunk)
        after = _state_hashes(native, env, tensor_hash)
        require(before["root"] == after["root"] and before["dof"] == after["dof"] and before["episode"] == after["episode"] and before["arm_history"] == after["arm_history"] and before["dog_history"] == after["dog_history"], "q0 reference load changed physics/lifecycle")
        state_events.append({"phase": "Q0_LOAD_REFERENCE_AFTER_FIRST_WBC", "before": before, "after": after, "unchanged": before == after})
        write_json(cond_root / "q0_load_reference.json", {"native": native_initial, "executor": executor_initial.as_dict(), "native_handoff_steps": 49, "executor_prefix_steps": prefix_steps, "action_sample_dt_s": action_dt, "physics_reset": False, "physics_teleport": False, "policy_history_reset": False})

        current_chunk = chunk
        update_count = 0
        for segment_index, step_count in enumerate(SCHEDULE):
            segment, sim_time, termination = helpers["step_rows"](
                native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=step_count, update_count=update_count,
                before_step=lambda *, step, sim_time, update_count, _condition=condition, _segment=segment_index: guard.reserve("explicit_control_tick", condition=_condition, phase="post_s0_control_tick", segment_index=_segment, tick_index=int(step), update_index=int(update_count), head=args.repository_head),
            )
            decorated = _decorate_rows(segment, condition)
            segments.append(decorated)
            rows.extend(decorated)
            # Persist every returned row before checking termination/length.
            write_json(cond_root / "rows.json", {"schema": "CORELAB_PI0162_EXPERT_ROWS_V1", "rows": rows, "segments": segments}, overwrite=True)
            if len(segment) != int(step_count) or termination is not None:
                raise CampaignError(f"{condition} segment {segment_index} terminated: {termination}, rows={len(segment)}")
            if segment_index == len(SCHEDULE) - 1:
                break
            next_snapshot, next_chunk = _capture_sync_snapshot(native=native, env=env, condition=condition, snapshot_id=f"q{segment_index + 1}", sim_time=sim_time, anchor=anchor, output=output, capture_scene=capture_scene, current_eef=current_eef, current_state=current_state, tensor_hash=tensor_hash, build_chunk=build_chunk, expert_cfg=expert_cfg, write_ppm=write_ppm)
            snapshots.append(next_snapshot)
            next_gate = gate.evaluate(next_chunk, now=sim_time, achieved_position=np.asarray(next_snapshot["eef_local_m"], dtype=np.float64), achieved_orientation=current_eef(native)[1])
            gates.append({"snapshot_id": f"q{segment_index + 1}", "result": next_gate.as_dict()})
            require(next_gate.decision.value != "REJECT", f"SafetyGate rejected q{segment_index + 1}: {next_gate.as_dict()}")
            current_position, current_quat = current_eef(native)
            old_position, old_quat = _pose_at(current_chunk, sim_time)
            guard.reserve("reference_update", condition=condition, phase="handoff_load_reference_chunk", update_index=update_count + 1, head=args.repository_head)
            before = _state_hashes(native, env, tensor_hash)
            executor_update = executor.submit_update(next_chunk, receive_timestamp=sim_time, execute_timestamp=sim_time, current_achieved_position=np.asarray(current_position, dtype=np.float64) - anchor, current_achieved_orientation=current_quat, previous_predicted_position=old_position, previous_predicted_orientation=old_quat, executed_prefix_steps=prefix_steps)
            native_update = native.load_reference_chunk(next_chunk)
            after = _state_hashes(native, env, tensor_hash)
            require(before["root"] == after["root"] and before["dof"] == after["dof"] and before["episode"] == after["episode"] and before["arm_history"] == after["arm_history"] and before["dog_history"] == after["dog_history"], f"reference update changed physics/lifecycle at q{segment_index + 1}")
            handoff = {"native": native_update, "executor": executor_update.as_dict(), "condition_id": condition, "snapshot_id": f"q{segment_index + 1}", "handoff_index": update_count + 1, "sim_time_s": sim_time, "native_handoff_steps": 49, "executor_prefix_steps": prefix_steps, "previous_chunk_id": current_chunk.chunk_id, "new_chunk_id": next_chunk.chunk_id, "reference_only": True, "physics_reset": False, "physics_teleport": False, "policy_history_reset": False, "gr00t_output_sent_to_wbc": False}
            handoffs.append(handoff)
            state_events.append({"phase": "RUNNING_UPDATE", "handoff_index": update_count + 1, "before": before, "after": after, "unchanged": before == after})
            write_json(cond_root / "handoffs.json", {"events": handoffs}, overwrite=True)
            current_chunk = next_chunk
            update_count += 1

        require(len(rows) == sum(SCHEDULE), f"row count drift: {len(rows)}")
        require(update_count == 3 and int(native.stage21_reference_update_count) == 4 and int(native.stage21_wbc_forward_count) == 1 + sum(SCHEDULE), "native counter drift")
        require(all(item["result"]["decision"] in {"ACCEPT", "WARN"} for item in gates), "inadmissible SafetyGate decision")
        native_time = float(native.traj_sim_time[0].detach().cpu().item())
        ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
        ref_l = float(native.traj_batch.L[0].detach().cpu().item())
        endpoint = bool(native_time >= float(current_chunk.validity_end - current_chunk.validity_start) - max(float(native.dt) * 0.5, 1e-9) and ref_s >= ref_l - max(1e-4, 1e-3 * max(ref_l, 1.0)))
        require(endpoint, f"reference endpoint not reached: time={native_time} s={ref_s} L={ref_l}")
        metrics = _metrics(snapshots, rows)
        metrics.update({"native_reference_time_s": native_time, "native_reference_s": ref_s, "native_reference_length": ref_l, "endpoint_reached": endpoint})
        write_json(cond_root / "expert_snapshots.json", {"schema": "CORELAB_PI0162_SYNCHRONIZED_SNAPSHOTS_V1", "snapshots": snapshots})
        write_json(cond_root / "safety_gate.json", {"schema": "CORELAB_PI0162_SAFETY_GATE_V1", "decisions": gates})
        write_json(cond_root / "state_events.json", {"events": state_events})
        write_json(cond_root / "metrics.json", metrics)
        write_json(cond_root / "condition_metadata.json", {"schema": "CORELAB_PI0162_EXPERT_CONDITION_V1", "condition_id": condition, "status": "PASS", "qualified_q0": True, "target_id": TARGET_ID, "delta_base_xy_m": list(DELTAS[condition]), "target_world_m": TARGET_WORLD.tolist(), "schedule": list(SCHEDULE), "bootstrap_ticks": 1, "wbc_step_count": int(native.stage21_wbc_forward_count), "reference_update_count": int(native.stage21_reference_update_count), "gr00t_forward_count": 0, "model_forward_count": 0, "gr00t_output_sent_to_wbc": False, "native_control_dt_s": float(native.dt), "safety_gate_decisions": [item["result"]["decision"] for item in gates], "metrics": metrics, "language": LANGUAGE})
        status = "PASS"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[-4000:]}
        status = "PARTIAL" if snapshots else "BLOCKED"
        # Persist all rows and the failure before returning/raising.  This is
        # intentionally a new evidence directory; old PI0160 rows are never
        # repaired or overwritten.
        write_json(cond_root / "rows.json", {"schema": "CORELAB_PI0162_EXPERT_ROWS_V1", "rows": rows, "segments": segments}, overwrite=True)
        write_json(cond_root / "expert_snapshots.json", {"schema": "CORELAB_PI0162_SYNCHRONIZED_SNAPSHOTS_V1", "snapshots": snapshots}, overwrite=True)
        write_json(cond_root / "safety_gate.json", {"schema": "CORELAB_PI0162_SAFETY_GATE_V1", "decisions": gates}, overwrite=True)
        write_json(cond_root / "state_events.json", {"events": state_events}, overwrite=True)
        write_json(cond_root / "failure.json", {"schema": "CORELAB_PI0162_EXPERT_FAILURE_V1", "status": status, "condition_id": condition, "error": error, "rows_saved": len(rows), "snapshots_saved": len(snapshots), "wbc_forward_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "gr00t_forward_count": 0, "no_retry_after_start": True})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {"condition_id": condition, "status": status, "qualified_q0": bool(snapshots), "error": error, "output_dir": str(cond_root), "rows": len(rows), "snapshots": len(snapshots), "wbc_steps": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_updates": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "gr00t_forward_count": 0, "model_forward_count": 0, "metrics_path": str(cond_root / "metrics.json") if (cond_root / "metrics.json").is_file() else None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    head = str(args.repository_head).lower()
    require(len(head) == 40 and all(c in "0123456789abcdef" for c in head), f"repository HEAD must be concrete SHA: {head}")
    snapshot = _nvidia_snapshot()
    gpu = _select_gpu(snapshot, args.gpu_uuid)
    runtime = _prepare_native_environment(output, args, str(gpu["uuid"]))
    ledger = BudgetLedger(args.budget_ledger, args.campaign_id, output_root=output)
    admission = {
        "schema": "CORELAB_PI0162_GUARDED_SYNC_ADMISSION_V1",
        "status": "ADMITTED_FOR_DIRECT_EXECUTION",
        "token": ADMISSION_TOKEN,
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {"branch": BRANCH, "head": head},
        "target": {"target_id": TARGET_ID, "target_world_m": TARGET_WORLD.tolist(), "language": LANGUAGE, "target_xyz_in_model_input": False},
        "conditions": [{"condition_id": c, "delta_base_xy_m": list(DELTAS[c])} for c in CONDITION_ORDER],
        "schedule": {"bootstrap_native_ticks": 1, "post_s0_native_ticks": list(SCHEDULE), "native_handoff_steps": 49, "reference_updates_per_condition": 4},
        "fixed_protocol": {"seed": SEED, "horizon": HORIZON, "action_sample_period_s": ACTION_PERIOD, "motion_duration_s": MOTION_DURATION, "source_frame": SOURCE_FRAME, "tcp": TCP},
        "gpu": {"selected": gpu, "snapshot": snapshot},
        "runtime": runtime,
        "budget": {"campaign_id": args.campaign_id, "ledger_path": str(Path(args.budget_ledger).resolve()), "caps": dict(ledger.budgets), "nominal": {"simulator_constructor": 2, "explicit_control_tick": 492, "reference_update": 8, "groot_constructor": 1, "conditioning": 4, "action_head": 14, "model_side_call": 18}},
        "history_audit": {"path": str(args.history_audit) if args.history_audit else None, "sha256": sha256_file(Path(args.history_audit)) if args.history_audit and Path(args.history_audit).is_file() else None, "pi0160_overage_preserved": True, "v1_v5_unknown_preserved": True},
        "execution_boundary": {"gr00t_forward": 0, "model_forward": 0, "training": False, "backward": False, "optimizer_step": 0, "wbc_expert_only": True, "gr00t_output_sent_to_wbc": False, "icl": False, "pbs": False, "h200": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "merge": False},
    }
    admission_sha = write_json(output / "admission.json", admission)
    from stage21_native_receding_horizon import _build_native, _capture_scene, _current_eef, _current_state_v1, _load_pose_file, _policy_step, _set_target
    from stage40_live_one_handoff import _step_rows, _tensor_hash
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.simulation import SameSceneExpertConfigV1, build_same_scene_expert_chunk, write_rgb_ppm
    trajectory = _load_pose_file(Path(args.trajectory))
    helpers = {"build_native": _build_native, "capture_scene": _capture_scene, "current_eef": _current_eef, "current_state": _current_state_v1, "policy_step": _policy_step, "set_target": _set_target, "tensor_hash": _tensor_hash, "step_rows": _step_rows, "gate": TrajectorySafetyGateV1, "expert_config": SameSceneExpertConfigV1, "build_chunk": build_same_scene_expert_chunk, "write_ppm": write_rgb_ppm}
    results = []
    try:
        for condition in CONDITION_ORDER:
            live = _select_gpu(_nvidia_snapshot(), str(gpu["uuid"]))
            write_json(output / f"{condition}_admission.json", {"schema": "CORELAB_PI0162_CONDITION_ADMISSION_V1", "condition_id": condition, "gpu": live, "model_attempt_started": False, "gr00t_forward_count": 0, "budget_campaign_id": args.campaign_id})
            result = _run_condition(args, ledger, condition, trajectory, output, helpers)
            results.append(result)
    finally:
        ledger_snapshot = ledger.snapshot()
        write_json(output / "budget_ledger_snapshot.json", ledger_snapshot)
        ledger.close("PI0162_EXPERT_PROCESS_CLOSED")
    passed = sum(item["status"] == "PASS" for item in results)
    qualified = sum(bool(item.get("qualified_q0")) for item in results)
    status = "COMPLETED" if passed == len(CONDITION_ORDER) else ("PARTIAL" if qualified else "BLOCKED")
    summary = {
        "schema": "CORELAB_PI0162_GUARDED_EXPERT_RESULT_V1",
        "status": status,
        "token": RETURN_TOKEN,
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {"branch": BRANCH, "head": head},
        "target": {"target_id": TARGET_ID, "target_world_m": TARGET_WORLD.tolist(), "language": LANGUAGE, "target_xyz_in_model_input": False},
        "conditions": results,
        "counters": {"conditions_started": len(results), "conditions_passed": passed, "conditions_expected": len(CONDITION_ORDER), "qualified_q0_count": qualified, "expert_bootstrap_ticks": int(sum(1 for _ in results if _["wbc_steps"] > 0)), "expert_wbc_steps": int(sum(item["wbc_steps"] for item in results)), "reference_updates": int(sum(item["reference_updates"] for item in results)), "gr00t_forward_count": 0, "model_forward_count": 0, "training_steps": 0},
        "budget_ledger": {"campaign_id": args.campaign_id, "path": str(Path(args.budget_ledger).resolve()), "snapshot_sha256": sha256_file(output / "budget_ledger_snapshot.json")},
        "admission_sha256": admission_sha,
        "gpu": {"selected": gpu, "initial_snapshot": snapshot},
        "runtime": runtime,
        "execution_boundary": {"gr00t_forward": 0, "model_forward": 0, "training": False, "backward": False, "optimizer_step": 0, "wbc_expert_only": True, "gr00t_output_sent_to_wbc": False, "icl": False, "pbs": False, "h200": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "merge": False},
        "claim_boundary": "two post-native-tick synchronized independent-expert references; no learned-policy/WBC handoff and no task-success claim",
        "recommendation": "If synchronized q0s remain valid while fixed N2 FR still degrades, authorize a matched visited-state supervision intervention; do not change seed/noise/stats automatically.",
    }
    result_sha = write_json(output / "pi0160_matched_expert_result.json", summary)
    summary["result_sha256"] = result_sha
    write_json(output / "evidence_hashes.json", {"admission_sha256": admission_sha, "result_sha256": result_sha, "budget_ledger_snapshot_sha256": summary["budget_ledger"]["snapshot_sha256"], "condition_metadata_sha256": {item["condition_id"]: sha256_file(Path(item["output_dir"]) / "condition_metadata.json") if Path(item["output_dir"], "condition_metadata.json").is_file() else None for item in results}})
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
    p.add_argument("--budget-ledger", required=True, type=Path)
    p.add_argument("--campaign-id", default="PI0162_CORELAB_20260913")
    p.add_argument("--history-audit", type=Path, default=None)
    p.add_argument("--gpu-uuid", default=None)
    return p


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "token": result["token"], "conditions": result["counters"]["conditions_passed"], "qualified_q0": result["counters"]["qualified_q0_count"], "expert_wbc_steps": result["counters"]["expert_wbc_steps"], "reference_updates": result["counters"]["reference_updates"], "gr00t_forward_count": 0}, sort_keys=True), flush=True)
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
