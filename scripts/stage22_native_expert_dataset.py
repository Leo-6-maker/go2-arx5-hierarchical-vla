#!/usr/bin/env python3
"""Collect same-scene native expert snapshots for Stage22.

The runner creates a new native IsaacGym episode for each target, captures the
native body/wrist cameras and StateV1, creates a deterministic smoothstep-to-
hold expert chunk from the current achieved EEF, and executes only that expert
reference through the delivered WBC. At each declared K=4 prefix boundary a
fresh snapshot-conditioned expert chunk is hot-swapped without resetting
physics or policy history.

The target is used by this simulator-side expert and is retained as provenance
only. It is never sent to a GR00T model input. The resulting snapshots are
later expanded into one-window official dataset fixtures by the CPU-only
materializer; the expansion is explicitly marked in metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1
from go2arx5_vla.data import SameSceneExpertSnapshotV1
from go2arx5_vla.metrics import chunk_overlap_metrics, chunk_profile
from go2arx5_vla.runtime import RecedingHorizonExecutorV1
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1
from go2arx5_vla.simulation import (
    SameSceneExpertConfigV1,
    build_same_scene_expert_chunk,
    write_rgb_ppm,
)
from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows

from stage21_native_receding_horizon import (
    LANGUAGE,
    SIMULATOR,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    _build_native,
    _capture_scene,
    _current_eef,
    _current_state_v1,
    _jsonable,
    _load_pose_file,
    _native_runtime_telemetry,
    _policy_step,
    _set_target,
    _tensor_hash,
)


DEFAULT_TARGETS = (
    ("red_target_train_a", (0.30, 0.10, 0.58)),
    ("red_target_train_b", (0.30, -0.10, 0.58)),
    ("red_target_train_c", (0.30, 0.05, 0.64)),
    ("red_target_train_d", (0.30, -0.05, 0.63)),
    ("red_target_holdout_e", (0.28, 0.15, 0.60)),
    ("red_target_holdout_f", (0.32, -0.15, 0.56)),
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(value)), indent=2) + "\n", encoding="utf-8")


def _pose_9d(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(position, dtype=np.float64).reshape(3), rotation_matrix_to_rot6d_rows(quat_to_matrix(quaternion))]
    )


def _base_pose(native: Any) -> list[list[float]] | None:
    base_position = getattr(native, "base_pos", None)
    base_quaternion = getattr(native, "base_quat", None)
    if base_position is None or base_quaternion is None:
        return None
    if hasattr(base_position, "detach"):
        position = base_position[0].detach().cpu().numpy()
    else:
        position = np.asarray(base_position)[0]
    if hasattr(base_quaternion, "detach"):
        quaternion = base_quaternion[0].detach().cpu().numpy()
    else:
        quaternion = np.asarray(base_quaternion)[0]
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        return None
    rotation = quat_to_matrix(quaternion)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform.tolist()


def _targets(path: Path | None, count: int | None) -> list[tuple[str, np.ndarray]]:
    if path is None:
        values = list(DEFAULT_TARGETS)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("targets", [])
        values = []
        for index, item in enumerate(payload):
            if isinstance(item, Mapping):
                target_id = str(item.get("target_id", f"target_{index:02d}"))
                position = item.get("position_world_m")
            else:
                target_id = f"target_{index:02d}"
                position = item
            values.append((target_id, tuple(position)))
    if count is not None:
        values = values[: int(count)]
    if not values:
        raise ValueError("at least one target is required")
    result = []
    for target_id, position in values:
        array = np.asarray(position, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"target {target_id!r} must contain finite world XYZ")
        result.append((str(target_id), array))
    return result


def _snapshot(
    *,
    native: Any,
    output_dir: Path,
    episode_id: str,
    target_id: str,
    target_world: np.ndarray,
    anchor_world: np.ndarray,
    observation_timestamp: float,
    expert_config: SameSceneExpertConfigV1,
    chunk_id: str,
) -> tuple[SameSceneExpertSnapshotV1, Any]:
    body_rgb, wrist_rgb = _capture_scene(native)
    achieved_world, achieved_quaternion = _current_eef(native)
    achieved_local = achieved_world - anchor_world
    target_local = target_world - anchor_world
    state = _current_state_v1(native)
    chunk = build_same_scene_expert_chunk(
        chunk_id=chunk_id,
        observation_timestamp=observation_timestamp,
        current_position=achieved_local,
        target_position=target_local,
        current_quaternion=achieved_quaternion,
        config=expert_config,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
    )
    stem = output_dir / "snapshots" / episode_id / chunk_id
    body_path = stem.with_name(stem.name + "_body.ppm")
    wrist_path = stem.with_name(stem.name + "_wrist.ppm")
    body_sha = write_rgb_ppm(body_path, body_rgb)
    wrist_sha = write_rgb_ppm(wrist_path, wrist_rgb)
    telemetry = _native_runtime_telemetry(native)
    telemetry["base_pose"] = _base_pose(native)
    telemetry["achieved_position_world_m"] = achieved_world.tolist()
    telemetry["achieved_position_local_m"] = achieved_local.tolist()
    telemetry["achieved_quaternion_xyzw"] = np.asarray(achieved_quaternion, dtype=np.float64).tolist()
    snapshot = SameSceneExpertSnapshotV1(
        example_id=f"{episode_id}:{chunk_id}",
        raw_episode_id=episode_id,
        simulator_instance_id=native.stage21_simulator_instance_id,
        observation_timestamp=float(observation_timestamp),
        body_frame_path=str(body_path),
        wrist_frame_path=str(wrist_path),
        state_v1=state,
        future_timestamps=chunk.point_timestamps,
        future_desired_eef_9d=np.concatenate(
            [chunk.eef_position, rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation))],
            axis=-1,
        ),
        future_gripper=np.asarray(chunk.gripper, dtype=np.float64).reshape(-1, 1),
        target_position_world_m=target_world,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        calibration_bundle_id="STAGE22_NATIVE_SIM_CALIBRATION_BUNDLE_V1",
        target_id=target_id,
        native_control_dt_s=float(native.dt),
        raw_telemetry=telemetry,
        provenance={
            "stage": "Stage22",
            "task_id": "SIM_VISUAL_REACH_V0",
            "language": LANGUAGE,
            "source": "native_isaacgym_same_scene_expert",
            "simulator": SIMULATOR,
            "native_wbc_active": True,
            "target_actor": "stage21_red_target",
            "target_excluded_from_model_input": True,
            "label_source": "deterministic_smoothstep_position_hold",
            "label_current_pose_source": "same_observation_achieved_eef",
            "body_frame_sha256": body_sha,
            "wrist_frame_sha256": wrist_sha,
            "action_sample_period_s": float(expert_config.action_sample_period_s),
            "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
            "no_physics_reset_or_teleport_at_replanning": True,
        },
    )
    _write_json(stem.with_suffix(".json"), snapshot.as_dict())
    return snapshot, chunk


def _run_episode(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray], target_id: str, target_world: np.ndarray) -> dict[str, Any]:
    episode_id = f"stage22_expert_{target_id}"
    output_dir = args.output_dir
    episode_dir = output_dir / "episodes" / episode_id
    native = env = None
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    termination = "NOT_STARTED"
    update_count = 0
    reset_events = 0
    result: dict[str, Any] = {
        "schema": "STAGE22_NATIVE_SAME_SCENE_EXPERT_EPISODE_V1",
        "stage": "Stage22",
        "episode_id": episode_id,
        "target_id": target_id,
        "target_position_world_m": target_world.tolist(),
        "mode": "SYNCHRONOUS_PAUSED_SIM_EXPERT_RECEDING_HORIZON",
        "simulation_only": True,
        "hardware_control": False,
        "fixed_reset_seed": int(args.seed),
        "prefix_steps_declared": [4, 6, 8, 10],
        "prefix_steps_frozen": int(args.prefix_steps),
        "target_excluded_from_model_input": True,
    }
    try:
        native_args = argparse.Namespace(**vars(args))
        native_args.mode = "gt_receding"
        native, env, actors, loads, _ = _build_native(native_args, trajectory)
        _set_target(native, target_world)
        native_dt = float(native.dt)
        if not np.isfinite(native_dt) or native_dt <= 0.0:
            raise ValueError(f"native control dt must be positive and finite, got {native_dt}")
        anchor_world = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
        expert_config = SameSceneExpertConfigV1(
            horizon=40,
            action_sample_period_s=float(args.action_period_s),
            motion_duration_s=float(args.motion_duration_s),
        )
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=SOURCE_FRAME,
                wbc_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_SEMANTIC_ID,
                stale_timeout_s=0.75,
            )
        )
        sim_time = 0.0
        current_world, current_q = _current_eef(native)
        current_local = current_world - anchor_world
        initial_snapshot, current_chunk = _snapshot(
            native=native,
            output_dir=output_dir,
            episode_id=episode_id,
            target_id=target_id,
            target_world=target_world,
            anchor_world=anchor_world,
            observation_timestamp=sim_time,
            expert_config=expert_config,
            chunk_id="expert-0000",
        )
        initial_gate = gate.evaluate(
            current_chunk,
            now=sim_time,
            achieved_position=current_local,
            achieved_orientation=current_q,
        )
        gates.append({"chunk_id": current_chunk.chunk_id, "result": initial_gate.as_dict()})
        if initial_gate.decision is GateDecision.REJECT:
            raise RuntimeError("EXPERT_SAME_SCENE_RECEDING_HORIZON_GATE_REJECTED_INITIAL")
        preview, _ = adapter.adapt(current_chunk, controller_timestamp=sim_time)
        binding = EpisodeReferenceBindingV1.from_external_chunk(current_chunk, episode_id=episode_id)
        identity_bound = binding.bound_chunk()
        current_chunk = binding.bind_external_chunk(current_chunk)
        # The binding preserves the expert values; the canonical WBC loader
        # receives only the bound external chunk.
        before_root = _tensor_hash(native.root_states)
        before_dof = _tensor_hash(native.dof_state)
        before_episode = int(native.episode_length_buf[0].detach().cpu().item())
        initial_load = native.load_reference_chunk(current_chunk)
        if not (
            before_root == _tensor_hash(native.root_states)
            and before_dof == _tensor_hash(native.dof_state)
            and before_episode == int(native.episode_length_buf[0].detach().cpu().item())
        ):
            raise RuntimeError("EXPERT_INITIAL_REFERENCE_CHANGED_PHYSICS_STATE")
        executor = RecedingHorizonExecutorV1(
            prefix_steps=int(args.prefix_steps),
            provenance={
                "stage": "Stage22",
                "mode": "SAME_SCENE_EXPERT_ORACLE",
                "schedule": "SYNCHRONOUS_PAUSED_SIM_EXPERT_RECEDING_HORIZON",
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
            },
        )
        initial_update = executor.submit_initial(
            current_chunk,
            receive_timestamp=sim_time,
            execute_timestamp=sim_time,
            current_achieved_position=current_local,
            current_achieved_orientation=current_q,
        )
        initial_entry = initial_snapshot.as_dict()
        initial_entry.update(
            {
                "profile": chunk_profile(current_chunk, achieved_position=current_local, achieved_quaternion=current_q),
                "binding": {
                    "episode_id": binding.episode_id,
                    "position_preserved_m": float(
                        np.max(np.linalg.norm(current_chunk.eef_position - identity_bound.eef_position, axis=-1))
                    ),
                    "preview_shape": list(preview.xyz_rot6d.shape),
                    "preview_finite": bool(np.isfinite(preview.xyz_rot6d).all()),
                },
                "native_load": initial_load,
                "executor_initial": initial_update.as_dict(),
            }
        )
        snapshots.append(initial_entry)
        max_steps = (int(args.max_prefix_updates) + 1) * int(args.prefix_steps)
        for step in range(max_steps):
            done, policy = _policy_step(native, env, actors)
            sim_time += native_dt
            event = native.pop_stage16_termination()
            current_world, current_q = _current_eef(native)
            current_local = current_world - anchor_world
            row = {
                "step": int(step),
                "simulation_time_s": float(sim_time),
                "chunk_id": current_chunk.chunk_id,
                "update_count": int(update_count),
                "done": bool(done),
                "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
                "episode_length": int(native.episode_length_buf[0].detach().cpu().item()),
                "traj_s": float(native.traj_s[0].detach().cpu().item()),
                "traj_s_ref": float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item()),
                "traj_d_lat": float(native.traj_d_lat[0].detach().cpu().item()),
                "traj_timing_err": float(native.traj_timing_err[0].detach().cpu().item()),
                "traj_L": float(native.traj_batch.L[0].detach().cpu().item()),
                "achieved_pos_world": current_world.tolist(),
                "achieved_pos_local": current_local.tolist(),
                "achieved_quat": current_q.tolist(),
                "base_pos": native.base_pos[0].detach().cpu().tolist(),
                "base_lin_vel": native.base_lin_vel[0].detach().cpu().tolist(),
                "base_ang_vel": native.base_ang_vel[0].detach().cpu().tolist(),
                "internal_body_plan_9d": policy["internal_body_plan_9d"],
                "arm_action_6d": policy["arm_action_6d"],
                "dog_action_12d": policy["dog_action_12d"],
                "termination_code": "NONE",
            }
            row.update(_native_runtime_telemetry(native))
            if event is not None:
                row["termination_event"] = event
                row["termination_code"] = event.get("termination_code", "UNKNOWN")
                termination = row["termination_code"]
                reset_events += 1
            elif done:
                row["termination_code"] = "NATIVE_TASK_TERMINATION"
                termination = "NATIVE_TASK_TERMINATION"
                reset_events += 1
            rows.append(row)
            if done:
                break
            if (step + 1) % int(args.prefix_steps) != 0 or update_count >= int(args.max_prefix_updates):
                continue

            previous_chunk = current_chunk
            previous_index = min(int(args.prefix_steps), previous_chunk.point_count - 1)
            previous_position = previous_chunk.eef_position[previous_index]
            previous_quaternion = previous_chunk.eef_orientation[previous_index]
            next_snapshot, next_chunk_raw = _snapshot(
                native=native,
                output_dir=output_dir,
                episode_id=episode_id,
                target_id=target_id,
                target_world=target_world,
                anchor_world=anchor_world,
                observation_timestamp=sim_time,
                expert_config=expert_config,
                chunk_id=f"expert-{update_count + 1:04d}",
            )
            next_gate = gate.evaluate(
                next_chunk_raw,
                now=sim_time,
                achieved_position=current_local,
                achieved_orientation=current_q,
            )
            gates.append({"chunk_id": next_chunk_raw.chunk_id, "result": next_gate.as_dict()})
            if next_gate.decision is GateDecision.REJECT:
                termination = "EXPERT_TRAJECTORY_REJECTED_BY_GATE"
                break
            next_binding = EpisodeReferenceBindingV1.from_external_chunk(next_chunk_raw, episode_id=episode_id)
            identity_bound = next_binding.bound_chunk()
            next_chunk = next_binding.bind_external_chunk(next_chunk_raw)
            overlap = chunk_overlap_metrics(previous_chunk, next_chunk)
            update = executor.submit_update(
                next_chunk,
                receive_timestamp=sim_time,
                execute_timestamp=sim_time,
                current_achieved_position=current_local,
                current_achieved_orientation=current_q,
                previous_predicted_position=previous_position,
                previous_predicted_orientation=previous_quaternion,
            )
            before_root = _tensor_hash(native.root_states)
            before_dof = _tensor_hash(native.dof_state)
            before_episode = int(native.episode_length_buf[0].detach().cpu().item())
            before_arm_history = _tensor_hash(env.arm_obs_history)
            before_dog_history = _tensor_hash(env.dog_obs_history)
            native_load = native.load_reference_chunk(next_chunk)
            invariant = {
                "physics_root_state_unchanged": before_root == _tensor_hash(native.root_states),
                "physics_dof_state_unchanged": before_dof == _tensor_hash(native.dof_state),
                "episode_length_unchanged": before_episode == int(native.episode_length_buf[0].detach().cpu().item()),
                "arm_policy_history_unchanged": before_arm_history == _tensor_hash(env.arm_obs_history),
                "dog_policy_history_unchanged": before_dog_history == _tensor_hash(env.dog_obs_history),
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
            }
            required_unchanged = (
                "physics_root_state_unchanged",
                "physics_dof_state_unchanged",
                "episode_length_unchanged",
                "arm_policy_history_unchanged",
                "dog_policy_history_unchanged",
            )
            if (
                not all(invariant[key] for key in required_unchanged)
                or invariant["physics_reset"]
                or invariant["physics_teleport"]
                or invariant["policy_history_reset"]
            ):
                raise RuntimeError(f"EXPERT_REFERENCE_UPDATE_INVARIANT_FAILED: {invariant}")
            handoff = {
                **native_load,
                "chunk_handoff": update.handoff.as_dict(),
                "lifecycle_status": update.lifecycle_status.value,
                "overlap": overlap,
                "invariant": invariant,
                "executor_update": update.as_dict(),
            }
            handoffs.append(handoff)
            entry = next_snapshot.as_dict()
            entry.update(
                {
                    "profile": chunk_profile(next_chunk, achieved_position=current_local, achieved_quaternion=current_q),
                    "overlap": overlap,
                "native_load": native_load,
                "executor_update": update.as_dict(),
                "binding_identity_position_preservation_m": float(
                    np.max(np.linalg.norm(next_chunk.eef_position - identity_bound.eef_position, axis=-1))
                ),
            }
            )
            snapshots.append(entry)
            current_chunk = next_chunk
            update_count += 1

        if termination == "NOT_STARTED":
            termination = "SCHEDULE_END"
        valid_rows = [item for item in rows if item.get("termination_code") == "NONE"]
        result.update(
            {
                "status": (
                    "EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS"
                    if update_count > 0 and reset_events == 0 and len(snapshots) == update_count + 1
                    and all(item["invariant"][key] for item in handoffs for key in (
                        "physics_root_state_unchanged",
                        "physics_dof_state_unchanged",
                        "episode_length_unchanged",
                        "arm_policy_history_unchanged",
                        "dog_policy_history_unchanged",
                    ))
                    else "EXPERT_SAME_SCENE_RECEDING_HORIZON_PARTIAL"
                ),
                "termination": termination,
                "steps": len(rows),
                "valid_steps": len(valid_rows),
                "reference_updates": int(update_count),
                "reset_events": int(reset_events),
                "snapshot_count": len(snapshots),
                "snapshots": snapshots,
                "handoffs": handoffs,
                "safety_gate": gates,
                "actor_loads": loads,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": actors[2].__version__,
                    "device": str(actors[3]),
                    "native_control_dt_s": native_dt,
                    "prefix_duration_native_s": native_dt * int(args.prefix_steps),
                    "action_sample_period_s": float(args.action_period_s),
                    "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
                    "expert_motion_duration_s": float(args.motion_duration_s),
                    "native_anchor_world_m": anchor_world.tolist(),
                },
                "claim_boundary": "same_scene_native_camera_and_expert_label; no_model_task_performance_claim",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "EXPERT_SAME_SCENE_RECEDING_HORIZON_BLOCKED" if not rows else "EXPERT_SAME_SCENE_RECEDING_HORIZON_PARTIAL",
                "blocker": {"type": type(exc).__name__, "message": str(exc)},
                "termination": termination,
                "steps": len(rows),
                "reference_updates": int(update_count),
                "reset_events": int(reset_events),
                "snapshot_count": len(snapshots),
                "snapshots": snapshots,
                "handoffs": handoffs,
                "safety_gate": gates,
            }
        )
    finally:
        if native is not None:
            try:
                result["native_load_events"] = native._stage21_load_events
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "expert_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row)) + "\n")
    _write_json(episode_dir / "expert_episode_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefix-steps", type=int, choices=(4, 6, 8, 10), default=4)
    parser.add_argument("--max-prefix-updates", type=int, default=5)
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--motion-duration-s", type=float, default=1.95)
    parser.add_argument("--targets-json", type=Path, default=None)
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--target-id", type=str, default=None)
    parser.add_argument("--target-position-world-m", type=float, nargs=3, default=None)
    args = parser.parse_args()
    if args.max_prefix_updates < 1 or args.action_period_s <= 0.0 or args.motion_duration_s <= 0.0:
        parser.error("prefix updates, action period, and motion duration must be positive")
    if args.target_position_world_m is not None:
        targets = [
            (
                args.target_id or "red_target_single",
                np.asarray(args.target_position_world_m, dtype=np.float64),
            )
        ]
    else:
        targets = _targets(args.targets_json, args.episode_count)
    trajectory = _load_pose_file(args.trajectory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target_id, target_world in targets:
        summaries.append(_run_episode(args, trajectory, target_id, target_world))
    pass_count = sum(item.get("status") == "EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS" for item in summaries)
    result = {
        "schema": "STAGE22_NATIVE_SAME_SCENE_EXPERT_DATASET_V1",
        "stage": "Stage22",
        "status": "PASS" if pass_count == len(summaries) else "PASS_WITH_DEVIATION",
        "expert_oracle_gate": "EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS" if pass_count == len(summaries) else "EXPERT_ORACLE_NOT_FULLY_QUALIFIED",
        "episode_count": len(summaries),
        "expert_pass_count": pass_count,
        "example_count": int(sum(item.get("snapshot_count", 0) for item in summaries)),
        "prefix_steps_declared": [4, 6, 8, 10],
        "prefix_steps_frozen": int(args.prefix_steps),
        "action_sample_period_s": float(args.action_period_s),
        "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
        "motion_duration_s": float(args.motion_duration_s),
        "target_excluded_from_model_input": True,
        "hardware_control": False,
        "simulation_only": True,
        "episodes": summaries,
    }
    _write_json(args.output_dir / "expert_dataset_collection_summary.json", result)
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return 0 if pass_count == len(summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
