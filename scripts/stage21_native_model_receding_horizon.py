#!/usr/bin/env python3
"""Run the actual model in a same-scene synchronous paused native rollout.

The native WBC process owns one IsaacGym world.  A small shared-filesystem
mailbox carries immutable camera/state observations to a separately loaded
GR00T process and carries decoded [1,40,10] actions back.  Physics is paused
while inference runs; no reset, teleport, policy-history reset, clipping,
smoothing, or retiming is performed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1, TrajectoryChunkV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.runtime import RecedingHorizonExecutorV1
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1
from go2arx5_vla.simulation import SameSceneTargetV1

from stage21_native_receding_horizon import (
    LANGUAGE,
    SIMULATOR,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    TEST_ACTION_PERIOD_S,
    _build_native,
    _capture_scene,
    _current_eef,
    _current_state_v1,
    _jsonable,
    _load_pose_file,
    _native_runtime_telemetry,
    _policy_step,
    _set_target,
    _sha256_array,
    _tensor_hash,
)
from stage21_same_scene_interface_seal import _profile


TARGET_POSITION_WORLD_M = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_observation_request(
    mailbox: Path,
    *,
    request_id: str,
    body_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: Any,
    target_position_world_m: np.ndarray,
    simulator_instance_id: str,
    simulation_time_s: float,
    seed: int,
) -> tuple[Path, Path, float]:
    requests = mailbox / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    npz_path = requests / f"{request_id}.npz"
    npz_tmp = requests / f"{request_id}.tmp.npz"
    np.savez_compressed(
        npz_tmp,
        body_rgb=np.asarray(body_rgb, dtype=np.uint8),
        wrist_rgb=np.asarray(wrist_rgb, dtype=np.uint8),
        state=state.as_array(),
        target_position_world_m=np.asarray(target_position_world_m, dtype=np.float64),
    )
    npz_tmp.replace(npz_path)
    request_path = requests / f"{request_id}.json"
    request_written_wall = time.perf_counter()
    _write_json_atomic(
        request_path,
        {
            "schema": "STAGE21_GR00T_WORKER_REQUEST_V1",
            "request_id": request_id,
            "observation_npz": str(npz_path),
            "language": LANGUAGE,
            "state_abi": "StateV1=EEF_XYZ_3+GR00T_FIRST_TWO_ROWS_ROT6D_6+GRIPPER_1",
            "source_frame": SOURCE_FRAME,
            "tcp_semantic_id": TCP_SEMANTIC_ID,
            "target_position_world_m": np.asarray(target_position_world_m).tolist(),
            "simulator_instance_id": simulator_instance_id,
            "simulation_time_s": float(simulation_time_s),
            "observation_timestamp": float(simulation_time_s),
            "seed": int(seed),
            "simulation_only": True,
            "hardware_control": False,
        },
    )
    return request_path, npz_path, request_written_wall


def _wait_response(mailbox: Path, request_id: str, *, timeout_s: float) -> tuple[dict[str, Any], float]:
    response_path = mailbox / "responses" / f"{request_id}.json"
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("status") != "PASS":
                raise RuntimeError(
                    f"GR00T worker failed for {request_id}: "
                    f"{response.get('error_type', 'ERROR')}: {response.get('error', 'unknown')}"
                )
            return response, time.perf_counter()
        time.sleep(0.10)
    raise TimeoutError(f"timed out waiting for GR00T response {request_id}")


def _model_chunk(response: dict[str, Any], *, chunk_id: str, observation_timestamp: float, period_s: float) -> tuple[TrajectoryChunkV1, Any]:
    with np.load(response["artifact"], allow_pickle=False) as data:
        decoded = np.asarray(data["decoded_physical_action"], dtype=np.float64)
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        raise ValueError(f"worker decoded action must be finite [1,40,10], got {decoded.shape}")
    provider = GR00TTrajectoryProvider(
        decoded,
        config=GR00TProviderConfig(
            source_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_SEMANTIC_ID,
            action_sample_period_s=float(period_s),
            timestamp_provenance="TEST_ONLY_STAGE21_SIMULATION",
        ),
        observation_timestamp=float(observation_timestamp),
        chunk_id=chunk_id,
        provenance={
            "stage": "Stage21",
            "source": "actual_GR00T_same_scene_worker",
            "worker_response": response["artifact"],
            "no_correction_applied": True,
        },
    )
    chunk = provider.next_chunk(float(observation_timestamp))
    if chunk is None:  # pragma: no cover
        raise RuntimeError("GR00T provider emitted no chunk")
    return chunk, decoded[0]


def _bind_for_native(chunk: TrajectoryChunkV1) -> tuple[EpisodeReferenceBindingV1, TrajectoryChunkV1]:
    binding = EpisodeReferenceBindingV1.from_external_chunk(chunk, episode_id=f"stage21-{chunk.chunk_id}")
    return binding, binding.bind_external_chunk(chunk)


def _model_receding(args: argparse.Namespace, trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mailbox = args.mailbox
    (mailbox / "requests").mkdir(parents=True, exist_ok=True)
    (mailbox / "responses").mkdir(parents=True, exist_ok=True)
    native = env = None
    rows: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    model_cases: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": "STAGE21_ACTUAL_GR00T_SAME_SCENE_RECEDING_HORIZON_V1",
        "stage": "Stage21",
        "status": "BLOCKED",
        "mode": "SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON",
        "simulation_only": True,
        "hardware_control": False,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "fixed_primary_reset_seed": int(args.seed),
        "prefix_steps_declared": [4, 6, 8, 10],
        "prefix_steps_frozen": int(args.prefix_steps),
        "action_sample_period_s": float(args.action_period_s),
        "action_sample_period_provenance": "TEST_ONLY_STAGE21_SIMULATION",
        "simulator": SIMULATOR,
        "mailbox": str(mailbox),
        "target_position_world_m": TARGET_POSITION_WORLD_M.tolist(),
        "model_outputs_executed_only_after_gate": True,
    }
    termination = "NOT_STARTED"
    update_count = 0
    reset_events = 0
    try:
        # The helper uses a validated GT chunk only to initialize native reset
        # and establish the reset anchor.  It is replaced by the first actual
        # model chunk before the first policy step.
        native_args = argparse.Namespace(**vars(args))
        native_args.mode = "model_receding"
        native, env, actors, loads, cfg = _build_native(native_args, trajectory)
        _set_target(native, TARGET_POSITION_WORLD_M)
        native_dt_s = float(native.dt)
        if not np.isfinite(native_dt_s) or native_dt_s <= 0.0:
            raise ValueError(f"native control dt must be positive and finite, got {native_dt_s}")
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=SOURCE_FRAME,
                wbc_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_SEMANTIC_ID,
                stale_timeout_s=0.75,
            )
        )
        executor: RecedingHorizonExecutorV1 | None = None
        sim_time = 0.0
        request_counter = 0
        current_chunk: TrajectoryChunkV1 | None = None

        def request_model() -> tuple[TrajectoryChunkV1, Any, dict[str, Any]]:
            nonlocal request_counter, sim_time
            request_id = f"model-{request_counter:04d}"
            request_counter += 1
            state = _current_state_v1(native)
            body_rgb, wrist_rgb = _capture_scene(native)
            request_path, observation_npz, request_wall = _write_observation_request(
                mailbox,
                request_id=request_id,
                body_rgb=body_rgb,
                wrist_rgb=wrist_rgb,
                state=state,
                target_position_world_m=TARGET_POSITION_WORLD_M,
                simulator_instance_id=native.stage21_simulator_instance_id,
                simulation_time_s=sim_time,
                seed=int(args.model_seed),
            )
            response, response_wall = _wait_response(mailbox, request_id, timeout_s=float(args.response_timeout_s))
            chunk, decoded = _model_chunk(
                response,
                chunk_id=request_id,
                observation_timestamp=sim_time,
                period_s=float(args.action_period_s),
            )
            current_position, current_quaternion = _current_eef(native)
            local_position = current_position - native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
            gate_result = gate.evaluate(
                chunk,
                now=sim_time,
                achieved_position=local_position,
                achieved_orientation=current_quaternion,
            )
            gate_results.append({"chunk_id": chunk.chunk_id, "result": gate_result.as_dict()})
            profile = _profile(decoded, float(args.action_period_s) * np.arange(40), state)
            case = {
                "chunk_id": chunk.chunk_id,
                "request_id": request_id,
                "request_path": str(request_path),
                "observation_npz": str(observation_npz),
                "response": response,
                "response_wait_wall_s": float(response_wall - request_wall),
                "observation_to_execution_latency_wall_s": float(response_wall - request_wall),
                "observation_timestamp": float(sim_time),
                "chunk_age_at_execution_s": 0.0,
                "state": state.as_array().tolist(),
                "body_sha256": _sha256_array(body_rgb),
                "wrist_sha256": _sha256_array(wrist_rgb),
                "decoded_sha256": _sha256_array(decoded),
                "profile": profile,
                "safety_gate": gate_result.as_dict(),
            }
            model_cases.append(case)
            if gate_result.decision is GateDecision.REJECT:
                raise RuntimeError(f"MODEL_TRAJECTORY_PROFILE_REJECTED_BY_GATE: {chunk.chunk_id}")
            binding, bound_chunk = _bind_for_native(chunk)
            # Constructing the preview is part of the chain audit; the native
            # loader consumes only the canonical bound chunk, never GR00T data.
            preview, gripper = adapter.adapt(bound_chunk, controller_timestamp=sim_time)
            case["binding"] = {
                "episode_id": binding.episode_id,
                "semantics": binding.binding_semantics,
                "pose_preservation_position_m": float(np.max(np.linalg.norm(bound_chunk.eef_position - chunk.eef_position, axis=-1))),
                "preview_shape": list(preview.xyz_rot6d.shape),
                "preview_finite": bool(np.isfinite(preview.xyz_rot6d).all()),
                "gripper_parallel_channel": gripper is not None,
            }
            return bound_chunk, decoded, case

        # Initial model request is made after native reset, before any physics
        # step; the dummy reset reference is replaced without resetting state.
        current_chunk, decoded, initial_case = request_model()
        before_root = _tensor_hash(native.root_states)
        before_dof = _tensor_hash(native.dof_state)
        before_episode = int(native.episode_length_buf[0].detach().cpu().item())
        before_arm_history = _tensor_hash(env.arm_obs_history)
        before_dog_history = _tensor_hash(env.dog_obs_history)
        native_initial_load = native.load_reference_chunk(current_chunk)
        initial_case["initial_reference_replacement"] = {
            **native_initial_load,
            "arm_policy_history_unchanged": before_arm_history == _tensor_hash(env.arm_obs_history),
            "dog_policy_history_unchanged": before_dog_history == _tensor_hash(env.dog_obs_history),
            "physics_root_state_unchanged": before_root == _tensor_hash(native.root_states),
            "physics_dof_state_unchanged": before_dof == _tensor_hash(native.dof_state),
            "episode_length_unchanged": before_episode == int(native.episode_length_buf[0].detach().cpu().item()),
        }
        executor = RecedingHorizonExecutorV1(
            prefix_steps=int(args.prefix_steps),
            provenance={
                "stage": "Stage21",
                "mode": "ACTUAL_GR00T",
                "schedule": "SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON",
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
            },
        )
        initial_position, initial_quaternion = _current_eef(native)
        initial_local = initial_position - native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
        initial_update = executor.submit_initial(
            current_chunk,
            receive_timestamp=sim_time,
            execute_timestamp=sim_time,
            current_achieved_position=initial_local,
            current_achieved_orientation=initial_quaternion,
        )
        initial_case["executor_initial"] = initial_update.as_dict()

        for step in range((int(args.max_prefix_updates) + 1) * int(args.prefix_steps)):
            done, policy = _policy_step(native, env, actors)
            sim_time += native_dt_s
            event = native.pop_stage16_termination()
            current_position, current_quaternion = _current_eef(native)
            native_reference_time = float(native.traj_sim_time[0].detach().cpu().item())
            reference_s = native.traj_batch.s_ref(native.traj_sim_time)
            desired_position = native.traj_batch.p_at(reference_s)[0].detach().cpu().tolist()
            desired_quaternion = native.traj_batch.quat_at(reference_s)[0].detach().cpu().tolist()
            row: dict[str, Any] = {
                "step": int(step),
                "sim_time": float(sim_time),
                "native_reference_local_time": native_reference_time,
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
                "desired_pos": desired_position,
                "desired_quat": desired_quaternion,
                "achieved_pos": current_position.tolist(),
                "achieved_quat": current_quaternion.tolist(),
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
            next_chunk, next_decoded, next_case = request_model()
            local_position = current_position - native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
            update = executor.submit_update(
                next_chunk,
                receive_timestamp=sim_time,
                execute_timestamp=sim_time,
                current_achieved_position=local_position,
                current_achieved_orientation=current_quaternion,
                previous_predicted_position=previous_position,
                previous_predicted_orientation=previous_quaternion,
            )
            before_root = _tensor_hash(native.root_states)
            before_dof = _tensor_hash(native.dof_state)
            before_episode = int(native.episode_length_buf[0].detach().cpu().item())
            before_arm_history = _tensor_hash(env.arm_obs_history)
            before_dog_history = _tensor_hash(env.dog_obs_history)
            native_update = native.load_reference_chunk(next_chunk)
            native_update.update(
                {
                    "chunk_handoff": update.handoff.as_dict(),
                    "lifecycle_status": update.lifecycle_status.value,
                    "arm_policy_history_unchanged": before_arm_history == _tensor_hash(env.arm_obs_history),
                    "dog_policy_history_unchanged": before_dog_history == _tensor_hash(env.dog_obs_history),
                    "physics_root_state_unchanged": before_root == _tensor_hash(native.root_states),
                    "physics_dof_state_unchanged": before_dof == _tensor_hash(native.dof_state),
                    "episode_length_unchanged": before_episode == int(native.episode_length_buf[0].detach().cpu().item()),
                    "executor_update": update.as_dict(),
                }
            )
            next_case["handoff"] = native_update
            handoffs.append(native_update)
            current_chunk = next_chunk
            update_count += 1

        valid_rows = [row for row in rows if row.get("termination_code") == "NONE"]
        if termination == "NOT_STARTED":
            termination = "SCHEDULE_END"
        result.update(
            {
                "status": "SIM_SAME_SCENE_MODEL_RECEDING_HORIZON_EXECUTED" if reset_events == 0 else "SIM_SAME_SCENE_MODEL_RECEDING_HORIZON_PARTIAL",
                "termination": termination,
                "steps": len(rows),
                "valid_steps": len(valid_rows),
                "reference_updates": int(update_count),
                "reset_events": int(reset_events),
                "camera_same_scene_available": True,
                "safety_gate": gate_results,
                "model_cases": model_cases,
                "handoffs": handoffs,
                "actor_loads": loads,
                "runtime": {
                    "python": __import__("platform").python_version(),
                    "torch": actors[2].__version__,
                    "device": str(actors[3]),
                    "native_control_dt_s": native_dt_s,
                    "prefix_duration_native_s": native_dt_s * int(args.prefix_steps),
                    "chunk_action_sample_period_s": float(args.action_period_s),
                    "chunk_to_control_rate_ratio": float(args.action_period_s) / native_dt_s,
                },
                "async_runtime": "ASYNC_RUNTIME_NOT_QUALIFIED",
                "claim_boundary": "same_scene_actual_GR00T_receding_horizon_execution; synchronous_paused_sim; no_real_time_or_task_success_claim",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "BLOCKED" if not rows else "SIM_SAME_SCENE_MODEL_RECEDING_HORIZON_PARTIAL",
                "blocker": {"type": type(exc).__name__, "message": str(exc)},
                "gate": "ONE_SPECIFIC_MODEL_PROFILE_BLOCKER_REMAINS" if "MODEL_TRAJECTORY_PROFILE_REJECTED" in str(exc) else "ONE_SPECIFIC_CHUNK_LIFECYCLE_BLOCKER_REMAINS",
                "termination": termination,
                "steps": len(rows),
                "reference_updates": int(update_count),
                "reset_events": int(reset_events),
                "safety_gate": gate_results,
                "model_cases": model_cases,
                "handoffs": handoffs,
                "async_runtime": "ASYNC_RUNTIME_NOT_QUALIFIED",
            }
        )
    finally:
        try:
            _write_json_atomic(mailbox / "STOP", {"reason": "native_runner_finished"})
        except Exception:
            pass
        if native is not None:
            try:
                result.setdefault("native_load_events", native._stage21_load_events)
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    with (output_dir / "model_receding_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row)) + "\n")
    result = _jsonable(result)
    (output_dir / "model_receding_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--mailbox", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-seed", type=int, default=2121)
    parser.add_argument("--prefix-steps", type=int, choices=(4, 6, 8, 10), default=4)
    parser.add_argument("--max-prefix-updates", type=int, default=5)
    parser.add_argument("--action-period-s", type=float, default=TEST_ACTION_PERIOD_S)
    parser.add_argument("--response-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.max_prefix_updates < 1 or args.action_period_s <= 0.0:
        parser.error("max-prefix-updates must be positive and action-period-s must be positive")
    trajectory = _load_pose_file(args.trajectory)
    result = _model_receding(args, trajectory)
    return 0 if str(result.get("status", "")).endswith("EXECUTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
