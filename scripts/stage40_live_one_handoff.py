#!/usr/bin/env python3
"""Stage40 synchronous one-handoff GR00T -> native WBC diagnostic.

This is intentionally a thin Stage21 reuse layer. The native simulator owns
the episode; the existing Stage21 worker owns GR00T inference. Physics is
paused while a fresh observation is exchanged. No controller, SafetyGate,
trajectory, or action representation is modified here.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.runtime import RecedingHorizonExecutorV1
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1

from stage21_native_receding_horizon import (
    LANGUAGE,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    _build_native,
    _capture_scene,
    _chunk_payload_equivalent,
    _current_eef,
    _current_state_v1,
    _jsonable,
    _load_pose_file,
    _make_chunk,
    _native_runtime_telemetry,
    _policy_step,
    _sha256_array,
    _tensor_hash,
)
from stage21_native_model_receding_horizon import (
    _bind_for_native,
    _model_chunk,
    _wait_response,
    _write_observation_request,
)


STAGE = "Stage40"
PI_GATE = "PI_GATE_STATE_0041"
HANDOFF_S = 0.9750000000000001
CHUNK_DURATION_S = 1.9500000000000002
SEEDS = (2223, 2228, 2233, 2238)


def _quat_error(q0: Any, q1: Any) -> float:
    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0)))


def _pose_at(chunk: Any, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(chunk.point_timestamps, dtype=np.float64)
    index = int(np.clip(np.searchsorted(ts, float(timestamp), side="right") - 1, 0, len(ts) - 1))
    return np.asarray(chunk.eef_position[index], dtype=np.float64), np.asarray(chunk.eef_orientation[index], dtype=np.float64)



def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(value)), indent=2) + "\n", encoding="utf-8")


def _request_model(*, native: Any, mailbox: Path, request_id: str, seed: int, sim_time: float,
                   gate: Any, adapter: Any, response_timeout_s: float) -> tuple[Any, np.ndarray, dict[str, Any]]:
    state = _current_state_v1(native)
    body_rgb, wrist_rgb = _capture_scene(native)
    request_path, observation_npz, request_wall = _write_observation_request(
        mailbox, request_id=request_id, body_rgb=body_rgb, wrist_rgb=wrist_rgb, state=state,
        target_position_world_m=np.asarray([0.30, 0.10, 0.58], dtype=np.float64),
        simulator_instance_id=native.stage21_simulator_instance_id,
        simulation_time_s=float(sim_time), seed=int(seed),
    )
    response, response_wall = _wait_response(mailbox, request_id, timeout_s=float(response_timeout_s))
    chunk, decoded = _model_chunk(response, chunk_id=request_id, observation_timestamp=float(sim_time), period_s=0.05)
    current_position, current_quaternion = _current_eef(native)
    anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
    local_position = current_position - anchor
    gate_result = gate.evaluate(chunk, now=float(sim_time), achieved_position=local_position, achieved_orientation=current_quaternion)
    binding, bound_chunk = _bind_for_native(chunk)
    preview, gripper = adapter.adapt(bound_chunk, controller_timestamp=float(sim_time))
    case = {
        "request_id": request_id, "request_path": str(request_path), "observation_npz": str(observation_npz),
        "observation_timestamp": float(sim_time), "simulator_instance_id": native.stage21_simulator_instance_id,
        "body_shape": list(body_rgb.shape), "body_dtype": str(body_rgb.dtype), "body_sha256": _sha256_array(body_rgb),
        "wrist_shape": list(wrist_rgb.shape), "wrist_dtype": str(wrist_rgb.dtype), "wrist_sha256": _sha256_array(wrist_rgb),
        "state_v1": state.as_array().tolist(), "decoded_shape": list(decoded.shape), "decoded_sha256": _sha256_array(decoded),
        "finite": bool(np.isfinite(decoded).all()), "response": response,
        "inference_latency_s": float(response.get("latency_s", response_wall - request_wall)),
        "mailbox_wall_s": float(response_wall - request_wall), "safety_gate": gate_result.as_dict(),
        "binding": {"episode_id": binding.episode_id, "semantics": binding.binding_semantics,
                    "preview_shape": list(preview.xyz_rot6d.shape), "preview_finite": bool(np.isfinite(preview.xyz_rot6d).all()),
                    "gripper_present": gripper is not None},
        "chunk": bound_chunk, "decoded": decoded,
    }
    return bound_chunk, decoded, case


def _step_rows(native: Any, env: Any, actors: Any, *, chunk: Any, sim_time: float, steps: int,
               update_count: int) -> tuple[list[dict[str, Any]], float, str | None]:
    rows: list[dict[str, Any]] = []
    termination: str | None = None
    dt = float(native.dt)
    for step in range(int(steps)):
        done, policy = _policy_step(native, env, actors)
        sim_time += dt
        event = native.pop_stage16_termination()
        current_position, current_quaternion = _current_eef(native)
        reference_s = native.traj_batch.s_ref(native.traj_sim_time)
        desired_position = native.traj_batch.p_at(reference_s)[0].detach().cpu().numpy()
        desired_quaternion = native.traj_batch.quat_at(reference_s)[0].detach().cpu().numpy()
        row: dict[str, Any] = {
            "step": int(step), "sim_time": float(sim_time),
            "native_reference_local_time": float(native.traj_sim_time[0].detach().cpu().item()),
            "chunk_id": chunk.chunk_id, "update_count": int(update_count), "done": bool(done),
            "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
            "episode_length": int(native.episode_length_buf[0].detach().cpu().item()),
            "traj_s": float(native.traj_s[0].detach().cpu().item()),
            "traj_s_ref": float(reference_s[0].detach().cpu().item()),
            "desired_pos": desired_position.tolist(), "desired_quat": desired_quaternion.tolist(),
            "achieved_pos": current_position.tolist(), "achieved_quat": current_quaternion.tolist(),
            "base_pos": native.base_pos[0].detach().cpu().tolist(),
            "base_lin_vel": native.base_lin_vel[0].detach().cpu().tolist(),
            "base_ang_vel": native.base_ang_vel[0].detach().cpu().tolist(),
            "arm_action_6d": policy["arm_action_6d"], "dog_action_12d": policy["dog_action_12d"],
            "termination_code": "NONE",
        }
        row.update(_native_runtime_telemetry(native))
        if event is not None:
            row["termination_event"] = event
            row["termination_code"] = event.get("termination_code", "UNKNOWN")
            termination = str(row["termination_code"])
        elif done:
            row["termination_code"] = "NATIVE_TASK_TERMINATION"
            termination = "NATIVE_TASK_TERMINATION"
        rows.append(row)
        if done or event is not None:
            break
    return rows, float(sim_time), termination


def run_b1(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    native = env = None
    result: dict[str, Any] = {"schema": "STAGE40_B1_NO_RESET_HANDOFF_V1", "stage": STAGE, "pi_gate": PI_GATE}
    try:
        native_args = argparse.Namespace(**vars(args))
        native_args.mode = "gt_receding"
        native, env, actors, loads, _ = _build_native(native_args, trajectory)
        requested_chunk_a = _make_chunk(trajectory, chunk_id="b1-a", observation_timestamp=0.0, start_time_s=0.0)
        installed_chunk_a = native.stage21_current_chunk
        if not _chunk_payload_equivalent(installed_chunk_a, requested_chunk_a):
            raise RuntimeError("INITIAL_REFERENCE_PAYLOAD_MISMATCH")
        chunk_a = installed_chunk_a
        chunk_b = _make_chunk(trajectory, chunk_id="b1-b", observation_timestamp=HANDOFF_S, start_time_s=HANDOFF_S)
        sim_time = 0.0
        before_rows, sim_time, before_term = _step_rows(native, env, actors, chunk=chunk_a, sim_time=sim_time,
                                                         steps=round(HANDOFF_S / float(native.dt)), update_count=0)
        before = {"root": _tensor_hash(native.root_states), "dof": _tensor_hash(native.dof_state),
                  "episode": int(native.episode_length_buf[0].detach().cpu().item())}
        native_update = native.load_reference_chunk(chunk_b)
        after = {"root": _tensor_hash(native.root_states), "dof": _tensor_hash(native.dof_state),
                 "episode": int(native.episode_length_buf[0].detach().cpu().item())}
        after_rows, sim_time, after_term = _step_rows(native, env, actors, chunk=chunk_b, sim_time=sim_time, steps=4, update_count=1)
        result.update({"status": "PASS" if before == after and before_term is None and after_term is None else "BLOCKED",
                       "physics_reset": False, "policy_history_reset": False, "handoff": native_update,
                       "state_before": before, "state_after": after, "state_continuous": before == after,
                       "rows_before": before_rows, "rows_after": after_rows, "loads": loads,
                       "native_control_dt_s": float(native.dt)})
    except Exception as exc:
        result.update({"status": "BLOCKED", "error": {"type": type(exc).__name__, "message": str(exc)}})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    _write_json(args.output_dir / "stage40_b1_result.json", result)
    return result


def run_live(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    native = env = None
    episodes: list[dict[str, Any]] = []
    loads_all: list[Any] = []
    runtime_info: dict[str, Any] = {}
    result: dict[str, Any] = {"schema": "STAGE40_LIVE_ONE_HANDOFF_V1", "stage": STAGE, "pi_gate": PI_GATE,
                              "mode": "SYNCHRONOUS_PAUSED_SIMULATION", "physics_reset_at_handoff": False,
                              "seeds": list(SEEDS), "handoff_target_s": HANDOFF_S,
                              "chunk_duration_s": CHUNK_DURATION_S}
    try:
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(expected_source_frame=SOURCE_FRAME,
            wbc_frame=SOURCE_FRAME, tcp_semantic_id=TCP_SEMANTIC_ID, stale_timeout_s=0.75))
        for seed in SEEDS:
            native_args = argparse.Namespace(**vars(args))
            native_args.mode = "model_receding"
            native_args.seed = int(seed)
            native, env, actors, loads, _ = _build_native(native_args, trajectory)
            loads_all.extend(loads)
            runtime_info = {"python": platform.python_version(), "torch": actors[2].__version__,
                            "device": str(actors[3]), "native_control_dt_s": float(native.dt),
                            "episode_seed": int(seed)}
            episode: dict[str, Any] = {"seed": int(seed), "status": "BLOCKED", "handoff_reset": False}
            sim_time = 0.0
            try:
                q0, _, case0 = _request_model(native=native, mailbox=args.mailbox,
                    request_id=f"stage40-s{seed}-q0", seed=seed, sim_time=sim_time, gate=gate,
                    adapter=adapter, response_timeout_s=args.response_timeout_s)
                episode["q0"] = {k: v for k, v in case0.items() if k not in {"chunk", "decoded"}}
                if case0["safety_gate"]["decision"] == GateDecision.REJECT.value:
                    episode["termination"] = "SAFETYGATE_REJECT_Q0"
                    episodes.append(episode)
                    continue
                native.initial_bind_reference(q0)
                executor = RecedingHorizonExecutorV1(prefix_steps=1, provenance={"stage": STAGE,
                    "mode": "LIVE_ONE_HANDOFF", "physics_reset": False, "policy_history_reset": False,
                    "synchronous_paused_simulation": True})
                anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
                current_position, current_quaternion = _current_eef(native)
                executor.submit_initial(q0, receive_timestamp=sim_time, execute_timestamp=sim_time,
                    current_achieved_position=current_position - anchor, current_achieved_orientation=current_quaternion)
                handoff_steps = max(1, int(round(HANDOFF_S / float(native.dt))))
                rows0, sim_time, term0 = _step_rows(native, env, actors, chunk=q0, sim_time=sim_time,
                                                     steps=handoff_steps, update_count=0)
                current_position, current_quaternion = _current_eef(native)
                local_position = current_position - anchor
                old_position, old_quaternion = _pose_at(q0, float(q0.point_timestamps[0] + sim_time))
                q1, _, case1 = _request_model(native=native, mailbox=args.mailbox,
                    request_id=f"stage40-s{seed}-q1", seed=seed, sim_time=sim_time, gate=gate,
                    adapter=adapter, response_timeout_s=args.response_timeout_s)
                episode["q1"] = {k: v for k, v in case1.items() if k not in {"chunk", "decoded"}}
                new_position = np.asarray(q1.eef_position[0], dtype=np.float64)
                new_quaternion = np.asarray(q1.eef_orientation[0], dtype=np.float64)
                episode["handoff_metrics"] = {"target_s": HANDOFF_S, "actual_switch_s": float(sim_time),
                    "switch_tick_error_s": float(sim_time - HANDOFF_S),
                    "old_reference_vs_achieved_xyz_m": float(np.linalg.norm(old_position - local_position)),
                    "new_pose0_vs_measured_xyz_m": float(np.linalg.norm(new_position - local_position)),
                    "new_pose0_vs_measured_so3_rad": _quat_error(new_quaternion, current_quaternion),
                    "old_vs_new_reference_xyz_jump_m": float(np.linalg.norm(old_position - new_position)),
                    "old_vs_new_reference_so3_jump_rad": _quat_error(old_quaternion, new_quaternion),
                    "traj_s_before_handoff": float(native.traj_s[0].detach().cpu().item()),
                    "traj_s_ref_before_handoff": float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item()),
                    "preview_discontinuity": case1["binding"], "no_smoothing_blending_retiming": True}
                if case1["safety_gate"]["decision"] == GateDecision.REJECT.value:
                    episode["termination"] = "SAFETYGATE_REJECT_Q1"
                    episodes.append(episode)
                    continue
                before = {"root": _tensor_hash(native.root_states), "dof": _tensor_hash(native.dof_state),
                          "episode": int(native.episode_length_buf[0].detach().cpu().item())}
                handoff = executor.submit_update(q1, receive_timestamp=sim_time, execute_timestamp=sim_time,
                    current_achieved_position=local_position, current_achieved_orientation=current_quaternion,
                    previous_predicted_position=old_position, previous_predicted_orientation=old_quaternion)
                native_update = native.load_reference_chunk(q1)
                after = {"root": _tensor_hash(native.root_states), "dof": _tensor_hash(native.dof_state),
                         "episode": int(native.episode_length_buf[0].detach().cpu().item())}
                episode["handoff"] = {**native_update, "executor": handoff.as_dict(),
                    "physics_state_unchanged": before == after, "policy_history_reset": False, "physics_reset": False}
                rows1, sim_time, term1 = _step_rows(native, env, actors, chunk=q1, sim_time=sim_time,
                                                     steps=max(1, int(math.ceil(CHUNK_DURATION_S / float(native.dt)))), update_count=1)
                all_rows = rows0 + rows1
                term = term1 or term0
                episode.update({"status": "PASS" if term is None and bool(episode["handoff"]["physics_state_unchanged"]) and len(rows1) > 0 else "FAIL",
                    "termination": term or "REFERENCE_COMPLETE", "chunk1_reference_complete": term is None,
                    "rows_before": rows0, "rows_after": rows1, "row_count": len(all_rows),
                    "telemetry_finite": all(bool(np.isfinite(np.asarray(row.get("achieved_pos", []), dtype=np.float64)).all()) for row in all_rows),
                    "joint_limit_termination_count": sum(bool(row.get("joint_limit_termination", False)) for row in all_rows),
                    "joint_limit_violation_count": sum(bool(row.get("joint_limit_violation", False)) for row in all_rows),
                    "fall_count": sum(bool(row.get("fall_flag", False)) for row in all_rows),
                    "invalid_state_count": sum(bool(row.get("invalid_state", False)) for row in all_rows)})
            except Exception as exc:
                episode.update({"status": "BLOCKED", "error": {"type": type(exc).__name__, "message": str(exc)}})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    native = env = None
            episodes.append(episode)
        result.update({"episodes": episodes, "episode_pass_count": sum(item.get("status") == "PASS" for item in episodes),
            "status": "PASS" if len(episodes) == len(SEEDS) and all(item.get("status") == "PASS" for item in episodes) else "NOT_QUALIFIED",
            "loads": loads_all, "runtime": runtime_info,
            "claim_boundary": "TINY1 single-fixture synchronous four-seed fresh-observation one-handoff diagnostic; not real-time, multi-handoff, generalization, or hardware"})
    except Exception as exc:
        result.update({"status": "BLOCKED", "error": {"type": type(exc).__name__, "message": str(exc)}, "episodes": episodes})
    finally:
        try:
            _write_json(args.mailbox / "STOP", {"reason": "stage40_native_finished"})
        except Exception:
            pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    _write_json(args.output_dir / "stage40_live_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("b1", "live"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--mailbox", type=Path, default=Path("stage40_mailbox"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-seed", type=int, default=2223)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mailbox.mkdir(parents=True, exist_ok=True)
    trajectory = _load_pose_file(args.trajectory)
    result = run_b1(args, trajectory) if args.mode == "b1" else run_live(args, trajectory)
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
