#!/usr/bin/env python3
"""Run one Stage43 controlled initial-condition M2 episode.

The runner is intentionally a single-condition executable.  The parent
launcher invokes it in the frozen C0..C4 order and stops on the first failure.
Every condition owns a fresh native simulator and GR00T worker, applies one
pre-S0 base-XY translation, and then performs the ordinary paused-observation
receding-horizon loop.  No handoff mutates physics or policy history.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage43_corelab_evidence as evidence  # noqa: E402
from stage40_corelab_phase import (  # noqa: E402
    HANDOFF_S,
    _array_sha,
    _build_native,
    _case_request,
    _case_summary,
    _check_rows,
    _current_eef,
    _current_state_v1,
    _initial_bind,
    _load_pose_file,
    _native_args,
    _relative_path,
    _snapshot_gpu,
    _start_worker,
    _step_rows,
    _stop_worker,
    _tensor_hash,
    _update_reference,
    _write_json,
)
from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sync_evidence_tree(root: Path) -> None:
    """Flush already-materialized evidence before native teardown."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())


def _write_durable_json(path: Path, value: Any) -> None:
    _write_json(path, value)
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _host_snapshot(path: Path, label: str) -> Dict[str, Any]:
    process: Dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                fields = value.strip().split()
                if fields:
                    process[key] = int(fields[0])
    except (OSError, ValueError):
        pass
    system: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields = value.strip().split()
                if fields:
                    system[key] = int(fields[0])
    except (OSError, ValueError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema": "CORELAB_STAGE43_HOST_RAM_SNAPSHOT_V1",
        "label": label,
        "process": {"status_kib": process, "maxrss": int(getattr(usage, "ru_maxrss", 0))},
        "system_kib": system,
    }
    _write_json(path, result)
    return result


def _telemetry_sample(root: Path, telemetry: List[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    gpu = _snapshot_gpu(root / ("gpu_" + label + ".json"), label)
    host = _host_snapshot(root / ("host_" + label + ".json"), label)
    sample = {"label": label, "gpu": gpu, "host": host, "wall_time": time.time()}
    telemetry.append(sample)
    return sample


def _state_hashes(native: Any, env: Any) -> Dict[str, Any]:
    return {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "wbc_forward_count": int(native.stage21_wbc_forward_count),
        "reference_update_count": int(native.stage21_reference_update_count),
    }


def _refresh_eef_from_rigid_body(native: Any, *, root_delta: Optional[np.ndarray] = None) -> None:
    """Refresh the WBC EEF view after an indexed root-state write.

    The WBC post-physics hook normally refreshes this view and applies the TCP
    grasp offset.  At this boundary the view already contains the delivered
    TCP pose from ``env.reset()``; repeating the offset here would double-count
    it.  We therefore synchronize only the derived position cache for the
    one-time base translation and leave orientation/velocity untouched.
    """

    rigid = native.rigid_body_state.view(native.num_envs, native.num_bodies, 13)
    # IsaacGym's indexed root-state setter updates the root tensor immediately,
    # but its derived rigid-body tensor remains the last fetched physics state
    # until a simulation step.  The initial-condition protocol deliberately
    # takes zero settling/physics steps before S0, so synchronize that derived
    # cache explicitly with the same one-time base translation.  This does not
    # alter velocities, orientations, DOF state, or simulator execution state;
    # the next physics step refreshes the cache from the simulator again.
    if root_delta is not None:
        import torch

        shift = torch.as_tensor(np.asarray(root_delta, dtype=np.float32), dtype=rigid.dtype, device=rigid.device)
        rigid[:, :, :3].add_(shift.view(1, 1, 3))


def _apply_initial_base_delta(native: Any, env: Any, condition_id: str, delta: Sequence[float]) -> Dict[str, Any]:
    """Apply exactly one pre-S0 base XY translation and prove its scope."""

    import torch

    delta_np = np.asarray(delta, dtype=np.float64)
    _require(delta_np.shape == (3,) and np.isfinite(delta_np).all(), "initial base delta is not finite [3]")
    _require(float(delta_np[2]) == 0.0, "initial base perturbation must be XY-only")
    _require(native.stage21_wbc_forward_count == 0 and native.stage21_reference_update_count == 0, "initial base perturbation occurred after execution")
    before = _state_hashes(native, env)
    root_before = native.root_states.detach().cpu().numpy().copy()
    dof_before = native.dof_state.detach().cpu().numpy().copy()
    target_index = int(native.stage21_target_root_index)
    target_before = root_before[target_index].copy()
    rigid_before = native.rigid_body_state.detach().cpu().numpy().copy()
    eef_before, quat_before = _current_eef(native)
    base_before = native.base_pos[0].detach().cpu().numpy().copy()
    target_pose_before = root_before[target_index, :3].copy()

    native.root_states[0, :2] += torch.as_tensor(delta_np[:2], dtype=native.root_states.dtype, device=native.root_states.device)
    robot_index = torch.as_tensor([0], device=native.device, dtype=torch.int32)
    native.gym.set_actor_root_state_tensor_indexed(
        native.sim,
        native._stage21_gymtorch.unwrap_tensor(native.root_states),
        native._stage21_gymtorch.unwrap_tensor(robot_index),
        1,
    )
    native.gym.refresh_actor_root_state_tensor(native.sim)
    # Do not refresh the derived rigid-body tensor here: before the first
    # physics step IsaacGym may repopulate it from the previous simulator
    # state, which is not the just-written root pose.  The explicit cache sync
    # below preserves the already-delivered post-reset kinematics and applies
    # only the declared base translation.
    _refresh_eef_from_rigid_body(native, root_delta=delta_np)

    root_after = native.root_states.detach().cpu().numpy().copy()
    dof_after = native.dof_state.detach().cpu().numpy().copy()
    target_after = root_after[target_index].copy()
    rigid_after = native.rigid_body_state.detach().cpu().numpy().copy()
    eef_after, quat_after = _current_eef(native)
    base_after = native.base_pos[0].detach().cpu().numpy().copy()
    after = _state_hashes(native, env)
    root_delta = root_after[0] - root_before[0]
    non_xy = np.delete(root_delta, [0, 1])
    rigid_delta = rigid_after - rigid_before
    rigid_position_delta = rigid_delta[..., :3]
    rigid_non_position_delta = np.concatenate((rigid_delta[..., 3:7], rigid_delta[..., 7:13]), axis=-1)
    rigid_position_delta_exact = bool(np.allclose(rigid_position_delta, delta_np, rtol=0.0, atol=1e-6))
    rigid_non_position_unchanged = bool(np.array_equal(rigid_non_position_delta, np.zeros_like(rigid_non_position_delta)))
    other_actor_unchanged = bool(np.array_equal(root_after[1:], root_before[1:]))
    dof_unchanged = bool(np.array_equal(dof_after, dof_before))
    event: Dict[str, Any] = {
        "schema": "CORELAB_STAGE43_INITIAL_BASE_CONDITION_V1",
        "condition_id": condition_id,
        "delta_base_xy_m": delta_np.tolist(),
        "mutation": "one indexed robot root-state write before S0",
        "physics_steps_before_S0": 0,
        "settling_steps": 0,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "root_state_before_sha256": _array_sha(root_before),
        "root_state_after_sha256": _array_sha(root_after),
        "rigid_body_state_before_sha256": _array_sha(rigid_before),
        "rigid_body_state_after_sha256": _array_sha(rigid_after),
        "dof_state_before_sha256": _array_sha(dof_before),
        "dof_state_after_sha256": _array_sha(dof_after),
        "target_root_before": target_before.tolist(),
        "target_root_after": target_after.tolist(),
        "target_pose_world_before_m": target_pose_before.tolist(),
        "target_pose_world_after_m": target_after[:3].tolist(),
        "base_position_before_m": base_before.tolist(),
        "base_position_after_m": base_after.tolist(),
        "eef_world_before_m": eef_before.tolist(),
        "eef_world_after_m": eef_after.tolist(),
        "eef_quaternion_before": quat_before.tolist(),
        "eef_quaternion_after": quat_after.tolist(),
        "root_delta_observed": root_delta.tolist(),
        "rigid_body_position_delta_observed": rigid_position_delta.tolist(),
        "state_before": before,
        "state_after": after,
        "root_xy_delta_exact": bool(np.allclose(root_delta[:2], delta_np[:2], rtol=0.0, atol=1e-6)),
        "root_non_xy_unchanged": bool(np.array_equal(non_xy, np.zeros_like(non_xy))),
        "rigid_body_position_delta_exact": rigid_position_delta_exact,
        "rigid_body_non_position_unchanged": rigid_non_position_unchanged,
        "rigid_body_cache_sync": "same one-time base XY delta applied to derived robot rigid-body position cache before S0",
        "other_actor_roots_unchanged": other_actor_unchanged,
        "dof_state_unchanged": dof_unchanged,
        "eef_translation_matches_delta": bool(np.allclose(eef_after - eef_before, delta_np, rtol=0.0, atol=1e-5)),
        "eef_orientation_unchanged": bool(np.array_equal(quat_after, quat_before)),
        "target_root_unchanged": bool(np.array_equal(target_after, target_before)),
        "lifecycle_unchanged": bool(before["episode"] == after["episode"] and before["reset_buf"] == after["reset_buf"]),
        "policy_histories_unchanged": bool(before["arm_history"] == after["arm_history"] and before["dog_history"] == after["dog_history"]),
        "wbc_forward_count_after": int(native.stage21_wbc_forward_count),
        "reference_update_count_after": int(native.stage21_reference_update_count),
    }
    _require(event["root_xy_delta_exact"] and event["root_non_xy_unchanged"], "base root mutation scope mismatch: xy=%s non_xy=%s delta=%s" % (event["root_xy_delta_exact"], event["root_non_xy_unchanged"], event["root_delta_observed"]))
    _require(event["other_actor_roots_unchanged"] and event["target_root_unchanged"], "initial base mutation changed target/other actors: other=%s target=%s" % (event["other_actor_roots_unchanged"], event["target_root_unchanged"]))
    _require(event["rigid_body_position_delta_exact"] and event["rigid_body_non_position_unchanged"], "derived rigid-body cache sync mismatch: position=%s non_position=%s" % (event["rigid_body_position_delta_exact"], event["rigid_body_non_position_unchanged"]))
    _require(event["dof_state_unchanged"] and event["eef_translation_matches_delta"] and event["eef_orientation_unchanged"], "initial base mutation changed an unauthorized state: dof=%s eef_translation=%s eef_orientation=%s" % (event["dof_state_unchanged"], event["eef_translation_matches_delta"], event["eef_orientation_unchanged"]))
    _require(event["lifecycle_unchanged"] and event["policy_histories_unchanged"], "initial base mutation changed lifecycle/history: lifecycle=%s history=%s" % (event["lifecycle_unchanged"], event["policy_histories_unchanged"]))
    _require(event["wbc_forward_count_after"] == 0 and event["reference_update_count_after"] == 0, "initial base mutation consumed execution counters")
    state_v1 = _current_state_v1(native)
    event["s0"] = {
        "eef_world_m": eef_after.tolist(),
        "state_v1": state_v1.as_array().tolist(),
        "state_v1_sha256": _array_sha(state_v1.as_array()),
        "target_world_m": evidence.TARGET_POSITION_WORLD_M.tolist(),
    }
    return event


def _counter_snapshot(cases: Sequence[Mapping[str, Any]], native: Any) -> Dict[str, int]:
    return {
        "gr00t_forward_count": int(len(cases)),
        "wbc_step_count": int(native.stage21_wbc_forward_count),
        "reference_update_count": int(native.stage21_reference_update_count),
    }


def _request_with_counters(
    *, native: Any, env: Any, mailbox: Path, request_id: str, seed: int, sim_time: float,
    gate: Any, adapter: Any, args: argparse.Namespace, cases: List[Mapping[str, Any]], counter_events: List[Mapping[str, Any]],
) -> Tuple[Any, Dict[str, Any]]:
    before = _counter_snapshot(cases, native)
    chunk, case = _case_request(native, mailbox, request_id, seed, sim_time, gate, adapter, args)
    cases.append(case)
    after = _counter_snapshot(cases, native)
    response = case["response"]
    event = {
        "event": "GR00T_FORWARD",
        "request_id": request_id,
        "sim_time_s": float(sim_time),
        "before": before,
        "after": after,
        "gr00t_forward_delta_observed": int(after["gr00t_forward_count"] - before["gr00t_forward_count"]),
        "worker_forward_count_delta": int(response.get("worker_forward_count_delta", -1)),
        "action_sha256": case.get("raw_sha256"),
        "state_v1_sha256": case.get("state_v1_sha256"),
        "safety_decision": case["safety_gate"]["decision"],
    }
    _require(event["gr00t_forward_delta_observed"] == 1 and event["worker_forward_count_delta"] == 1, request_id + " did not produce one real worker forward")
    _require(case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_" + request_id)
    counter_events.append(event)
    return chunk, case


def _metrics(setup: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    initial = np.asarray(setup["s0"]["eef_world_m"], dtype=np.float64)
    positions = np.asarray([row["achieved_pos"] for row in rows], dtype=np.float64)
    desired = np.asarray([row["desired_pos"] for row in rows], dtype=np.float64)
    _require(initial.shape == (3,) and positions.ndim == 2 and positions.shape[1] == 3, "invalid EEF metric positions")
    _require(desired.shape == positions.shape, "desired/achieved EEF metric shape mismatch")
    all_positions = np.vstack((initial[None, :], positions))
    distances = np.linalg.norm(all_positions - evidence.TARGET_POSITION_WORLD_M[None, :], axis=1)
    tracking = np.linalg.norm(positions - desired, axis=1)
    _require(np.isfinite(distances).all() and np.isfinite(tracking).all(), "target metrics are non-finite")
    minimum_index = int(np.argmin(distances))
    d0 = float(distances[0]); dmin = float(distances[minimum_index]); dfinal = float(distances[-1])
    return {
        "sample_count_including_S0": int(len(distances)),
        "initial_target_error_m": d0,
        "minimum_target_error_m": dmin,
        "minimum_error_sample_index": minimum_index,
        "final_target_error_m": dfinal,
        "absolute_progress_m": float(d0 - dfinal),
        "best_progress_m": float(d0 - dmin),
        "normalized_progress": float((d0 - dfinal) / d0) if d0 > 0.0 else None,
        "initial_eef_world_m": initial.tolist(),
        "final_eef_world_m": all_positions[-1].tolist(),
        "tracking_rms_m": float(np.sqrt(np.mean(tracking * tracking))),
        "tracking_max_m": float(np.max(tracking)),
        "progress_sign": "POSITIVE" if d0 > dfinal else ("ZERO" if np.isclose(d0, dfinal, atol=1e-12) else "NEGATIVE"),
        "threshold_used": None,
        "target_world_m": evidence.TARGET_POSITION_WORLD_M.tolist(),
        "frame": "world",
        "eef_point": evidence.TCP_SEMANTIC_ID,
        "definition": "S0 is captured after the one pre-episode base translation; subsequent samples are actual native achieved_pos rows",
    }


def _artifact_checks(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    file_keys = ("structured_stats", "modality", "trajectory", "wbc_arm_checkpoint", "wbc_dog_checkpoint")
    for key in file_keys:
        item = bundle["artifacts"][key]
        path = Path(str(item["path"]))
        _require(path.is_file() and not path.is_symlink(), "Stage43 artifact missing: " + key)
        digest = evidence.sha256_file(path)
        _require(digest == item["sha256"], "Stage43 artifact hash drift: " + key)
        checks[key] = {"path": str(path), "sha256": digest, "expected_sha256": item["sha256"], "status": "PASS"}
    directory_keys = ("structured_checkpoint", "tiny1_dataset", "cosmos_snapshot", "wbc_source", "stage10_root", "base_model_path", "model_path", "dataset_dir", "cosmos_snapshot_path")
    for key in directory_keys:
        item = bundle["artifacts"][key]
        path = Path(str(item["path"]))
        _require(path.is_dir() and not path.is_symlink(), "Stage43 artifact directory missing: " + key)
        checks[key] = {"path": str(path), "status": "PASS", "declared_identity": {k: v for k, v in item.items() if k not in {"path"}}}
    return checks


def _materialize(
    *, args: argparse.Namespace, bundle: Mapping[str, Any], root: Path, mailbox: Path, condition_id: str,
    setup: Optional[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], rows_before: Sequence[Sequence[Mapping[str, Any]]],
    rows_after: Sequence[Sequence[Mapping[str, Any]]], bind_events: Sequence[Mapping[str, Any]], handoffs: Sequence[Mapping[str, Any]],
    executor_events: Sequence[Mapping[str, Any]], state_events: Sequence[Mapping[str, Any]], counter_events: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], native: Any, worker_return_code: Optional[int], status: str,
    completion: Optional[Mapping[str, Any]], metrics: Optional[Mapping[str, Any]], artifact_checks: Optional[Mapping[str, Any]],
    error: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    summaries = [_case_summary(case, mailbox) for case in cases]
    body_values: Dict[str, np.ndarray] = {}; wrist_values: Dict[str, np.ndarray] = {}
    raw_values: Dict[str, np.ndarray] = {}; decoded_values: Dict[str, np.ndarray] = {}
    for case, summary in zip(cases, summaries):
        case_id = str(case["request_id"])
        with np.load(case["observation_npz"], allow_pickle=False) as data:
            body = np.asarray(data["body_rgb"], dtype=np.uint8); wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
        with np.load(case["response"]["artifact"], allow_pickle=False) as data:
            raw = np.asarray(data["raw_universal_action"])
        body_values[case_id] = body; wrist_values[case_id] = wrist; raw_values[case_id] = raw; decoded_values[case_id] = np.asarray(case["decoded"], dtype=np.float64)
        summary["body_red_pixel_count"] = int(evidence._red_pixel_count(body)); summary["wrist_red_pixel_count"] = int(evidence._red_pixel_count(wrist))
    np.savez_compressed(root / "observation_body.npz", **body_values)
    np.savez_compressed(root / "observation_wrist.npz", **wrist_values)
    np.savez_compressed(root / "raw_action_d132.npz", **raw_values)
    np.savez_compressed(root / "decoded_action_d10_h40.npz", **decoded_values)
    _write_json(root / "condition_setup.json", {"schema": "CORELAB_STAGE43_CONDITION_SETUP_EVIDENCE_V1", "event": setup, "artifact_checks": artifact_checks})
    _write_json(root / "state_v1.json", {"schema": "CORELAB_STAGE43_STATE_V1_EVIDENCE_V1", "observations": summaries})
    _write_json(root / "worker_request.json", {"schema": "CORELAB_STAGE43_WORKER_REQUEST_EVIDENCE_V1", "requests": [item["request"] for item in summaries]})
    _write_json(root / "worker_response.json", {"schema": "CORELAB_STAGE43_WORKER_RESPONSE_EVIDENCE_V1", "responses": [item["response"] for item in summaries]})
    _write_json(root / "safety_gate.json", {"schema": "CORELAB_STAGE43_SAFETYGATE_EVIDENCE_V1", "decisions": [item["safety_gate"]["decision"] for item in summaries], "cases": [item["safety_gate"] for item in summaries]})
    _write_json(root / "counter_observations.json", {"schema": "CORELAB_STAGE43_COUNTER_OBSERVATION_V1", "events": list(counter_events), "final": {"gr00t_forward_count": len(cases), "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0}})
    _write_json(root / "mailbox.json", {"schema": "CORELAB_STAGE43_MAILBOX_EVIDENCE_V1", "mailbox": str(mailbox), "requests": sorted(str(path) for path in mailbox.glob("requests/*")), "responses": sorted(str(path) for path in mailbox.glob("responses/*")), "worker_return_code": worker_return_code})
    _write_json(root / "logs.json", {"schema": "CORELAB_STAGE43_LOG_EVIDENCE_V1", "condition_id": condition_id, "status": status, "worker_log": str(root / "worker.log"), "error": error})
    _write_json(root / "telemetry.json", {"schema": "CORELAB_STAGE43_TELEMETRY_EVIDENCE_V1", "samples": list(telemetry)})
    _write_json(root / "initial_bind.json", {"schema": "CORELAB_STAGE43_INITIAL_BIND_EVIDENCE_V1", "events": list(bind_events)})
    _write_json(root / "root_state.json", {"schema": "CORELAB_STAGE43_ROOT_STATE_EVIDENCE_V1", "setup": setup, "events": list(state_events)})
    _write_json(root / "dof_state.json", {"schema": "CORELAB_STAGE43_DOF_STATE_EVIDENCE_V1", "setup": setup, "events": list(state_events)})
    _write_json(root / "executor_bookkeeping.json", {"schema": "CORELAB_STAGE43_EXECUTOR_BOOKKEEPING_EVIDENCE_V1", "events": list(executor_events)})
    _write_json(root / "rows_before_handoff.json", {"schema": "CORELAB_STAGE43_ROWS_BEFORE_HANDOFF_EVIDENCE_V1", "segments": [list(segment) for segment in rows_before]})
    _write_json(root / "rows_after_handoff.json", {"schema": "CORELAB_STAGE43_ROWS_AFTER_HANDOFF_EVIDENCE_V1", "segments": [list(segment) for segment in rows_after]})
    _write_json(root / "handoff.json", {"schema": "CORELAB_STAGE43_HANDOFF_EVIDENCE_V1", "events": list(handoffs), "completion": completion})
    _write_json(root / "condition_metrics.json", {"schema": "CORELAB_STAGE43_TARGET_METRICS_V1", "metrics": metrics})
    metadata = {
        "schema": "CORELAB_STAGE43_CONDITION_RUN_V1", "stage": evidence.STAGE_ID, "condition_id": condition_id, "status": status,
        "bundle": str(args.bundle), "bundle_sha256": evidence.sha256_file(args.bundle), "seed": int(args.seed),
        "gpu_uuid": args.gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "native_control_dt_s": float(native.dt) if native is not None else None,
        "gr00t_forward_count": len(cases), "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
        "wbc_segment_lengths": [len(segment) for segment in list(rows_before) + list(rows_after)],
        "training": False, "hardware_control": False, "forbidden_operations": evidence.expected_forbidden_operations(),
        "completion": completion, "metrics": metrics, "error": error,
        "platform": {"python": platform.python_version(), "hostname": platform.node()},
    }
    _write_json(root / "run_metadata.json", metadata)
    roles = {
        "condition_setup": "condition_setup.json", "observation_body": "observation_body.npz", "observation_wrist": "observation_wrist.npz", "state_v1": "state_v1.json",
        "worker_request": "worker_request.json", "worker_response": "worker_response.json", "raw_action_d132": "raw_action_d132.npz", "decoded_action_d10_h40": "decoded_action_d10_h40.npz",
        "safety_gate": "safety_gate.json", "counter_observations": "counter_observations.json", "mailbox": "mailbox.json", "logs": "logs.json", "telemetry": "telemetry.json",
        "initial_bind": "initial_bind.json", "root_state": "root_state.json", "dof_state": "dof_state.json", "executor_bookkeeping": "executor_bookkeeping.json",
        "rows_before_handoff": "rows_before_handoff.json", "rows_after_handoff": "rows_after_handoff.json", "handoff": "handoff.json", "condition_metrics": "condition_metrics.json", "run_metadata": "run_metadata.json",
    }
    manifest = evidence.build_condition_manifest(root, roles)
    manifest_sha = evidence.write_canonical_json(root / "condition_manifest.json", manifest)
    result: Dict[str, Any] = {
        "schema": "CORELAB_STAGE43_CONDITION_RESULT_V1", "stage": evidence.STAGE_ID, "condition_id": condition_id, "status": status,
        "bundle_sha256": evidence.sha256_file(args.bundle), "condition_manifest_sha256": manifest_sha, "condition_manifest": "condition_manifest.json",
        "condition_delta_base_xy_m": list(evidence.COHORT_DELTAS_M[condition_id]), "schedule_native_ticks": list(evidence.COHORT_SEGMENT_SCHEDULE),
        "cases": len(cases), "gr00t_forward_count": len(cases), "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
        "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "wbc_segment_lengths": [len(segment) for segment in list(rows_before) + list(rows_after)], "safety_gate_decisions": [item["safety_gate"]["decision"] for item in summaries],
        "completion": completion, "metrics": metrics, "setup": setup, "error": error, "manifest": manifest,
        "forbidden_operations": evidence.expected_forbidden_operations(),
    }
    # Keep the result immutable once serialized.  The seal binds these final
    # bytes; adding a self-hash to the result would create a needless cycle.
    result_sha = evidence.write_canonical_json(root / "condition_result.json", result)
    if status == "PASS":
        seal = {
            "schema": evidence.CONDITION_SEAL_SCHEMA, "stage": evidence.STAGE_ID, "condition_id": condition_id, "status": "PASS",
            "bundle_sha256": evidence.sha256_file(args.bundle), "condition_manifest_sha256": manifest_sha, "result_sha256": result_sha,
        }
        seal_sha = evidence.write_canonical_json(root / "condition_seal.json", seal)
        (root / "condition_seal.json.sha256").write_text("{}  {}\n".format(seal_sha, "condition_seal.json"), encoding="utf-8")
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "result": result, "result_sha256": result_sha}


def _run(args: argparse.Namespace) -> int:
    condition_id = str(args.condition_id)
    _require(condition_id in evidence.COHORT_ORDER, "unknown Stage43 condition")
    root = Path(args.output_dir).resolve()
    _require(not root.exists(), "condition output directory already exists")
    root.mkdir(parents=True)
    mailbox = root / "mailbox"; mailbox.mkdir(parents=True, exist_ok=True)
    (mailbox / "requests").mkdir(); (mailbox / "responses").mkdir()
    bundle_path = Path(args.bundle).resolve(strict=True)
    _require(bundle_path.read_bytes() == evidence.canonical_json_bytes(evidence.read_json(bundle_path)), "Stage43 bundle is not canonical JSON")
    bundle = evidence.read_json(bundle_path)
    evidence.verify_stage43_bundle_files(bundle, repo_root=REPO_ROOT)
    _require(bundle["repository"]["head"] == args.repository_head, "Stage43 runner repository HEAD argument drift")
    artifact_checks = _artifact_checks(bundle)
    trajectory = _load_pose_file(Path(bundle["artifacts"]["trajectory"]["path"]))
    worker = None; native = env = None
    cases: List[Dict[str, Any]] = []; rows_before: List[List[Mapping[str, Any]]] = []; rows_after: List[List[Mapping[str, Any]]] = []
    bind_events: List[Mapping[str, Any]] = []; handoffs: List[Mapping[str, Any]] = []; executor_events: List[Mapping[str, Any]] = []; state_events: List[Mapping[str, Any]] = []
    counter_events: List[Mapping[str, Any]] = []; telemetry: List[Mapping[str, Any]] = []
    setup: Optional[Mapping[str, Any]] = None; completion: Optional[Mapping[str, Any]] = None; metrics: Optional[Mapping[str, Any]] = None; error: Optional[Mapping[str, Any]] = None
    status = "BLOCKED"; worker_return_code = None
    try:
        _telemetry_sample(root, telemetry, "before_condition")
        worker_args = argparse.Namespace(**vars(args))
        worker_args.stage10_root = Path(bundle["artifacts"]["stage10_root"]["path"])
        worker_args.base_model_path = Path(bundle["artifacts"]["base_model_path"]["path"])
        worker_args.model_path = Path(bundle["artifacts"]["model_path"]["path"])
        worker_args.dataset_dir = Path(bundle["artifacts"]["dataset_dir"]["path"])
        worker_args.modality_config = Path(bundle["artifacts"]["modality_config"]["path"])
        worker_args.cosmos_snapshot = Path(bundle["artifacts"]["cosmos_snapshot_path"]["path"])
        worker_args.structured_stats = Path(bundle["artifacts"]["structured_stats"]["path"])
        worker_args.trajectory = Path(bundle["artifacts"]["trajectory"]["path"])
        worker_args.wbc_source_root = Path(bundle["artifacts"]["wbc_source"]["path"])
        worker_args.arm = Path(bundle["artifacts"]["wbc_arm_checkpoint"]["path"])
        worker_args.dog = Path(bundle["artifacts"]["wbc_dog_checkpoint"]["path"])
        worker_args.mode = "model_receding"
        worker_args.source_root = Path(bundle["artifacts"]["wbc_source"]["path"])
        native, env, actors, _, _ = _build_native(_native_args(worker_args), trajectory)
        _require(float(native.dt) > 0.0 and np.isfinite(float(native.dt)), "native dt is invalid")
        handoff_steps = int(round(HANDOFF_S / float(native.dt)))
        _require((handoff_steps, handoff_steps, handoff_steps, 2 * handoff_steps) == evidence.COHORT_SEGMENT_SCHEDULE, "observed native dt does not yield frozen 49/49/49/98 schedule")
        worker = _start_worker(worker_args, mailbox, root / "worker.log")
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(expected_source_frame=evidence.SOURCE_FRAME, wbc_frame=evidence.SOURCE_FRAME, tcp_semantic_id=evidence.TCP_SEMANTIC_ID, stale_timeout_s=0.75))
        setup = _apply_initial_base_delta(native, env, condition_id, evidence.COHORT_DELTAS_M[condition_id])
        sim_time = 0.0
        current_chunk, _ = _request_with_counters(native=native, env=env, mailbox=mailbox, request_id=condition_id + "-q0", seed=args.seed, sim_time=sim_time, gate=gate, adapter=adapter, args=worker_args, cases=cases, counter_events=counter_events)
        executor, initial_bind_event, state_event = _initial_bind(native, env, current_chunk, sim_time, {"stage": evidence.STAGE_ID, "condition_id": condition_id, "mode": "CORELAB_STAGE43_CONTROLLED_COHORT"})
        bind_events.append(initial_bind_event); state_events.append(state_event); executor_events.append(initial_bind_event.get("executor_initial"))
        for handoff_index in range(evidence.COHORT_REFERENCE_UPDATE_COUNT):
            before_counter = _counter_snapshot(cases, native)
            segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=handoff_steps, update_count=handoff_index)
            after_counter = _counter_snapshot(cases, native)
            rows_before.append(segment)
            counter_events.append({"event": "WBC_SEGMENT", "condition_id": condition_id, "segment_index": handoff_index, "steps_expected": handoff_steps, "steps_observed": len(segment), "before": before_counter, "after": after_counter, "termination": termination})
            _check_rows(segment, "Stage43 %s segment %d" % (condition_id, handoff_index))
            _require(len(segment) == handoff_steps and after_counter["wbc_step_count"] - before_counter["wbc_step_count"] == handoff_steps, "Stage43 WBC segment count mismatch")
            state_before = {"eef_world_m": _current_eef(native)[0].tolist(), "state_v1": _current_state_v1(native).as_array().tolist(), "state_v1_sha256": _array_sha(_current_state_v1(native).as_array())}
            next_chunk, _ = _request_with_counters(native=native, env=env, mailbox=mailbox, request_id=condition_id + "-q" + str(handoff_index + 1), seed=args.seed, sim_time=sim_time, gate=gate, adapter=adapter, args=worker_args, cases=cases, counter_events=counter_events)
            update_before = _counter_snapshot(cases, native)
            update_event, update_state = _update_reference(native, env, executor, current_chunk, next_chunk, sim_time, handoff_steps, handoff_index + 1)
            update_after = _counter_snapshot(cases, native)
            update_event = dict(update_event); update_event.update({"condition_id": condition_id, "observation_state_after_segment": state_before, "reference_counter_before": update_before["reference_update_count"], "reference_counter_after": update_after["reference_update_count"], "reference_update_delta_observed": update_after["reference_update_count"] - update_before["reference_update_count"]})
            _require(update_event["reference_update_delta_observed"] == 1 and update_event.get("reference_only") is True, "Stage43 reference update was not one reference-only update")
            handoffs.append(update_event); state_events.append(update_state); executor_events.append(update_event.get("executor_update")); current_chunk = next_chunk
        before_counter = _counter_snapshot(cases, native)
        final_segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=2 * handoff_steps, update_count=evidence.COHORT_REFERENCE_UPDATE_COUNT)
        after_counter = _counter_snapshot(cases, native)
        rows_after.append(final_segment)
        counter_events.append({"event": "WBC_COMPLETION_SEGMENT", "condition_id": condition_id, "steps_expected": 2 * handoff_steps, "steps_observed": len(final_segment), "before": before_counter, "after": after_counter, "termination": termination})
        _check_rows(final_segment, "Stage43 %s completion" % condition_id)
        _require(len(final_segment) == 2 * handoff_steps and after_counter["wbc_step_count"] - before_counter["wbc_step_count"] == 2 * handoff_steps, "Stage43 completion segment count mismatch")
        native_time = float(native.traj_sim_time[0].detach().cpu().item()); ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item()); ref_L = float(native.traj_batch.L[0].detach().cpu().item()); ref_duration = float(current_chunk.validity_end - current_chunk.validity_start)
        endpoint = bool(native_time >= ref_duration - max(float(native.dt) * 0.5, 1e-9) and ref_s >= ref_L - max(1e-4, 1e-3 * max(ref_L, 1.0)))
        completion = {"native_time_s": native_time, "reference_duration_s": ref_duration, "traj_s_ref": ref_s, "traj_L": ref_L, "endpoint_reached": endpoint, "termination": termination, "native_handoff_steps": handoff_steps, "schedule_native_ticks": list(evidence.COHORT_SEGMENT_SCHEDULE), "sim_time_s": sim_time}
        _require(endpoint, "Stage43 reference endpoint was not reached")
        _require(len(cases) == evidence.COHORT_FORWARD_COUNT and int(native.stage21_reference_update_count) == evidence.COHORT_REFERENCE_UPDATE_COUNT and int(native.stage21_wbc_forward_count) == sum(evidence.COHORT_SEGMENT_SCHEDULE), "Stage43 final counter contract mismatch")
        _require(all(case["safety_gate"]["decision"] in {GateDecision.ACCEPT.value, GateDecision.WARN.value} for case in cases), "Stage43 SafetyGate decision vector is inadmissible")
        flat_rows = [row for segment in rows_before + rows_after for row in segment]
        metrics = _metrics(setup, flat_rows)
        status = "PASS"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}; status = "BLOCKED"
    finally:
        try:
            if native is not None:
                _telemetry_sample(root, telemetry, "after_native")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        try:
            worker_return_code = _stop_worker(worker, mailbox)
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        try:
            _telemetry_sample(root, telemetry, "after_condition")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        try:
            materialized = _materialize(args=args, bundle=bundle, root=root, mailbox=mailbox, condition_id=condition_id, setup=setup, cases=cases, rows_before=rows_before, rows_after=rows_after, bind_events=bind_events, handoffs=handoffs, executor_events=executor_events, state_events=state_events, counter_events=counter_events, telemetry=telemetry, native=native, worker_return_code=worker_return_code, status=status, completion=completion, metrics=metrics, artifact_checks=artifact_checks, error=error)
            _sync_evidence_tree(root)
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}; status = "BLOCKED"; materialized = None
            _write_json(root / "materialize_error.json", error)
        result_payload = {"schema": "CORELAB_STAGE43_CONDITION_RESULT_POINTER_V1", "condition_id": condition_id, "status": status, "error": error, "materialized": None if materialized is None else {"result_sha256": materialized["result_sha256"], "manifest_sha256": materialized["manifest_sha256"]}}
        _write_durable_json(root / "run_result.json", result_payload)
        print(json.dumps({"stage": evidence.STAGE_ID, "condition_id": condition_id, "status": status, "output_dir": str(root), "cases": len(cases), "wbc_steps": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_updates": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "error": error}, sort_keys=True), flush=True)
        # Keep native teardown as the final native operation.  A SIGSEGV here
        # is recoverable only when the persisted pointer and cell evidence are
        # already complete; the parent verifier enforces that boundary.
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return 0 if status == "PASS" else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-id", choices=evidence.COHORT_ORDER, required=True)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--arm", type=Path)
    parser.add_argument("--dog", type=Path)
    parser.add_argument("--stage10-root", type=Path)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--cosmos-snapshot", type=Path)
    parser.add_argument("--structured-stats", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--wbc-source-root", type=Path)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=evidence.DEV_SEED)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=1200.0)
    args = parser.parse_args(argv)
    if args.seed != evidence.DEV_SEED:
        parser.error("Stage43 development seed is fixed at 4242")
    if args.model_python is None:
        args.model_python = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/runtime/venv/bin/python")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
