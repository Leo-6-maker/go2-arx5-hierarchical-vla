#!/usr/bin/env python3
"""PI0252: reuse PI0226 canary, complete its reference and retain actual frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


TARGET = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
SUCCESS_RADIUS = 0.035
ACTION_PERIOD = 0.05
MODEL_SEED = 22301


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _distance_stats(positions: np.ndarray, target: np.ndarray, dt: float) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(positions - target.reshape(1, 3), axis=1)
    first_hit = np.flatnonzero(distances <= SUCCESS_RADIUS)
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.asarray([])
    speeds = jumps / float(dt) if jumps.size else np.asarray([])
    return {
        "count": int(len(positions)),
        "min_distance_m": float(np.min(distances)) if len(distances) else None,
        "min_distance_tick": int(np.argmin(distances)) if len(distances) else None,
        "endpoint_distance_m": float(distances[-1]) if len(distances) else None,
        "first_hit_tick": int(first_hit[0]) if len(first_hit) else None,
        "path_length_m": float(np.sum(jumps)) if jumps.size else 0.0,
        "max_linear_speed_mps": float(np.max(speeds)) if speeds.size else 0.0,
        "finite": bool(np.isfinite(positions).all()),
    }


def _load_oracle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        candidates = ("eef_position", "position", "xyz", "eef_xyz")
        for key in candidates:
            if key in data:
                value = np.asarray(data[key], dtype=np.float64)
                if value.ndim == 2 and value.shape[1] == 3:
                    return value, keys
    raise RuntimeError(f"reference file has no [N,3] position field: {path}")


def _array_attr(native: Any, name: str) -> Any:
    value = getattr(native, name, None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value[0].detach().cpu().numpy()
    else:
        value = np.asarray(value)
        if value.ndim > 1:
            value = value[0]
    return _finite(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--mailbox", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--target-position-world-m", nargs=3, type=float, default=TARGET.tolist())
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22301)
    parser.add_argument("--response-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    assert args.seed == MODEL_SEED
    binding_path = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0252/pre_full_sim/sim_binding.json')
    binding = json.loads(binding_path.read_text())
    assert binding['status'] == 'PASS_CPU_BINDING'
    assert binding['checkpoint_revision'] == 'efecea485ae77000599d19171b02491f8d514f42'

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {args.output_dir}")
    if args.mailbox.exists() and any(path.is_file() for path in args.mailbox.rglob("*")):
        raise RuntimeError(f"refusing non-empty mailbox: {args.mailbox}")
    cell_target = np.asarray(args.target_position_world_m, dtype=np.float64)
    checkpoint_dir = args.checkpoint_dir

    code = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    import stage21_native_model_receding_horizon as runner
    from go2arx5_vla.benchmark.lifecycle import reference_is_complete

    captured: dict[str, Any] = {'frames': []}
    original_build = runner._build_native
    original_capture = runner._capture_scene

    def capture_actual_frame(native):
        pair = original_capture(native)
        captured['frames'].append(pair)
        return pair

    runner._capture_scene = capture_actual_frame

    def build_and_install_stop(native_args: Any, trajectory: Any):
        result = original_build(native_args, trajectory)
        native = result[0]
        captured["native"] = native
        captured['native_dt_s'] = float(native.dt)
        captured["anchor_world_m"] = (
            native.stage21_anchor_world.detach().cpu().numpy().reshape(3).copy()
        )
        original_pop = native.pop_stage16_termination

        def pop_with_protocol_stop():
            event = original_pop()
            if event is not None:
                return event
            reference_time = float(native.traj_sim_time[0].detach().cpu().item())
            s_ref = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
            reference_length = float(native.traj_batch.L[0].detach().cpu().item())
            duration_value = getattr(native.traj_batch, "T", None)
            if duration_value is None:
                duration = float(native.stage21_current_chunk.point_timestamps[-1])
            else:
                if hasattr(duration_value, "detach"):
                    duration = float(duration_value[0].detach().cpu().item())
                else:
                    duration = float(np.asarray(duration_value).reshape(-1)[0])
            if reference_is_complete(
                sim_time=reference_time,
                reference_duration=duration,
                s_ref=s_ref,
                reference_length=reference_length,
                dt=float(native.dt),
            ):
                return {
                    "termination_code": "REFERENCE_COMPLETE",
                    "stage21_wrapper_stop": "REFERENCE_COMPLETE",
                    "reference_time_s": reference_time,
                    "reference_duration_s": duration,
                }
            return None

        native.pop_stage16_termination = pop_with_protocol_stop
        return result

    original_telemetry = runner._native_runtime_telemetry
    original_executor = runner.RecedingHorizonExecutorV1
    original_policy_step = runner._policy_step

    class OneChunkExecutor(original_executor):
        def __init__(self, *executor_args: Any, **executor_kwargs: Any):
            if int(executor_kwargs.get("prefix_steps", 0)) == 500:
                executor_kwargs["prefix_steps"] = 4
            super().__init__(*executor_args, **executor_kwargs)

    def telemetry_with_wbc_commands(native: Any) -> dict[str, Any]:
        telemetry = dict(original_telemetry(native))
        for name in ("base_feedforward_cmd", "delta_velocity_cmd", "commands_dog"):
            value = _array_attr(native, name)
            if value is not None:
                telemetry[name] = value
        return telemetry

    def policy_step_with_protocol_stop(native: Any, env: Any, actors: Any):
        done, policy = original_policy_step(native, env, actors)
        capture_actual_frame(native)
        if done:
            return done, policy
        reference_time = float(native.traj_sim_time[0].detach().cpu().item())
        s_ref = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
        reference_length = float(native.traj_batch.L[0].detach().cpu().item())
        duration_value = getattr(native.traj_batch, "T", None)
        if duration_value is None:
            duration = float(native.stage21_current_chunk.point_timestamps[-1])
        elif hasattr(duration_value, "detach"):
            duration = float(duration_value[0].detach().cpu().item())
        else:
            duration = float(np.asarray(duration_value).reshape(-1)[0])
        if reference_is_complete(
            sim_time=reference_time,
            reference_duration=duration,
            s_ref=s_ref,
            reference_length=reference_length,
            dt=float(native.dt),
        ):
            return True, policy
        return done, policy

    runner._build_native = build_and_install_stop
    runner._native_runtime_telemetry = telemetry_with_wbc_commands
    runner.RecedingHorizonExecutorV1 = OneChunkExecutor
    runner._policy_step = policy_step_with_protocol_stop
    runner.TARGET_POSITION_WORLD_M = cell_target.copy()
    native_args = argparse.Namespace(
        source_root=args.source_root,
        trajectory=args.trajectory,
        arm=args.arm,
        dog=args.dog,
        mailbox=args.mailbox,
        output_dir=args.output_dir,
        sim_device=args.sim_device,
        seed=int(args.seed),
        model_seed=MODEL_SEED,
        prefix_steps=500,
        max_prefix_updates=0,
        action_period_s=ACTION_PERIOD,
        response_timeout_s=float(args.response_timeout_s),
        mode="model_receding",
    )

    result = runner._model_receding(native_args, runner._load_pose_file(args.trajectory))
    summary_path = args.output_dir / "model_receding_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if captured['frames']:
        np.savez_compressed(args.output_dir / 'actual_camera_frames.npz',
            body_rgb=np.stack([f[0] for f in captured['frames']]),
            wrist_rgb=np.stack([f[1] for f in captured['frames']]))
    rows = [
        json.loads(line)
        for line in (args.output_dir / "model_receding_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchor = np.asarray(captured["anchor_world_m"], dtype=np.float64)
    model_case = (summary.get("model_cases") or [{}])[0]
    response = model_case.get("response") or {}
    sampler_path = args.mailbox / 'responses' / (model_case['request_id'] + '_sampler.json')
    sampler = json.loads(sampler_path.read_text())
    assert sampler['status'] == 'PASS' and sampler['head_calls'] == 4 and sampler['noise_count'] == 1
    assert sampler['query_seed'] == MODEL_SEED and response['worker_forward_count_delta'] == 1
    assert sampler['noise'][0]['dtype'] == 'torch.bfloat16' and sampler['noise'][0]['device'] == 'cuda:0'
    with np.load(sampler['noise_artifact'], allow_pickle=False) as data:
        assert hashlib.sha256(data['raw_bytes'].tobytes()).hexdigest() == sampler['noise'][0]['raw_tensor_sha256']
    artifact = Path(response["artifact"])
    with np.load(artifact, allow_pickle=False) as data:
        decoded = np.asarray(data["decoded_physical_action"], dtype=np.float64)
    decoded = decoded[0] if decoded.ndim == 3 else decoded
    model_world = decoded[:, :3] + anchor.reshape(1, 3)
    oracle_local, oracle_keys = _load_oracle(args.trajectory)
    oracle_world = oracle_local + anchor.reshape(1, 3)
    target_local = cell_target - anchor

    physical_world = np.asarray([row["achieved_pos"] for row in rows], dtype=np.float64).reshape(-1, 3)
    physical_stats = _distance_stats(physical_world, cell_target, captured['native_dt_s'])
    model_stats = _distance_stats(model_world, cell_target, ACTION_PERIOD)
    oracle_stats = _distance_stats(oracle_world, cell_target, ACTION_PERIOD)
    desired_world = np.asarray([row["desired_pos"] for row in rows], dtype=np.float64).reshape(-1, 3)
    tracking_error = np.linalg.norm(physical_world - desired_world, axis=1)
    wrapper_stops = [
        row.get("termination_event", {}).get("stage21_wrapper_stop")
        for row in rows
        if isinstance(row.get("termination_event"), dict)
        and row.get("termination_event", {}).get("stage21_wrapper_stop")
    ]
    native_events = [
        row.get("termination_event")
        for row in rows
        if isinstance(row.get("termination_event"), dict)
        and not row.get("termination_event", {}).get("stage21_wrapper_stop")
    ]
    invalid = []
    for row in rows:
        event = row.get("termination_event")
        protocol_stop = isinstance(event, dict) and bool(event.get("stage21_wrapper_stop"))
        invalid.append(
            bool(row.get("invalid_state", False))
            or bool(row.get("fall_flag", False))
            or bool(row.get("joint_limit_termination", False))
            or (bool(row.get("done", False)) and not protocol_stop)
        )
    stop = wrapper_stops[0] if wrapper_stops else None
    physical_failure = bool(native_events) or bool(any(invalid)) or not bool(np.isfinite(physical_world).all())
    reference_hit = bool(model_stats["min_distance_m"] is not None and model_stats["min_distance_m"] <= SUCCESS_RADIUS)
    achieved_hit = bool(physical_stats["min_distance_m"] is not None and physical_stats["min_distance_m"] <= SUCCESS_RADIUS)
    gate_rejected = any(e['result']['decision'] == 'REJECT' for e in summary.get('safety_gate', []))
    complete_duration = stop == 'REFERENCE_COMPLETE'
    if gate_rejected:
        cell_decision = 'VALID_SIM_GATE_FAIL'
    elif physical_failure:
        cell_decision = "PI0225_NATIVE_TERMINATION_OR_PHYSICAL_FAILURE_STOP"
    elif reference_hit and achieved_hit:
        cell_decision = "ONE_CHUNK_CELL_TARGET_HIT"
    elif reference_hit and not achieved_hit:
        cell_decision = "MODEL_REFERENCE_TASK_CORRECT_BUT_WBC_TRACKING_INSUFFICIENT_FOR_MODEL_PROFILE"
    elif not reference_hit and not achieved_hit:
        cell_decision = "MODEL_TASK_TRAJECTORY_ACCURACY_LIMITATION"
    else:
        cell_decision = "MODEL_REFERENCE_MISS_BUT_ACHIEVED_TCP_HIT_DIAGNOSTIC"

    manifest = {
        "schema": "PI0252_PRE_FULL_G_ONE_CHUNK_V1",
        "goal": "PI0252_R2_G_QUALIFIED_SIM_FULL_CONTINUATION_GO",
        "cell": args.label,
        "decision": cell_decision,
        "qualification": 'PASS' if achieved_hit and complete_duration and not physical_failure and not gate_rejected else 'FAIL',
        "development_checkpoint": 'G2000',
        "formal_selected_checkpoint": 'NONE',
        "checkpoint_revision": 'efecea485ae77000599d19171b02491f8d514f42',
        "checkpoint_tree_sha256": 'cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226',
        "protocol": {
            "checkpoint": checkpoint_dir.name,
            "model_forward_count": 1,
            "native_reference_updates": 0,
            "persistent_reference": True,
            "fresh_rgb_state_v1": True,
            "dense_relative_matched_decode": True,
            "safety_gate_unchanged": True,
            "wbc_unchanged": True,
            "target_position_world_m": cell_target.tolist(),
            "success_radius_m": SUCCESS_RADIUS,
            "action_period_s": ACTION_PERIOD,
            "native_control_dt_s": captured['native_dt_s'],
            "complete_reference_duration": complete_duration,
            "no_replan": True,
            "no_reference_hot_swap": True,
            "no_progress_reset_after_initial_bind": True,
            "target_xyz_model_input": False,
            "dev_e_unread": True,
            "final_f_unread": True,
        },
        "identity": {
            "sim_binding_sha256": _sha256_file(binding_path),
            "actual_sampler_receipt": sampler,
            "actual_sampler_receipt_sha256": _sha256_file(sampler_path),
            "model_seed": MODEL_SEED,
            "native_reset_seed": int(args.seed),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_index_sha256": _sha256_file(checkpoint_dir / "model.safetensors.index.json"),
            "full_checkpoint_readback": '/home/liuyu-nfs/Projects/go2-arx5/PI0252/evidence/G2000/hf_readback.json',
            "arm_checkpoint_sha256": _sha256_file(args.arm),
            "dog_checkpoint_sha256": _sha256_file(args.dog),
            "legacy_manifest_dog_sha256_field": "51776ddfd7173444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
            "legacy_manifest_dog_sha256_field_valid": False,
            "wbc_source_tree_sha256_sealed": "9d653c39c73d5755a29d46f4dff91b69120244f660f41a977db471d58ba8b193",
            "trajectory_path": str(args.trajectory),
            "trajectory_sha256": _sha256_file(args.trajectory),
            "anchor_world_m": anchor.tolist(),
            "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
            "tcp_semantic_id": "x5_link6_gripper_center",
        },
        "model_reference": {
            "artifact": str(artifact),
            "raw_sha256": response.get("raw_sha256"),
            "decoded_sha256": response.get("decoded_sha256"),
            "worker_forward_count_delta": response.get("worker_forward_count_delta"),
            "decoded_shape": list(decoded.shape),
            "decoded_world_target_stats": model_stats,
            "decoded_local_first_point_jump_m": float(
                np.linalg.norm(decoded[0, :3] - np.asarray(model_case["state"], dtype=np.float64)[:3])
            ),
            "oracle_world_target_stats": oracle_stats,
            "oracle_npz_keys": oracle_keys,
            "oracle_vs_model_position_rmse_m": float(
                np.sqrt(np.mean(np.sum((model_world - oracle_world[: len(model_world)]) ** 2, axis=1)))
            ),
            "profile_from_matched_decoder": model_case.get("profile"),
            "safety_gate": model_case.get("safety_gate"),
            "target_xyz_consumed_by_inference": False,
            "inference_input_fields": ["external_rgb", "wrist_rgb", "eef_9d", "gripper_position", "language"],
        },
        "physical_wbc": {
            "status": 'NOT_EXECUTED_GATE_REJECT' if gate_rejected else ("PHYSICAL_CLEAN" if not physical_failure else "PHYSICAL_FAILURE"),
            "tracking_position_frame": 'WORLD',
            "achieved_world_target_stats": physical_stats,
            "first_hit_tick": physical_stats["first_hit_tick"],
            "termination": summary.get("termination"),
            "protocol_stop": stop,
            "native_events": native_events,
            "native_termination_observed": bool(native_events) or summary.get("termination") == "NATIVE_TASK_TERMINATION",
            "tracking_error_max_m": float(np.max(tracking_error)) if len(tracking_error) else None,
            "tracking_error_mean_m": float(np.mean(tracking_error)) if len(tracking_error) else None,
            "tracking_error_final_m": float(tracking_error[-1]) if len(tracking_error) else None,
            "rows": int(len(rows)),
            "wbc_forward_count": int(len(rows)),
            "reference_updates": int(summary.get("reference_updates", 0)),
            "telemetry_fields": sorted(
                key for key in (rows[0].keys() if rows else []) if key not in {"step", "sim_time"}
            ),
            "tracking_trace": str(args.output_dir / "model_receding_rows.jsonl"),
        },
        "source_outputs": {
            "stage21_summary": str(summary_path),
            "stage21_summary_sha256": _sha256_file(summary_path),
            "stage21_rows": str(args.output_dir / "model_receding_rows.jsonl"),
            "stage21_rows_sha256": _sha256_file(args.output_dir / "model_receding_rows.jsonl"),
            "worker_mailbox": str(args.mailbox),
        },
    }
    manifest_path = args.output_dir / "PI0252_G_ONE_CHUNK_MANIFEST.json"
    manifest_path.write_text(json.dumps(_finite(manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_finite({
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "cell": args.label,
        "decision": cell_decision,
        "model_min_distance_m": model_stats["min_distance_m"],
        "achieved_min_distance_m": physical_stats["min_distance_m"],
        "protocol_stop": stop,
        "rows": len(rows),
    })), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
