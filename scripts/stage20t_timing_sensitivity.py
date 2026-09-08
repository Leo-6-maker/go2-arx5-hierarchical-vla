#!/usr/bin/env python3
"""Run the matched Stage20T native-WBC timing attribution matrix.

The experiment has three reference families:

* A: the exact decoded Stage20S model path;
* B: the future desired EEF label from the same observation;
* C: a qualified Stage16 native control.

For A and B, only the timestamp grid is scaled.  The spatial arrays are
copied byte-for-byte into every derived input, and every non-1.0 scale is
labelled ``DERIVED_TIMING_DIAGNOSTIC``.  The provider/binding/adapter chain is
executed before the delivered native launcher.  This script never changes the
native WBC source, checkpoint, timing threshold, or termination semantics.

All rollout artifacts are intended for a node-local output root.  The Git
repository receives only this harness and a small manifest/report, never model
weights, WBC source, or rollout tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1, TrajectoryChunkV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
)


SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"
TIMING_THRESHOLD_FRACTION = 0.70
DEFAULT_SCALES = (0.75, 1.0, 1.25, 1.50, 2.0)
DEFAULT_SEEDS = (13, 14, 15)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stats(values: Any) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def scale_timestamps(timestamps: Any, scale: float) -> np.ndarray:
    """Scale duration about the first point without changing spatial data."""

    ts = np.asarray(timestamps, dtype=np.float64)
    factor = float(scale)
    if ts.ndim != 1 or ts.size < 2 or not np.isfinite(ts).all():
        raise ValueError("timestamps must be a finite [N] array with N >= 2")
    if not np.all(np.diff(ts) > 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("scale must be positive and finite")
    return ts[0] + (ts - ts[0]) * factor


def spatial_geometry_hash(position: Any, quaternion: Any) -> str:
    """Hash position and SO(3), making quaternion sign irrelevant."""

    pos = np.asarray(position, dtype=np.float64)
    rot = quat_to_matrix(np.asarray(quaternion, dtype=np.float64))
    packed = np.concatenate([pos.reshape(pos.shape[0], -1), rot.reshape(rot.shape[0], -1)], axis=-1)
    return sha256_array(packed)


def _trajectory_audit(
    position: Any,
    quaternion: Any,
    timestamps: Any,
    achieved_position: Any,
    achieved_orientation: Any,
    *,
    lambda_m_per_rad: float = 0.15,
) -> dict[str, Any]:
    pos = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    achieved_pos = np.asarray(achieved_position, dtype=np.float64)
    achieved_quat = np.asarray(achieved_orientation, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or quat.shape != (len(pos), 4):
        raise ValueError("trajectory must contain position [N,3] and quaternion [N,4]")
    if ts.shape != (len(pos),) or len(pos) < 2 or not np.all(np.diff(ts) > 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing [N]")
    matrices = quat_to_matrix(quat)
    dt = np.diff(ts)
    translation_steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    rotation_steps = geodesic_angle(matrices[:-1], matrices[1:])
    linear_speed = translation_steps / dt
    angular_speed = rotation_steps / dt
    se3_steps = np.sqrt(translation_steps**2 + (float(lambda_m_per_rad) * rotation_steps) ** 2)

    curvature = np.asarray([], dtype=np.float64)
    if len(pos) >= 3:
        segments = np.diff(pos, axis=0)
        segment_length = np.linalg.norm(segments, axis=-1)
        valid = segment_length > 1e-12
        tangent = np.zeros_like(segments)
        tangent[valid] = segments[valid] / segment_length[valid, None]
        dot = np.sum(tangent[:-1] * tangent[1:], axis=-1)
        turn = np.arccos(np.clip(dot, -1.0, 1.0))
        denominator = np.maximum((segment_length[:-1] + segment_length[1:]) * 0.5, 1e-12)
        curvature = np.where(valid[:-1] & valid[1:], turn / denominator, 0.0)

    initial_rotation = float(
        geodesic_angle(quat_to_matrix(achieved_quat), matrices[0])
    )
    return {
        "finite": bool(np.isfinite(pos).all() and np.isfinite(quat).all() and np.isfinite(ts).all()),
        "point_count": int(len(pos)),
        "duration_s": float(ts[-1] - ts[0]),
        "timestamp_start_s": float(ts[0]),
        "timestamp_end_s": float(ts[-1]),
        "initial_position_discontinuity_m": float(np.linalg.norm(pos[0] - achieved_pos)),
        "initial_rotation_discontinuity_rad": initial_rotation,
        "translation_path_length_m": float(np.sum(translation_steps)),
        "se3_arc_length_m": float(np.sum(se3_steps)),
        "translation_step_m": _stats(translation_steps),
        "rotation_step_rad": _stats(rotation_steps),
        "linear_speed_mps": _stats(linear_speed),
        "angular_speed_radps": _stats(angular_speed),
        "curvature_proxy_1_per_m": _stats(curvature),
        "workspace_min_m": np.min(pos, axis=0).tolist(),
        "workspace_max_m": np.max(pos, axis=0).tolist(),
        "spatial_geometry_sha256": spatial_geometry_hash(pos, quat),
        "timestamp_sha256": sha256_array(ts),
    }


def _load_model_output(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        decoded = np.asarray(data["decoded_physical_action"], dtype=np.float64)
        state = np.asarray(data["input_state"], dtype=np.float64).reshape(-1)
        desired = np.asarray(data["desired_action"], dtype=np.float64) if "desired_action" in data else None
        raw = np.asarray(data["raw_universal_action"], dtype=np.float64) if "raw_universal_action" in data else None
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        raise ValueError(f"expected finite decoded_physical_action [1,40,10], got {decoded.shape}")
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite input_state [10], got {state.shape}")
    if desired is None:
        raise ValueError("Stage20T requires desired_action from the same model observation")
    if desired.ndim == 3 and desired.shape[0] == 1:
        desired = desired[0]
    if desired.shape != (40, 10) or not np.isfinite(desired).all():
        raise ValueError(f"expected finite desired_action [40,10], got {desired.shape}")
    return {
        "decoded": decoded[0],
        "desired": desired,
        "state": state,
        "raw": np.asarray(raw) if raw is not None else np.empty((0,), dtype=np.float64),
    }


def _load_benchmark(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        timestamps = np.asarray(data["timestamp"], dtype=np.float64)
        position = np.asarray(data["position"], dtype=np.float64)
        quaternion = np.asarray(data["quaternion"], dtype=np.float64)
        gripper = np.asarray(data["gripper"], dtype=np.float64) if "gripper" in data else None
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("benchmark timestamp must be strictly increasing [N]")
    if position.shape != (len(timestamps), 3) or quaternion.shape != (len(timestamps), 4):
        raise ValueError("benchmark position/quaternion shapes do not match timestamps")
    if gripper is not None and gripper.shape not in {(len(timestamps),), (len(timestamps), 1)}:
        raise ValueError("benchmark gripper shape does not match timestamps")
    return {
        "timestamps": timestamps,
        "position": position,
        "quaternion": quaternion,
        "gripper": gripper,
    }


def _build_model_chunk(
    decoded_action: np.ndarray,
    timestamps: np.ndarray,
    *,
    chunk_id: str,
    provenance: Mapping[str, Any],
) -> TrajectoryChunkV1:
    provider = GR00TTrajectoryProvider(
        decoded_action,
        config=GR00TProviderConfig(
            source_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_ID,
            action_sample_period_s=None,
            timestamp_provenance=str(provenance["timestamp_provenance"]),
        ),
        point_timestamps=timestamps,
        observation_timestamp=float(timestamps[0]),
        chunk_id=chunk_id,
        provenance=dict(provenance),
    )
    chunk = provider.next_chunk(float(timestamps[0]))
    if chunk is None:  # pragma: no cover - defensive
        raise RuntimeError("GR00T provider emitted no chunk")
    return chunk


def _build_benchmark_chunk(
    timestamps: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    *,
    chunk_id: str,
    gripper: np.ndarray | None,
    provenance: Mapping[str, Any],
) -> TrajectoryChunkV1:
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="BenchmarkTrajectoryProvider",
        observation_timestamp=float(timestamps[0]),
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=quaternion,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_ID,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=gripper,
        provenance=dict(provenance),
    )


def _interface_seal(
    chunk: TrajectoryChunkV1,
    *,
    achieved_position: np.ndarray,
    achieved_orientation: np.ndarray,
    episode_id: str,
) -> dict[str, Any]:
    gate = TrajectorySafetyGateV1().evaluate(
        chunk,
        now=float(chunk.observation_timestamp),
        achieved_position=achieved_position,
        achieved_orientation=achieved_orientation,
    )
    result: dict[str, Any] = {
        "schema": "STAGE20T_INTERFACE_SEAL_V1",
        "simulation_only": True,
        "hardware_control": False,
        "actual_wbc_execution": False,
        "chunk": {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "point_count": chunk.point_count,
            "point_timestamps": chunk.point_timestamps.tolist(),
            "position_shape": list(chunk.eef_position.shape),
            "orientation_shape": list(chunk.eef_orientation.shape),
            "gripper_shape": None if chunk.gripper is None else list(np.asarray(chunk.gripper).shape),
            "source_frame": chunk.source_frame,
            "tcp_semantic_id": chunk.tcp_semantic_id,
            "provenance": dict(chunk.provenance),
        },
        "safety_gate": gate.as_dict(),
        "correction_applied": False,
        "no_post_model_spatial_transform": True,
        "binding": None,
        "preview": None,
    }
    if gate.accepted:
        binding = EpisodeReferenceBindingV1.from_external_chunk(chunk, episode_id=episode_id)
        bound = binding.bind_external_chunk(chunk)
        preview, gripper = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=SOURCE_FRAME,
                wbc_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_ID,
                stale_timeout_s=0.75,
            )
        ).adapt(bound, controller_timestamp=float(chunk.observation_timestamp))
        result["binding"] = {
            "episode_id": binding.episode_id,
            "binding_semantics": binding.binding_semantics,
            "source_frame": bound.source_frame,
            "tcp_semantic_id": bound.tcp_semantic_id,
            "position_preservation_max_m": float(
                np.max(np.linalg.norm(bound.eef_position - chunk.eef_position, axis=-1))
            ),
            "rotation_preservation_max_rad": float(
                np.max(
                    geodesic_angle(
                        quat_to_matrix(bound.eef_orientation),
                        quat_to_matrix(chunk.eef_orientation),
                    )
                )
            ),
        }
        result["preview"] = {
            "shape": list(preview.xyz_rot6d.shape),
            "finite": bool(np.isfinite(preview.xyz_rot6d).all()),
            "source_frame": preview.source_frame,
            "tcp_semantic_id": preview.tcp_semantic_id,
            "endpoint_clamped": preview.endpoint_clamped.tolist(),
            "parallel_gripper_channel": gripper is not None,
        }
    return result


def _save_candidate(path: Path, chunk: TrajectoryChunkV1) -> None:
    kwargs: dict[str, Any] = {
        "timestamp": chunk.point_timestamps,
        "position": chunk.eef_position,
        "quaternion": chunk.eef_orientation,
    }
    if chunk.gripper is not None:
        kwargs["gripper"] = np.asarray(chunk.gripper, dtype=np.float64)
    np.savez_compressed(path, **kwargs)


def _run_native(
    *,
    native_script: Path,
    source_root: Path,
    arm: Path,
    dog: Path,
    candidate: Path,
    output_dir: Path,
    metadata_path: Path,
    seconds: float,
    seed: int,
    sim_device: str,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(native_script),
        "--source-root",
        str(source_root),
        "--arm",
        str(arm),
        "--dog",
        str(dog),
        "--reference-mode",
        "BENCHMARK_CONTINUOUS",
        "--trajectory",
        str(candidate),
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(float(seconds)),
        "--seed",
        str(int(seed)),
        "--sim-device",
        str(sim_device),
        "--metadata-json",
        str(metadata_path),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(timeout_s),
        )
        stdout = completed.stdout + completed.stderr
        timed_out = False
        returncode: int | None = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") + (exc.stderr or "")
        timed_out = True
        returncode = None
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stage20t_native_stdout.log"
    log_path.write_text(stdout, encoding="utf-8")
    summary_path = output_dir / "native_external_summary.json"
    rows_path = output_dir / "native_external_rows.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    if summary is None:
        execution_status = "TIMEOUT" if timed_out else "LAUNCH_FAILED"
        termination = execution_status
        reference_completed = False
    else:
        reference_completed = bool(summary.get("reference_completed"))
        termination = str(summary.get("termination_code") or summary.get("termination") or "UNKNOWN")
        execution_status = "REFERENCE_COMPLETE" if reference_completed else termination
    return {
        "command": command,
        "process_returncode": returncode,
        "elapsed_s": float(elapsed),
        "timed_out": bool(timed_out),
        "stdout_log": str(log_path),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
        "rows_path": str(rows_path) if rows_path.is_file() else None,
        "execution_status": execution_status,
        "termination_code": termination,
        "reference_completed": reference_completed,
        "summary": summary,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _build_run_plan(scales: tuple[float, ...], seeds: tuple[int, ...], control_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for condition in ("A_MODEL", "B_GT"):
        for scale in scales:
            for seed in seeds:
                plan.append(
                    {
                        "condition": condition,
                        "scale": float(scale),
                        "seed": int(seed),
                        "derived_timing_diagnostic": not np.isclose(scale, 1.0),
                    }
                )
    for seed in control_seeds:
        plan.append(
            {
                "condition": "C_STAGE16_CONTROL",
                "scale": 1.0,
                "seed": int(seed),
                "derived_timing_diagnostic": False,
            }
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--stage16-trajectory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-script", type=Path, default=Path("scripts/native_wbc_benchmark.py"))
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--original-period-s", type=float, required=True)
    parser.add_argument("--original-period-provenance", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--control-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--native-timeout-s", type=float, default=240.0)
    parser.add_argument("--seconds-margin-s", type=float, default=0.35)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.original_period_s <= 0.0 or not np.isfinite(args.original_period_s):
        parser.error("--original-period-s must be positive and finite")
    if args.native_timeout_s <= 0.0 or args.seconds_margin_s < 0.0:
        parser.error("native timeout and seconds margin are invalid")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "stage20t_timing_sensitivity_manifest.json"
    existing: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_runs = {str(row["run_id"]): row for row in existing.get("runs", [])}

    model = _load_model_output(args.model_output.expanduser().resolve())
    benchmark = _load_benchmark(args.stage16_trajectory.expanduser().resolve())
    original_timestamps = np.arange(40, dtype=np.float64) * float(args.original_period_s)
    plan = _build_run_plan(
        tuple(float(x) for x in args.scales),
        tuple(int(x) for x in args.seeds),
        tuple(int(x) for x in args.control_seeds),
    )
    manifest: dict[str, Any] = {
        "schema": "STAGE20T_TIMING_SENSITIVITY_MANIFEST_V1",
        "stage": "STAGE20T",
        "created_by": str(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "simulation_only": True,
        "hardware_control": False,
        "actual_gr00t_closed_loop": False,
        "scientific_declarations": {
            "matched_spatial_geometry": True,
            "spatial_geometry_changed_by_harness": False,
            "non_unit_scales_are": "DERIVED_TIMING_DIAGNOSTIC",
            "original_model_path_is_unchanged": True,
            "controller_semantics_changed": False,
            "wbc_checkpoint_changed": False,
            "timing_threshold_changed": False,
            "native_timing_threshold_fraction": TIMING_THRESHOLD_FRACTION,
            "production_action_period_decision": "NOT_MADE",
        },
        "inputs": {
            "model_output": str(args.model_output.expanduser().resolve()),
            "model_output_sha256": sha256_file(args.model_output.expanduser().resolve()),
            "stage16_trajectory": str(args.stage16_trajectory.expanduser().resolve()),
            "stage16_trajectory_sha256": sha256_file(args.stage16_trajectory.expanduser().resolve()),
            "source_root": str(args.source_root),
            "arm_checkpoint": str(args.arm),
            "dog_checkpoint": str(args.dog),
            "native_script": str(args.native_script),
        },
        "timing_grid": {
            "original_period_s": float(args.original_period_s),
            "original_period_provenance": str(args.original_period_provenance),
            "scales": [float(x) for x in args.scales],
            "seed_sets": {"model_and_gt": [int(x) for x in args.seeds], "stage16_control": [int(x) for x in args.control_seeds]},
        },
        "runs": [],
    }

    native_script = args.native_script.expanduser().resolve()
    run_root = output_root / "runs"
    for item in plan:
        condition = str(item["condition"])
        scale = float(item["scale"])
        seed = int(item["seed"])
        scale_tag = f"{scale:.2f}".replace(".", "p")
        run_id = f"{condition.lower()}_scale{scale_tag}_seed{seed}"
        if args.resume and run_id in existing_runs:
            manifest["runs"].append(existing_runs[run_id])
            print(json.dumps({"event": "resume_existing", "run_id": run_id}), flush=True)
            continue

        derived = bool(item["derived_timing_diagnostic"])
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        episode_id = f"stage20t-{run_id}"
        if condition == "A_MODEL":
            decoded = model["decoded"]
            timestamps = scale_timestamps(original_timestamps, scale)
            timestamp_provenance = (
                str(args.original_period_provenance)
                if not derived
                else "DERIVED_TIMING_DIAGNOSTIC"
            )
            chunk = _build_model_chunk(
                decoded,
                timestamps,
                chunk_id=episode_id,
                provenance={
                    "stage": "STAGE20T",
                    "condition": "A_UNCHANGED_MODEL" if not derived else "A_MODEL_RETIMED",
                    "source_model_artifact_sha256": sha256_file(args.model_output.expanduser().resolve()),
                    "original_spatial_geometry_sha256": spatial_geometry_hash(
                        decoded[:, :3], matrix_to_quat(rot6d_rows_to_matrix(decoded[:, 3:9]))
                    ),
                    "timestamp_provenance": timestamp_provenance,
                    "time_scale": scale,
                    "derived_timing_diagnostic": derived,
                    "no_spatial_correction": True,
                },
            )
            achieved_position = model["state"][:3]
            achieved_orientation = matrix_to_quat(rot6d_rows_to_matrix(model["state"][3:9][None, :])[0])
            source_kind = "ACTUAL_GR00T_DECODED_PHYSICAL_ACTION"
        elif condition == "B_GT":
            decoded = model["desired"]
            timestamps = scale_timestamps(original_timestamps, scale)
            timestamp_provenance = (
                str(args.original_period_provenance)
                if not derived
                else "DERIVED_TIMING_DIAGNOSTIC"
            )
            chunk = _build_model_chunk(
                decoded,
                timestamps,
                chunk_id=episode_id,
                provenance={
                    "stage": "STAGE20T",
                    "condition": "B_GT_DESIRED_LABEL" if not derived else "B_GT_DESIRED_LABEL_RETIMED",
                    "source_model_artifact_sha256": sha256_file(args.model_output.expanduser().resolve()),
                    "label_role": "future_desired_eef_same_observation",
                    "original_spatial_geometry_sha256": spatial_geometry_hash(
                        decoded[:, :3], matrix_to_quat(rot6d_rows_to_matrix(decoded[:, 3:9]))
                    ),
                    "timestamp_provenance": timestamp_provenance,
                    "time_scale": scale,
                    "derived_timing_diagnostic": derived,
                    "no_spatial_correction": True,
                },
            )
            achieved_position = model["state"][:3]
            achieved_orientation = matrix_to_quat(rot6d_rows_to_matrix(model["state"][3:9][None, :])[0])
            source_kind = "STAGE20S_SAME_OBSERVATION_FUTURE_DESIRED_LABEL"
        else:
            timestamps = benchmark["timestamps"]
            chunk = _build_benchmark_chunk(
                timestamps,
                benchmark["position"],
                benchmark["quaternion"],
                chunk_id=episode_id,
                gripper=benchmark["gripper"],
                provenance={
                    "stage": "STAGE20T",
                    "condition": "C_STAGE16_KNOWN_GOOD_CONTROL",
                    "parent_trajectory_sha256": sha256_file(args.stage16_trajectory.expanduser().resolve()),
                    "timestamp_provenance": "STAGE16_NATIVE_BENCHMARK_ORIGINAL",
                    "derived_timing_diagnostic": False,
                    "no_spatial_correction": True,
                },
            )
            achieved_position = chunk.eef_position[0]
            achieved_orientation = chunk.eef_orientation[0]
            source_kind = "STAGE16_QUALIFIED_NATIVE_BENCHMARK"

        candidate_path = run_dir / "reference_candidate.npz"
        _save_candidate(candidate_path, chunk)
        interface = _interface_seal(
            chunk,
            achieved_position=achieved_position,
            achieved_orientation=achieved_orientation,
            episode_id=episode_id,
        )
        interface_path = run_dir / "stage20t_interface_seal.json"
        _write_json(interface_path, interface)
        audit = _trajectory_audit(
            chunk.eef_position,
            chunk.eef_orientation,
            chunk.point_timestamps,
            achieved_position,
            achieved_orientation,
        )
        metadata = {
            "schema": "STAGE20T_NATIVE_RUN_METADATA_V1",
            "run_id": run_id,
            "condition": condition,
            "source_kind": source_kind,
            "scale": scale,
            "seed": seed,
            "derived_timing_diagnostic": derived,
            "original_period_s": float(args.original_period_s),
            "original_period_provenance": str(args.original_period_provenance),
            "candidate_sha256": sha256_file(candidate_path),
            "spatial_geometry_sha256": audit["spatial_geometry_sha256"],
            "timestamp_sha256": audit["timestamp_sha256"],
            "interface_seal": str(interface_path),
            "safety_gate_decision": interface["safety_gate"]["decision"],
            "simulation_only": True,
            "hardware_control": False,
        }
        metadata_path = run_dir / "native_run_metadata.json"
        _write_json(metadata_path, metadata)
        gate_decision = interface["safety_gate"]["decision"]
        native: dict[str, Any]
        if gate_decision == "REJECT":
            native = {
                "execution_status": "SKIPPED_SAFETY_GATE_REJECT",
                "termination_code": "SAFETY_GATE_REJECT",
                "reference_completed": False,
                "process_returncode": None,
            }
        elif args.prepare_only:
            native = {
                "execution_status": "SKIPPED_PREPARE_ONLY",
                "termination_code": "NOT_RUN",
                "reference_completed": False,
                "process_returncode": None,
            }
        else:
            seconds = max(2.2, float(audit["duration_s"]) + float(args.seconds_margin_s))
            native = _run_native(
                native_script=native_script,
                source_root=args.source_root.expanduser().resolve(),
                arm=args.arm.expanduser().resolve(),
                dog=args.dog.expanduser().resolve(),
                candidate=candidate_path,
                output_dir=run_dir / "native_wbc",
                metadata_path=metadata_path,
                seconds=seconds,
                seed=seed,
                sim_device=args.sim_device,
                timeout_s=args.native_timeout_s,
            )
        record = {
            "run_id": run_id,
            "condition": condition,
            "source_kind": source_kind,
            "scale": scale,
            "seed": seed,
            "original_timing": bool(not derived and np.isclose(scale, 1.0)),
            "derived_timing_diagnostic": derived,
            "candidate": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "interface_seal": str(interface_path),
            "trajectory_audit": audit,
            "interface": {
                "safety_gate_decision": gate_decision,
                "safety_gate": interface["safety_gate"],
                "binding": interface["binding"],
                "preview": interface["preview"],
                "correction_applied": False,
            },
            "native": native,
        }
        manifest["runs"].append(record)
        _write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_id": run_id,
                    "gate": gate_decision,
                    "execution": native.get("execution_status"),
                    "termination": native.get("termination_code"),
                }
            ),
            flush=True,
        )

    _write_json(manifest_path, manifest)
    counts: dict[str, int] = {}
    for run in manifest["runs"]:
        status = str(run.get("native", {}).get("execution_status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"manifest": str(manifest_path), "run_count": len(manifest["runs"]), "execution_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
