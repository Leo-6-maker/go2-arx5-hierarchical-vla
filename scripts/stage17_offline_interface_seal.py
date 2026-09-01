#!/usr/bin/env python3
"""Seal one decoded actual-GR00T output at the WBC preview boundary.

The script intentionally stops before any simulator/controller call.  A gate
REJECT is a valid descriptive result for an unadapted base-model trajectory;
the script records it without modifying the trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def descriptors(chunk) -> dict[str, float | list[float]]:
    positions = np.asarray(chunk.eef_position, dtype=np.float64)
    q = np.asarray(chunk.eef_orientation, dtype=np.float64)
    from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix

    matrices = quat_to_matrix(q)
    dp = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    dtheta = geodesic_angle(matrices[:-1], matrices[1:])
    dt = np.diff(chunk.point_timestamps)
    translation_speed = dp / dt
    angular_speed = dtheta / dt
    path_length = float(dp.sum())
    return {
        "point_count": int(chunk.point_count),
        "duration_s": float(chunk.point_timestamps[-1] - chunk.point_timestamps[0]),
        "translation_path_length_m": path_length,
        "translation_speed_mps_max": float(np.max(translation_speed)) if len(translation_speed) else 0.0,
        "translation_speed_mps_mean": float(np.mean(translation_speed)) if len(translation_speed) else 0.0,
        "angular_speed_radps_max": float(np.max(angular_speed)) if len(angular_speed) else 0.0,
        "angular_speed_radps_mean": float(np.mean(angular_speed)) if len(angular_speed) else 0.0,
        "orientation_range_rad": float(np.sum(dtheta)),
        "workspace_min_m": positions.min(axis=0).tolist(),
        "workspace_max_m": positions.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    actual_manifest = json.loads(args.manifest.read_text())
    artifact = Path(actual_manifest["output_artifact"])
    arrays = np.load(artifact, allow_pickle=False)
    decoded_raw = np.asarray(arrays["decoded_physical_action"])
    decoded = np.asarray(decoded_raw, dtype=np.float64)
    state = np.asarray(arrays["input_state"], dtype=np.float64)
    observation_timestamp = 1000.0
    period = float(actual_manifest["action_sample_period_s"])
    source_frame = "STAGE10_SYNTHETIC_EEF_FRAME"
    tcp_id = "STAGE10_SYNTHETIC_TCP_UNRESOLVED"
    provider = GR00TTrajectoryProvider(
        decoded,
        config=GR00TProviderConfig(
            source_frame=source_frame,
            tcp_semantic_id=tcp_id,
            action_sample_period_s=period,
            timestamp_provenance=str(actual_manifest["action_sample_period_provenance"]),
            source="GR00T_ACTUAL_BASE_CHECKPOINT",
        ),
        chunk_id="stage17-actual-groot-episode0",
        observation_timestamp=observation_timestamp,
        provenance={
            "actual_output_manifest": str(args.manifest),
            "raw_universal_sha256": actual_manifest["raw_universal_sha256"],
            "decoded_physical_sha256": actual_manifest["decoded_physical_sha256"],
        },
    )
    chunk = provider.next_chunk(observation_timestamp)
    if chunk is None:
        raise RuntimeError("provider did not emit its one chunk")
    achieved_q = matrix_to_quat(rot6d_rows_to_matrix(state[3:9]))
    gate = TrajectorySafetyGateV1().evaluate(
        chunk,
        now=observation_timestamp,
        achieved_position=state[:3],
        achieved_orientation=achieved_q,
    )
    seal: dict = {
        "schema": "STAGE17_ACTUAL_MODEL_INTERFACE_SEAL_V1",
        "actual_model_manifest": str(args.manifest),
        "raw_universal_shape": actual_manifest["output_shapes"]["raw_universal"],
        "decoded_physical_shape": actual_manifest["output_shapes"]["decoded_physical"],
        "decoded_physical_sha256": sha256_array(decoded_raw),
        "decoded_physical_sha256_matches_inference": sha256_array(decoded_raw) == actual_manifest["decoded_physical_sha256"],
        "provider": {
            "class": "GR00TTrajectoryProvider",
            "output_contract": "TrajectoryChunkV1",
            "chunk_id": chunk.chunk_id,
            "point_count": chunk.point_count,
            "position_shape": list(chunk.eef_position.shape),
            "orientation_shape": list(chunk.eef_orientation.shape),
            "gripper_shape": None if chunk.gripper is None else list(np.asarray(chunk.gripper).shape),
            "point_timestamp_shape": list(chunk.point_timestamps.shape),
            "source_frame": chunk.source_frame,
            "tcp_semantic_id": chunk.tcp_semantic_id,
            "eef_rot6d_layout": "input GR00T_FIRST_TWO_ROWS -> canonical quaternion/FIRST_TWO_COLUMNS",
            "timestamp_provenance": chunk.provenance["timestamp_provenance"],
        },
        "trajectory_descriptors": descriptors(chunk),
        "safety_gate": gate.as_dict(),
        "wbc_executed_from_actual_model": False,
        "stop_boundary": "WBC_PREVIEW_ONLY",
    }
    if gate.decision.value in {"ACCEPT", "WARN"}:
        binding = EpisodeReferenceBindingV1.from_external_chunk(
            chunk,
            episode_id="stage17-actual-groot-offline",
        )
        bound = binding.bind_external_chunk(chunk)
        adapter = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=source_frame,
                wbc_frame=source_frame,
                tcp_semantic_id=tcp_id,
                stale_timeout_s=0.75,
            )
        )
        preview, gripper = adapter.adapt(bound, controller_timestamp=observation_timestamp)
        if not np.isfinite(preview.xyz_rot6d).all():
            raise RuntimeError("offline WBC preview contains non-finite values")
        seal["binding"] = {
            "class": "EpisodeReferenceBindingV1",
            "episode_id": binding.episode_id,
            "semantics": binding.binding_semantics,
            "bound_chunk_source": bound.source,
            "pose_preservation_max_position_error_m": float(np.max(np.linalg.norm(bound.eef_position - chunk.eef_position, axis=-1))),
        }
        seal["wbc_preview"] = {
            "shape": list(preview.xyz_rot6d.shape),
            "position_shape": list(preview.eef_position.shape),
            "rot6d_shape": list(preview.eef_rot6d.shape),
            "finite": bool(np.isfinite(preview.xyz_rot6d).all()),
            "gripper_parallel_channel": None if gripper is None else gripper.normalized_command,
        }
        seal["result"] = "ACTUAL_GR00T_INTERFACE_PASS"
    else:
        seal["binding"] = {"skipped": True, "reason": "trajectory rejected by safety gate"}
        seal["wbc_preview"] = {"skipped": True, "reason": "trajectory rejected by safety gate"}
        seal["result"] = "ACTUAL_GR00T_TRAJECTORY_REJECTED_BY_GATE"

    output = args.output or args.manifest.with_name("actual_model_interface_seal.json")
    output.write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()
