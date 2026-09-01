#!/usr/bin/env python3
"""Audit exact same-scene expert labels without invoking a model or WBC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.data.stage22_expert import load_snapshot
from go2arx5_vla.metrics import chunk_profile
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix, rot6d_rows_to_matrix


def _stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "p95": float(np.percentile(array, 95.0)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
    }


def _chunk(snapshot: Any) -> TrajectoryChunkV1:
    return TrajectoryChunkV1(
        chunk_id=f"expert-label:{snapshot.example_id}",
        source="STAGE22_SAME_SCENE_EXPERT_LABEL",
        observation_timestamp=float(snapshot.observation_timestamp),
        point_timestamps=snapshot.future_timestamps,
        eef_position=snapshot.future_desired_eef_9d[:, :3],
        eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(snapshot.future_desired_eef_9d[:, 3:9])),
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        validity_start=float(snapshot.future_timestamps[0]),
        validity_end=float(snapshot.future_timestamps[-1]),
        gripper=snapshot.future_gripper.reshape(-1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots = [
        load_snapshot(path)
        for path in sorted(args.input_dir.glob("snapshots/*/expert-*.json"))
    ]
    if not snapshots:
        parser.error(f"no snapshots found under {args.input_dir}")
    rows = []
    initial_position_errors = []
    initial_rotation_errors = []
    for snapshot in snapshots:
        state = snapshot.state_v1.as_array()
        chunk = _chunk(snapshot)
        state_q = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
        profile = chunk_profile(
            chunk,
            achieved_position=state[:3],
            achieved_quaternion=state_q,
            provenance={"profile_role": "exact_same_scene_expert_label"},
        )
        initial_position_errors.append(profile["initial_position_discontinuity_m"])
        initial_rotation_errors.append(profile["initial_rotation_discontinuity_rad"])
        rows.append(
            {
                "example_id": snapshot.example_id,
                "target_id": snapshot.target_id,
                "raw_episode_id": snapshot.raw_episode_id,
                "observation_timestamp_s": snapshot.observation_timestamp,
                "profile": profile,
                "first_label_state_position_error_m": float(np.linalg.norm(chunk.eef_position[0] - state[:3])),
                "first_label_state_rotation_error_rad": float(
                    geodesic_angle(quat_to_matrix(chunk.eef_orientation[0]), quat_to_matrix(state_q))
                ),
                "target_excluded_from_model_input": snapshot.target_excluded_from_model_input,
            }
        )
    payload = {
        "schema": "STAGE22_EXPERT_LABEL_AUDIT_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "snapshot_count": len(snapshots),
        "episode_count": len({item.raw_episode_id for item in snapshots}),
        "target_count": len({item.target_id for item in snapshots}),
        "target_excluded_from_model_input": all(item.target_excluded_from_model_input for item in snapshots),
        "exact_same_scene_observation": True,
        "initial_position_discontinuity_m": _stats(initial_position_errors),
        "initial_rotation_discontinuity_rad": _stats(initial_rotation_errors),
        "expert_label_path_length_m": _stats([row["profile"]["path_length_m"] for row in rows]),
        "expert_max_linear_speed_mps": _stats(
            [row["profile"]["linear_speed_mps"]["max"] for row in rows]
        ),
        "expert_mean_linear_speed_mps": _stats(
            [row["profile"]["linear_speed_mps"]["mean"] for row in rows]
        ),
        "rows": rows,
        "claim_boundary": "expert_oracle_and_label_integrity; no model task-performance claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
