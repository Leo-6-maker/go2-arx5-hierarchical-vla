from __future__ import annotations

import numpy as np
import pytest

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.data import (
    CalibrationBundleV1,
    SameSceneExpertSnapshotV1,
    StateV1,
    snapshot_to_raw_episode,
    snapshot_to_training_example,
)
from go2arx5_vla.metrics import chunk_overlap_metrics, chunk_profile
from go2arx5_vla.simulation import SameSceneExpertConfigV1, build_same_scene_expert_chunk
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def _state(position=(0.0, 0.0, 0.5)) -> StateV1:
    return StateV1(
        eef_xyz=np.asarray(position, dtype=np.float64),
        eef_rot6d_rows=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        gripper=np.asarray([0.5]),
    )


def _snapshot() -> SameSceneExpertSnapshotV1:
    chunk = build_same_scene_expert_chunk(
        chunk_id="expert-0000",
        observation_timestamp=2.0,
        current_position=[0.0, 0.0, 0.5],
        target_position=[0.05, -0.02, 0.55],
        current_quaternion=[0.0, 0.0, 0.0, 1.0],
        config=SameSceneExpertConfigV1(),
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
    )
    return SameSceneExpertSnapshotV1(
        example_id="episode-0:expert-0000",
        raw_episode_id="episode-0",
        simulator_instance_id="native-test-1",
        observation_timestamp=2.0,
        body_frame_path="/tmp/body.ppm",
        wrist_frame_path="/tmp/wrist.ppm",
        state_v1=_state(),
        future_timestamps=chunk.point_timestamps,
        future_desired_eef_9d=np.concatenate(
            [chunk.eef_position, np.broadcast_to(np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (40, 6))], axis=1
        ),
        future_gripper=np.full((40, 1), 0.5),
        target_position_world_m=[0.3, 0.1, 0.58],
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
        calibration_bundle_id="stage22-calibration",
        target_id="target-a",
    )


def test_expert_chunk_is_smooth_and_anchored() -> None:
    chunk = build_same_scene_expert_chunk(
        chunk_id="chunk-a",
        observation_timestamp=0.0,
        current_position=[0.0, 0.0, 0.5],
        target_position=[0.1, 0.0, 0.5],
        current_quaternion=[0.0, 0.0, 0.0, 1.0],
        source_frame="frame",
        tcp_semantic_id="tcp",
    )
    assert chunk.point_count == 40
    assert np.allclose(chunk.eef_position[0], [0.0, 0.0, 0.5])
    assert np.allclose(chunk.eef_position[-1], [0.1, 0.0, 0.5])
    assert np.allclose(chunk.eef_orientation, [0.0, 0.0, 0.0, 1.0])
    profile = chunk_profile(chunk, achieved_position=[0.0, 0.0, 0.5], achieved_quaternion=[0, 0, 0, 1])
    assert profile["initial_position_discontinuity_m"] == pytest.approx(0.0)
    assert profile["path_length_m"] == pytest.approx(0.1, abs=1e-8)


def test_overlap_metrics_compare_common_future_without_correction() -> None:
    first = build_same_scene_expert_chunk(
        chunk_id="first",
        observation_timestamp=0.0,
        current_position=[0.0, 0.0, 0.5],
        target_position=[0.1, 0.0, 0.5],
        current_quaternion=[0, 0, 0, 1],
        source_frame="frame",
        tcp_semantic_id="tcp",
    )
    second = build_same_scene_expert_chunk(
        chunk_id="second",
        observation_timestamp=0.2,
        current_position=[0.02, 0.0, 0.5],
        target_position=[0.1, 0.0, 0.5],
        current_quaternion=[0, 0, 0, 1],
        source_frame="frame",
        tcp_semantic_id="tcp",
    )
    result = chunk_overlap_metrics(first, second)
    assert result["overlap_available"] is True
    assert result["overlap_point_count"] >= 2
    assert result["no_trajectory_correction"] is True
    assert result["position_error_m"]["max"] is not None


def test_snapshot_keeps_target_out_of_model_record_and_expands_raw_window() -> None:
    snapshot = _snapshot()
    calibration = CalibrationBundleV1(
        bundle_id="stage22-calibration",
        version="sim",
        status="SIMULATION_ONLY",
        real_robot_calibration=False,
    )
    raw = snapshot_to_raw_episode(snapshot, calibration=calibration)
    example = snapshot_to_training_example(snapshot)
    record = example.to_lerobot_record()
    assert raw.frame_count == 40
    assert raw.metadata["snapshot_expanded_horizon"] is True
    assert raw.metadata["target_excluded_from_model_input"] is True
    assert np.asarray(record["observation.state"]).shape == (10,)
    assert np.asarray(record["future_action"]).shape == (40, 10)
    assert "target_position_world_m" not in record


def test_snapshot_rejects_non_monotonic_future_timestamps() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="strictly increasing"):
        SameSceneExpertSnapshotV1(
            **{
                **snapshot.__dict__,
                "future_timestamps": np.asarray(snapshot.future_timestamps)[::-1],
            }
        )
