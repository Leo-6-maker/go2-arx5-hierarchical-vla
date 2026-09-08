import numpy as np
import pytest

from go2arx5_vla.data import (
    CameraStreamV1,
    MaterializationConfigV1,
    RawEpisodeV1,
    RecorderConfigV1,
    RecorderV1,
    TeleopCommandV1,
    run_synchronization_stress,
)
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def _identity_pose(count: int) -> np.ndarray:
    rows = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (count, 1))
    positions = np.column_stack(
        [np.linspace(0.2, 0.3, count), np.zeros((count, 2))]
    )
    return np.concatenate([positions, rows], axis=1)


def _raw_for_stress() -> RawEpisodeV1:
    native_timestamps = np.arange(6, dtype=np.float64) * 0.02
    pose = _identity_pose(6)
    cameras = {
        name: CameraStreamV1(
            name=name,
            frame_paths=tuple(f"{name}-{index}.png" for index in range(3)),
            timestamps=np.array([0.0, 0.04, 0.08]),
            declared_fps=20.0,
            frame="sim_world",
        )
        for name in ("body", "wrist")
    }
    return RawEpisodeV1(
        episode_id="stage19_stress_fixture",
        trajectory_id="stage19_stress_trajectory",
        source="D3 simulated fixture",
        sim_level="ARM_KINEMATIC",
        task={"instruction": "move the end effector"},
        native_timestamps=native_timestamps,
        state_eef_9d=pose,
        state_gripper=np.zeros((6, 1)),
        desired_eef_9d=pose.copy(),
        desired_gripper=np.zeros((6, 1)),
        achieved_eef_9d=pose.copy(),
        achieved_gripper=np.zeros((6, 1)),
        camera_streams=cameras,
        calibration_bundle_id="D3_SIMULATION_CALIBRATION_BUNDLE_V1",
        frame_graph={"pose_frame": "sim_world", "tcp": "SIM_PROVISIONAL_TCP"},
        lifecycle_events=({"type": "episode_start"}, {"type": "episode_end"}),
        metadata={"representation": "D3"},
    )


def _decoded_action(count: int = 4) -> np.ndarray:
    action = np.zeros((count, 10), dtype=np.float64)
    action[:, :3] = [0.2, 0.0, 0.4]
    action[:, 0] += np.linspace(0.0, 0.05, count)
    action[:, 3:9] = rotation_matrix_to_rot6d_rows(
        np.repeat(np.eye(3)[None, ...], count, axis=0)
    )
    action[:, 9] = np.linspace(0.1, 0.8, count)
    return action


def test_groot_provider_requires_explicit_timestamp_provenance_and_splits_gripper():
    timestamps = np.array([4.0, 4.2, 4.7, 5.0])
    provider = GR00TTrajectoryProvider(
        _decoded_action(),
        config=GR00TProviderConfig(
            source_frame="sim_world",
            tcp_semantic_id="SIM_PROVISIONAL_TCP",
            timestamp_provenance="TEST_EXPLICIT_POINT_TIMESTAMPS",
        ),
        point_timestamps=timestamps,
    )
    chunk = provider.build_chunk(observation_timestamp=4.0)
    assert chunk.point_timestamps.tolist() == timestamps.tolist()
    assert chunk.eef_position.shape == (4, 3)
    assert chunk.gripper.shape == (4,)
    assert chunk.provenance["output_contract"] == "TrajectoryChunkV1"
    command = provider.gripper_command(chunk, timestamp=4.7)
    assert command is not None
    assert command.normalized_command == pytest.approx(0.5666666667)


def test_teleop_and_recorder_keep_desired_and_achieved_channels_separate():
    recorder = RecorderV1(
        RecorderConfigV1(
            episode_id="episode",
            source="MOCK_REPLAY",
            calibration_bundle_id="CAL_REAL_TBD",
            frame_graph={"pose_frame": "robot_world", "tcp": "arx5_tcp"},
        )
    )
    for index in range(3):
        timestamp = index * 0.02
        desired = np.concatenate([_identity_pose(3)[index], [0.5]])
        achieved = desired.copy()
        achieved[0] += 0.01
        recorder.append_state(timestamp, achieved_eef_9d=achieved[:9])
        recorder.append_command(
            TeleopCommandV1(
                command_id=f"command-{index}",
                timestamp=timestamp,
                eef_9d=desired[:9],
                gripper=desired[9:10],
                frame="robot_world",
                tcp_semantic_id="arx5_tcp",
                source="MOCK_TELEOP_LABEL",
            )
        )
        recorder.append_camera("body", timestamp, f"body-{index}.png")
    recorded = recorder.finalize()
    assert recorded.state_count == 3
    assert len(recorded.desired_commands) == 3
    assert recorded.state_samples[0]["achieved_eef_9d"][0] != recorded.desired_commands[0].eef_9d[0]
    assert recorded.as_dict()["actuation_backend"] == "NONE_READ_ONLY_RECORDING_ONLY"
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.append_camera("body", 0.04, "body-duplicate-time.png")


def test_synchronization_stress_preserves_source_and_rejects_malformed_cases():
    raw = _raw_for_stress()
    results = run_synchronization_stress(raw)
    assert len(results) == 1
    envelope = results[0]
    assert envelope["source_unchanged"] is True
    cases = {row["case"]: row for row in envelope["case_results"]}
    assert len(cases) == 8
    assert cases["camera_jitter"]["synchronizer_error"] is None
    assert cases["bounded_clock_offset"]["synchronizer_error"] is None
    assert cases["duplicated_wrist_timestamp"]["quality_decision"] == "REJECT"
    assert cases["non_monotonic_wrist_timestamp"]["quality_decision"] == "REJECT"
    assert cases["body_wrist_async_reject"]["quality_decision"] == "REJECT"


def test_materialization_config_marks_fixture_horizon_and_production_timing_explicitly(tmp_path):
    config = MaterializationConfigV1(output_dir=tmp_path / "fixture", horizon=40)
    payload = config.as_dict()
    assert payload["horizon_provenance"] == "TEST_ONLY_D3_FIXTURE"
    assert payload["production_action_timing"] == "TBD_BY_ADAPTATION_DATASET"
