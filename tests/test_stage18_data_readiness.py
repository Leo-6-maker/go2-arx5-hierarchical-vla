import numpy as np
import pytest

from go2arx5_vla.data import (
    CalibrationBundleV1,
    CameraStreamV1,
    EpisodeGateDecision,
    EpisodeQualityGateConfig,
    EpisodeQualityGateV1,
    GR00TTrainingExampleV1,
    RawEpisodeV1,
    StateV1,
    SynchronizerV1,
    interpolate_pose_rot6d_rows,
)
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


def _raw(*, camera_fps=20.0, include_wrist=True):
    timestamps = np.arange(6, dtype=np.float64) * 0.02
    identity_rows = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (6, 1))
    positions = np.column_stack([np.linspace(0.2, 0.25, 6), np.zeros((6, 2))])
    pose = np.concatenate([positions, identity_rows], axis=1)
    cameras = {
        "body": CameraStreamV1(
            name="body",
            frame_paths=tuple(f"body-{i}.png" for i in range(3)),
            timestamps=np.array([0.0, 0.04, 0.08]),
            declared_fps=camera_fps,
            frame="sim_world",
        )
    }
    if include_wrist:
        cameras["wrist"] = CameraStreamV1(
            name="wrist",
            frame_paths=tuple(f"wrist-{i}.png" for i in range(3)),
            timestamps=np.array([0.0, 0.04, 0.08]),
            declared_fps=camera_fps,
            frame="sim_world",
        )
    return RawEpisodeV1(
        episode_id="d3_fixture_episode",
        trajectory_id="trajectory",
        source="D3 simulated fixture",
        sim_level="ARM_KINEMATIC",
        task={"instruction": "move the end effector"},
        native_timestamps=timestamps,
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


def test_raw_episode_is_native_rate_and_arrays_are_read_only():
    raw = _raw()
    assert raw.frame_count == 6
    assert raw.state_eef_9d.shape == (6, 9)
    assert raw.state_eef_9d.flags.writeable is False
    with pytest.raises(ValueError):
        raw.state_eef_9d[0, 0] = 99.0
    assert raw.desired_eef_9d is not raw.achieved_eef_9d


def test_state_v1_is_exactly_the_primary_10d_abi():
    state = StateV1.from_array(_raw().state_eef_9d[0].tolist() + [0.25])
    assert state.as_array().shape == (10,)
    assert state.as_array()[-1] == pytest.approx(0.25)
    assert state.rot6d_layout == "GR00T_FIRST_TWO_ROWS"


def test_synchronizer_preserves_timestamps_and_surfaces_rate_mismatch():
    result = SynchronizerV1().synchronize(_raw())
    assert result.sample_timestamps.tolist() == [0.0, 0.04, 0.08]
    assert result.effective_rates_hz["body"] == pytest.approx(25.0)
    assert result.declared_rates_hz["body"] == pytest.approx(20.0)
    assert "DECLARED_EFFECTIVE_RATE_MISMATCH" in result.warnings
    assert result.metrics["max_abs_skew_s"] == pytest.approx(0.0)
    assert result.metrics["command_state_skew_s"]["max"] == pytest.approx(0.0)
    assert result.provenance["production_action_rate"] == "TBD_BY_ADAPTATION_DATASET"


def test_synchronizer_does_not_assume_20hz_when_timestamps_are_explicit():
    raw = _raw(camera_fps=None)
    result = SynchronizerV1().synchronize(raw, sample_timestamps=[0.01, 0.03, 0.07])
    assert result.sample_timestamps.tolist() == [0.01, 0.03, 0.07]
    # 0.03 is exactly between the 50 Hz samples; the synchronizer's
    # deterministic tie rule chooses the earlier source sample.
    assert result.indices["state"].tolist() == [0, 1, 4]


def test_rotation_interpolation_decodes_and_reencodes_so3():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    matrices = quat_to_matrix(np.stack([q0, q1]))
    pose = np.concatenate([np.zeros((2, 3)), rotation_matrix_to_rot6d_rows(matrices)], axis=1)
    midpoint = interpolate_pose_rot6d_rows([0.0, 1.0], pose, [0.5])
    decoded = quat_to_matrix(matrix_to_quat(rot6d_rows_to_matrix(midpoint[:, 3:])))
    assert decoded.shape == (1, 3, 3)
    assert np.linalg.det(decoded[0]) == pytest.approx(1.0)
    assert geodesic_angle(decoded[0], matrices[0]) == pytest.approx(np.pi / 4.0, abs=1e-5)


def test_training_example_uses_desired_future_labels_and_explicit_timestamps():
    raw = _raw()
    sync = SynchronizerV1().synchronize(raw)
    example = GR00TTrainingExampleV1.from_raw(raw, sync, horizon=4)
    assert example.state_v1.as_array().shape == (10,)
    assert example.future_desired_eef_9d.shape == (4, 9)
    assert example.future_gripper.shape == (4, 1)
    assert example.future_timestamps.tolist() == [0.0, 0.02, 0.04, 0.06]
    record = example.to_lerobot_record()
    assert len(record["action"]) == 10
    assert record["metadata"]["future_label"] == "desired"
    assert record["metadata"]["achieved_stream_in_raw_episode"] is True


def test_calibration_bundle_keeps_simulation_and_real_tbd_separate():
    bundle = CalibrationBundleV1(
        bundle_id="D3_SIMULATION_CALIBRATION_BUNDLE_V1",
        version="D3_V0",
        status="SIMULATION_ONLY",
    )
    assert bundle.real_robot_calibration is False
    assert bundle.T_go2_arx5 is None
    assert bundle.as_dict()["provenance"] == {}


def test_quality_gate_warns_on_d3_declared_rate_mismatch_without_repair():
    raw = _raw()
    sync = SynchronizerV1().synchronize(raw)
    result = EpisodeQualityGateV1().evaluate(
        raw,
        synchronization=sync,
        calibration=CalibrationBundleV1(
            bundle_id=raw.calibration_bundle_id,
            version="D3_V0",
            status="SIMULATION_ONLY",
        ),
    )
    assert result.decision is EpisodeGateDecision.WARN
    assert any(issue.code == "D3_CAMERA_FPS_METADATA_MISMATCH" for issue in result.issues)
    assert result.provenance["corrective_action"] == "NONE"


def test_quality_gate_does_not_allow_simulation_bundle_for_real_gate():
    raw = _raw()
    result = EpisodeQualityGateV1(EpisodeQualityGateConfig(simulation_fixture_allowed=False)).evaluate(
        raw,
        synchronization=SynchronizerV1().synchronize(raw),
        calibration=CalibrationBundleV1(
            bundle_id=raw.calibration_bundle_id,
            version="D3_V0",
            status="SIMULATION_ONLY",
        ),
    )
    assert result.decision is EpisodeGateDecision.REJECT
    assert any(issue.code == "SIMULATION_CALIBRATION_NOT_ALLOWED" for issue in result.issues)


def test_quality_gate_rejects_missing_required_camera_stream():
    raw = _raw(include_wrist=False)
    result = EpisodeQualityGateV1().evaluate(raw, synchronization=SynchronizerV1().synchronize(raw))
    assert result.decision is EpisodeGateDecision.REJECT
    assert any(issue.code == "REQUIRED_STREAM_MISSING" for issue in result.issues)


def test_quality_gate_rejects_nonfinite_data_at_boundary():
    raw = _raw()
    object.__setattr__(raw, "desired_eef_9d", np.full((6, 9), np.nan))
    result = EpisodeQualityGateV1().evaluate(raw, synchronization=SynchronizerV1().synchronize(_raw()))
    assert result.decision is EpisodeGateDecision.REJECT
    assert any(issue.code == "NON_FINITE_STATE_OR_LABEL" for issue in result.issues)
