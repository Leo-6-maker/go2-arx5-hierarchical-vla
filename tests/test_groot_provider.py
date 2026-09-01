import numpy as np
import pytest

from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def _decoded_action(points=4):
    action = np.zeros((points, 10), dtype=np.float64)
    action[:, 0] = np.linspace(0.2, 0.3, points)
    action[:, 1] = -0.1
    action[:, 2] = 0.4
    action[:, 3:9] = rotation_matrix_to_rot6d_rows(np.repeat(np.eye(3)[None], points, axis=0))
    action[:, 9] = np.linspace(0.1, 0.8, points)
    return action


def _config(period=0.1):
    return GR00TProviderConfig(
        source_frame="test_frame",
        tcp_semantic_id="test_tcp",
        action_sample_period_s=period,
        timestamp_provenance="TEST_ONLY",
    )


def test_provider_splits_40x10_physical_output_and_uses_explicit_period():
    provider = GR00TTrajectoryProvider(
        _decoded_action(),
        config=_config(),
        chunk_id="g0",
    )
    chunk = provider.next_chunk(10.0)
    assert chunk is not None
    assert chunk.point_timestamps.tolist() == [10.0, 10.1, 10.2, 10.3]
    assert chunk.eef_position.shape == (4, 3)
    assert chunk.eef_orientation.shape == (4, 4)
    assert chunk.gripper.shape == (4,)
    assert np.allclose(chunk.eef_orientation[:, 3], 1.0)
    assert chunk.provenance["eef_slice"] == [0, 9]
    assert chunk.provenance["gripper_slice"] == [9, 10]
    assert provider.next_chunk(10.0) is None
    gripper = provider.gripper_command(chunk, timestamp=10.2)
    assert gripper is not None
    assert gripper.normalized_command == pytest.approx(0.5666666667)


def test_provider_accepts_batched_model_output_with_explicit_timestamps():
    timestamps = np.array([4.0, 4.2, 4.7, 5.0])
    provider = GR00TTrajectoryProvider(
        _decoded_action()[None, ...],
        config=GR00TProviderConfig(
            source_frame="test_frame",
            tcp_semantic_id="test_tcp",
            timestamp_provenance="DATASET_EXPLICIT",
        ),
        point_timestamps=timestamps,
        batch_index=0,
    )
    chunk = provider.build_chunk(observation_timestamp=4.0)
    assert np.allclose(chunk.point_timestamps, timestamps)
    assert chunk.point_count == 4


def test_provider_rejects_implicit_production_timing_and_bad_timestamps():
    with pytest.raises(ValueError, match="point_timestamps are required"):
        GR00TTrajectoryProvider(
            _decoded_action(),
            config=GR00TProviderConfig(source_frame="f", tcp_semantic_id="t"),
            observation_timestamp=0.0,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        GR00TTrajectoryProvider(
            _decoded_action(),
            config=GR00TProviderConfig(source_frame="f", tcp_semantic_id="t", timestamp_provenance="TEST_ONLY"),
            point_timestamps=[0.0, 0.1, 0.1, 0.2],
        )
