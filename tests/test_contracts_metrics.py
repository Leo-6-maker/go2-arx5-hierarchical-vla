import numpy as np
import pytest

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.metrics import (
    CONTROLLER_TRACKING_ERROR,
    END_TO_END_TASK_SPACE_ERROR,
    REFERENCE_RECONSTRUCTION_ERROR,
    controller_tracking_error,
    end_to_end_task_space_error,
    reference_reconstruction_error,
    task_space_error,
)
from go2arx5_vla.providers import BenchmarkTrajectoryProvider
from go2arx5_vla.trajectory.so3 import matrix_to_quat


def make_chunk():
    return TrajectoryChunkV1(
        chunk_id="c0",
        source="test",
        observation_timestamp=0.0,
        point_timestamps=np.array([0.0, 0.1]),
        eef_position=np.zeros((2, 3)),
        eef_orientation=matrix_to_quat(np.repeat(np.eye(3)[None], 2, axis=0)),
        source_frame="world",
        tcp_semantic_id="tcp",
        validity_start=0.0,
        validity_end=0.1,
    )


def test_chunk_validation_and_provider_boundary():
    chunk = make_chunk()
    assert chunk.point_count == 2
    assert BenchmarkTrajectoryProvider([chunk]).next_chunk(0.0) is chunk
    with pytest.raises(ValueError, match="position"):
        TrajectoryChunkV1(
            chunk_id="bad",
            source="test",
            observation_timestamp=0.0,
            point_timestamps=np.array([0.0, 0.1]),
            eef_position=np.zeros((3, 3)),
            eef_orientation=np.repeat(np.asarray([[0, 0, 0, 1]], dtype=float), 2, axis=0),
            source_frame="world",
            tcp_semantic_id="tcp",
            validity_start=0.0,
            validity_end=0.1,
        )


def test_error_names_remain_separate():
    eye = np.repeat(np.eye(3)[None], 3, axis=0)
    p = np.zeros((3, 3))
    q = p.copy()
    q[:, 0] = 0.1
    for name in (REFERENCE_RECONSTRUCTION_ERROR, CONTROLLER_TRACKING_ERROR, END_TO_END_TASK_SPACE_ERROR):
        result = task_space_error(p, q, eye, eye, metric_name=name)
        assert result["metric_name"] == name
        assert result["position_m"]["mean"] == pytest.approx(0.1)


def test_named_metric_apis_accept_quaternions_and_preserve_boundaries():
    p0 = np.zeros((2, 3))
    p1 = np.ones((2, 3)) * 0.01
    q = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    assert reference_reconstruction_error(p0, p1, q, q)["metric_name"] == REFERENCE_RECONSTRUCTION_ERROR
    assert controller_tracking_error(p0, p1, q, q)["metric_name"] == CONTROLLER_TRACKING_ERROR
    assert end_to_end_task_space_error(p0, p1, q, q)["metric_name"] == END_TO_END_TASK_SPACE_ERROR
