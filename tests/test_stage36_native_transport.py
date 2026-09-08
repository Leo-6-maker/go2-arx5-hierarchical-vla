from __future__ import annotations

import numpy as np

from go2arx5_vla.trajectory.native_transport import (
    NATIVE_DT_S,
    NATIVE_DURATION_S,
    NATIVE_SAMPLE_COUNT,
    materialize_timestamped_pose_to_native,
)
from go2arx5_vla.trajectory.so3 import quat_to_matrix


def _z_quaternion(angle: float) -> np.ndarray:
    return np.asarray(
        [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)], dtype=np.float64
    )


def test_nonuniform_source_is_materialized_in_source_time() -> None:
    source_timestamps = np.asarray([0.0, 0.25, 1.25, 2.0], dtype=np.float64)
    source_position = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    source_quaternion = np.asarray(
        [_z_quaternion(0.0), _z_quaternion(np.pi / 2.0), _z_quaternion(np.pi / 2.0), _z_quaternion(np.pi)],
        dtype=np.float64,
    )
    original_timestamps = source_timestamps.copy()
    original_position = source_position.copy()
    original_quaternion = source_quaternion.copy()

    result = materialize_timestamped_pose_to_native(
        source_timestamps, source_position, source_quaternion
    )

    assert result.sample_count == NATIVE_SAMPLE_COUNT
    assert result.timestamps[0] == 0.0
    assert result.timestamps[-1] == NATIVE_DURATION_S
    assert np.allclose(np.diff(result.timestamps), NATIVE_DT_S, rtol=0.0, atol=1e-12)
    assert np.array_equal(result.position[0], source_position[0])
    assert np.array_equal(result.position[-1], source_position[-1])
    assert np.array_equal(result.quaternion[0], source_quaternion[0])
    assert np.array_equal(result.quaternion[-1], source_quaternion[-1])
    assert np.allclose(result.position[30, 0], 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(result.position[90, 0], 5.0 / 3.0, rtol=0.0, atol=1e-12)
    assert np.isfinite(result.position).all()
    assert np.isfinite(result.quaternion).all()
    assert np.allclose(np.linalg.norm(result.quaternion, axis=-1), 1.0, atol=1e-12)
    assert np.allclose(
        quat_to_matrix(result.quaternion[90]),
        quat_to_matrix(_z_quaternion(2.0 * np.pi / 3.0)),
        atol=1e-12,
    )
    assert np.array_equal(source_timestamps, original_timestamps)
    assert np.array_equal(source_position, original_position)
    assert np.array_equal(source_quaternion, original_quaternion)


def test_transport_rejects_noncanonical_duration() -> None:
    timestamps = np.asarray([0.0, 1.0, 2.1], dtype=np.float64)
    position = np.zeros((3, 3), dtype=np.float64)
    quaternion = np.repeat(_z_quaternion(0.0)[None, :], 3, axis=0)

    try:
        materialize_timestamped_pose_to_native(timestamps, position, quaternion)
    except ValueError as exc:
        assert "end" in str(exc)
    else:
        raise AssertionError("transport accepted a source trajectory with the wrong duration")
