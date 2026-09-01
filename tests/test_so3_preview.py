import numpy as np

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.trajectory.preview import DEFAULT_WBC_PREVIEW_OFFSETS_M, se3_arc_length_preview
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d,
)


def z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_chunk(position, rotations=None, times=None, observation=0.0):
    position = np.asarray(position, dtype=float)
    if times is None:
        times = np.linspace(0.0, 1.0, len(position))
    if rotations is None:
        rotations = np.repeat(np.eye(3)[None], len(position), axis=0)
    return TrajectoryChunkV1(
        chunk_id="test",
        source="unit_test",
        observation_timestamp=observation,
        point_timestamps=times,
        eef_position=position,
        eef_orientation=matrix_to_quat(np.asarray(rotations)),
        source_frame="test_world",
        tcp_semantic_id="test_tcp",
        validity_start=float(times[0]),
        validity_end=float(times[-1]),
    )


def test_rot6d_round_trip_uses_first_two_columns():
    rotations = np.asarray([z_rotation(0.2), z_rotation(-1.1)])
    decoded = rot6d_to_matrix(rotation_matrix_to_rot6d(rotations))
    assert np.max(np.abs(decoded - rotations)) < 1e-12
    assert np.max(geodesic_angle(decoded, rotations)) < 1e-12


def test_straight_arc_length_preview_and_endpoint_clamp():
    chunk = make_chunk([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    preview = se3_arc_length_preview(chunk)
    assert np.allclose(preview.eef_position[:, 0], DEFAULT_WBC_PREVIEW_OFFSETS_M)
    assert not preview.endpoint_clamped[0]
    assert preview.endpoint_clamped[-1]
    assert preview.xyz_rot6d.shape == (9, 9)


def test_stationary_path_holds_pose():
    chunk = make_chunk(np.repeat([[0.2, -0.1, 0.4]], 4, axis=0))
    preview = se3_arc_length_preview(chunk)
    assert np.allclose(preview.eef_position, [0.2, -0.1, 0.4])
    assert preview.endpoint_clamped.all()


def test_rotation_only_path_has_se3_progress():
    rotations = np.asarray([z_rotation(0.0), z_rotation(np.pi / 2), z_rotation(np.pi)])
    chunk = make_chunk(np.zeros((3, 3)), rotations=rotations)
    preview = se3_arc_length_preview(chunk)
    preview_rot = rot6d_rows_to_matrix(preview.eef_rot6d)
    angles = geodesic_angle(np.eye(3), preview_rot)
    assert np.all(np.isfinite(preview.xyz_rot6d))
    assert angles[-1] > angles[0]


def test_short_horizon_is_finite_and_clamped():
    chunk = make_chunk([[0, 0, 0], [0.01, 0, 0]])
    preview = se3_arc_length_preview(chunk)
    assert np.isfinite(preview.xyz_rot6d).all()
    assert preview.endpoint_clamped[-1]


def test_high_curvature_keeps_valid_rotations():
    t = np.linspace(0.0, 1.0, 21)
    p = np.c_[0.25 * t, 0.08 * np.sin(8 * np.pi * t), 0.02 * np.cos(6 * np.pi * t)]
    rotations = np.asarray([z_rotation(1.2 * np.sin(6 * np.pi * x)) for x in t])
    preview = se3_arc_length_preview(make_chunk(p, rotations=rotations))
    assert np.isfinite(preview.xyz_rot6d).all()
    assert np.allclose(np.linalg.det(rot6d_rows_to_matrix(preview.eef_rot6d)), 1.0, atol=1e-10)
