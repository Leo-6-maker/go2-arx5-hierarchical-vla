"""Geometric and angular identities behind the registered dynamic admission margins."""

import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from dynamic_margin import quat_error_deg, sat_lower_bound


def test_separating_lower_bound_contact_and_diagonal():
    coords = np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [3.0, 3, 0]])
    axes = np.repeat(np.eye(3)[None], len(coords), axis=0)
    got = sat_lower_bound(coords, axes, np.ones(3), np.zeros(3), np.eye(3), np.ones(3))
    assert np.allclose(got, [0, 0, 1, 1])
    assert got[-1] < np.sqrt(2)


def test_rotated_boxes_world_frame_invariance():
    a = Rotation.from_rotvec([0.3, -0.1, 0.2]).as_matrix()
    b = Rotation.from_rotvec([-0.1, 0.3, 0.1]).as_matrix()
    frame = Rotation.from_rotvec([0.2, 0.3, -0.4]).as_matrix()
    pts = np.array([[3.0, 1, 2], [0.6, 1, 1]])
    rots = np.broadcast_to(a, (len(pts), 3, 3))
    other = np.array([0.0, 0.3, 0.4])

    def evaluate(p, r, o, s):
        return sat_lower_bound(p, r, [1, 0.7, 0.8], o, s, [0.5, 0.5, 0.3])

    np.testing.assert_allclose(
        evaluate(pts, rots, other, b),
        evaluate(pts @ frame.T, frame @ rots, frame @ other, frame @ b),
    )


def test_quaternion_geodesic_sign_and_small_rotation():
    q = Rotation.from_rotvec([0.2, 0.7, -0.8]).as_quat()
    tiny = (
        Rotation.from_rotvec([0, np.deg2rad(0.002), 0]) * Rotation.from_quat(q)
    ).as_quat()
    assert quat_error_deg(q, -q) == pytest.approx(0, abs=1e-14)
    assert quat_error_deg(q, -tiny) == pytest.approx(0.002, abs=1e-10)
    with pytest.raises(ValueError):
        quat_error_deg(q, q * 2)
