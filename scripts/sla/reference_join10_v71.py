"""Experimental offline reference connection; never applied to the raw policy.

Only the two completed reference poses and the proposed H40 are inputs. The
tenth future knot and the suffix remain byte-identical; no retiming or grip
change is made. This is a separately scored adapter candidate, not a gate.
"""
import numpy as np

from go2arx5_vla.trajectory.so3 import (
    matrix_to_quat, quat_to_matrix, rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows,
)

METHOD = 'REFERENCE_JOIN10_TWO_HISTORY_QUINTIC_V71'


def _log(rotation):
    q = matrix_to_quat(rotation)
    if q[3] < 0:
        q = -q
    sine = np.linalg.norm(q[:3])
    return q[:3] * (2*np.arctan2(sine, q[3])/sine if sine else 2.)


def _exp(vector):
    angle = np.linalg.norm(vector)
    xyz = vector * (np.sin(angle/2)/angle if angle else .5)
    return quat_to_matrix(np.r_[xyz, np.cos(angle/2)])


def correction_weights():
    # c(0)=0; c(.1)=d1; c(.2)=d2; c(1)=c'(1)=c''(1)=0.
    # The physical times are .05, .10 and .50 seconds, respectively.
    matrix = np.vstack((np.polynomial.polynomial.polyvander([0., .1, .2, 1.], 5),
                        [0., 1., 2., 3., 4., 5.], [0., 0., 2., 6., 12., 20.]))
    rhs = np.zeros((6, 2)); rhs[1, 0] = rhs[2, 1] = 1.
    coefficients = np.linalg.solve(matrix, rhs)
    weights = np.polynomial.polynomial.polyvander(np.arange(1, 11)/10., 5) @ coefficients
    weights[0], weights[1], weights[-1] = [1., 0.], [0., 1.], [0., 0.]
    return weights


def connect(reference_H40, previous_reference_poses):
    baseline = np.asarray(reference_H40, dtype=np.float64)
    past = np.asarray(previous_reference_poses, dtype=np.float64)
    if baseline.shape != (40, 10) or past.shape != (2, 4, 4):
        raise ValueError('EXACT_H40_AND_TWO_COMPLETED_REFERENCES_REQUIRED')
    if not np.isfinite(baseline).all() or not np.isfinite(past).all():
        raise ValueError('FINITE_REFERENCE_CONNECTION_INPUTS_REQUIRED')
    if (not np.allclose(past[:, 3, :], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(np.swapaxes(past[:, :3, :3], -1, -2) @ past[:, :3, :3],
                               np.eye(3), rtol=0, atol=1e-10)
            or not np.allclose(np.linalg.det(past[:, :3, :3]), 1., rtol=0, atol=1e-10)):
        raise ValueError('ORIGINAL_RIGID_REFERENCE_POSES_REQUIRED')
    result = baseline.copy()
    weights = correction_weights()
    displacement = past[-1, :3, 3] - past[-2, :3, 3]
    desired_xyz = past[-1, :3, 3] + np.arange(1, 3)[:, None]*displacement
    result[:9, :3] += (weights @ (desired_xyz-baseline[:2, :3]))[:9]

    rotations = rot6d_rows_to_matrix(baseline[:, 3:9])
    increment = _log(past[-2, :3, :3].T @ past[-1, :3, :3])
    desired_rotations = [past[-1, :3, :3] @ _exp(k*increment) for k in (1, 2)]
    corrections = np.stack([_log(wanted @ actual.T)
                            for wanted, actual in zip(desired_rotations, rotations[:2], strict=True)])
    correction_vectors = weights @ corrections
    result[:9, 3:9] = rotation_matrix_to_rot6d_rows(np.stack(
        [_exp(v) @ r for v, r in zip(correction_vectors[:9], rotations[:9], strict=True)]))
    assert np.array_equal(result[9:], baseline[9:]) and np.array_equal(result[:, 9], baseline[:, 9])
    return result
