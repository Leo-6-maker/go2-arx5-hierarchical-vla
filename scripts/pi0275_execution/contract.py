"""PI0275 CPU boundaries. This module never advances a simulator or a model."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix

AUTH = 'PI0275_OBJECT_TRANSFER_PHYSICS_AND_DATA_PILOT_GO_V1'
RELEASE = 'b900283549a984197b68e857ef05b801a2f6fd15'
DT = 1 / 120


def bound_source(name, expected_sha):
    path = REPO / 'scripts' / (name + '.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('prepared dependency drift: ' + str(path))
    spec = importlib.util.spec_from_file_location('pi0275_bound_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_opening(width):
    x = np.asarray(width, float)
    if not np.isfinite(x).all() or np.any(x < -1e-5) or np.any(x > .08 + 1e-5):
        raise ValueError('NATIVE_OPENING_LIMIT_FAILURE')
    return x / .08  # Preserve unmodified native measurement, no clipping.


def guard(label, time, measured_p, measured_r, previous=None):
    x = np.asarray(label, float)
    if x.shape != (10,) or not np.isfinite(x).all() or not np.isfinite(time):
        return ['NONFINITE_COMMAND']
    if not 0 <= x[9] <= 1:
        return ['COMMAND_OPENING_LIMIT']
    try:
        r = rot6d_rows_to_matrix(x[3:9])
    except ValueError:
        return ['COMMAND_NOT_SO3']
    # A decoded but nonorthogonal raw row pair is not accepted as a rotation.
    if not np.allclose(r[:2].ravel(), x[3:9], atol=1e-6, rtol=0):
        return ['COMMAND_NOT_SO3']
    reasons = []
    measured_p, measured_r = np.asarray(measured_p, float), np.asarray(measured_r, float)
    if measured_p.shape != (3,) or measured_r.shape != (3, 3) or not np.isfinite(measured_p).all() or not np.isfinite(measured_r).all():
        return ['NONFINITE_MEASUREMENT']
    if np.any(measured_p < [-.55, -.55, .65]) or np.any(measured_p > [.55, .55, 1.10]):
        reasons.append('ACTUAL_TCP_WORKSPACE')
    if np.any(x[:3] < [-.55, -.55, .65]) or np.any(x[:3] > [.55, .55, 1.10]):
        reasons.append('COMMAND_WORKSPACE')
    if np.linalg.norm(x[:3] - measured_p) > .05 + 1e-9 or geodesic_angle(r, measured_r) > .20 + 1e-9:
        reasons.append('TRACKING_GAP')
    if previous is not None:
        t0, old = previous
        dt = time - t0
        if not 0 < dt <= DT + 1e-8:
            reasons.append('COMMAND_CLOCK')
        elif (np.linalg.norm(x[:3] - old[:3]) > .10 * dt + 1e-9
              or geodesic_angle(r, rot6d_rows_to_matrix(old[3:9])) > .5 * dt + 1e-9
              or abs(x[9] - old[9]) * .08 > .04 * dt + 1e-9):
            reasons.append('COMMAND_RATE')
    elif abs(time) > 1e-9:
        reasons.append('COMMAND_CLOCK')
    return reasons


def clip_norm(vector, limit):
    v = np.asarray(vector, float)
    return v * min(1., limit / max(np.linalg.norm(v), 1e-12))


def rotation_jacobian(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.column_stack(([0, 0, 1], [-sy, cy, 0], [cy * cp, sy * cp, -sp]))


def efforts(label, measured, gravity_force, gravity_torque, rotation_error):
    """Native caller supplies measured COM/Jacobian gravity compensation and left SO3 error."""
    q, qd = measured['q'], measured['qd']
    force = clip_norm(600 * (label[:3] - measured['p']) - 40 * measured['v'] + gravity_force, 20.)
    torque = clip_norm(4 * rotation_error - .3 * measured['omega'] + gravity_torque, 2.)
    angular = rotation_jacobian(q[3], q[4]).T @ torque
    fingers = np.clip(2000 * (label[9] * .04 - q[6:]) - 5 * qd[6:], -10, 10)
    return np.r_[np.clip(force, -20, 20), np.clip(angular, -2, 2), fingers], force, torque


def obb_depth(p0, r0, h0, p1, r1, h1):
    """Minimum separating translation on normalized OBB axes; 0 for disjoint boxes."""
    axes = [r0[:, i] for i in range(3)] + [r1[:, i] for i in range(3)]
    axes += [np.cross(r0[:, i], r1[:, j]) for i in range(3) for j in range(3)]
    depth = float('inf')
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis / norm
        overlap = np.dot(np.abs(r0.T @ axis), h0) + np.dot(np.abs(r1.T @ axis), h1) - abs(np.dot(p1 - p0, axis))
        if overlap <= 0:
            return 0.
        depth = min(depth, overlap)
    return float(depth)


def containment(position, rotation, half, bounds, actual_support):
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    corners = position[..., None, :] + np.einsum('...ij,kj->...ki', rotation, signs * half)
    strict = ((corners >= bounds[0] - 1e-9) & (corners <= bounds[1] + 1e-9)).all(axis=(-1, -2))
    lower = np.broadcast_to(bounds[0] - 1e-5, position.shape).copy()
    lower[..., 2] = bounds[0, 2] - np.where(actual_support, .002, 1e-5)
    supported = ((corners >= lower[..., None, :]) & (corners <= bounds[1] + 1e-5)).all(axis=(-1, -2))
    return strict, supported


def self_check():
    r = np.diag([1., -1., -1.]); x = np.r_[[0, -.38, .85], r[:2].ravel(), 1.]
    assert not guard(x, 0., x[:3], r)
    bad = x.copy(); bad[0] += .06
    assert 'TRACKING_GAP' in guard(bad, 0., x[:3], r)
    assert 'COMMAND_RATE' in guard(bad, DT, bad[:3], r, (0., x))
    assert abs(measured_opening(-1e-6) + 1.25e-5) < 1e-12
    try:
        measured_opening(-2e-5)
    except ValueError:
        pass
    else:
        raise AssertionError('opening tolerance expanded')
    i = np.eye(3); half = np.array([.02, .015, .015]); bounds = np.array([[-.08, -.08, .65], [.08, .08, .69]])
    assert abs(obb_depth(np.zeros(3), i, half, np.array([.039, 0, 0]), i, half) - .001) < 1e-12
    assert obb_depth(np.zeros(3), i, half, np.array([.041, 0, 0]), i, half) == 0
    for p, support, accepted in [([0, 0, .664], True, True), ([0, 0, .664], False, False),
                                 ([0, 0, .662], True, False), ([.061, 0, .665], True, False),
                                 ([0, 0, .686], True, False)]:
        strict, refined = containment(np.array(p), i, half, bounds, support)
        assert bool(refined) == accepted
        if p[2] == .664:
            assert not strict
    # Virtual-work identity and angular finite-difference axes, including -pi roll.
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, quat_to_matrix
    q = np.array([.4, -.35, -np.pi])
    def fk(a):
        rz = np.array([[np.cos(a[0]), -np.sin(a[0]), 0], [np.sin(a[0]), np.cos(a[0]), 0], [0, 0, 1]])
        ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
        rx = np.array([[1, 0, 0], [0, np.cos(a[2]), -np.sin(a[2])], [0, np.sin(a[2]), np.cos(a[2])]])
        return rz @ ry @ rx
    j = rotation_jacobian(*q[:2]); v = np.array([.1, -.2, .3]); torque = np.array([.4, -.5, .6])
    assert abs(np.dot(j @ v, torque) - np.dot(v, j.T @ torque)) < 1e-12
    for k in range(3):
        a = q.copy(); a[k] += 1e-6; delta = fk(a) @ fk(q).T
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]) / 2e-6
        assert np.allclose(axis, j[:, k], atol=1e-6)
    assert np.allclose(quat_to_matrix(matrix_to_quat(fk(q))), fk(q))
    return dict(status='PASS_PI0275_CPU_CONTRACT_ONLY', physical_starts=0, physics_ticks=0, renders=0,
                learned_calls=0, native_checks='NOT_RUN', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), sort_keys=True))
