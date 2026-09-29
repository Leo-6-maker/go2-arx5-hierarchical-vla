"""Small NumPy-only SO(3) implementation used by the CPU contract tests."""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < eps):
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode first two rotation columns packed as ``[c0, c1]``.

    This is the canonical/GR00T-side representation used when a trajectory
    chunk supplies ``eef_rot6d``.  The delivered native WBC has a separate
    historical wire convention; use :func:`rot6d_rows_to_matrix` for it.
    """
    x = np.asarray(rot6d, dtype=np.float64)
    if x.shape[-1] != 6:
        raise ValueError("rot6d last dimension must be 6")
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = _unit(a1)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _unit(a2_orth)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Pack the first two rotation columns as ``[c0, c1]``."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode the delivered native WBC's PyTorch3D-0.3 row convention.

    The archived controller calls ``matrix_to_rotation_6d`` from its bundled
    PyTorch3D compatibility layer.  That implementation flattens
    ``matrix[..., :2, :]`` in row-major order, and its inverse constructs the
    orthonormal basis as matrix rows.  It is therefore semantically different
    from the column convention above even though both use six numbers.
    """
    x = np.asarray(rot6d, dtype=np.float64)
    if x.shape[-1] != 6:
        raise ValueError("rot6d last dimension must be 6")
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = _unit(a1)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _unit(a2_orth)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def rotation_matrix_to_rot6d_rows(matrix: np.ndarray) -> np.ndarray:
    """Pack the first two rotation rows using the native WBC wire layout."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return r[..., :2, :].reshape(r.shape[:-2] + (6,))


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternion last dimension must be 4 in xyzw order")
    q = _unit(q)
    x, y, z, w = [q[..., i] for i in range(4)]
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _matrix_to_quat_one(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
        w = (r[2, 1] - r[1, 2]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
        w = (r[0, 2] - r[2, 0]) / s
    else:
        s = np.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
        w = (r[1, 0] - r[0, 1]) / s
    return np.asarray([x, y, z, w], dtype=np.float64)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    if r.ndim == 2:
        return _unit(_matrix_to_quat_one(r))
    flat = r.reshape((-1, 3, 3))
    out = np.stack([_matrix_to_quat_one(x) for x in flat], axis=0)
    return _unit(out).reshape(r.shape[:-2] + (4,))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    a = _unit(np.asarray(q0, dtype=np.float64))
    b = _unit(np.asarray(q1, dtype=np.float64))
    tt = np.asarray(t, dtype=np.float64)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    linear = dot > 0.9995
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    tt_exp = np.expand_dims(tt, axis=-1) if tt.ndim == a.ndim - 1 else tt
    first = np.sin((1.0 - tt_exp) * theta) / np.where(sin_theta < 1e-12, 1.0, sin_theta)
    second = np.sin(tt_exp * theta) / np.where(sin_theta < 1e-12, 1.0, sin_theta)
    result = first * a + second * b
    result_linear = (1.0 - tt_exp) * a + tt_exp * b
    result = np.where(linear, result_linear, result)
    return _unit(result)


def geodesic_angle(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    a = np.asarray(r0, dtype=np.float64)
    b = np.asarray(r1, dtype=np.float64)
    if a.shape[-2:] != (3, 3) or b.shape[-2:] != (3, 3):
        raise ValueError("geodesic inputs must be rotation matrices")
    rel = np.einsum("...ji,...jk->...ik", a, b)
    c = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(c)
