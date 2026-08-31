"""Small, explicit SE(3) helpers for benchmark shape/binding contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .so3 import matrix_to_quat, quat_to_matrix


def _as_pose(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape[-2:] != (4, 4):
        raise ValueError("poses must end in [4,4]")
    if not np.isfinite(pose).all():
        raise ValueError("poses must be finite")
    if not np.allclose(pose[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("poses must be homogeneous")
    rotations = pose[..., :3, :3]
    if not np.allclose(
        np.matmul(np.swapaxes(rotations, -1, -2), rotations),
        np.eye(3),
        atol=1e-6,
    ):
        raise ValueError("pose rotation blocks must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6):
        raise ValueError("pose rotation blocks must have determinant one")
    return pose


def pose_from_position_quaternion(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if p.shape[-1] != 3 or q.shape[-1] != 4 or p.shape[:-1] != q.shape[:-1]:
        raise ValueError("position and quaternion batch shapes must agree")
    out = np.zeros(p.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = quat_to_matrix(q)
    out[..., :3, 3] = p
    out[..., 3, 3] = 1.0
    return _as_pose(out)


def pose_position(pose: Any) -> np.ndarray:
    return _as_pose(pose)[..., :3, 3]


def pose_quaternion(pose: Any) -> np.ndarray:
    return matrix_to_quat(_as_pose(pose)[..., :3, :3])


def compose_poses(*poses: Any) -> np.ndarray:
    if not poses:
        return np.eye(4, dtype=np.float64)
    result = _as_pose(poses[0])
    for pose in poses[1:]:
        result = _as_pose(np.matmul(result, _as_pose(pose)))
    return result


def invert_pose(pose: Any) -> np.ndarray:
    p = _as_pose(pose)
    out = np.zeros_like(p)
    r_t = np.swapaxes(p[..., :3, :3], -1, -2)
    out[..., :3, :3] = r_t
    out[..., :3, 3] = -np.matmul(r_t, p[..., :3, 3][..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def relative_poses(absolute_poses: Any, *, reference_index: int = 0) -> np.ndarray:
    absolute = _as_pose(absolute_poses)
    if absolute.ndim != 3:
        raise ValueError("absolute_poses must be [N,4,4]")
    if not 0 <= reference_index < len(absolute):
        raise ValueError("reference_index is outside the pose sequence")
    return compose_poses(invert_pose(absolute[reference_index]), absolute)


def apply_binding(T_world_anchor: Any, T_anchor_task: Any, T_task_eef: Any) -> np.ndarray:
    """Evaluate ``T_world_eef(t)=T_world_anchor*T_anchor_task*T_task_eef(t)``."""
    world_anchor = _as_pose(T_world_anchor)
    anchor_task = _as_pose(T_anchor_task)
    task_eef = _as_pose(T_task_eef)
    if task_eef.ndim != 3:
        raise ValueError("T_task_eef must be [N,4,4]")
    return np.matmul(np.matmul(world_anchor, anchor_task), task_eef)
