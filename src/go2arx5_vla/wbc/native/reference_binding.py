"""Pure math for the delivered native reset/reach anchor.

This mirrors the source-backed native operation without importing IsaacGym or
the private lab package. It deliberately returns a world-aligned translation
anchor: the native loader adds this point to the local trajectory positions
and does not rotate those positions or the local orientations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ...trajectory.relative_se3 import pose_from_position_quaternion


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("anchor direction must be non-zero")
    return vector / norm


@dataclass(frozen=True)
class NativeAnchorInputs:
    base_position: np.ndarray
    base_quaternion_xyzw: np.ndarray
    arm_mount_translation_body: np.ndarray
    anchor_offset_body: np.ndarray
    rho_star: float
    reach_radius_m: Optional[float] = None
    reach_max_m: Optional[float] = None


def native_translation_anchor(inputs: NativeAnchorInputs) -> dict[str, Any]:
    """Reproduce native ``shoulder + direction * rho_star * R_max`` math."""
    base_pose = pose_from_position_quaternion(inputs.base_position, inputs.base_quaternion_xyzw)
    base_rotation = base_pose[:3, :3]
    mount = np.asarray(inputs.arm_mount_translation_body, dtype=np.float64)
    direction_body = _normalize(np.asarray(inputs.anchor_offset_body, dtype=np.float64))
    shoulder_position = np.asarray(inputs.base_position, dtype=np.float64) + base_rotation @ mount
    direction_world = base_rotation @ direction_body
    r_max = inputs.reach_max_m
    if r_max is None:
        if inputs.reach_radius_m is None:
            raise ValueError("reach_radius_m or reach_max_m is required")
        r_max = float(inputs.reach_radius_m)
    anchor_position = shoulder_position + direction_world * float(inputs.rho_star) * float(r_max)
    # The native trajectory local axes are world-aligned. Its loader performs
    # translation-only anchoring, so this pose has identity orientation.
    anchor_pose = pose_from_position_quaternion(anchor_position, [0.0, 0.0, 0.0, 1.0])
    return {
        "base_pose": base_pose,
        "shoulder_position": shoulder_position,
        "direction_world": direction_world,
        "reach_max_m": float(r_max),
        "anchor_position": anchor_position,
        "T_world_anchor": anchor_pose,
        "anchor_rotation_semantics": "WORLD_ALIGNED_TRANSLATION_ONLY",
    }
