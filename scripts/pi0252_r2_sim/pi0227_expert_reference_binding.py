"""Exact original PI0227 target/anchor binding used by the native expert collector."""
from __future__ import annotations
from typing import Mapping
import numpy as np

def _remap_trajectory_to_target(
    trajectory: Mapping[str, np.ndarray], anchor_world: np.ndarray, target_world: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep the generator shape while binding its interior waypoint to target."""

    timestamps = np.asarray(trajectory["timestamp"], dtype=np.float64)
    position = np.asarray(trajectory["position"], dtype=np.float64)
    quaternion = np.asarray(trajectory["quaternion"], dtype=np.float64)
    target_local = np.asarray(target_world, dtype=np.float64) - np.asarray(anchor_world, dtype=np.float64)
    target_index = int(np.argmin(np.abs(timestamps - 1.0)))
    source_delta = position[target_index] - position[0]
    desired_delta = target_local - position[0]
    denominator = float(np.dot(source_delta, source_delta))
    if denominator < 1e-10:
        raise ValueError("trajectory realization has degenerate target displacement")
    transform = np.eye(3, dtype=np.float64) + np.outer(desired_delta - source_delta, source_delta) / denominator
    remapped = position[0] + (position - position[0]) @ transform.T
    remapped[target_index] = target_local
    return {
        "timestamp": timestamps.copy(),
        "position": remapped,
        "quaternion": quaternion.copy(),
    }

