"""Offline Reach proposal profiles. All inputs are available at query time."""

import numpy as np
from scipy.spatial.transform import Rotation

PREFIX = 16  # indices 0..15 inclusive, at 30 Hz
MAX_STEP = 0.020
CORRIDOR = 0.008
MAX_ROTATION = np.deg2rad(2)


def visual_goal(obj, actual):
    if obj["status"] != "measured":
        raise ValueError("vision: " + obj["status"])
    for comp in obj["views"].values():
        x0, y0, x1, y1 = comp["bbox_xyxy"]
        if min(x0, y0) <= 0 or max(x1, y1) >= 255:
            raise ValueError("vision: color component touches image boundary")
    allowance = 0.005 + 0.0005 * obj["separation_mm"]
    if not np.isfinite(allowance) or allowance >= CORRIDOR:
        raise ValueError("vision: uncertainty exceeds corridor")
    center = np.asarray(obj["center_world_m"])
    delta = center + [0, 0, 0.080] - actual[:3, 3]
    distance = np.linalg.norm(delta)
    if not np.isfinite(distance) or distance <= allowance:
        raise ValueError("vision: direction unresolved")
    return delta / distance, min(MAX_STEP, distance - allowance), CORRIDOR - allowance


def shortened(ref, path):
    delta = path[:PREFIX, :3, 3] - ref[:3, 3]
    alpha = min(1.0, MAX_STEP / max(np.linalg.norm(delta, axis=1).max(), 1e-12))
    out = path[:PREFIX].copy()
    out[:, :3, 3] = ref[:3, 3] + alpha * delta
    relative = Rotation.from_matrix(out[:, :3, :3] @ ref[:3, :3].T).as_rotvec()
    out[:, :3, :3] = Rotation.from_rotvec(alpha * relative).as_matrix() @ ref[:3, :3]
    return out, alpha


def geometric_control(actual, ref, obj):
    axis, cap, _ = visual_goal(obj, actual)
    u = np.linspace(0, 1, PREFIX)
    smooth = u**3 * (10 - 15 * u + 6 * u * u)
    out = np.repeat(ref[None], PREFIX, axis=0)
    out[:, :3, 3] += smooth[:, None] * cap * axis
    return out


def propose(profile, world, actual, ref, obj):
    if profile == "RAW":
        return world.copy(), 1.0
    rel = ref @ np.linalg.solve(actual, world)
    if profile == "REANCHOR":
        return rel, 1.0
    short, alpha = shortened(ref, rel)
    if profile == "SHORT":
        return short, alpha
    if profile != "PROJECT":
        raise ValueError("unknown treatment")
    axis, cap, radius = visual_goal(obj, actual)
    delta = short[:, :3, 3] - ref[:3, 3]
    parallel = delta @ axis
    if parallel.max() <= 0.0:
        raise ValueError("no forward component in SLA prefix")
    perp = delta - parallel[:, None] * axis
    # Closest feasible point in the axial interval x transverse disk. The axial
    # cap can only shorten positive SLA motion, never supply missing progress.
    scale = np.minimum(1.0, radius / np.maximum(1e-12, np.linalg.norm(perp, axis=1)))
    out = short.copy()
    out[:, :3, 3] = (
        ref[:3, 3] + np.clip(parallel, 0, cap)[:, None] * axis + perp * scale[:, None]
    )
    rot = Rotation.from_matrix(out[:, :3, :3] @ ref[:3, :3].T).as_rotvec()
    rot *= np.minimum(
        1.0, MAX_ROTATION / np.maximum(1e-12, np.linalg.norm(rot, axis=1))
    )[:, None]
    out[:, :3, :3] = Rotation.from_rotvec(rot).as_matrix() @ ref[:3, :3]
    return out, alpha
