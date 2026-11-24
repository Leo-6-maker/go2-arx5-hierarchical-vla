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
    return geometry_line(ref, axis, cap)


def geometry_line(ref, axis, cap):
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


def direct_fit(world, actual, ref, obj):
    axis, cap, radius = visual_goal(obj, actual)
    return fit_with_bounds(world, actual, ref, axis, cap, radius, MAX_ROTATION)


def fit_with_bounds(world, actual, ref, axis, cap, radius, rotation_limit):
    path = ref @ np.linalg.solve(actual, world[:PREFIX])
    proposal = path[:, :3, 3] - ref[:3, 3]
    axial_cap = np.clip(proposal @ axis, 0.0, cap)
    if axial_cap[1:].max() <= 1e-12:
        raise ValueError("no forward component in SLA prefix")

    def project(x):
        axial = x @ axis
        perp = x - axial[:, None] * axis
        scale = np.minimum(
            1.0, radius / np.maximum(1e-12, np.linalg.norm(perp, axis=1))
        )
        out = np.clip(axial, 0, axial_cap)[:, None] * axis + perp * scale[:, None]
        out[0] = 0.0
        return out

    d2 = np.diff(np.eye(len(path)), n=2, axis=0)
    hessian = np.eye(len(path)) + d2.T @ d2
    x = project(proposal)
    for iterations in range(1, 501):
        next_x = project(x - (hessian @ x - proposal) / 17.0)
        residual = float(np.linalg.norm(next_x - x, axis=1).max())
        x = next_x
        if residual <= 1e-10:
            break
    else:
        raise ValueError("direct-fit solver failed to converge")
    out = path.copy()
    out[:, :3, 3] = ref[:3, 3] + x
    rot = Rotation.from_matrix(path[:, :3, :3] @ ref[:3, :3].T).as_rotvec()
    rot *= np.minimum(
        1.0, rotation_limit / np.maximum(1e-12, np.linalg.norm(rot, axis=1))
    )[:, None]
    rot[0] = 0
    out[:, :3, :3] = Rotation.from_rotvec(rot).as_matrix() @ ref[:3, :3]
    return out, dict(
        iterations=iterations,
        residual_m=residual,
        rms_translation_edit_mm=float(
            1000 * np.sqrt(np.mean(np.sum((x - proposal) ** 2, axis=1)))
        ),
        proposed_final_axial_mm=float(1000 * proposal[-1] @ axis),
        retained_final_axial_mm=float(1000 * x[-1] @ axis),
    )


class ReachEpisode:
    """Keep the visual entry axis and total excursion fixed across feedback queries."""

    def __init__(self, epoch, actual, ref, obj):
        self.epoch = epoch
        self.actual = np.array(actual, copy=True)
        self.ref = np.array(ref, copy=True)
        self.center = np.array(obj["center_world_m"], copy=True)
        self.axis, self.cap, self.radius = visual_goal(obj, actual)

    def plan(self, epoch, actual, ref, obj, *, method, model_world=None):
        if method not in ("DIRECT", "GEOMETRY") or (model_world is None) != (
            method == "GEOMETRY"
        ):
            raise ValueError("missing model output or wrong Reach treatment")
        if epoch != self.epoch:
            raise ValueError("different Reach reset")
        _, _, now_radius = visual_goal(obj, actual)
        if (
            np.linalg.norm(np.asarray(obj["center_world_m"]) - self.center)
            > 2 * CORRIDOR - self.radius - now_radius
        ):
            raise ValueError("visual center inconsistent with fixed Reach target")
        offsets = np.stack(
            [actual[:3, 3] - self.actual[:3, 3], ref[:3, 3] - self.ref[:3, 3]]
        )
        projections = offsets @ self.axis
        transverse = np.linalg.norm(offsets - projections[:, None] * self.axis, axis=1)
        # Keep both physical and reference paths inside the original episode allowance.
        radius = min(self.radius, now_radius) - max(transverse)
        rot = Rotation.from_matrix(
            np.stack(
                [
                    actual[:3, :3] @ self.actual[:3, :3].T,
                    ref[:3, :3] @ self.ref[:3, :3].T,
                ]
            )
        ).magnitude()
        angular = MAX_ROTATION - max(rot)
        used = max(0.0, max(projections))
        if min(radius, angular, self.cap - used) < -1e-9:
            raise ValueError(
                f"outside original Reach bounds: transverse {1000 * max(transverse):.3f}/{1000 * min(self.radius, now_radius):.3f} mm, rotation {np.degrees(max(rot)):.3f}/{np.degrees(MAX_ROTATION):.3f} deg, advance {1000 * used:.3f}/{1000 * self.cap:.3f} mm"
            )
        remaining = max(0.0, self.cap - used)
        if remaining <= 1e-9 or method == "GEOMETRY":
            return geometry_line(ref, self.axis, remaining)
        return fit_with_bounds(
            model_world,
            actual,
            ref,
            self.axis,
            remaining,
            max(0.0, radius),
            max(0.0, angular),
        )[0]
