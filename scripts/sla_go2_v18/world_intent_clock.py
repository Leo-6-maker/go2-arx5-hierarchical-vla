"""Preserve world-frame pose increments and time a connected, source-faithful prefix."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from compiler import ease, matrices, normmax, relative_angle, vectors


def world_offset(world, actual, reference):
    out = np.asarray(world, dtype=float).copy()
    out[..., :3, 3] = reference[:3, 3] + (out[..., :3, 3] - actual[:3, 3])
    out[..., :3, :3] = out[..., :3, :3] @ actual[:3, :3].T @ reference[:3, :3]
    return out


def h40_fraction_to_width(fraction):
    return 0.044 * np.asarray(fraction)


def sensed_width_to_state(width):
    return np.asarray(width) / 0.044


def action_fraction_to_width(fraction):
    return 0.002 + 0.040 * np.asarray(fraction)


def command_width_to_action(width):
    width = np.asarray(width)
    if (
        not np.isfinite(width).all()
        or np.any(width < 0.002 - 1e-12)
        or np.any(width > 0.042 + 1e-12)
    ):
        raise ValueError("command outside calibrated Go2 working interval")
    return np.clip((width - 0.002) / 0.040, 0.0, 1.0)


class PrefixClock:
    """Fit in source phase, then choose one quintic time law for the entire prefix."""

    def __init__(self, source_at, limits, *, max_intervals=15):
        for end in range(max_intervals, 0, -1):
            k = np.arange(end + 1)
            p = source_at(k)
            self.xyz = CubicSpline(k, p[:, :3, 3], axis=0)
            self.rot = RotationSpline(k, Rotation.from_matrix(p[:, :3, :3]))
            dense = np.linspace(0, end, 32 * end + 1)
            truth = source_at(dense)
            approx = self.pose(dense)
            dx = normmax(approx[:, :3, 3] - truth[:, :3, 3])
            dr = float(np.max(relative_angle(approx[:, :3, :3], truth[:, :3, :3])))
            if dx <= 0.0005 and dr <= np.deg2rad(0.25):
                break
        else:
            raise ValueError("no source-faithful continuous prefix")
        self.end_index = end
        self.shape_error_mm = 1000 * dx
        self.shape_error_deg = np.degrees(dr)
        u = np.linspace(0, 1, 64 * end + 1)
        pose = self.pose(end * ease(u))
        v, w = vectors(pose, 1 / (len(u) - 1))
        rate = max(
            normmax(v) / limits["linear_velocity"],
            normmax(w) / limits["angular_velocity"],
        )
        accel = max(
            normmax(np.diff(v, axis=0)) * (len(u) - 1) / limits["linear_acceleration"],
            normmax(np.diff(w, axis=0)) * (len(u) - 1) / limits["angular_acceleration"],
        )
        self.duration = 0.05 * max(
            1, int(np.ceil(max(rate / 0.9, np.sqrt(accel / 0.9)) / 0.05))
        )

    def pose(self, index):
        return matrices(self.xyz(index), self.rot(index).as_matrix())

    def at(self, time):
        return self.pose(
            self.end_index * ease(np.clip(np.asarray(time) / self.duration, 0, 1))
        )
