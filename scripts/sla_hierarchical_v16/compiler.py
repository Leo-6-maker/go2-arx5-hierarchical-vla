"""Continuous world-TCP references for sealed SLA actions, in SI units and seconds."""

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

DT = 0.05
END = 49 / 30


def ease(u):
    u = np.clip(u, 0, 1)
    return u**3 * (10 + u * (-15 + 6 * u))


def rounded(t):
    return max(DT, math.ceil(t / DT - 1e-12) * DT)


def normmax(v):
    return float(np.linalg.norm(v, axis=-1).max())


def vectors(p, dt):
    v = np.diff(p[:, :3, 3], axis=0) / dt
    w = (
        Rotation.from_matrix(
            p[1:, :3, :3] @ p[:-1, :3, :3].transpose(0, 2, 1)
        ).as_rotvec()
        / dt
    )
    return v, w


def relative_angle(a, b):
    return Rotation.from_matrix(a @ np.swapaxes(b, -1, -2)).magnitude()


def matrices(x, r):
    out = np.tile(np.eye(4), (len(x), 1, 1))
    out[:, :3, 3] = x
    out[:, :3, :3] = r
    return out


class SourceCurve:
    def __init__(self, raw, origin, tool):
        raw = np.asarray(raw, float)
        if raw.shape != (50, 16) or not np.isfinite(raw).all():
            raise ValueError("invalid official actions")
        self.arm = raw[:, 8:].copy()
        if np.max(abs(np.linalg.norm(self.arm[:, 3:7], axis=1) - 1)) > 1e-4 or np.any(
            (self.arm[:, 7] < 0) | (self.arm[:, 7] > 1)
        ):
            raise ValueError("invalid official rotation or width")
        self.origin, self.tool = (np.asarray(m, float) for m in (origin, tool))
        for m in (self.origin, self.tool):
            if (
                m.shape != (4, 4)
                or not np.isfinite(m).all()
                or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(
                    m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-8, rtol=0
                )
                or abs(np.linalg.det(m[:3, :3]) - 1) > 1e-8
            ):
                raise ValueError("invalid frame")
        self.indices = np.arange(50)
        self.rot = Slerp(self.indices, Rotation.from_quat(self.arm[:, [4, 5, 6, 3]]))
        self.knots = self.at(self.indices)
        self.events = []
        self.protected = set()
        for i, (a, b) in enumerate(zip(self.arm[:-1, 7], self.arm[1:, 7])):
            level, kind = (0.55, "close") if a > 0.55 >= b else (0.65, "open")
            if kind == "close" or a < 0.65 <= b:
                self.events.append((kind, i + (level - a) / (b - a)))
                self.protected.update(range(max(0, i - 1), min(49, i + 2)))

    def at(self, index):
        t = np.clip(index, 0, 49)
        x = np.stack([np.interp(t, self.indices, c) for c in self.arm[:, :3].T], axis=1)
        return self.origin @ matrices(x, self.rot(t).as_matrix()) @ self.tool

    def grip(self, index):
        return np.interp(index, self.indices, self.arm[:, 7])


class CurveBlock:
    def __init__(self, source, start, end, rounded_geometry):
        self.source, self.start, self.end = source, start, end
        self.spline = rounded_geometry and end - start > 1
        if self.spline:
            k = np.arange(start, end + 1)
            p = source.knots[k]
            self.pos = CubicSpline(k, p[:, :3, 3], axis=0)
            self.rot = RotationSpline(k, Rotation.from_matrix(p[:, :3, :3]))

    def at(self, phase):
        if self.spline:
            return matrices(self.pos(phase), self.rot(phase).as_matrix())
        return self.source.at(phase)

    def unit(self, t):
        phase = self.start + (self.end - self.start) * ease(t)
        return self.at(phase), phase

    def distortion(self):
        phase = np.linspace(self.start, self.end, 24 * (self.end - self.start) + 1)
        fitted, original = self.at(phase), self.source.at(phase)
        return normmax(fitted[:, :3, 3] - original[:, :3, 3]), float(
            np.max(relative_angle(fitted[:, :3, :3], original[:, :3, :3]))
        )


def blocks(source, bounded, contact):
    group = []

    def add(i, j):
        b = CurveBlock(source, i, j, bounded and not contact)
        dx, da = b.distortion()
        if b.spline and (dx > 0.0005 or da > np.deg2rad(0.25)):
            m = (i + j) // 2
            add(i, m)
            add(m, j)
        else:
            group.append(b)

    i = 0
    while i < 49:
        j = i + 1
        if bounded and not contact and i not in source.protected:
            while j < 49 and j - i < 6 and j not in source.protected:
                j += 1
        add(i, j)
        i = j
    return group


def size_duration(unit, grip, limits, n):
    t = np.linspace(0, 1, n + 1)
    p, phase = unit(t)
    v, w = vectors(p, 1 / n)
    maxima = [normmax(v), normmax(w)]
    accel = [normmax(np.diff(x, axis=0) * n) for x in (v, w)]
    worst = []
    for vel, acc, key in zip(maxima, accel, ("linear", "angular")):
        a = limits[key + "_acceleration"]
        worst.extend(
            [
                vel / (0.9 * min(limits[key + "_velocity"], a * DT)),
                math.sqrt(acc / (0.9 * a)),
            ]
        )
    if grip is not None:
        worst.append(
            float(np.max(abs(np.diff(grip(phase))))) * 0.044 * n / (0.9 * 0.04)
        )
    return rounded(max(worst))


class CruiseClock:
    """C² scalar phase for a line and shortest slerp, with equal smooth speed ramps."""

    def __init__(self, translation, angle, limits, minimum_duration):
        if (
            not np.isfinite([translation, angle, minimum_duration]).all()
            or min(translation, angle) < 0
        ):
            raise ValueError("invalid alignment distances or time")
        inverse_speed = 0.0
        inverse_acceleration = 0.0
        for distance, kind in ((translation, "linear"), (angle, "angular")):
            velocity = limits[kind + "_velocity"]
            acceleration = limits[kind + "_acceleration"]
            if (
                not np.isfinite([velocity, acceleration]).all()
                or min(velocity, acceleration) <= 0
            ):
                raise ValueError("invalid alignment limits")
            inverse_speed = max(
                inverse_speed, distance / (0.9 * min(velocity, acceleration * DT))
            )
            inverse_acceleration = max(
                inverse_acceleration, distance / (0.9 * acceleration)
            )
        if inverse_speed == 0:
            self.duration = rounded(minimum_duration)
            self.ramp_fraction = 0.5
            return
        # Each smoothstep velocity ramp covers half the distance of a cruise at peak speed.
        inverse_peak = max(inverse_speed, math.sqrt(1.5 * inverse_acceleration))
        ramp = 1.5 * inverse_acceleration / inverse_peak
        fastest = inverse_peak + ramp
        self.duration = rounded(max(fastest, minimum_duration))
        self.ramp_fraction = min(0.5, ramp / fastest)

    def phase(self, t):
        u = np.clip(t, 0, 1)
        r = self.ramp_fraction
        q = np.minimum(np.minimum(u, 1 - u) / r, 1.0)
        ramp_distance = r * q**3 * (1 - 0.5 * q) / (1 - r)
        middle = (u - 0.5 * r) / (1 - r)
        return np.clip(
            np.where(
                u < r, ramp_distance, np.where(u > 1 - r, 1 - ramp_distance, middle)
            ),
            0,
            1,
        )


class Compiler:
    """The alignment has its own clock. No teacher future pose or contact target enters the fit."""

    def __init__(
        self,
        source,
        anchor,
        incoming,
        limits,
        *,
        bounded=False,
        protect_all=False,
        cruise_from_width_m=None,
    ):
        self.source = source
        self.anchor = np.array(anchor, float)
        self.v, self.w = np.asarray(incoming, float)
        self.brake = rounded(
            max(
                0.1,
                1.875 * np.linalg.norm(self.v) / limits["linear_acceleration"],
                1.875 * np.linalg.norm(self.w) / limits["angular_acceleration"],
            )
        )
        self.stopped = self.braking(np.array([self.brake]))[0]
        self.first = source.knots[0]
        self.rotation = Slerp(
            [0.0, 1.0], Rotation.from_matrix([self.stopped[:3, :3], self.first[:3, :3]])
        )
        self.cruise_clock = None
        if cruise_from_width_m is None:
            self.align = size_duration(self.alignment, None, limits, 96)
        else:
            first_width_m = 0.002 + 0.040 * source.arm[0, 7]
            # The legacy H40 interprets this same fractional change over 44 mm travel.
            width_time = (
                1.875
                * abs(first_width_m - cruise_from_width_m)
                * (0.044 / 0.040)
                / (0.9 * 0.040)
            )
            self.cruise_clock = CruiseClock(
                np.linalg.norm(self.first[:3, 3] - self.stopped[:3, 3]),
                relative_angle(self.first[:3, :3], self.stopped[:3, :3]),
                limits,
                width_time - self.brake,
            )
            self.align = self.cruise_clock.duration
        self.lead = self.brake + self.align
        self.parts = blocks(source, bounded, protect_all)
        durations = [
            size_duration(b.unit, source.grip, limits, max(96, 48 * (b.end - b.start)))
            for b in self.parts
        ]
        self.boundaries = self.lead + np.r_[0, np.cumsum(durations)]
        self.end = float(self.boundaries[-1])

    def braking(self, t):
        u = np.clip(t / self.brake, 0, 1)
        s = self.brake * (u - 2.5 * u**4 + 3 * u**5 - u**6)
        return matrices(
            self.anchor[:3, 3] + s[:, None] * self.v,
            Rotation.from_rotvec(s[:, None] * self.w).as_matrix() @ self.anchor[:3, :3],
        )

    def alignment(self, t):
        h = (
            np.clip(ease(t), 0, 1)
            if self.cruise_clock is None
            else self.cruise_clock.phase(t)
        )
        x = self.stopped[:3, 3] + h[:, None] * (self.first[:3, 3] - self.stopped[:3, 3])
        return matrices(x, self.rotation(h).as_matrix()), np.zeros_like(t)

    def at(self, times):
        t = np.atleast_1d(times).astype(float)
        if not np.isfinite(t).all() or min(t) < -1e-9:
            raise ValueError("invalid execution clock")
        index = np.searchsorted(self.boundaries, t, side="right") - 1
        p = np.repeat(self.first[None], len(t), axis=0)
        phase = np.zeros_like(t)
        brake = t < self.brake
        if np.any(brake):
            p[brake] = self.braking(t[brake])
        align = (~brake) & (index < 0)
        if np.any(align):
            p[align] = self.alignment((t[align] - self.brake) / self.align)[0]
        for i in np.unique(index[index >= 0]):
            mask = index == i
            if i == len(self.parts):
                p[mask], phase[mask] = self.source.knots[-1], 49
            else:
                a, b = self.boundaries[i : i + 2]
                p[mask], phase[mask] = self.parts[i].unit((t[mask] - a) / (b - a))
        return p, self.source.grip(phase), phase / 30
