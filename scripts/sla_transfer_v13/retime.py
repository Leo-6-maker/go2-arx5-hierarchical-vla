"""Parameterize an unchanged source flange path. Constraints are supplied from the frozen gate."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DT = 0.05
SOURCE_DT = 1 / 30
END = 49 / 30
NAMES = ("linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration")


def twist(previous, current, dt):
    linear = (current[..., :3, 3] - previous[..., :3, 3]) / dt
    angular = (
        Rotation.from_matrix(
            current[..., :3, :3] @ np.swapaxes(previous[..., :3, :3], -1, -2)
        ).as_rotvec()
        / dt
    )
    return linear, angular


def rates(poses, dt):
    v, w = twist(poses[:-1], poses[1:], dt)

    def acceleration(x):
        return np.diff(np.vstack([np.zeros(3), x, np.zeros(3)]), axis=0) / dt

    return dict(zip(NAMES, [v, w, acceleration(v), acceleration(w)]))


class SourcePath:
    def __init__(self, selected, origin, offset):
        self.selected = selected
        self.t = np.arange(50) / 30
        self.origin = np.asarray(origin)
        self.offset = np.asarray(offset)
        self.slerp = Slerp(self.t, Rotation.from_quat(selected[:, [4, 5, 6, 3]]))
        self.raw = self.at(self.t)
        self.dist = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(self.raw[:, :3, 3], axis=0), axis=1))
        ]

    def at(self, times):
        u = np.asarray(times)
        if min(u) < -1e-12 or max(u) > END + 1e-12:
            raise ValueError("Source extrapolation")
        u = np.clip(u, 0, END)  # floating-point endpoint tolerance only
        relative = np.tile(np.eye(4), (len(u), 1, 1))
        relative[:, :3, :3] = self.slerp(u).as_matrix()
        relative[:, :3, 3] = np.stack(
            [np.interp(u, self.t, c) for c in self.selected[:, :3].T], axis=1
        )
        return self.origin @ relative @ self.offset

    def arc(self, u):
        return np.interp(u, self.t, self.dist)


def select_timing(path, limits, previous, angular_transport, *, uniform=False, max_ticks=1046):
    tangent = twist(path.raw[0], path.at([0.0001])[0], 0.0001)
    initial = np.r_[
        tangent[0] / limits[NAMES[0]], angular_transport @ tangent[1] / limits[NAMES[1]]
    ]
    incoming = np.r_[previous[0] / limits[NAMES[0]], previous[1] / limits[NAMES[1]]]
    projection = float(initial @ incoming / max(initial @ initial, 1e-20))
    for steps in range(33, max_ticks + 1):
        duration = steps * DT
        z = np.minimum(np.arange(max(40, steps + 1)) * DT / duration, 1.0)
        if uniform:
            phase = END * z
        else:
            # Monotone cubic Hermite, in [0, 3] for a zero end slope.
            slope = np.clip(projection, 0, min(1.0, 3 * END / duration))
            phase = END * (3 * z**2 - 2 * z**3) + duration * slope * z * (1 - z) ** 2
        poses = path.at(phase)
        velocity = twist(poses[0], poses[1], DT)
        seam = [
            np.linalg.norm(velocity[0] - previous[0]) / DT,
            np.linalg.norm(angular_transport @ velocity[1] - previous[1]) / DT,
        ]
        maxima = {k: max(np.linalg.norm(vec, axis=1)) for k, vec in rates(poses, DT).items()}
        # The frozen H40 treats the end of EACH preview as a rest boundary.
        for k, vec in zip(NAMES[2:], twist(poses[38], poses[39], DT)):
            maxima[k] = max(maxima[k], np.linalg.norm(vec) / DT)
        dynamic = all(maxima[k] <= limits[k] * (1 + 1e-10) for k in NAMES)
        boundary = all(x <= limits[k] * (1 + 1e-10) for k, x in zip(NAMES[2:], seam))
        if dynamic and (uniform or boundary):
            return dict(
                duration_s=duration,
                full_ticks=steps,
                phase=phase[:40],
                poses=poses[:40],
                full_phase=phase[: steps + 1],
                maxima=maxima,
                entry_acceleration=seam,
                boundary_pass=bool(boundary),
                projected_initial_rate=projection,
                planning_updates=int(np.ceil(duration / 0.5)),
                preview_lengths=int(np.ceil(duration / 1.95)),
            )
    return None


def crossing(g):
    result = []
    for i, (a, b) in enumerate(zip(g, g[1:])):
        if a > 0.55 >= b:
            result.append(("close", i + (0.55 - a) / (b - a)))
        if a < 0.65 <= b:
            result.append(("open", i + (0.65 - a) / (b - a)))
    return result


def geometric_progress(path, times, compiled, teacher, half_edge, stage, grip):
    source_arc = path.arc(times)
    travel = float(np.sum(np.linalg.norm(np.diff(compiled[:10, :3, 3], axis=0), axis=1)))
    delta = compiled[9, :3, 3] - compiled[0, :3, 3]
    ref = teacher[9, :3, 3] - teacher[0, :3, 3] if teacher is not None else np.zeros(3)
    teacher_travel = (
        None
        if teacher is None
        else float(np.linalg.norm(np.diff(teacher[:10, :3, 3], axis=0), axis=1).sum())
    )
    floor = max(0.05 * half_edge, 0.25 * (teacher_travel or 0.0))
    directional = float(delta @ ref / max(np.linalg.norm(ref), 1e-16))
    moving = travel >= floor and (
        np.linalg.norm(ref) < floor or directional >= 0.25 * np.linalg.norm(ref)
    )
    if stage == "LIFT" and ref[2] > 0:
        moving = moving and delta[2] >= 0.25 * ref[2]
    se = crossing(path.selected[:, 7])
    ce = crossing(grip)

    def positions(events, clock):
        return [
            (
                k,
                float(
                    path.arc(np.interp(i, np.arange(len(clock)), clock)) / max(path.dist[-1], 1e-15)
                ),
            )
            for k, i in events
        ]

    original, adapted = positions(se, path.t), positions(ce, times)
    event_ok = len(original) == len(adapted) and all(
        k == l and abs(s - t) <= 0.10 for (k, s), (l, t) in zip(original, adapted)
    )
    return dict(
        P10_travel_m=travel,
        P10_source_arc_m=float(source_arc[9]),
        P10_min_m=floor,
        teacher_P10_travel_m=teacher_travel,
        projected_m=directional,
        P10_net_m=delta.tolist(),
        teacher_net_m=ref.tolist(),
        motion_pass=bool(moving) if teacher is not None else None,
        events_source=original,
        events_compiled=adapted,
        event_pass=event_ok,
        preview_fraction=float(source_arc[-1] / max(path.dist[-1], 1e-15)),
        full_source_travel_m=float(path.dist[-1]),
    )
