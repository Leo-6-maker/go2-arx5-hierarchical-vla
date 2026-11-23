"""A0/A1 proposal kernel for an offline rehearsal on the 2 ms native clock.

The caller owns the receiver, verifies actual accept/install/consumption evidence
and certifies every proposed window before sending. Proposing has no actuator effect.
"""

import numpy as np

from go2arx5_vla.trajectory.so3 import matrix_to_quat, quat_to_matrix

START = 5703
OPEN = (0.044 - 0.002) / 0.044


def require(ok, why):
    if not ok:
        raise ValueError(why)


def pose(p):
    p = np.asarray(p, dtype=float)
    require(
        p.shape == (4, 4)
        and np.isfinite(p).all()
        and np.array_equal(p[3], [0, 0, 0, 1])
        and np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), rtol=0, atol=1e-6)
        and abs(np.linalg.det(p[:3, :3]) - 1) < 1e-6,
        "INVALID_WORLD_POSE",
    )
    return p


def rotvec(r):
    q = matrix_to_quat(r)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    return q[:3] * (2 * np.arctan2(s, q[3]) / s if s > 1e-14 else 2.0)


def rotation(v):
    a = np.linalg.norm(v)
    return quat_to_matrix(np.r_[0.5 * np.sinc(a / (2 * np.pi)) * v, np.cos(a / 2)])


def capped(v, limit):
    return v * min(1.0, limit / max(1e-15, np.linalg.norm(v)))


def boundary(episode, tick, expected_wire, receipt, measurement):
    """A receipt is constructed from receiver events plus the last completed atom."""
    require(receipt["episode"] == measurement["episode"] == episode, "WRONG_EPISODE")
    require(
        type(tick) is int
        and measurement["tick"] == receipt["end"] == tick
        and receipt["start"] == tick - 25
        and receipt["source"] <= tick - 25
        and tick - receipt["source"] < 200,
        "WRONG_CLOCK",
    )
    require(
        bool(expected_wire)
        and receipt["accepted"]
        == receipt["installed"]
        == receipt["consumed"]
        == expected_wire
        and receipt["solver_echo"],
        "UNVERIFIED_REFERENCE",
    )
    require(measurement["safe"] and not measurement["contact"], "CONTACT_OR_STOP")
    require(
        receipt["finger_fraction"] == OPEN and receipt["finger_owner"] == episode,
        "GRIPPER_OWNERSHIP",
    )
    return pose(receipt["end_pose"]), pose(measurement["TCP"])


class ReferenceManager:
    def __init__(self, episode, old, config):
        self.episode = episode
        self.origin = pose(old).copy()
        self.last_wire = None
        self.anchor = self.origin.copy()
        self.next_tick = START
        self.c = config

    def propose(self, mode, tick, expected_wire, receipt, measurement, target):
        require(tick == self.next_tick, "DUPLICATE_OR_SKIPPED_UPDATE")
        anchor, actual = boundary(
            self.episode, tick, expected_wire, receipt, measurement
        )
        require(
            (self.last_wire is None or self.last_wire == expected_wire)
            and np.allclose(anchor, self.anchor, rtol=0, atol=2e-7),
            "ACK_ANCHOR_MISMATCH",
        )
        goal = pose(target)
        require(mode in ("A0", "A1"), "UNKNOWN_MODE")
        if mode == "A1":
            dx = capped(
                self.c["translation_gain_per_update"] * (goal[:3, 3] - actual[:3, 3]),
                self.c["max_step_mm"] / 1000,
            )
            dr = capped(
                self.c["rotation_gain_per_update"]
                * rotvec(goal[:3, :3] @ actual[:3, :3].T),
                np.deg2rad(self.c["max_step_deg"]),
            )
        else:
            # A0 is the unchanged known static goal at the continued reference.
            dx = np.zeros(3)
            dr = np.zeros(3)
        knots = np.repeat(anchor[None], 41, axis=0)
        for ix, fraction in ((1, 0.5), (2, 1.0)):
            knots[ix, :3, 3] += dx * fraction
            knots[ix, :3, :3] = rotation(dr * fraction) @ anchor[:3, :3]
        knots[3:] = knots[2]
        previous = pose(receipt["start_pose"])
        prior_dp = anchor[:3, 3] - previous[:3, 3]
        prior_dr = rotvec(anchor[:3, :3] @ previous[:3, :3].T)
        require(
            np.linalg.norm(prior_dp) <= 0.05 * 0.05 + 1e-10
            and np.linalg.norm(prior_dr) <= 0.25 * 0.05 + 1e-10
            and np.linalg.norm(dx / 2 - prior_dp) <= 0.25 * 0.05**2 + 1e-10
            and np.linalg.norm(dr / 2 - prior_dr) <= 1.0 * 0.05**2 + 1e-10,
            "CONSUMED_SEAM",
        )
        require(
            1000 * np.linalg.norm(knots[2, :3, 3] - self.origin[:3, 3])
            <= self.c["max_translation_from_initial_ref_mm"] + 1e-9
            and np.degrees(
                np.linalg.norm(rotvec(knots[2, :3, :3] @ self.origin[:3, :3].T))
            )
            <= self.c["max_rotation_from_initial_ref_deg"] + 1e-9,
            "WORKSPACE_CAP",
        )
        return knots

    def consumed(self, tick, expected_wire, proposed, receipt, measurement):
        require(tick == self.next_tick + 50, "DUPLICATE_OR_SKIPPED_CONSUMPTION")
        endpoint, _ = boundary(self.episode, tick, expected_wire, receipt, measurement)
        require(
            receipt["source"] == self.next_tick
            and np.allclose(endpoint, proposed[2], rtol=0, atol=2e-7),
            "COMMAND_ENDPOINT_MISMATCH",
        )
        self.anchor = endpoint.copy()
        self.last_wire = expected_wire
        self.next_tick = tick


class StopMonitor:
    def __init__(self):
        self.errors = []
        self.hold = None
        self.flat = np.zeros(2, dtype=int)
        self.reversals = np.zeros(6, dtype=int)
        self.extreme = np.zeros(6)
        self.sign = np.zeros(6)

    def observe(self, t_ms, e_mm, e_deg, position_limit, angle_limit):
        # A new monotonic sample, including the zero-time sample and every 2 ms.
        require(t_ms == 2 * len(self.errors), "WRONG_FEEDBACK_GRID")
        e = np.r_[e_mm, e_deg]
        size = np.array([np.linalg.norm(e_mm), np.linalg.norm(e_deg)])
        limits = np.array([position_limit, angle_limit])
        self.errors.append(size)
        if np.all(size <= limits):
            if self.hold is None:
                # A hold can start only at a 10 Hz command boundary.
                if t_ms % 100:
                    return "CONTINUE"
                self.hold = t_ms
            return "HOLD_COMPLETE" if t_ms - self.hold >= 1000 else "HOLD"
        if self.hold is not None:
            return "HOLD_LOST"
        if t_ms == 0:
            self.extreme = e.copy()
        diff = e - self.extreme
        band = np.array([1.0] * 3 + [0.3] * 3)
        changed = (np.sign(diff) != self.sign) & (np.abs(diff) > band)
        self.reversals += changed & (self.sign != 0)
        self.sign[changed] = np.sign(diff[changed])
        forward = (self.sign * diff > 0) | changed
        self.extreme[forward] = e[forward]
        if (self.reversals >= 3).any():
            return "OSCILLATION"
        if t_ms and t_ms % 1000 == 0:
            limited = (size > limits) & (self.errors[-501] - size < [0.3, 0.08])
            self.flat = np.where(limited, self.flat + 1, 0)
            if (self.flat >= 3).any():
                return "NO_PROGRESS"
        return "CONTINUE"
