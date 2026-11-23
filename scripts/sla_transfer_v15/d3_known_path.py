"""Standing D3 analytic references on the receiver's native 50 ms sample lattice."""

import numpy as np
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import quintic

DURATIONS = {"slow": 1.90, "fast": 1.35}
FAMILIES = ("translation", "rotation", "curve")


def poses_at(start, family, rate, seconds):
    """Reference attached to a fixed world pose. The final point is held."""
    if family not in FAMILIES or rate not in DURATIONS:
        raise ValueError("unknown D3 condition")
    t = np.atleast_1d(np.asarray(seconds, dtype=float))
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("invalid D3 clock")
    u = t / DURATIONS[rate]
    s = quintic(u)
    result = np.repeat(np.asarray(start, dtype=float)[None], len(t), axis=0)
    if family == "translation":
        # Excite horizontal and vertical directions in separate halves.
        result[:, 0, 3] += 0.004 * quintic(2 * u)
        result[:, 2, 3] += 0.004 * quintic(2 * u - 1)
        return result
    if family == "curve":
        theta = np.pi * 0.5 * s
        result[:, 0, 3] += 0.005 * np.sin(theta)
        result[:, 2, 3] += 0.005 * (1 - np.cos(theta))
    # Left multiplication rotates about world Y, with translation set separately.
    angle = np.deg2rad(1.75) * s
    rotations = np.repeat(np.eye(3)[None], len(t), axis=0)
    rotations[:, 0, 0] = rotations[:, 2, 2] = np.cos(angle)
    rotations[:, 0, 2] = np.sin(angle)
    rotations[:, 2, 0] = -np.sin(angle)
    result[:, :3, :3] = rotations @ result[:, :3, :3]
    return result


class KnownPath:
    """ONE binds the full trajectory once. P10 recreates the remaining future, in the original frame."""

    def __init__(self, start, family, rate, mode):
        if mode not in ("ONE_SHOT_INTENT", "P10_REVISION"):
            raise ValueError("unknown high-level update mode")
        self.start = np.array(start, dtype=float, copy=True)
        self.family, self.rate, self.mode = family, rate, mode
        self.atom_times = np.arange(81) * 0.05
        self.poses = poses_at(self.start, family, rate, self.atom_times)
        self.revisions = 0
        self.next_install = 0

    def window(self, atom):
        if type(atom) is not int or atom != 2 * self.next_install or not 0 <= atom < 40:
            raise ValueError(
                "D3 requires one fresh 10 Hz installation on the original clock"
            )
        if self.mode == "P10_REVISION" and atom and atom % 10 == 0:
            self.poses[atom + 1 :] = poses_at(
                self.start, self.family, self.rate, self.atom_times[atom + 1 :]
            )
            self.revisions += 1
        self.next_install += 1
        return self.poses[atom : atom + 41].copy()
