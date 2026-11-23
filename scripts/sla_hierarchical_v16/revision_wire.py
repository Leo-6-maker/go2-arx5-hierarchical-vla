"""Append a distinct prediction after the last protected wire knot, in CPU shadow."""

import math

import numpy as np

from compiler import Compiler, DT, vectors
from prefix_replay import WireProgram


def first_publication(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid return clock")
    return math.ceil(seconds * 10 - 1e-10)


class RevisionWire:
    """The epoch identifies the episode. Each segment keeps its own prediction and phase."""

    @staticmethod
    def protected_knot(last_install):
        return 2 * last_install + 41

    def __init__(
        self, previous, previous_hash, next_hash, last_install, source, limits
    ):
        if last_install < 0 or type(last_install) is not int:
            raise ValueError("missing prior installation")
        if not previous_hash or not next_hash or previous_hash == next_hash:
            raise ValueError("distinct identified returns required")
        self.identity = previous.identity
        self.previous = previous
        self.prior_hash = previous_hash
        self.next_hash = next_hash
        # Each 10 Hz installation commits 41 points, plus one 50 ms join segment.
        self.cut = self.protected_knot(last_install)
        boundary = previous.at(DT * np.array([self.cut - 1, self.cut]))[0]
        twist = np.stack([axis[0] for axis in vectors(boundary, DT)])
        self.compiler = Compiler(source, boundary[-1], twist, limits)
        ticks = np.arange(round(self.compiler.end / DT) + 1)
        self.future = WireProgram(self.identity, *self.compiler.at(ticks * DT))
        self.execution_times = [0, self.cut * DT + self.compiler.end]

    def at(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        if t.ndim != 1 or not np.isfinite(t).all():
            raise ValueError("invalid revision clock")
        original = self.previous.at(t)
        k = np.rint(t / DT).astype(int) - self.cut
        tail = self.future.at(np.maximum(k, 0) * DT)
        return tuple(
            np.where((k > 0).reshape((-1,) + (1,) * (old.ndim - 1)), new, old)
            for old, new in zip(original, tail, strict=True)
        )

    def check_gripper_handoff(self):
        """Fail the join if it invents a source event or exceeds the target G7 speed."""
        _, width, _ = self.at(DT * np.array([self.cut, self.cut + 1]))
        before, after = width
        if before > 0.55 >= after or before < 0.65 <= after:
            raise ValueError("unattributed gripper event at revision")
        if abs(after - before) * 0.040 / DT > 0.040 + 1e-10:
            raise ValueError("revision gripper slew above 40 mm/s")

    def source_hash_at(self, t):
        # at() enforces clock alignment. Use integer knots at the floating-point boundary.
        t = np.atleast_1d(np.asarray(t, float))
        self.previous.at(t)
        return np.where(np.rint(t / DT) > self.cut, self.next_hash, self.prior_hash)


class NewSourcePhaseView:
    """The V1 transport checker scores progress of one source. Old knots contribute zero."""

    def __init__(self, revision):
        self.revision = revision
        self.identity = revision.identity
        self.execution_times = revision.execution_times

    def at(self, t):
        pose, grip, phase = self.revision.at(t)
        eligible = self.revision.source_hash_at(t) == self.revision.next_hash
        return pose, grip, np.where(eligible, phase, 0.0)
