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
        self,
        previous,
        previous_hash,
        next_hash,
        last_install,
        source,
        limits,
        *,
        cruise_alignment=False,
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
        if isinstance(previous, RevisionWire) and (
            previous.next_hash != previous_hash or self.cut <= previous.cut
        ):
            raise ValueError("revision must extend the currently installed source")
        boundary, widths, _ = previous.at(DT * np.array([self.cut - 1, self.cut]))
        twist = np.stack([axis[0] for axis in vectors(boundary, DT)])
        self.compiler = Compiler(
            source,
            boundary[-1],
            twist,
            limits,
            cruise_from_width_m=(0.002 + 0.040 * widths[-1])
            if cruise_alignment
            else None,
        )
        ticks = np.arange(round(self.compiler.end / DT) + 1)
        self.future = WireProgram(self.identity, *self.compiler.at(ticks * DT))
        self.execution_times = [0, self.cut * DT + self.compiler.end]

    def _sections(self, t):
        """Route each global knot to the original source, including older protected prefixes."""
        t = np.atleast_1d(np.asarray(t, float))
        if t.ndim != 1 or not np.isfinite(t).all():
            raise ValueError("invalid revision clock")
        knots = np.rint(t / DT).astype(int)
        if np.any(knots < 0) or np.any(abs(t - DT * knots) > 1e-8):
            raise ValueError("invalid wire clock")
        remaining = np.arange(len(t))
        plan = self
        while len(remaining) and isinstance(plan, RevisionWire):
            future = knots[remaining] > plan.cut
            index = remaining[future]
            if len(index):
                yield index, plan.future, (knots[index] - plan.cut) * DT, plan.next_hash
            remaining = remaining[~future]
            previous_hash = plan.prior_hash
            plan = plan.previous
        if len(remaining):
            yield remaining, plan, t[remaining], previous_hash

    def at(self, t):
        n = len(np.atleast_1d(t))
        pose, width, phase = np.empty((n, 4, 4)), np.empty(n), np.empty(n)
        for index, program, local, _ in self._sections(t):
            pose[index], width[index], phase[index] = program.at(local)
        return pose, width, phase

    def check_gripper_handoff(self):
        """Fail the join if it invents a source event or exceeds the target G7 speed."""
        _, width, _ = self.at(DT * np.array([self.cut, self.cut + 1]))
        before, after = width
        if before > 0.55 >= after or before < 0.65 <= after:
            raise ValueError("unattributed gripper event at revision")
        if abs(after - before) * 0.040 / DT > 0.040 + 1e-10:
            raise ValueError("revision gripper slew above 40 mm/s")

    def source_hash_at(self, t):
        sources = np.empty(len(np.atleast_1d(t)), dtype=object)
        for index, _, _, source_hash in self._sections(t):
            sources[index] = source_hash
        return sources.astype(str)


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
