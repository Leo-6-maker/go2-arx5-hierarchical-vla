"""Terminal clock-fault handling for a single, serially dispatched receiver episode."""

import math
from uuid import uuid4

from std_msgs.msg import String


def new_receiver_output(parent):
    """The site uses the output basename as the wire episode at both endpoints."""
    output = parent / ("native-" + uuid4().hex)
    output.mkdir(mode=0o700, exist_ok=False)
    return output


class ClockGuard:
    """Compose ahead of the site's initialized receiver. Each instance has one epoch."""

    def __init__(self):
        self.last_clock = {}
        self.clock_fault = None
        super().__init__()

    def _accept_clock(self, stream, now):
        if self.clock_fault is not None:
            return False
        previous = self.last_clock.get(stream)
        if math.isfinite(now) and (previous is None or now >= previous):
            self.last_clock[stream] = now
            return True

        self.clock_fault = {
            "stream": stream,
            "previous_s": previous,
            "observed_s": now if math.isfinite(now) else None,
            "reason": "rewind" if math.isfinite(now) else "nonfinite",
        }
        # Invalidate before requesting HOLD. If delivery fails, this instance
        # remains terminal, including queued publication and pending callbacks.
        self.reference_enabled = self.trajectory_started = False
        self.trajectory = self.trajectory_source = self.bound_start = None
        self._r4_pending = self._r4_deadline = self._r4_ref = None
        self.ee_state = self.ee_time = self.observation = None
        self.goal = self.base_goal = None
        self._r4_mode.publish(String(data="HOLD"))
        return False

    def on_observation(self, msg):
        if self._accept_clock("mpc", float(msg.time)):
            super().on_observation(msg)

    def on_ee_state(self, msg):
        if self._accept_clock("ee", float(msg.time)):
            super().on_ee_state(msg)

    def on_trajectory_json(self, msg):
        if self.clock_fault is None:
            super().on_trajectory_json(msg)

    def _send(self, msg):
        if self.clock_fault is None:
            super()._send(msg)
