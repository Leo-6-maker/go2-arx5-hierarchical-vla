"""Selected only for the two registered, simulator-only technical cards."""

import os
import time
from pathlib import Path

from clock_guard import ClockGuard
from identity import read, sha
from ocs2_msgs.msg import MpcObservation
from rclpy.clock import Clock, ClockType
from scripts.wm02 import receiver_lifecycle

from scripts.pi0292.durable import write
from scripts.pi0294 import native_receiver


def run_guarded(ctx, factory, lifecycle):
    out = Path(os.environ["PI0294_PLANNER_TRACE"]).parent
    epoch = os.environ["PI0294_EPISODE"]
    if (
        epoch != out.name
        or sha(Path(native_receiver.__file__)) != os.environ["B2_RECEIVER_PIN"]
    ):
        raise ValueError("receiver identity mismatch")

    class Receiver(ClockGuard, factory):
        def __init__(self):
            super().__init__()
            if os.environ["B2_CARD"] == "T1_CLOCK_FAULT_STOP":
                self.b2_timer = self.create_timer(
                    0.05,
                    self.check_fault_request,
                    clock=Clock(clock_type=ClockType.STEADY_TIME),
                )
            write(out / "GUARD_READY.json", {"episode": epoch, "pid": os.getpid()})

        def _accept_clock(self, stream, now):
            previous = self.clock_fault
            ok = super()._accept_clock(stream, now)
            if previous is None and self.clock_fault is not None:
                write(
                    out / "GUARD_TERMINAL.json",
                    {
                        "episode": epoch,
                        "fault": self.clock_fault,
                        "hold_request_published": True,
                        "monotonic_s": time.monotonic(),
                        "reference_invalidated": self.trajectory is None
                        and self._r4_pending is None
                        and not self.reference_enabled
                        and not self.trajectory_started,
                    },
                )
            return ok

        def check_fault_request(self):
            path = out / "B2_REWIND_REQUEST.json"
            if self.clock_fault is not None or not path.exists():
                return
            req = read(path)
            if req["episode"] != epoch or req["step"] != 6678:
                raise ValueError("unmatched clock fault request")
            obs = MpcObservation()
            obs.time = self.last_clock["mpc"] - 0.1
            self.on_observation(obs)

    return lifecycle(ctx, Receiver)


def main():
    original = receiver_lifecycle.run
    receiver_lifecycle.run = lambda ctx, factory: run_guarded(ctx, factory, original)
    native_receiver.main()


if __name__ == "__main__":
    main()
