"""CPU callback regression against the sealed site receiver; never start a ROS graph."""

import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from clock_guard import ClockGuard, new_receiver_output
from identity import record, sha
from ocs2_msgs.msg import MpcObservation
from shadow import (
    EETrajectoryManager,
    OfflineManager,
    check,
    message,
    native_receiver,
    observe,
    packet,
)
from std_msgs.msg import Empty


class GuardedManager(ClockGuard, OfflineManager):
    pass


@contextmanager
def episode(name, out):
    """Restore the site's class hooks between distinct simulated receiver lifetimes."""
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                os.environ,
                PI0294_EPISODE=name,
                PI0294_PLANNER_TRACE=str(out / "CALLBACKS.private.jsonl"),
            )
        )
        for attr in ("__init__", "on_trajectory_json", "on_observation", "_send"):
            stack.enter_context(
                patch.object(
                    EETrajectoryManager, attr, getattr(EETrajectoryManager, attr)
                )
            )
        lifecycle = stack.enter_context(patch("scripts.wm02.receiver_lifecycle.run"))
        native_receiver.main()
        assert lifecycle.call_count == 1
        yield


def signal(node, stream, t):
    obs = MpcObservation()
    obs.time = t
    obs.mode = 1
    obs.state.value = (
        [0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0] if stream == "ee" else [0.0] * 12
    )
    (node.on_ee_state if stream == "ee" else node.on_observation)(obs)


def issue(node, epoch, t, seq):
    doc = packet(t, seq)
    doc["_pi0294"]["episode"] = epoch
    node.on_trajectory_json(message(doc))


def replay(out):
    receiver_hash = sha(Path(native_receiver.__file__))
    if (
        receiver_hash
        != "d381a1de67f63d5b00ca28130144610374ef1d497054a8815a5c12ee06a544e2"
    ):
        raise ValueError("Receiver differs from B0")
    results = []
    old_output = new_receiver_output(out)
    new_output = new_receiver_output(out)
    assert old_output.name != new_output.name
    with episode(old_output.name, old_output):
        baseline = OfflineManager()
        observe(baseline, 10.0)
        issue(baseline, old_output.name, 10.0, 0)
        observe(baseline, 9.8)
        baseline.publish()
        assert (
            baseline.target_pub[-1].time_trajectory[0] == 9.8 and not baseline._r4_mode
        )
        for stream in ("mpc", "ee"):
            for pending in (False, True):
                for fault in (9.8, float("nan"), float("inf")):
                    node = GuardedManager()
                    observe(node, 10.0)
                    issue(node, old_output.name, 10.0, 0)
                    if pending:
                        issue(node, old_output.name, 10.02, 1)
                        assert node._r4_pending is not None
                    count = len(node.target_pub)
                    previous = node.target_pub[-1]
                    signal(node, stream, fault)
                    # Exercise every way the old path could still emit or reactivate.
                    observe(node, 10.02)
                    issue(node, old_output.name, 10.02, 1)
                    node.on_start_trajectory(Empty())
                    node.publish()
                    node._send(previous)
                    assert len(node.target_pub) == count and [
                        m.data for m in node._r4_mode
                    ] == ["HOLD"]
                    assert (
                        node.trajectory
                        is node.trajectory_source
                        is node._r4_pending
                        is node._r4_deadline
                        is None
                    )
                    assert (
                        node.clock_fault["stream"] == stream
                        and not node.trajectory_started
                    )
                    results.append(
                        {
                            "stream": stream,
                            "pending_before": pending,
                            "stimulus": str(fault),
                            "hold_requests": 1,
                            "targets_after_fault": 0,
                        }
                    )
        node = GuardedManager()
        signal(node, "ee", 9.99)
        signal(node, "mpc", 10.0)
        signal(node, "ee", 9.99)
        issue(node, old_output.name, 10.0, 0)
        assert node.clock_fault is None and len(node.target_pub) == 1
        issue(node, old_output.name, 10.02, 1)
        observe(node, 10.02)
        assert (
            node._r4_pending is None
            and node.bound_start == 10.02
            and len(node.target_pub) == 2
        )
        observe(node, 10.42)
        assert node._r4_mode[-1].data == "HOLD" and not node.trajectory_started
    with episode(new_output.name, new_output):
        new = GuardedManager()
        observe(new, 9.8, mode=0)
        issue(new, new_output.name, 9.8, 0)
        assert not new.target_pub and not new.reference_enabled
        observe(new, 9.82)
        issue(new, old_output.name, 9.82, 0)
        assert not new.target_pub and new._r4_sequence == -1
        issue(new, new_output.name, 9.82, 0)
        target = check(new.target_pub[-1], 7)
        assert (
            len(target) == 41 and new.bound_start == 9.82 and len(new.target_pub) == 1
        )
    record(
        out / "CLOCK_GUARD_ACTUAL.json",
        {
            "receiver_sha256": receiver_hash,
            "guard_sha256": sha(Path(__file__).with_name("clock_guard.py")),
            "unguarded": {"republished_target_s": 9.8, "hold_requests": 0},
            "fault_cases": results,
            "ordinary_callbacks": "equal timestamps, asynchronous streams, pending delivery and TTL preserved",
            "cold_receiver": {
                "old_epoch_accepted": 0,
                "new_epoch_accepted": 1,
                "first_mpc_s": 9.82,
                "assumptions": "Fresh isolated receiver and source streams, simulated disabled then enabled feedback",
            },
            "live_stop_ack": None,
            "post_stop_native_ticks": None,
            "actual_tcp": None,
            "new_native_requests": 0,
            "new_physics_ticks": 0,
        },
    )


if __name__ == "__main__":
    replay(Path(sys.argv[1]))
