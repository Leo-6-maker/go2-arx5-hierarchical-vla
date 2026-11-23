"""Invoke the archived and site planner callbacks with serialized ROS messages, without a ROS graph."""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from go2_x5_planning import ee_trajectory, ee_trajectory_node
from go2_x5_planning.ee_trajectory_node import EETrajectoryManager
from identity import record, sha
from ocs2_msgs.msg import MpcObservation, MpcTargetTrajectories
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
from std_msgs.msg import String

from scripts.pi0294 import native_receiver


class Sink(list):
    def publish(self, message):
        self.append(deserialize_message(serialize_message(message), type(message)))


class OfflineManager(EETrajectoryManager):
    """Only the ROS graph plumbing and RViz display are replaced."""

    def __init__(self):
        self.params, self.qos = {}, {}
        with patch.object(Node, "__init__", lambda *_: None):
            super().__init__()

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def create_publisher(self, msg_type, topic, qos):
        self.qos[topic] = str(qos)
        return Sink()

    def create_subscription(self, *args):
        pass

    def create_timer(self, *args):
        pass

    def publish_planned_path(self):
        pass

    def get_logger(self):
        return logging.getLogger("offline")


def observe(node, time, x=0.3, mode=1, quat=(0.0, 0.0, 0.0, 1.0)):
    state = MpcObservation()
    state.time, state.mode = float(time), mode
    state.state.value = [float(x), 0.0, 0.5, *quat]
    node.on_ee_state(state)
    obs = MpcObservation()
    obs.time = float(time)
    obs.state.value = [0.0] * 12
    node.on_observation(obs)


def part(times, origin=0.3, frame="world", speed=0.01):
    return {
        "frame": frame,
        "waypoints": [
            {
                "time": t,
                "position": [origin + speed * t, 0.0, 0.5 if frame == "world" else 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            for t in times
        ],
    }


def message(doc):
    return String(data=json.dumps(doc, allow_nan=False))


def check(msg, width):
    assert isinstance(msg, MpcTargetTrajectories)
    t = np.array(msg.time_trajectory)
    assert len(t) == len(msg.state_trajectory) == len(msg.input_trajectory)
    assert np.all(np.diff(t) > 0)
    assert all(
        len(v.value) == width and np.isfinite(v.value).all()
        for v in msg.state_trajectory
    )
    assert all(len(v.value) == 12 and not any(v.value) for v in msg.input_trajectory)
    return np.array([v.value for v in msg.state_trajectory], dtype=np.float64)


def baseline():
    path = {"ee": part([0.0, 4.0])}
    one, repeat = OfflineManager(), OfflineManager()
    details = []
    for k in range(5):
        t = 10.0 + 0.5 * k
        for node in (one, repeat):
            observe(node, t, 0.3 + 0.005 * k)
            if node is repeat or k == 0:
                node.on_trajectory_json(message(path))
            node.publish()
        left, right = one.target_pub[-1], repeat.target_pub[-1]
        a, b = check(left, 7), check(right, 7)
        assert (
            len(a) == len(b) == 41
            and left.time_trajectory[0] == t
            and left.time_trajectory[-1] == t + 2
        )
        details.append(
            {
                "t": t,
                "bound_one": one.bound_start,
                "bound_repeat": repeat.bound_start,
                "first_x_m": [a[0, 0], b[0, 0]],
                "first_v_m_s": [(a[1, 0] - a[0, 0]) / 0.05, (b[1, 0] - b[0, 0]) / 0.05],
                "horizon_x_m": [a[-1, 0], b[-1, 0]],
            }
        )
    node = OfflineManager()
    observe(node, 10.0)
    node.on_trajectory_json(
        message({"ee": part([0.0, 0.373, 3.0]), "base": part([0.0, 3.0], origin=0.0)})
    )
    node.publish()
    corner = node.target_pub[-1]
    assert len(check(corner, 14)) == 42 and 10.373 in corner.time_trajectory
    previous = node.trajectory_source
    node.on_trajectory_json(message({"ee": part([])}))
    assert node.trajectory_source is previous
    observe(node, 11.0, mode=0)
    node.publish()
    assert len(check(node.target_pub[-1], 7)) == 1 and not node.trajectory_started
    observe(node, 11.1)
    size = len(node.target_pub)
    node.publish()
    assert len(node.target_pub) == size
    # Independent frame landmarks. The rotated EE-local offset is +Y; world_offset is +X.
    frame_landmarks = {}
    for frame, expected in (
        ("world", [0.1, 0.0, 0.5]),
        ("world_offset", [0.4, 0.0, 0.5]),
        ("ee_start", [0.3, 0.1, 0.5]),
    ):
        node = OfflineManager()
        observe(node, 10.0, quat=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
        node.on_trajectory_json(
            message({"ee": part([0.0, 3.0], origin=0.1, frame=frame, speed=0.0)})
        )
        node.publish()
        actual = check(node.target_pub[-1], 7)[-1, :3]
        assert np.allclose(actual, expected, atol=1e-6)
        frame_landmarks[frame] = actual.tolist()
    observe(one, 9.0)
    one.publish()
    assert one.target_pub[-1].time_trajectory[0] == 9.0
    return {
        "one_vs_full_resend": details,
        "regular_points": 41,
        "extra_knot_points": 42,
        "extra_knot_s": 10.373,
        "archive_mode0_hold_points": 1,
        "empty_replacement": "prior retained",
        "archive_backward_clock_first_time": 9.0,
        "frame_landmarks": frame_landmarks,
        "params": {
            k: v
            for k, v in one.params.items()
            if k
            in (
                "publish_rate",
                "window_horizon",
                "sample_dt",
                "target_topic",
                "input_dim",
                "observation_topic",
                "ee_state_topic",
            )
        },
    }


def packet(t, seq, origin=0.3, base=False):
    grid = [i * 0.05 for i in range(41)]
    doc = {
        "ee": part(grid, origin),
        "_pi0294": {
            "episode": "offline",
            "source_mpc_s": t,
            "sequence": seq,
            "ttl_s": 0.4,
            "operation": "install",
        },
    }
    if base:
        doc["base"] = part(grid, 0.0, speed=0.0)
    return doc


def rolling_intent():
    """A ten-second world path, with independent policy and transport clocks."""
    results = {}
    one_shot_windows = []
    max_field_error = 0.0
    differing_byte_offsets = set()
    unequal_typed_payloads = 0
    for label in ("one_intent", "remaining_path_at_P10"):
        node = OfflineManager()
        source_t = np.array([0.0, 10.0])
        source_x = np.array([0.3, 0.4])
        digest = hashlib.sha256()
        endpoints = []
        seam_error = []
        for i in range(101):
            phase = i / 10
            t = 20.0 + phase
            x = float(np.interp(phase, source_t, source_x))
            observe(node, t, x)
            old = (
                None
                if node.bound_start is None
                else node.trajectory.window(t, 2.0, 0.05)[1][0].position[0]
            )
            if label == "remaining_path_at_P10" and i % 5 == 0 and i < 100:
                source_t = np.array([phase, 10.0])
                source_x = np.array([x, 0.4])
            doc = packet(t, i)
            for point in doc["ee"]["waypoints"]:
                point["position"][0] = float(
                    np.interp(phase + point["time"], source_t, source_x)
                )
            node.on_trajectory_json(message(doc))
            msg = node.target_pub[-1]
            values = check(msg, 7)
            assert (
                node.bound_start == t
                and not node._r4_mode
                and np.isclose(values[0, 0], x)
            )
            wire = serialize_message(msg)
            digest.update(wire)
            typed = (
                np.asarray(msg.time_trajectory, dtype="<f8").tobytes()
                + np.asarray(values, dtype="<f4").tobytes()
                + np.asarray(
                    [v.value for v in msg.input_trajectory], dtype="<f4"
                ).tobytes()
            )
            if label == "one_intent":
                one_shot_windows.append(
                    (list(msg.time_trajectory), values, wire, typed)
                )
            else:
                old_times, old_values, old_wire, old_typed = one_shot_windows[i]
                assert (
                    old_times == list(msg.time_trajectory)
                    and old_values.shape == values.shape
                )
                max_field_error = max(
                    max_field_error, float(np.abs(old_values - values).max())
                )
                unequal_typed_payloads += old_typed != typed
                differing_byte_offsets.update(
                    j
                    for j, (a, b) in enumerate(zip(old_wire, wire, strict=True))
                    if a != b
                )
            endpoints.append(float(values[0, 0]))
            if old is not None:
                seam_error.append(abs(old - values[0, 0]))
        results[label] = {
            "windows": len(endpoints),
            "start_x_m": endpoints[0],
            "end_x_m": endpoints[-1],
            "transport_gap_s": 0.1,
            "max_seam_delta_m": max(seam_error),
            "wire_sha256": digest.hexdigest(),
            "hold_requests": len(node._r4_mode),
        }
    results["identical_serialized_byte_streams"] = (
        results["one_intent"]["wire_sha256"]
        == results["remaining_path_at_P10"]["wire_sha256"]
    )
    results["max_abs_state_component_difference"] = max_field_error
    results["unequal_typed_payloads"] = unequal_typed_payloads
    results["differing_CDR_byte_offsets"] = sorted(differing_byte_offsets)
    assert unequal_typed_payloads == 0 and max_field_error == 0.0
    return results


def site(out):
    trace = out / "B1_SHADOW_CALLBACKS.private.jsonl"
    os.environ.update(PI0294_EPISODE="offline", PI0294_PLANNER_TRACE=str(trace))
    # Run the site's exact class-patching entry point, stopping before rclpy.init.
    with patch("scripts.wm02.receiver_lifecycle.run") as lifecycle:
        native_receiver.main()
        assert (
            lifecycle.call_count == 1
            and lifecycle.call_args.args[1] is EETrajectoryManager
        )
    node = OfflineManager()
    observe(node, 10.0)
    node.on_trajectory_json(message(packet(10.0, 0)))
    assert node.bound_start == 10.0 and len(check(node.target_pub[-1], 7)) == 41
    observe(node, 10.399)
    assert node.trajectory_started
    observe(node, 10.4)
    assert not node.trajectory_started and node._r4_mode[-1].data == "HOLD"
    # Shorter-than-TTL rolling updates are an offline control, not a changed safety budget.
    node = OfflineManager()
    compare = []
    for i in range(6):
        t = 10.0 + 0.2 * i
        observe(node, t, 0.3 + 0.01 * (t - 10))
        previous = node.trajectory
        old_at_t = (
            None
            if node.bound_start is None
            else previous.window(t, 2.0, 0.05)[1][0].as_state()[0]
        )
        node.on_trajectory_json(message(packet(t, i, 0.3 + 0.01 * (t - 10), base=True)))
        points = check(node.target_pub[-1], 14)
        assert node.bound_start == t and np.isclose(points[0, 0], 0.3 + 0.01 * (t - 10))
        compare.append(
            {
                "time": t,
                "old_x_m": old_at_t,
                "new_x_m": points[0, 0],
                "first_v_m_s": (points[1, 0] - points[0, 0]) / 0.05,
            }
        )
    assert not node._r4_mode and all(
        abs(r["old_x_m"] - r["new_x_m"]) < 1e-6 for r in compare[1:]
    )
    cases = {}
    for case in (
        "long_path",
        "relative_frame",
        "stale",
        "duplicate_sequence",
        "disabled",
        "future_observation",
        "clock_backwards",
    ):
        node = OfflineManager()
        observe(node, 10.0)
        node.on_trajectory_json(message(packet(10.0, 0)))
        doc = packet(10.1, 1)
        observe(node, 10.1, mode=0 if case == "disabled" else 1)
        if case == "long_path":
            doc["ee"] = part([0.0, 1.0, 10.0])
        if case == "relative_frame":
            doc["ee"]["frame"] = "ee_start"
        if case == "stale":
            doc["_pi0294"]["source_mpc_s"] = 10.0
        if case == "duplicate_sequence":
            doc["_pi0294"]["sequence"] = 0
        if case == "future_observation":
            doc["_pi0294"]["source_mpc_s"] = 10.12
        if case == "clock_backwards":
            observe(node, 9.8)
            node.publish()
            cases[case] = {
                "bound": node.bound_start,
                "first_time": node.target_pub[-1].time_trajectory[0],
                "hold": bool(node._r4_mode),
            }
            continue
        node.on_trajectory_json(message(doc))
        if case == "future_observation":
            assert node._r4_pending is not None
            observe(node, 10.12)
            assert node.bound_start == 10.12
            cases[case] = "pending then accepted at source clock"
        else:
            assert not node.trajectory_started and node._r4_mode[-1].data == "HOLD"
            cases[case] = "rejected; native HOLD requested"
    return {
        "expiry_s": 10.4,
        "next_P10_s": 10.5,
        "nominal_gap_after_expiry_s": 0.1,
        "continuous_200ms_reference": compare,
        "cases": cases,
        "ten_second_rolling": rolling_intent(),
        "scope": "CPU callback HOLD request; no actual mode change, ROS graph or physics",
    }


def main(out, abi_file):
    archive = baseline()
    core = site(out)
    sources = {
        Path(m.__file__).name: sha(Path(m.__file__))
        for m in (ee_trajectory, ee_trajectory_node, native_receiver)
    }
    expected = json.loads(abi_file.read_text())
    for filename in ("ee_trajectory.py", "ee_trajectory_node.py"):
        assert expected[filename] == sources[filename]
    record(
        out / "B1_OCS2_ABI_WINDOW_AND_REPLACE.json",
        {
            "status": "BACKEND_ABI_BLOCKED_FOR_LONG_PATH_OR_P10_ONLY",
            "archive": archive,
            "site": core,
            "sources": sources,
            "serialization": "ROS generated CDR serialized/deserialized at every output sink",
            "matched_archive_files": expected,
            "native_requests": 0,
            "native_ticks": 0,
            "actual_tcp": None,
            "solver_calls": 0,
        },
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]), Path(sys.argv[2]))
