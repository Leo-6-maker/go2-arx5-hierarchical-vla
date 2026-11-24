"""Own one frozen-policy Reach request and distinguish alignment from consumed source motion."""

import json
from pathlib import Path

import numpy as np
from compiler import Compiler, SourceCurve, vectors
from demonstrated_grip import read, require
from reach_forward import predict
from reach_policy_input import sensor_input
from scipy.spatial.transform import Rotation


def scope_metrics(poses, origin, direction):
    shift = (poses[:, :3, 3] - origin[:3, 3]) * 1000
    phase = shift @ direction
    lateral = np.linalg.norm(shift - phase[:, None] * direction, axis=1)
    rotation = np.degrees(
        Rotation.from_matrix(poses[:, :3, :3] @ origin[:3, :3].T).magnitude()
    )
    return {
        "cross_track_mm": float(max(lateral)),
        "end_overrun_mm": float(max(0, max(phase) - 20, -min(phase))),
        "attitude_deg": float(max(rotation)),
    }


class Student:
    def __init__(self, root):
        self.root = root
        self.compiled = None
        self.reference_tick = None
        self.calls = 0
        self.motion = None
        self.first_source = None
        self.last_source_tick = None
        self.post_motion = False
        self.refusal = None
        self.path = None
        self.prev_output = None
        self.last_origin = None

    def log(self, kind, **payload):
        with self.path.open("a") as fp:
            fp.write(json.dumps(dict(kind=kind, **payload), allow_nan=False) + "\n")

    def ask(self, adapter, tick, receipt):
        case = adapter.output.parent
        self.path = case / "POLICY_OWNER.jsonl"
        obs, info = sensor_input(self.root, case, adapter.output, tick)
        require(
            receipt["end"] == tick and receipt["solver_echo"],
            "unverified observation boundary",
        )
        raw = predict(self.root, case, tick, self.calls, obs, info)
        origin, tool = map(np.asarray, (info["world_origin"], info["link6_tcp"]))
        curve = SourceCurve(raw, origin, tool)
        delta = None
        if self.prev_output is not None:
            old_rebound = SourceCurve(self.prev_output, origin, tool)
            delta = float(
                1000
                * np.linalg.norm(
                    curve.knots[:, :3, 3] - old_rebound.knots[:, :3, 3], axis=1
                ).mean()
            )
            self.post_motion = (
                self.motion is not None
                and self.motion["tick"] <= tick
                and self.last_origin != info["input_sha256"]
            )
        self.calls += 1
        self.log(
            "RETURN",
            tick=tick,
            consumed=receipt["consumed"],
            input=info["input_sha256"],
            sensor=info,
            same_state_old_output_mean_difference_mm=delta,
            post_motion=self.post_motion,
        )
        limits = read(
            Path(__file__).resolve().parents[2]
            / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json"
        )["limits"]
        velocity, angular = vectors(
            np.asarray([receipt["start_pose"], receipt["end_pose"]]), 0.05
        )
        self.compiled = Compiler(
            curve,
            receipt["end_pose"],
            (velocity[0], angular[0]),
            limits,
            cruise_from_width_m=0.042,
        )
        self.reference_tick = tick
        self.last_origin = info["input_sha256"]
        self.prev_output = raw
        self.first_source = None
        self.log(
            "COMPILED",
            tick=tick,
            lead_s=self.compiled.lead,
            raw_working_width_min_m=float(0.002 + 0.040 * min(curve.arm[:, 7])),
            raw_working_width_max_m=float(0.002 + 0.040 * max(curve.arm[:, 7])),
            open_gripper_m=0.042,
        )

    def preview(self, adapter, tick, receipt, sample, monitor, anchor, direction):
        require(monitor.last == tick and receipt["end"] == tick, "stale paused state")
        if self.compiled is None or (self.calls == 1 and self.motion is not None):
            self.ask(adapter, tick, receipt)
        elapsed = (tick - self.reference_tick) * 0.002
        poses, _, phase = self.compiled.at(elapsed + np.arange(41) * 0.05)
        poses[-1] = poses[-2]
        observed = scope_metrics(poses, anchor, direction)
        self.log(
            "PREVIEW",
            tick=tick,
            source_s=phase[:3].tolist(),
            start=poses[0, :3, 3].tolist(),
            task_geometry=observed,
        )
        return poses

    def consumed(self, tick, receipt, sample):
        if self.compiled is None:
            return
        elapsed = (tick - self.reference_tick) * 0.002
        times = elapsed + np.asarray([-0.05, 0.0])
        _, _, phase = self.compiled.at(times)
        if phase[1] <= phase[0] or times[0] < self.compiled.lead - 1e-9:
            return
        actual = sample["tcp"][:3, 3]
        if self.first_source is None:
            self.first_source = (tick, actual.copy(), receipt["end_pose"][:3, 3].copy())
            return
        first, p0, r0 = self.first_source
        delta = receipt["end_pose"][:3, 3] - r0
        distance = float(1000 * np.linalg.norm(delta))
        if distance <= 1:
            return
        progress = float(1000 * (actual - p0) @ delta / np.linalg.norm(delta))
        if progress >= 1 and self.motion is None:
            self.motion = {
                "start": first,
                "tick": tick,
                "actual_projection_mm": progress,
                "installed_mm": distance,
                "consumed": receipt["consumed"],
            }
            self.log("SOURCE_MOTION", **self.motion)
        self.last_source_tick = tick

    def summary(self):
        return {
            "accepted_returns": self.calls,
            "source_motion": self.motion,
            "post_motion_forward": self.post_motion,
            "refusal": self.refusal,
        }
