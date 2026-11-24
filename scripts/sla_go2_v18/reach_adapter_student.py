"""Paired Reach controllers using the same measured visual goal and frozen receiver."""

import numpy as np
from scipy.spatial.transform import Rotation

from demonstrated_grip import read, require
from compiler import SourceCurve
from reach_forward import predict
from reach_intent_replay import BASE, compile_path
from reach_policy_input import sensor_input
from reach_student import Student
from standing_reach import ReachProposalRejected
from task_space_intent import MAX_ROTATION, ReachEpisode

LIMITS = BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json"


def governor(episode, poses):
    delta = poses[:, :3, 3] - episode.ref[:3, 3]
    axial = delta @ episode.axis
    lateral = np.linalg.norm(delta - axial[:, None] * episode.axis, axis=1)
    rotation = Rotation.from_matrix(
        poses[:, :3, :3] @ episode.ref[:3, :3].T
    ).magnitude()
    if (
        min(axial) < -1e-7
        or max(axial) > episode.cap + 1e-7
        or max(lateral) > episode.radius + 1e-7
        or max(rotation) > MAX_ROTATION + 1e-7
    ):
        raise ReachProposalRejected(
            f"compiled Reach bounds: axial {min(axial) * 1000:.3f}..{max(axial) * 1000:.3f}/{episode.cap * 1000:.3f} mm, "
            f"lateral {max(lateral) * 1000:.3f}/{episode.radius * 1000:.3f} mm, "
            f"rotation {np.degrees(max(rotation)):.3f}/{np.degrees(MAX_ROTATION):.3f} deg"
        )


class AdapterStudent(Student):
    def __init__(self, root, treatment):
        super().__init__(root)
        require(treatment in ("DIRECT", "GEOMETRY"), "unregistered adapter treatment")
        self.treatment = treatment
        self.episode_goal = None
        self.proposal_count = 0

    def ask(self, adapter, tick, receipt):
        case = adapter.output.parent
        self.path = case / "POLICY_OWNER.jsonl"
        obs, info = sensor_input(
            self.root, case, adapter.output, tick, with_visual_goal=True
        )
        require(
            receipt["end"] == tick and receipt["solver_echo"],
            "missing own observed boundary",
        )
        actual = np.asarray(info["actual_tcp"])
        epoch = adapter.output.name
        ref = receipt["end_pose"]
        obj = info["red_rgb"]
        try:
            if self.episode_goal is None:
                self.episode_goal = ReachEpisode(epoch, actual, ref, obj)
            # Check measured and consumed bounds before spending a new forward.
            self.episode_goal.plan(epoch, actual, ref, obj, method="GEOMETRY")
        except ValueError as e:
            raise ReachProposalRejected(str(e)) from e
        raw = None
        curve = None
        if self.treatment == "DIRECT":
            raw = predict(self.root, case, tick, self.calls, obs, info)
            curve = SourceCurve(
                raw, np.asarray(info["world_origin"]), np.asarray(info["link6_tcp"])
            )
            self.calls += 1
            self.log(
                "RETURN",
                tick=tick,
                consumed=receipt["consumed"],
                input=info["input_sha256"],
                sensor=info,
            )
        self.post_motion = (
            self.motion is not None and self.last_origin != info["input_sha256"]
        )
        try:
            path = self.episode_goal.plan(
                epoch,
                actual,
                ref,
                obj,
                method=self.treatment,
                model_world=None if curve is None else curve.knots,
            )
        except ValueError as e:
            raise ReachProposalRejected(str(e)) from e
        limits = read(LIMITS)["limits"]
        # Reach executes a fixed open command. Widths remain in the archived official return.
        self.compiled = compile_path(path, None, ref, receipt["start_pose"], limits)
        self.reference_tick = tick
        self.proposal_count += 1
        self.last_origin = info["input_sha256"]
        self.first_source = None
        self.log(
            "VISUAL_REACH",
            tick=tick,
            phase=self.proposal_count,
            profile=self.treatment,
            input=info["input_sha256"],
            sensor=info,
            lead_s=self.compiled.lead,
            goal_axis=self.episode_goal.axis.tolist(),
            goal_cap_mm=1000 * self.episode_goal.cap,
            goal_radius_mm=1000 * self.episode_goal.radius,
            post_own_motion=self.post_motion,
        )

    def preview(self, adapter, tick, receipt, sample, monitor, anchor, direction):
        require(monitor.last == receipt["end"] == tick, "stale observation")
        if self.compiled is None or (
            self.proposal_count == 1 and self.motion is not None
        ):
            self.ask(adapter, tick, receipt)
        elapsed = (tick - self.reference_tick) * 0.002
        poses, _, phase = self.compiled.at(elapsed + np.arange(41) * 0.05)
        poses[-1] = poses[-2]
        governor(self.episode_goal, poses)
        self.log("ADAPTED_PREVIEW", tick=tick, source_phase_s=phase[:3].tolist())
        return poses

    def summary(self):
        return {
            **super().summary(),
            "treatment": self.treatment,
            "post_motion_forward": bool(self.post_motion and self.calls == 2),
            "post_motion_geometric_update": bool(
                self.post_motion and self.treatment == "GEOMETRY"
            ),
            "visual_replans": self.proposal_count,
        }
