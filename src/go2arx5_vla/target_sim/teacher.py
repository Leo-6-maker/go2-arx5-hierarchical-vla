"""Object-relative teacher: commands EEF and gripper through the frozen WBC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix
from .scorer import finite_array, physical_grasp, pose_matrix


@dataclass(frozen=True)
class TeacherCommand:
    tcp_world: np.ndarray
    gripper_fraction: float
    stage_before: str
    stage_after: str
    time_s: float
    failed: bool
    failure_reason: str | None


class Teacher:
    """One continuous episode; no retries, resets, or object attachments.

    ``object_to_tcp`` is the geometry/calibration-derived grasp transform.
    It is explicit because the Go2-X5's grasp frame must not be inferred from
    a Panda pose. Workspace and orientation envelope freeze before pilots.
    Physical grasp must be calibrated before any quality run; this class
    performs no simulation or policy inference itself.
    """

    stages = ("approach", "align", "close", "lift", "carry", "lower", "release", "withdraw", "hold")

    def __init__(self, initial_tcp, initial_object_pose, goal_center, object_to_tcp,
                 workspace_min, workspace_max, *, linear_speed_m_s=.15,
                 angular_speed_rad_s=1., orientation_limit_rad=math.pi,
                 clearance_m=.08, approach_m=.05, withdraw_m=.10):
        self.command = pose_matrix(finite_array(initial_tcp, (4, 4), "initial_tcp"))
        self.initial_tcp = self.command.copy()
        self.initial_object = pose_matrix(finite_array(initial_object_pose, (4, 4), "initial_object_pose"))
        self.object_to_tcp = pose_matrix(finite_array(object_to_tcp, (4, 4), "object_to_tcp"))
        self.goal = finite_array(goal_center, (3,), "goal_center")
        self.minimum = finite_array(workspace_min, (3,), "workspace_min")
        self.maximum = finite_array(workspace_max, (3,), "workspace_max")
        values = (linear_speed_m_s, angular_speed_rad_s, orientation_limit_rad, clearance_m, approach_m, withdraw_m)
        if not np.isfinite(values).all() or min(values) <= 0 or np.any(self.minimum >= self.maximum):
            raise ValueError("teacher envelope and speed limits must be finite and positive")
        if orientation_limit_rad > math.pi or clearance_m < .02 or withdraw_m < .08:
            raise ValueError("invalid orientation/lift/withdraw envelope")
        if np.linalg.norm(self.goal[:2] - self.initial_object[:2, 3]) < .15:
            raise ValueError("source-goal separation must be at least 0.15 m")
        self.linear_speed = float(linear_speed_m_s)
        self.angular_speed = float(angular_speed_rad_s)
        self.orientation_limit = float(orientation_limit_rad)
        self.clearance = float(clearance_m)
        self.approach = float(approach_m)
        self.withdraw = float(withdraw_m)
        self.stage = "approach"
        self.stage_start = 0.
        self.ready_since = None
        self.last_time = None
        self.gripper = 1.
        self.failed = False
        self.failure_reason = None
        self.events = []
        self.release_tcp = None
        self.terminal_tcp = None
        self._check_target(self.command)

    def _check_target(self, target):
        if (np.any(target[:3, 3] < self.minimum) or np.any(target[:3, 3] > self.maximum)
                or geodesic_angle(self.initial_tcp[:3, :3], target[:3, :3]) > self.orientation_limit + 1e-8):
            raise ValueError("teacher target outside frozen pose envelope")

    def _ready(self, condition, timestamp, duration=.10):
        if not condition:
            self.ready_since = None
            return False
        if self.ready_since is None:
            self.ready_since = timestamp
        return timestamp - self.ready_since >= duration - 1e-8

    def _transition(self, stage, timestamp):
        self.events.append({"time_s": timestamp, "stage_before": self.stage, "stage_after": stage})
        self.stage, self.stage_start, self.ready_since = stage, timestamp, None

    def query(self, time_s, feedback):
        timestamp = float(time_s)
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > 32 + 1e-8:
            raise ValueError("teacher time must be within complete 0..32 second episode")
        if self.last_time is None:
            if abs(timestamp) > 1e-8:
                raise ValueError("first teacher query must be at task t=0")
            dt = .05
        else:
            dt = timestamp - self.last_time
            if abs(dt - .05) > 1e-8:
                raise ValueError("teacher query requires contiguous 20Hz feedback")
        tcp = pose_matrix(finite_array(feedback["tcp_world"], (4, 4), "tcp_world"))
        obj = pose_matrix(finite_array(feedback["object_pose"], (4, 4), "object_pose"))
        grasp = bool(physical_grasp(feedback))
        for field in ("safety_valid", "native_abort", "illegal_contact", "joint_limit_penetration"):
            if not isinstance(feedback[field], (bool, np.bool_)):
                raise ValueError(f"{field} must be a boolean physical predicate")
        opening = float(feedback["gripper_fraction"])
        if not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("actual gripper fraction must be valid")
        linear = finite_array(feedback["object_linear_velocity"], (3,), "object_linear_velocity")
        angular = finite_array(feedback["object_angular_velocity"], (3,), "object_angular_velocity")
        before = self.stage
        if not feedback["safety_valid"] or any(feedback[field] for field in ("native_abort", "illegal_contact", "joint_limit_penetration")):
            self.failed, self.failure_reason = True, "NATIVE_SAFETY_STOP"
        elif not all(feedback[field] for field in ("object_present", "contact_valid", "gripper_state_valid")):
            self.failed, self.failure_reason = True, "PHYSICAL_SENSOR_INVALID"
        elif before in ("lift", "carry", "lower") and not grasp:
            self.failed, self.failure_reason = True, "PHYSICAL_GRASP_LOST"
        elif before == "close" and timestamp - self.stage_start > 2.:
            self.failed, self.failure_reason = True, "PHYSICAL_GRASP_TIMEOUT"
        target = obj @ self.object_to_tcp
        gripper = 1.
        if before == "approach":
            target[2, 3] += self.approach
        elif before in ("close", "lift", "carry", "lower"):
            gripper = 0.
            if before in ("lift", "carry", "lower"):
                desired_object = self.initial_object.copy()
                if before in ("carry", "lower"):
                    desired_object[:3, 3] = self.goal
                if before != "lower":
                    desired_object[2, 3] += self.clearance
                target = desired_object @ self.object_to_tcp
        elif before == "release":
            target = self.release_tcp.copy()
        elif before in ("withdraw", "hold"):
            target = self.terminal_tcp.copy()
        if not self.failed:
            try:
                self._check_target(target)
            except ValueError:
                self.failed, self.failure_reason = True, "POSE_ENVELOPE_REJECTED"
        if self.failed:
            # ponytail: stop this episode on failure; no hidden quality retries.
            self.command = tcp.copy()
            self.gripper = opening
        else:
            arrived = (np.linalg.norm(tcp[:3, 3] - target[:3, 3]) <= .005
                       and geodesic_angle(tcp[:3, :3], target[:3, :3]) <= .06)
            if before in ("approach", "align") and self._ready(arrived, timestamp):
                self._transition("align" if before == "approach" else "close", timestamp)
            elif before == "close" and self._ready(grasp, timestamp, .30):
                # Capture measured object/TCP relation only after sustained physical grasp.
                self.object_to_tcp = np.linalg.inv(obj) @ tcp
                self._transition("lift", timestamp)
            elif before == "lift" and self._ready(grasp and obj[2, 3] - self.initial_object[2, 3] >= .02 and arrived, timestamp, .30):
                self._transition("carry", timestamp)
            elif before == "carry" and self._ready(grasp and np.linalg.norm(obj[:2, 3] - self.goal[:2]) <= .005 and arrived, timestamp):
                self._transition("lower", timestamp)
            elif before == "lower" and self._ready(grasp and np.linalg.norm(obj[:3, 3] - self.goal) <= .004 and arrived, timestamp):
                self.release_tcp = tcp.copy()
                self._transition("release", timestamp)
            elif before == "release" and self._ready(not grasp and not feedback["native_grasp"] and opening >= .9
                    and np.linalg.norm(linear) <= .02 and np.linalg.norm(angular) <= .10, timestamp, .30):
                self.terminal_tcp = self.release_tcp.copy()
                self.terminal_tcp[2, 3] += self.withdraw
                self._transition("withdraw", timestamp)
            elif before == "withdraw" and self._ready(arrived and np.linalg.norm(tcp[:3, 3] - obj[:3, 3]) >= .08, timestamp):
                self._transition("hold", timestamp)
            delta = target[:3, 3] - self.command[:3, 3]
            self.command[:3, 3] += delta * min(1., self.linear_speed * dt / max(np.linalg.norm(delta), 1e-12))
            angle = float(geodesic_angle(self.command[:3, :3], target[:3, :3]))
            self.command[:3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(self.command[:3, :3]),
                matrix_to_quat(target[:3, :3]), min(1., self.angular_speed * dt / max(angle, 1e-12))))
            self.gripper = gripper
        self.last_time = timestamp
        return TeacherCommand(self.command.copy(), self.gripper, before, self.stage, timestamp,
                              self.failed, self.failure_reason)
