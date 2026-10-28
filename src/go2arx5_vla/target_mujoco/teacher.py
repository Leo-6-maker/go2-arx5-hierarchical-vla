"""Object-relative CPU teacher. No robot API, physics, or joint oracle."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..target_sim.teacher import Teacher as StageTeacher, TeacherCommand
from ..trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix
from .scorer import physical_grasp, pose_matrix

SPEC_FIELDS = ("geometry", "travel", "closing_direction", "mass_com_inertia", "tcp",
               "speed_limit", "force_limit")


def require_gripper_spec(spec, *, formal=True):
    if not isinstance(spec, dict):
        raise ValueError("BLOCKED_GRIPPER_SPEC: calibrated target gripper spec required")
    if spec.get("fingers"):
        fingers = spec["fingers"]
        missing = []
        if len(fingers) != 2 or np.shape(spec.get("tcp_xyz")) != (3,):
            missing.append("two target fingers and calibrated TCP")
        for finger in fingers:
            for key in ("collisions", "range", "axis", "mass", "com", "fullinertia", "force_model_N", "speed_model_m_s"):
                if not finger.get(key): missing.append(key)
        missing.extend(spec.get("missing", []))
        if not formal:
            missing = [key for key in missing if key not in spec.get("missing", [])]
        verified = spec.get("status") == "HARDWARE_SPEC_VERIFIED"
    else:
        missing = [key for key in SPEC_FIELDS if not spec.get(key)]
        verified = spec.get("status") == "VERIFIED_DEPLOYMENT_SPEC"
    if missing or (formal and not verified):
        raise ValueError("BLOCKED_GRIPPER_SPEC: " + ",".join(missing or ["verified source identity"]))
    if formal and not (spec.get("source_sha256") and spec.get("source_path")):
        raise ValueError("BLOCKED_GRIPPER_SPEC: source path/hash required")
    if formal and spec.get("fingers") and not spec.get("hardware_source"):
        raise ValueError("BLOCKED_GRIPPER_SPEC: verified deployment hardware source required")
    return spec


@dataclass(frozen=True)
class TeacherPlan:
    time_s: float
    point_times_s: np.ndarray
    tcp_world: np.ndarray
    gripper_fraction: np.ndarray
    stage: str
    version: str
    failed: bool


class Teacher(StageTeacher):
    """Reuse contact-driven phases; freeze explicit geometry and two versions.

    query() consumes real 20Hz feedback. plan() is called after that query at
    0.4s intervals; future poses extrapolate the current target only, without
    inventing future contacts or claiming those planned points were accepted.
    """
    def __init__(self, *args, gripper_spec, version="PI0288_TEACHER_V1",
                 declared_versions=("PI0288_TEACHER_V1",), provisional=False, **kwargs):
        if (len(set(declared_versions)) != len(declared_versions) or
                not 1 <= len(declared_versions) <= 2 or version not in declared_versions):
            raise ValueError("At most two declared teacher versions are allowed")
        require_gripper_spec(gripper_spec, formal=not provisional)
        kwargs.setdefault("linear_speed_m_s", .05)
        kwargs.setdefault("angular_speed_rad_s", .20)
        if not 0 < kwargs["linear_speed_m_s"] <= .05 or not 0 < kwargs["angular_speed_rad_s"] <= .20:
            raise ValueError("PI0288 teacher speed exceeds frozen G1 envelope")
        super().__init__(*args, **kwargs)
        self.version, self.gripper_spec = version, gripper_spec
        self.provisional, self.last_plan_time = provisional, None
        self.plan_count = 0
        self.current_plan = None

    def query(self, time_s, feedback):
        checked = dict(feedback)
        checked["left_object_contact"] = bool(physical_grasp(feedback))
        if feedback["numerical_bound_warning"]:
            checked["safety_valid"] = False
        t = float(time_s)
        command = super().query(t, checked)
        if not command.failed and self.current_plan is not None:
            index = round((t - self.current_plan.time_s) * 20)
            if not 0 <= index < 40:
                raise ValueError("Current reference expired; cannot continue without replan")
            # Phase changes use real feedback but cannot rewrite an issued EEF chunk.
            self.command = self.current_plan.tcp_world[index].copy()
            command = TeacherCommand(self.command.copy(), command.gripper_fraction,
                command.stage_before, command.stage_after, command.time_s,
                command.failed, command.failure_reason)
        elif abs(t) <= 1e-8:
            self.command = self.initial_tcp.copy()
            command = TeacherCommand(self.command.copy(), command.gripper_fraction,
                command.stage_before, command.stage_after, command.time_s,
                command.failed, command.failure_reason)
        return command

    def plan(self, time_s, feedback):
        t = float(time_s)
        if self.last_time is None or abs(t - self.last_time) > 1e-8:
            raise ValueError("plan requires same-time real feedback query first")
        if abs(t / .4 - round(t / .4)) > 1e-8 or (self.last_plan_time is not None and abs(t - self.last_plan_time - .4) > 1e-8):
            raise ValueError("teacher plans must be distinct contiguous 0.4s decisions")
        obj = pose_matrix(feedback["object_pose"])
        target = obj @ self.object_to_tcp
        grip = 1.
        if self.stage == "approach":
            target[2, 3] += self.approach
        elif self.stage in ("close", "lift", "carry", "lower"):
            grip = 0.
            if self.stage != "close":
                desired = self.initial_object.copy()
                if self.stage in ("carry", "lower"):
                    desired[:3, 3] = self.goal
                if self.stage != "lower":
                    desired[2, 3] += self.clearance
                target = desired @ self.object_to_tcp
        elif self.stage == "release":
            target = self.release_tcp.copy()
        elif self.stage in ("withdraw", "hold"):
            target = self.terminal_tcp.copy()
        self._check_target(target)
        initial = self.command.copy()
        if self.failed:
            target, grip = initial, float(feedback["gripper_fraction"])
        offsets = np.arange(40, dtype=float) / 20
        poses = np.repeat(initial[None], 40, axis=0)
        delta = target[:3, 3] - initial[:3, 3]
        distance = np.linalg.norm(delta)
        poses[:, :3, 3] += np.minimum(1., self.linear_speed * offsets / max(distance, 1e-12))[:, None] * delta
        angle = float(geodesic_angle(initial[:3, :3], target[:3, :3]))
        for i, offset in enumerate(offsets):
            poses[i, :3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(initial[:3, :3]),
                matrix_to_quat(target[:3, :3]), min(1., self.angular_speed * offset / max(angle, 1e-12))))
        self.last_plan_time, self.plan_count = t, self.plan_count + 1
        self.current_plan = TeacherPlan(t, t + offsets, poses, np.full(40, grip), self.stage, self.version, self.failed)
        return self.current_plan
