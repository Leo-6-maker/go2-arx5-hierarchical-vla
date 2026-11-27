"""High-release Teacher for the registered mobile container qualification layouts."""

import numpy as np

from go2arx5_vla.world_model.mobile_grasp_budget_v3 import BudgetTeacher
from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION, LIFT, OPEN_FRACTION
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher


RIM_CLEARANCE_M = .015
COMMAND_CLEARANCE_M = .016
CENTER_TOLERANCE_M = .004
RETREAT_M = .060
OBJECT_SPEED_LIMIT_M_S = .02
QUALIFICATION = {
    "EF5_Q_CONTAINER_01": ("F1", "red", "left"),
    "EF5_Q_CONTAINER_02": ("F2", "blue", "right"),
}


class ContainerTeacher(MultitaskTeacher):
    def __init__(self, initial, condition, geometry, identity, bowl_xy, bowl_rim_z):
        family, color, destination = QUALIFICATION[identity]
        if condition["family"] != family or condition["mode"] != "M_FREE":
            raise ValueError("Container qualification must use its registered M_FREE family")
        super().__init__(initial, condition, geometry, f"HOLD-{color}")
        self.qualification_instruction = f"CONTAINER-{color}-{destination}"
        self.center = np.asarray(bowl_xy, dtype=float)
        self.rim_z = float(bowl_rim_z)
        if self.center.shape != (2,) or not np.isfinite([*self.center, self.rim_z]).all():
            raise ValueError("The measured bowl center and rim are required")
        self.half_size = np.asarray(geometry["object_half_sizes"][color], dtype=float)

    def goal(self, sample):
        tcp_pose = np.asarray(sample["tcp"])
        tcp = tcp_pose[:3, 3]
        target = sample["objects"][self.task.color]
        obj_pose = np.asarray(target["pose"])
        obj = obj_pose[:3, 3]
        z_half = abs(obj_pose[2, :3]) @ self.half_size
        if self.stage in ("PREGRASP", "DESCEND", "CLOSE", "LIFT", "HOLD"):
            pickup = BudgetTeacher.goal(self, sample)
            if self.stage != "HOLD" or not target["native_grasp"] or obj[2] - self.initial_cube_z < LIFT:
                return pickup
            self.stage = "CARRY"
            self.rotation = tcp_pose[:3, :3].copy()

        if self.stage == "CARRY":
            if not target["native_grasp"]:
                return tcp.copy()
            clearance = obj[2] - z_half - self.rim_z
            required_height = tcp[2] + COMMAND_CLEARANCE_M - clearance
            previous_height = tcp[2] if self.carry_height is None else self.carry_height
            self.carry_height = max(required_height, previous_height)
            if clearance < RIM_CLEARANCE_M:
                if self.raise_xy is None:
                    self.raise_xy = tcp[:2].copy()
                return np.r_[self.raise_xy, self.carry_height]
            target_tcp, unobstructed = self.carry_goal(sample, self.center)
            moved = (np.asarray(sample["base_pose"])[:2, 3] - self.base_origin) @ self.direction
            if (not unobstructed or moved < BASE_MOTION
                    or np.linalg.norm(obj[:2] - self.center) > CENTER_TOLERANCE_M
                    or np.linalg.norm(target["linear_velocity"]) >= OBJECT_SPEED_LIMIT_M_S):
                return target_tcp
            self.stage = "OPEN"
            self.placement_TCP = tcp.copy()
            self.open_start = sample["native_step"]

        if self.stage == "OPEN":
            if ((sample["native_step"] - self.open_start) * .002 < 2.
                    or sample["gripper_open_fraction"] < OPEN_FRACTION
                    or target["hand_contact"] or target["native_grasp"]):
                return self.placement_TCP
            self.stage = "WITHDRAW"
            self.withdrawal_goal = self.placement_TCP + [0., 0., RETREAT_M]
        if self.stage == "WITHDRAW" and np.linalg.norm(tcp - self.withdrawal_goal) <= CENTER_TOLERANCE_M:
            self.stage = "SETTLE"
        return self.withdrawal_goal

    def decide(self, sample, reference, base):
        program, evidence = super().decide(sample, reference, base)
        evidence.update(teacher_version="V64_MFREE_CONTAINER_HIGH_RELEASE",
                        instruction=self.qualification_instruction,
                        container_release_rim_clearance_m=RIM_CLEARANCE_M,
                        container_command_rim_clearance_m=COMMAND_CLEARANCE_M)
        return program, evidence
