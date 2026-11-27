"""Score no-contact Reach/Orient on the frozen mobile Teacher loop."""

import numpy as np

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import ContinuousTeacher, rotation_vector
from go2arx5_vla.world_model.multitask_carry_geometry_v6 import world_yaw


YAW_DEG = {"EF5_Q_REACH_01": 5., "EF5_Q_REACH_01_R2": 5., "EF5_Q_REACH_02": -5.}
TRAIN_YAW_DEG = {"EF5_T_REACH_01": 5., "EF5_T_REACH_02": -5., "EF5_T_REACH_03": 5., "EF5_T_REACH_04": -5.}
POSITION_LIMIT_M = .005
ROTATION_LIMIT_RAD = np.deg2rad(2.)
DWELL_TICKS = 250
GOAL_OFFSET = np.array([0., -.0017, .080])


class ReachTeacher(ContinuousTeacher):
    def __init__(self, initial, position, rotation):
        super().__init__(initial)
        self.position_goal = position
        self.rotation = rotation
        self.stage = "PREGRASP"

    def goal(self, sample):
        return self.position_goal

    def grip_trajectory(self, start):
        return np.ones(40)


class ReachScore:
    def __init__(self, initial, target, position, rotation):
        self.start = initial["native_step"]
        self.last = None
        self.position_goal = position
        self.rotation_goal = rotation
        self.target = target
        self.moved_m = 0.
        self.dwell_start = None
        self.longest_dwell = 0
        self.bad_contact_ticks = 0
        self.gaps = 0
        self.unsafe_ticks = 0
        self.position_error = float(np.linalg.norm(initial["tcp"][:3, 3] - position))
        self.rotation_error = float(np.linalg.norm(rotation_vector(rotation @ initial["tcp"][:3, :3].T)))
        self.initial_target_xyz = initial["objects"][target]["pose"][:3, 3].copy()

    def update(self, sample):
        step = sample["native_step"]
        expected = self.start if self.last is None else self.last + 1
        if step != expected:
            self.gaps += abs(step-expected)
        self.last = step
        tcp = sample["tcp"]
        self.position_error = float(np.linalg.norm(tcp[:3, 3] - self.position_goal))
        self.rotation_error = float(np.linalg.norm(rotation_vector(self.rotation_goal @ tcp[:3, :3].T)))
        self.moved_m = max(self.moved_m, float(np.linalg.norm(
            sample["objects"][self.target]["pose"][:3, 3] - self.initial_target_xyz)))
        self.bad_contact_ticks += any(obj["hand_contact"] for obj in sample["objects"].values())
        self.unsafe_ticks += not (sample["safety_valid"] and sample["gripper_state_valid"])
        at_goal = self.position_error <= POSITION_LIMIT_M and self.rotation_error <= ROTATION_LIMIT_RAD
        if at_goal and not (self.bad_contact_ticks or self.gaps or self.unsafe_ticks):
            if self.dwell_start is None:
                self.dwell_start = step
            self.longest_dwell = max(self.longest_dwell, step - self.dwell_start)
        else:
            self.dwell_start = None

    def result(self):
        passed = (self.longest_dwell >= DWELL_TICKS and not
                  (self.bad_contact_ticks or self.gaps or self.unsafe_ticks))
        return dict(task_success=passed, native_step=self.last, position_error_m=self.position_error,
                    rotation_error_rad=self.rotation_error, longest_dwell_s=.002*self.longest_dwell,
                    object_hand_contact_ticks=self.bad_contact_ticks, unsafe_ticks=self.unsafe_ticks,
                    missing_native_intervals=self.gaps, max_target_translation_m=self.moved_m,
                    ever_target_grasp=False, maximum_lift_m=0., longest_grasp_lift_hold_s=0.)


def goals(initial, identity):
    target = np.asarray(initial["object_pose"])
    tcp = np.asarray(initial["tcp"])
    yaw = (YAW_DEG | TRAIN_YAW_DEG)[identity]
    return target[:3, 3] + GOAL_OFFSET, world_yaw(np.deg2rad(yaw)) @ tcp[:3, :3]


def train_language(target, identity):
    turn = "counterclockwise" if TRAIN_YAW_DEG[identity] > 0 else "clockwise"
    return f"Reach above the {target} cube, turn the wrist 5 degrees {turn} around the vertical axis, and hold without contact."


def episode(adapter):
    from go2arx5_vla.world_model.multitask_teacher_episode_v6 import episode as teacher_episode

    spec = adapter.spec
    identity = spec["panel_entry_id"]
    train = identity in TRAIN_YAW_DEG
    if (identity not in (YAW_DEG | TRAIN_YAW_DEG) or spec["collection_split"] != ("TRAIN" if train else "QUAL")
            or spec["expert_training_eligible"] != train
            or spec["scope"] != "SLA_CORE_HOPPER_V1_EXPERT" or spec["WBC_changes"] or spec["safety_changes"]):
        raise PermissionError("Reach/Orient must use its own fixed split and roster")
    target = adapter.task.color

    def construct(initial):
        position, orientation = goals(initial, identity)
        return ReachTeacher(initial, position, orientation), ReachScore(initial, target, position, orientation)

    result = teacher_episode(adapter, reach_qualification=construct,
                             capture_language=train_language(target, identity) if train else None)
    reached = result["task_success"]
    task = f"REACH-ORIENT-{target}"
    result = dict(result, schema="V64_REACH_ORIENT_TRAIN_ACTUAL" if train else "V64_REACH_ORIENT_QUAL_ACTUAL",
                task_success=False, reach_orientation_success=reached,
                terminal_reason="REACH_ORIENTATION_SUCCESS" if reached else result["terminal_reason"],
                scored_as_original_hold=False, training_admission=False)
    result["training_instruction" if train else "qualification_instruction"] = task
    return result
