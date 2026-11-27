"""Prepare the first bounded Teacher prefix at an archived early Recovery boundary."""

import numpy as np

from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_CLOSED_M, WORKING_OPEN_M
from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import DT
from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION, LIFT
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher

from .v64_recovery_carry import accepted_pose, checked_plan, exact_boundary


OPENING_SECONDS = 2.
G7_PREFLIGHT_REFUSAL = ("L1_BOUNDED_TRANSPORT_REJECTED: G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED: "
                       "BOUNDED_CONTINUITY_EXHAUSTED: ORIGINAL_G7_RATE")


class EarlyTeacher(MultitaskTeacher):
    """Keep the consumed jaw position when taking over and use the original arm planner."""

    def __init__(self, initial, sample, condition, geometry, instruction, previous_width):
        super().__init__(initial, condition, geometry, instruction)
        self.holding_at_handoff = bool(sample["native_grasp"])
        self.handoff_step = sample["native_step"]
        self.start_fraction = (previous_width - WORKING_CLOSED_M) / (WORKING_OPEN_M - WORKING_CLOSED_M)
        if not 0. <= self.start_fraction <= 1.:
            raise ValueError("Require the actual last emitted jaw width in the original working stroke")
        if self.holding_at_handoff:
            tcp = np.asarray(sample["tcp"])
            self.rotation = tcp[:3, :3].copy()
            self.lift_origin = tcp[:3, 3].copy()
            self.stage = "LIFT"

    def goal(self, sample):
        if (self.holding_at_handoff and self.stage in ("LIFT", "HOLD", "CARRY", "LOWER")
                and not sample["native_grasp"]):
            raise ValueError("The early Teacher lost the measured grasp before release")
        return super().goal(sample)

    def grip_trajectory(self, start):
        if self.holding_at_handoff:
            if self.open_start is None:
                return np.full(40, self.start_fraction)
            return self.start_fraction + (1. - self.start_fraction) * super().grip_trajectory(start)
        # Reopen across the original two-second quintic and retain its absolute clock across replans.
        elapsed = (start - self.handoff_step) * .002
        if elapsed >= OPENING_SECONDS:
            return super().grip_trajectory(start)
        if self.close_start is not None:
            raise ValueError("Early Teacher closing started before the opening curve completed")
        u = np.clip((elapsed + np.arange(40) * DT) / OPENING_SECONDS, 0., 1.)
        return self.start_fraction + (1. - self.start_fraction) * u**3 * (10. + u * (-15. + 6. * u))


def rejected_boundary(rejected, consumed, step):
    decision = rejected["rejection"]
    if (decision["kind"] != "ORIGINAL_FULL_H40_POLICY_REJECTED" or decision["reason"] != G7_PREFLIGHT_REFUSAL
            or not decision["full_H40_checked"] or decision["installed"] or decision["full_H40_compiled"]
            or rejected["actual_model_generations"] != 1 or rejected["install_ACKs"] or rejected["confirmed_atoms"]
            or rejected["native_start_step"] != step or rejected["query"] != decision["query"]
            or consumed["query"] != rejected["query"] - 1):
        raise ValueError("Require the original uninstalled G7 proposal rejection after the preceding complete P10")
    return exact_boundary(consumed, step)


def prepare_early(initial, sample, result, spec, rejected, consumed, previous, base):
    """Compile a fresh prefix from a real, safe, paused early state and its installed knot."""
    if (sample["native_step"] != result["native_step"] or not sample["safety_valid"] or not sample["gripper_state_valid"]
            or result["base_directional_motion_m"] < BASE_MOTION or result["ever_target_lift20mm"]
            or any(result[key] for key in ("unsafe_states", "missing_native_intervals", "wrong_object_contact_ever",
                                          "wrong_object_grasp_ever")) or result["maximum_lift_m"] >= LIFT):
        raise ValueError("Require a safe paused early state after real BASE motion")
    if result["target_grasp"] != sample["native_grasp"] or (result["ever_target_grasp"] and not result["target_grasp"]):
        raise ValueError("Grasp evidence must agree and a lost grasp requires its own qualification")
    rejected_boundary(rejected, consumed, sample["native_step"])
    reference = accepted_pose(sample, consumed, previous)
    teacher = EarlyTeacher(initial, sample, spec["mobile_condition"], spec["multitask_geometry"],
                           spec["task_instruction_id"], previous["width_m"])
    return checked_plan(teacher, sample, reference, previous, base)
