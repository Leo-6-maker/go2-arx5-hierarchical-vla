"""Classify a measured, safely stalled carry and compile its first Teacher H40."""

from collections import deque
from itertools import product

import numpy as np

from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.trajectory.so3 import quat_to_matrix
from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_CLOSED_M, WORKING_OPEN_M
from go2arx5_vla.world_model.multitask_carry_geometry_v6 import DistalCarryGeometry
from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION, LIFT
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher


ATOM_TICKS, PREFIX_TICKS = 25, 250
HELD_MIN_S, STALL_POINTS, MIN_PROGRESS_M = .3, 4, .002


class CarryTeacher(MultitaskTeacher):
    """Preserve the issued holding width at the student-to-Teacher seam."""

    def __init__(self, initial, sample, condition, geometry, instruction, prior_width_m, *, model):
        super().__init__(initial, condition, geometry, instruction)
        if not WORKING_CLOSED_M - 1e-12 <= prior_width_m <= WORKING_OPEN_M + 1e-12:
            raise ValueError("The actually consumed gripper command left the working stroke")
        self.holding_fraction = float(np.clip(
            (prior_width_m - WORKING_CLOSED_M) / (WORKING_OPEN_M - WORKING_CLOSED_M), 0., 1.))
        self.stage = "CARRY"
        self.rotation = np.asarray(sample["tcp"])[:3, :3].copy()
        self.carry_height = float(np.asarray(sample["tcp"])[2, 3])
        if self.landing is not None:
            self.distal_clearance = DistalCarryGeometry(model, self.task.color).clearance

    def grip_trajectory(self, start):
        if self.open_start is None:
            return np.full(40, self.holding_fraction)
        return self.holding_fraction + (1-self.holding_fraction)*super().grip_trajectory(start)


class CarryStall:
    """Use consecutive P10 results and original score flags to detect a carry stall."""

    def __init__(self, task, geometry):
        if task.operation != "PLACE":
            raise ValueError("The carry Recovery pilot requires an original PLACE task")
        self.zone = geometry["zones"][task.destination]
        self.half_size = np.asarray(geometry["object_half_sizes"][task.color])
        self.corners = np.asarray(list(product((-1., 1.), repeat=3))) * self.half_size
        self.history = deque(maxlen=STALL_POINTS)

    def observe(self, score, sample, consumed):
        if (score.latest is None or score.last != sample["native_step"]
                or score.latest["native_step"] != sample["native_step"]):
            raise ValueError("Score must cover the measured paused state")
        step = sample["native_step"]
        if any((score.gaps, score.unsafe_states, score.wrong_grasp, score.wrong_contact)):
            self.history.clear()
            return None
        row = score.latest
        held = 0. if score.hold_since is None else sample["time_s"] - score.hold_since
        if (not sample["safety_valid"] or not row["target_grasp"] or row["lift_m"] < LIFT or held < HELD_MIN_S
                or row["base_directional_motion_m"] < BASE_MOTION or row["inside_selected_region"]):
            self.history.clear()
            return None
        boundary = exact_boundary(consumed, step)
        pose = np.asarray(sample["object_pose"])
        footprint = self.corners @ pose[:3, :3].T + pose[:3, 3]
        gap = float(np.max(np.abs(footprint[:, :2] - self.zone["center_world_XY"]) - self.zone["half_size_XY"]))
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError("The actual outside-region footprint differs from the original score")
        if self.history and step != self.history[-1][0] + PREFIX_TICKS:
            self.history.clear()
        self.history.append((step, gap))
        if len(self.history) < STALL_POINTS:
            return None
        improvement = self.history[0][1] - min(x[1] for x in self.history)
        if improvement >= MIN_PROGRESS_M:
            return None
        return dict(native_step=step, native_time_s=sample["time_s"], held_s=held, lift_m=row["lift_m"],
                    footprint_outside_m=gap, best_progress_m=improvement, requested_samples=STALL_POINTS,
                    original_wire_sha256=boundary["wire_sha256"])


def exact_boundary(consumed, step):
    atoms = consumed["confirmed_intervals"]
    start = consumed["native_start_step"]
    if (consumed["actual_model_generations"] != 1 or not consumed["full_P10"]
            or consumed["confirmed_atoms"] != len(atoms) or len(atoms) != PREFIX_TICKS//ATOM_TICKS
            or consumed["terminal_native_step"] != step or step != start+PREFIX_TICKS
            or consumed["unconfirmed_native_ticks"] != 0 or consumed["mock_consumption"]):
        raise ValueError("A complete genuinely consumed student P10 is required")
    for i, atom in enumerate(atoms):
        boundary = atom["accepted_boundary"]
        if (atom["native_start_step"] != start+i*ATOM_TICKS
                or atom["native_end_step"] != start+(i+1)*ATOM_TICKS
                or boundary["native_step"] != atom["native_end_step"] or not boundary["exact_receiver_knot"]
                or atom["wire_sha256"] != boundary["accepted_wire_sha256"]):
            raise ValueError("P10 carries a partial or differently accepted interval")
    return atoms[-1]


def prepare(initial, sample, condition, geometry, instruction, consumed, previous, base, model):
    """Use measured state and the accepted receiver knot, without emitting a command."""
    atom = exact_boundary(consumed, sample["native_step"])
    if (atom["emitted_gripper_width_m"] != previous["width_m"]
            or atom["emitted_gripper_source_native_s"] != previous["source_native_s"]
            or abs(previous["query_native_s"]-sample["time_s"]) > 1e-8):
        raise ValueError("The original last consumed gripper command was not handed over")
    pose = np.asarray(atom["accepted_boundary"]["accepted_pose7"])
    reference = np.eye(4)
    reference[:3, 3] = pose[:3]
    reference[:3, :3] = quat_to_matrix(pose[3:])
    teacher = CarryTeacher(initial, sample, condition, geometry, instruction, previous["width_m"], model=model)
    program, evidence = teacher.decide(sample, reference, base)
    margin = label_margin_report(program.decoded_H40, previous)
    if margin["training_margin_exceeded"] or margin["original_execution_speed_exceeded"]:
        raise ValueError("The Recovery Teacher attempt failed the original complete H40 check")
    return teacher, program, evidence, margin
