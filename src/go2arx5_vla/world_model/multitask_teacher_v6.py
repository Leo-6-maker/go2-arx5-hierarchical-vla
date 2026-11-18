"""Feedback Teacher for the six declared tasks and original H40 transport."""
import numpy as np

from .mobile_continuous_teacher_v2 import DT
from .mobile_grasp_budget_v3 import BudgetTeacher
from .multitask_score_v6 import BASE_MOTION, LIFT, OPEN_FRACTION, Task

TRANSFER_CLEARANCE_M = .008
F1_RED_HOLD_GRASP_RAISE_M = .016
F1_RED_LEFT_CARRY_RAISE_M = .036


def swept_footprints_overlap(start, finish, obstacle, combined_half_size):
    """Check the whole transported footprint along a straight XY segment."""
    delta = finish - start
    offset = start - obstacle
    first, last = 0., 1.
    for d, start_axis, half_size in zip(delta, offset, combined_half_size, strict=True):
        if abs(d) < 1e-12:
            if abs(start_axis) > half_size:
                return False
            continue
        entry, exit = sorted(((-half_size - start_axis) / d, (half_size - start_axis) / d))
        first, last = max(first, entry), min(last, exit)
        if first > last:
            return False
    return True


class MultitaskTeacher(BudgetTeacher):
    def __init__(self, initial, condition, geometry, instruction):
        super().__init__(initial, condition)
        self.task = Task.parse(instruction)
        self.f1_red_hold_clearance = (
            self.task.instruction == 'HOLD-red'
            and condition['family'] == 'F1'
            and condition['mode'] == 'M_FREE'
        )
        if self.f1_red_hold_clearance:
            self.raise_m = F1_RED_HOLD_GRASP_RAISE_M
        self.geometry = geometry
        self.base_origin = np.asarray(initial['base_pose'])[:2,3].copy()
        self.direction = np.asarray(geometry['direction_world_XY'])
        self.open_start = None
        self.withdrawal_goal = None
        self.placement_TCP = None
        self.carry_height = None
        self.raise_xy = None
        self.carry_footprint_overlap = False
        self.distal_clearance = None
        self.distal_raise_height = None
        self.distal_clearance_evidence = None

    def distal_carry_goal(self, sample, zone, tcp, obj):
        gap = self.distal_clearance(sample, zone)
        if gap < TRANSFER_CLEARANCE_M and self.distal_raise_height is None:
            self.distal_raise_height = tcp[2] + F1_RED_LEFT_CARRY_RAISE_M
            self.carry_height = max(self.carry_height, self.distal_raise_height)
        waiting = gap < TRANSFER_CLEARANCE_M
        self.distal_clearance_evidence = dict(sampled_gap_m=gap, raise_height_m=self.distal_raise_height,
                                             waiting_for_measured_clearance=waiting)
        if waiting:
            if self.raise_xy is None:
                self.raise_xy = tcp[:2].copy()
            return np.r_[self.raise_xy, self.carry_height], False
        self.raise_xy = None
        can_lower = False
        if np.linalg.norm(obj[:2] - zone) <= .004:
            support_z = self.geometry['table_top_world_z'] + self.geometry['object_half_sizes'][self.task.color][2]
            can_lower = self.distal_clearance(sample, zone, support_z=support_z) > 0.
        return np.r_[tcp[:2] + zone - obj[:2], self.carry_height], can_lower

    def carry_goal(self, sample, zone):
        target = self.task.color
        other = 'blue' if target == 'red' else 'red'
        target_pose = np.asarray(sample['objects'][target]['pose'])
        other_pose = np.asarray(sample['objects'][other]['pose'])
        half_sizes = self.geometry['object_half_sizes']
        target_half = abs(target_pose[:3,:3]) @ half_sizes[target]
        other_half = abs(other_pose[:3,:3]) @ half_sizes[other]
        tcp = np.asarray(sample['tcp'])[:3,3]
        obj, obstacle = target_pose[:3,3], other_pose[:3,3]
        half_planar = (target_half + other_half)[:2]
        self.carry_footprint_overlap = swept_footprints_overlap(obj[:2], zone, obstacle[:2], half_planar)
        if self.carry_height is None:
            self.carry_height = self.lift_origin[2] + .026
        if self.distal_clearance is not None:
            if not sample['native_grasp']:
                self.distal_clearance_evidence = dict(sampled_gap_m=None, native_grasp=False)
                return tcp, False
            return self.distal_carry_goal(sample, zone, tcp, obj)
        if sample['native_grasp'] and self.carry_footprint_overlap:
            gap = (obj[2] - target_half[2]) - (obstacle[2] + other_half[2])
            self.carry_height = max(self.carry_height, tcp[2] + TRANSFER_CLEARANCE_M - gap)
            if gap < TRANSFER_CLEARANCE_M:
                if self.raise_xy is None:
                    self.raise_xy = tcp[:2].copy()
                return np.r_[self.raise_xy, self.carry_height], False
        self.raise_xy = None
        lower_clear = np.any(abs(obj[:2] - obstacle[:2]) > half_planar)
        return np.r_[tcp[:2] + zone - obj[:2], self.carry_height], lower_clear

    def goal(self, sample):
        if self.task.operation == 'HOLD':
            return super().goal(sample)
        tcp = np.asarray(sample['tcp'])[:3,3]
        obj = np.asarray(sample['object_pose'])[:3,3]
        if self.stage in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):
            grasp_goal = super().goal(sample)
            if self.stage != 'HOLD' or not sample['native_grasp']:
                return grasp_goal
            if obj[2]-self.initial_cube_z < LIFT:
                return grasp_goal
            self.stage = 'CARRY'
        zone = np.asarray(self.geometry['zones'][self.task.destination]['center_world_XY'])
        if self.stage == 'CARRY':
            goal, lower_clear = self.carry_goal(sample, zone)
            displacement = np.asarray(sample['base_pose'])[:2,3]-self.base_origin
            if lower_clear and np.linalg.norm(obj[:2]-zone) <= .004 and displacement @ self.direction >= BASE_MOTION:
                self.stage = 'LOWER'
            else:
                return goal
        if self.stage == 'LOWER':
            half_height = self.geometry['object_half_sizes'][self.task.color][2]
            support_z = self.geometry['table_top_world_z']+half_height
            goal = tcp+np.r_[zone-obj[:2],support_z-obj[2]]
            target = sample['objects'][self.task.color]
            if (target['table_contact'] and abs(obj[2]-support_z) <= .005
                    and np.linalg.norm(obj[:2]-zone) <= .004):
                self.stage = 'OPEN'
                self.open_start = sample['native_step']
                self.placement_TCP = tcp.copy()
            else:
                return goal
        if self.stage == 'OPEN':
            if (sample['native_step']-self.open_start)*.002 < 2. or (
                    sample['gripper_open_fraction'] < OPEN_FRACTION or sample['native_grasp']):
                return self.placement_TCP
            self.stage = 'WITHDRAW'
            self.withdrawal_goal = self.placement_TCP+np.array([0.,0.,.060])
        if self.stage == 'WITHDRAW' and np.linalg.norm(tcp-self.withdrawal_goal) <= .004:
            self.stage = 'SETTLE'
        if self.stage in ('WITHDRAW','SETTLE'):
            return self.withdrawal_goal
        raise ValueError('DECLARED_PLACE_PHASE_REQUIRED')

    def grip_trajectory(self,start):
        if self.open_start is None:
            return super().grip_trajectory(start)
        u = np.clip(((start-self.open_start)*.002+np.arange(40)*DT)/2.,0.,1.)
        # The 40mm working stroke over this nominal curve has analytical
        # peak speed 37.5mm/s. The unchanged full preview gate still applies.
        return u**3*(10+u*(-15+6*u))

    def decide(self,sample,reference,base):
        plan, evidence = super().decide(sample,reference,base)
        version = ('MFREE_MULTITASK_TEACHER_V6_F1_RED_HOLD_CLEARANCE'
                   if self.f1_red_hold_clearance else 'MFREE_MULTITASK_TEACHER_V6_TRANSFER_CLEARANCE')
        if self.distal_clearance is not None:
            version = 'MFREE_MULTITASK_TEACHER_V6_F1_RED_LEFT_DISTAL_CARRY'
        evidence.update(teacher_version=version,
                        instruction=self.task.instruction, release_start_step=self.open_start,
                        teacher_carry_footprint_overlap=self.carry_footprint_overlap,
                        teacher_carry_waiting_for_measured_clearance=(
                            self.raise_xy is not None and self.distal_clearance is None))
        return plan,evidence
