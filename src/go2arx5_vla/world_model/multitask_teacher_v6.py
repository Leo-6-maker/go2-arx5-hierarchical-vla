"""Feedback Teacher for the six declared tasks and original H40 transport."""
import numpy as np

from .mobile_continuous_teacher_v2 import DT
from .mobile_grasp_budget_v3 import BudgetTeacher
from .multitask_score_v6 import BASE_MOTION, LIFT, OPEN_FRACTION, Task


class MultitaskTeacher(BudgetTeacher):
    def __init__(self, initial, condition, geometry, instruction):
        super().__init__(initial, condition)
        self.task = Task.parse(instruction)
        self.geometry = geometry
        self.base_origin = np.asarray(initial['base_pose'])[:2,3].copy()
        self.direction = np.asarray(geometry['direction_world_XY'])
        self.open_start = None
        self.withdrawal_goal = None
        self.placement_TCP = None

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
            goal = np.r_[tcp[:2]+zone-obj[:2], self.lift_origin[2]+.026]
            displacement = np.asarray(sample['base_pose'])[:2,3]-self.base_origin
            if np.linalg.norm(obj[:2]-zone) <= .004 and displacement @ self.direction >= BASE_MOTION:
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
        evidence.update(teacher_version='MFREE_MULTITASK_TEACHER_V6',
                        instruction=self.task.instruction, release_start_step=self.open_start)
        return plan,evidence
