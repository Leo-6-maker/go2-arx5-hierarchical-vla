"""Cold F2 red HOLD candidate using the measured pregrasp detour."""
import numpy as np

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import rotation_vector
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from scripts.sla.core_hopper_red_approach import world_pitch

VERSION = 'F2_RED_HOLD_PITCH_DETOUR_V1'
PITCH_DEG = 20
AWAY_M = .024
RISE_M = .030
REACHED_M = .004
TURN_REACHED_RAD = np.deg2rad(2.)


def detour_goal(grasp, red, blue):
    away = np.asarray(red)[:2] - np.asarray(blue)[:2]
    distance = float(np.linalg.norm(away))
    if distance <= 2 * REACHED_M:
        raise ValueError('F2 red detour requires two distinct object centers')
    return np.asarray(grasp) + np.r_[AWAY_M * away / distance, RISE_M]


class RedHoldDetour(MultitaskTeacher):
    def __init__(self, initial, condition, geometry, instruction, model):
        if (condition['family'], condition['mode'], instruction) != ('F2', 'M_FREE', 'HOLD-red'):
            raise ValueError('The wrist detour is restricted to the diagnosed F2 red HOLD cell')
        super().__init__(initial, condition, geometry, instruction)
        self.turn_anchor = None
        self.turn_complete = False
        self.detour_complete = False

    def goal(self, sample):
        if self.stage != 'PREGRASP':
            return super().goal(sample)
        self.cube = np.asarray(sample['object_pose']).copy()
        actual = np.asarray(sample['tcp'])
        xyz = actual[:3, 3]
        if self.turn_anchor is None:
            pregrasp = self.cube[:3, 3] + [0., -.0017, .080]
            if np.linalg.norm(xyz - pregrasp) > REACHED_M:
                return pregrasp
            self.turn_anchor = xyz.copy()
            self.rotation = world_pitch(np.deg2rad(PITCH_DEG)) @ self.rotation
        if not self.turn_complete:
            angle = np.linalg.norm(rotation_vector(self.rotation @ actual[:3, :3].T))
            if angle > TURN_REACHED_RAD or np.linalg.norm(xyz - self.turn_anchor) > REACHED_M:
                return self.turn_anchor
            self.turn_complete = True
        blue = np.asarray(sample['objects']['blue']['pose'])[:3, 3]
        waypoint = detour_goal(self.grasp_goal(), self.cube[:3, 3], blue)
        if np.linalg.norm(xyz - waypoint) > REACHED_M:
            return waypoint
        self.detour_complete = True
        self.stage = 'DESCEND'
        return super().goal(sample)

    def decide(self, sample, reference, base):
        plan, evidence = super().decide(sample, reference, base)
        evidence.update(teacher_version=VERSION, detour_pitch_deg=PITCH_DEG,
                        detour_away_m=AWAY_M, detour_elevation_m=RISE_M,
                        detour_turn_complete=self.turn_complete, detour_reached=self.detour_complete)
        return plan, evidence
