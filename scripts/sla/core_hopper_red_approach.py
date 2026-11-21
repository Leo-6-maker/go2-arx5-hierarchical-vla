"""Fixed world-frame grasp pitch for the registered F2 red TRAIN repairs."""
import numpy as np

from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher

CELL = 'CHV1-TRAIN-F2-HOLD-red'
PILOT = CELL+'-02'
PITCH_DEGREES = 20
KIND = 'F2_RED_HOLD_PITCH'
PLACE_KIND = 'F2_RED_PLACE_PITCH'


def world_pitch(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])


class RedHoldPitch(MultitaskTeacher):
    instructions = ('HOLD-red',)
    repair_kind = KIND

    def __init__(self, initial, condition, geometry, instruction, model):
        if (condition['family'],condition['mode'])!=('F2','M_FREE') or instruction not in self.instructions:
            raise ValueError('Pitch change is bound to its diagnosed F2 red cell')
        super().__init__(initial,condition,geometry,instruction)
        self.rotation=world_pitch(np.deg2rad(PITCH_DEGREES)) @ self.rotation

    def decide(self,sample,reference,base):
        program,evidence=super().decide(sample,reference,base)
        evidence.update(teacher_version=self.repair_kind,grasp_world_pitch_change_deg=PITCH_DEGREES)
        return program,evidence


class RedPlacePitch(RedHoldPitch):
    instructions = ('PLACE-red-left','PLACE-red-right')
    repair_kind = PLACE_KIND
