"""One task-gripper endpoint/reset correction; no gain or WBC intervention.

V1 reset and commanded raw mechanical endpoints had zero safety margin. V2
retains mechanical stroke, masses, damping, force and velocity. Its working
stroke reserves exactly one 20Hz command interval of declared finger travel.
Action/state units remain mechanical fraction q/.044; no extra rescaling at
the actuator and no clipping of measured state or student output is allowed.
"""
import numpy as np

VERSION='FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE'
MECHANICAL_TRAVEL=.044
SPEED=.04
COMMAND_DT=.05
MARGIN=SPEED*COMMAND_DT
WORKING_CLOSED=MARGIN
WORKING_OPEN=MECHANICAL_TRAVEL-MARGIN


def teacher_physical_fraction(logical_opening):
    value=np.asarray(logical_opening,dtype=float)
    if not np.isfinite(value).all() or np.any(value<0) or np.any(value>1):
        raise ValueError('LOGICAL_TEACHER_OPENING_RANGE')
    return (WORKING_CLOSED+(WORKING_OPEN-WORKING_CLOSED)*value)/MECHANICAL_TRAVEL


def require_working_target(physical_fraction):
    value=np.asarray(physical_fraction,dtype=float)
    if (not np.isfinite(value).all() or np.any(value<WORKING_CLOSED/MECHANICAL_TRAVEL)
            or np.any(value>WORKING_OPEN/MECHANICAL_TRAVEL)):
        raise ValueError('OUTSIDE_DECLARED_GRIPPER_WORKING_TARGET_RANGE_NO_CLIP')
