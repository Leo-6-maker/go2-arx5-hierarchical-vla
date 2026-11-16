"""One intervention: cyclically relocate the existing three parameter blocks.

The 14/13/13 scalar repetitions, affine statistics, event parameters and
robot-time decoder remain identical. No predicted physical path is filtered.
"""
import numpy as np
from . import gripper_time_v1 as original

VERSION = 'SLA_REFERENCE_CONTINUED_GRIP_TIME_SHIFT7_V1'
SHIFT = 7
SLOTS = tuple(tuple((np.arange(a, b) + SHIFT) % 40) for a, b in original.BLOCKS)


def relocate(values):
    values = np.asarray(values)
    if values.shape[-1:] != (40,) or not np.isfinite(values).all():
        raise ValueError('FINITE_40_SLOT_PARAMETER_ARRAY_REQUIRED')
    return np.roll(values, SHIFT, axis=-1)


def decode(values, context, low, high):
    values = np.asarray(values)
    if values.shape != (40,):
        raise ValueError('SINGLE_FINITE_40_SLOT_CONTAINER_REQUIRED')
    width, detail = original.decode(np.roll(values, -SHIFT), context, low, high)
    return width, dict(detail, version=VERSION, parameter_slot_shift=SHIFT)


def label_parameters(width, context, low, high):
    target, detail = original.label_parameters(width, context, low, high)
    return relocate(target), dict(detail, version=VERSION, parameter_slot_shift=SHIFT)
