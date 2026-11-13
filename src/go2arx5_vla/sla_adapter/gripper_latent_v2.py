"""SLA gripper scalar transform shared by the codec and native qualification."""
import numpy as np

VERSION = 'SLA_GRIP_LOGIT_V2'
EPSILON = 1e-4


def label_latent(nominal):
    value = np.asarray(nominal, dtype=np.float64)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError('ORIGINAL_FINITE_NOMINAL_LABEL_RANGE_REQUIRED')
    # This is a declared derived-label transform, never an inference clip.
    bounded = np.clip(value, EPSILON, 1 - EPSILON)
    return np.log(bounded) - np.log1p(-bounded)


def decode_latent(latent):
    value = np.asarray(latent, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('FINITE_GRIPPER_LATENT_REQUIRED_NO_NAN_REPAIR')
    small = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + small), small / (1 + small))
