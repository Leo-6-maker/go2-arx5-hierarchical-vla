"""Explicit GRIP_LOGIT_V2 action representation; V1 states/labels stay intact."""
import numpy as np

from .codec_v1 import Codec

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


class GripLogitCodec(Codec):
    """Input is already inverse-normalized; apply sigmoid exactly once here.

    Output remains original nominal EEF10. The existing native consumer owns
    its original nominal-to-mechanical mapping. Measured state grip is intact.
    """
    def encode(self, state10, nominal_h40):
        state, actions = super().encode(state10, nominal_h40)
        actions[:, 6] = label_latent(actions[:, 6])
        return state, actions

    def decode(self, state10, latent32):
        value = np.array(latent32, dtype=np.float64, copy=True)
        if value.shape != (40, 32) or not np.isfinite(value).all():
            raise ValueError('FULL_FINITE_INVERSE_NORMALIZED_LATENT_H40_REQUIRED')
        value[:, 6] = decode_latent(value[:, 6])
        return super().decode(state10, value)


def range_audit(grip):
    value = np.asarray(grip, dtype=np.float64)
    if value.shape != (40,) or not np.isfinite(value).all():
        raise ValueError('ORIGINAL_FINITE_H40_GRIPPER_REQUIRED')
    low, high = value < 0, value > 1
    outside = low | high
    return dict(full_H40_legal=bool(not outside.any()),
                first_legal=bool(not outside[0]), last_legal=bool(not outside[-1]),
                endpoints_legal=bool(not outside[[0, -1]].any()),
                interior_legal=bool(not outside[1:-1].any()),
                first_outside=int(outside[0]), last_outside=int(outside[-1]),
                interior_outside=int(outside[1:-1].sum()),
                below_zero=int(low.sum()), above_one=int(high.sum()),
                maximum_lower_violation=float(np.maximum(-value, 0).max()),
                maximum_upper_violation=float(np.maximum(value - 1, 0).max()),
                minimum=float(value.min()), maximum=float(value.max()))
