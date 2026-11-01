"""Declared two-prototype book diagnostic, separate from raw-policy DEV."""
import numpy as np

KIND = 'V7_BOOK_BINARY_GRIP_DIAGNOSTIC'
VERSION = 'BOOK_EXPERT_TWO_PROTOTYPE_NEAREST_V1'
CLOSED = .002 / .044
OPEN = .042 / .044


def decode(actions):
    """Use model output alone; never inspect task/contact/support truth."""
    value = np.asarray(actions)
    if value.shape != (40, 10) or not np.isfinite(value).all():
        raise ValueError('FINITE_H40_EEF10_REQUIRED')
    result = value.copy()
    result[:, 9] = np.where(value[:, 9] < .5, CLOSED, OPEN)
    return result


def validate_manifest(manifest):
    if manifest['kind'] != KIND or manifest.get('grip_decoder') != VERSION:
        raise ValueError('EXPLICIT_VERSIONED_GRIP_DIAGNOSTIC_REQUIRED')
    if manifest['limits']['optimizer_updates'] != 0:
        raise ValueError('DECODER_DIAGNOSTIC_HAS_NO_TRAINING')
