"""Source ABI and prospective target label masks; never converts EEF10 to13."""
import numpy as np

from .codec import eef10

SOURCE_ABI = dict(frame='PI0291_MUJOCO_WORLD', tcp='ARX5_NATIVE_SITE_x5_ee',
    position_units='m', orientation='ROW_ROT6D', gripper='0_CLOSED_1_OPEN',
    disk_action='ABSOLUTE_WORLD', observation_hz=20, horizon=40, prefix=8)


def validate_source_abi(binding):
    if binding != SOURCE_ABI:
        raise ValueError('SOURCE_UNITS_TCP_FRAME_GRIPPER_OR_RELATIVE_REPRESENTATION_CHANGED')


def validate_label_mask(action, mask, *, domain, base_preference_valid):
    a, m = np.asarray(action), np.asarray(mask)
    width = 10 if domain == 'FIXED_BASE_ARX5_TASK_SIM' else 13 if domain == 'FULL_GO2_ARX5_TASK_SIM_EEF13_STATE19_V1' else None
    if (width is None or a.shape != (40, width) or m.shape != a.shape or m.dtype != np.bool_
            or not np.isfinite(a).all() or not m[:, :10].all() or type(base_preference_valid) is not bool):
        raise ValueError('EXPLICIT_DOMAIN_FULL_HORIZON_AND_DIMENSION_MASK_REQUIRED')
    eef10(a[:, :10])
    if width == 10:
        if base_preference_valid:
            raise ValueError('SOURCE_EEF10_HAS_NO_BASE_LABELS')
    elif base_preference_valid:
        if not m[:, 10:].all():raise ValueError('VALID_BASE_PREFERENCE_REQUIRES_REAL_LABEL_MASK')
    elif m[:, 10:].any() or np.any(a[:, 10:] != 0):
        raise ValueError('ABSENT_BASE_MUST_REMAIN_MASKED_STRUCTURAL_PADDING_NOT_SOFT_ZERO_LABEL')
    return m.copy()
