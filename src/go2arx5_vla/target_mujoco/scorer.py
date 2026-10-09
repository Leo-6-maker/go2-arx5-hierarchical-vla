"""PI0288 physical scorer; reuse only engine-independent task geometry."""
from __future__ import annotations

import numpy as np

from ..target_sim.scorer import (BOOL_FIELDS as BASE_BOOL_FIELDS, boolean_array,
    finite_array, physical_grasp as bilateral_grasp, pose_matrix, score_episode as base_score,
    sustained)

BOOL_FIELDS = (*BASE_BOOL_FIELDS, "opposing_pad_contact", "relative_motion_valid",
               "numerical_bound_warning")


def physical_grasp(data):
    """Contact calibration must prove opposition and measured relative motion."""
    base = bilateral_grasp(data)
    opposing = boolean_array(data["opposing_pad_contact"], np.shape(base), "opposing_pad_contact")
    relative = boolean_array(data["relative_motion_valid"], np.shape(base), "relative_motion_valid")
    return base & opposing & relative


def score_episode(data, **kwargs):
    times = np.asarray(data["time_s"], dtype=float)
    grasp = physical_grasp(data)
    warning = boolean_array(data["numerical_bound_warning"], times.shape, "numerical_bound_warning")
    adapted = dict(data, left_object_contact=np.asarray(data["left_object_contact"]) & grasp)
    result = base_score(adapted, **kwargs)
    result.update(task_id="GO2_MJ_PLACE_V1", numerical_bound_warning=bool(warning.any()))
    result["safety_valid"] &= not bool(warning.any())
    result["complete_32s"] &= result["safety_valid"]
    result["persistent_success"] &= result["safety_valid"]
    return result
