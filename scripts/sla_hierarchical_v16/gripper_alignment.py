"""Working-range width alignment in the already scheduled, pre-source pose interval."""

import numpy as np

from compiler import ease


def alignment_widths(t, phase, raw, previous_m, lead, contact):
    t, phase, raw = (np.asarray(x, dtype=float) for x in (t, phase, raw))
    if (
        raw.ndim != 1
        or not len(raw)
        or t.shape != raw.shape
        or phase.shape != raw.shape
        or t[0] != 0
        or not all(np.isfinite(x).all() for x in (t, phase, raw, [previous_m, lead]))
        or np.any(np.diff(t) <= 0)
        or np.any((raw < 0) | (raw > 1))
        or lead <= 0
    ):
        raise ValueError("invalid scheduled width data")
    before = (previous_m - 0.002) / 0.040
    delta_m = 0.002 + 0.040 * raw[0] - previous_m
    if delta_m == 0:
        before = raw[0]
    peak = 1.875 * abs(delta_m) / lead
    reasons = []
    if not 0.002 - 1e-12 <= previous_m <= 0.042 + 1e-12:
        reasons.append("PREVIOUS_COMMAND_OUTSIDE_WORKING_RANGE")
    # In a held state even a numerically slow event-free width change can release the object.
    if contact and abs(delta_m) > 1e-12:
        reasons.append("CONTACT_OR_GRASP_AT_HANDOFF")
    if (previous_m > 0.002 + 0.040 * 0.55 and raw[0] <= 0.55) or (
        previous_m < 0.002 + 0.040 * 0.65 and raw[0] >= 0.65
    ):
        reasons.append("UNATTRIBUTED_THRESHOLD_CROSSING")
    if peak > 0.04 + 1e-12:
        reasons.append("INSUFFICIENT_EXISTING_ALIGNMENT_TIME")
    mask = t < lead - 1e-9
    if np.any(phase[mask] != 0) or np.any(raw[mask] != raw[0]):
        raise ValueError("alignment would alter source motion")
    evidence = dict(
        previous_width_mm=1000 * previous_m,
        proposed_first_width_mm=1000 * (0.002 + 0.040 * raw[0]),
        continuous_peak_mm_s=1000 * peak,
        available_s=float(lead),
        rejections=reasons,
    )
    if reasons:
        return None, evidence
    changed = raw.copy()
    changed[mask] = before + (raw[0] - before) * ease(t[mask] / lead)
    # A conversion roundoff at the 2 or 42 mm endpoint is not a new gripper command.
    changed[mask] = np.clip(changed[mask], 0, 1)
    return changed, evidence
