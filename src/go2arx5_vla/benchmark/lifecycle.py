"""Explicit native episode/reference lifecycle semantics.

The delivered environment exposes several reset channels through one
``reset_buf`` tensor.  This module keeps the benchmark's episode outcome
separate from that tensor and, in particular, does not treat an achieved
projection at ``traj_s == traj_L`` as reference completion.  Completion is
defined by the reference time law (``s_ref``/``T``), while controller aborts
are defined by the native termination flags captured before reset.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

REFERENCE_COMPLETE = "REFERENCE_COMPLETE"
TIME_LIMIT = "TIME_LIMIT"
FALL = "FALL"
INVALID_STATE = "INVALID_STATE"
JOINT_LIMIT = "JOINT_LIMIT"
TRACKING_ABORT = "TRACKING_ABORT"
NATIVE_TASK_TERMINATION = "NATIVE_TASK_TERMINATION"
UNKNOWN = "UNKNOWN"


def _flag(flags: Mapping[str, Any], name: str) -> bool:
    value = flags.get(name, False)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.asarray(value).any())
    return bool(value)


def reference_is_complete(
    *,
    sim_time: float,
    reference_duration: float,
    s_ref: float | None = None,
    reference_length: float | None = None,
    dt: float = 0.02,
) -> bool:
    """Return whether the *reference* reached its endpoint.

    ``s_ref`` is the time-law projection, not the achieved ``traj_s``.  A
    small control-step tolerance makes the decision robust to the native
    ``dt`` increment.  The duration check remains authoritative when a time
    law is present.
    """

    eps = max(1e-6, 1.5 * abs(float(dt)))
    if np.isfinite(reference_duration) and sim_time >= reference_duration - eps:
        return True
    if (
        s_ref is not None
        and reference_length is not None
        and np.isfinite(s_ref)
        and np.isfinite(reference_length)
        and reference_length >= 0.0
        and s_ref >= reference_length - max(1e-6, 1e-5 * max(reference_length, 1.0))
        and np.isfinite(reference_duration)
        and reference_duration <= eps
    ):
        return True
    return False


def classify_termination(
    flags: Mapping[str, Any], *, reference_complete: bool = False
) -> str:
    """Map pre-reset native flags to one stable benchmark termination code."""

    if reference_complete:
        return REFERENCE_COMPLETE
    if _flag(flags, "invalid_state") or _flag(flags, "non_finite_state"):
        return INVALID_STATE
    # ``joint_limit_violation`` is a diagnostic margin check.  The delivered
    # runtime does not reset an episode for that check (the gripper can sit
    # just outside its soft limit), so it must not be promoted to a lifecycle
    # outcome.  A source-provided termination flag remains authoritative.
    if _flag(flags, "joint_limit") or _flag(flags, "joint_limit_termination"):
        return JOINT_LIMIT
    if _flag(flags, "traj_early_term") or _flag(flags, "tracking_abort"):
        return TRACKING_ABORT
    if _flag(flags, "body_height_buf") or _flag(flags, "fall"):
        return FALL
    if _flag(flags, "time_out_buf") or _flag(flags, "timeout"):
        return TIME_LIMIT
    if _flag(flags, "native_task_termination") or _flag(flags, "reset_buf"):
        return NATIVE_TASK_TERMINATION
    return UNKNOWN


def lifecycle_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize row-level lifecycle codes without silently filling gaps."""

    rows = list(rows)
    codes = []
    for row in rows:
        code = row.get("termination_code")
        if not code:
            code = classify_termination(row)
        codes.append(str(code))
    counts = Counter(codes)
    return {
        "row_count": len(rows),
        "termination_codes": dict(sorted(counts.items())),
        "reference_completion_rows": int(counts.get(REFERENCE_COMPLETE, 0)),
        "tracking_abort_rows": int(counts.get(TRACKING_ABORT, 0)),
        "unknown_rows": int(counts.get(UNKNOWN, 0)),
    }
