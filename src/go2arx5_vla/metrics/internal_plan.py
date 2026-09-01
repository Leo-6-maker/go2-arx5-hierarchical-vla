"""Analysis helpers for the delivered WBC's structured internal 9-D plan."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


PLAN_CHANNEL_NAMES = (
    "dv_x",
    "dv_y",
    "dv_yaw",
    "body_height",
    "body_pitch",
    "body_roll",
    "gait_frequency",
    "stance_width",
    "stance_length",
)


def _stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def analyze_internal_plan(
    data: Mapping[str, Any],
    *,
    burden: Optional[Sequence[float]] = None,
    boundary_time_s: Optional[float] = None,
) -> dict[str, Any]:
    """Summarize bounds and burden response without calling the plan latent."""

    if "internal_body_plan_9d" not in data:
        return {"status": "NOT_LOGGED", "channel_names": list(PLAN_CHANNEL_NAMES)}
    plan = np.asarray(data["internal_body_plan_9d"], dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 9:
        raise ValueError("internal_body_plan_9d must have shape [T,9]")
    finite = np.isfinite(plan).all(axis=1)
    result: dict[str, Any] = {
        "status": "FINITE" if bool(finite.all()) else "NON_FINITE",
        "channel_names": list(PLAN_CHANNEL_NAMES),
        "finite_fraction": float(np.mean(finite)),
        "channels": {name: _stats(plan[:, index]) for index, name in enumerate(PLAN_CHANNEL_NAMES)},
    }
    plan_norm = np.linalg.norm(plan[:, :3], axis=-1)
    result["delta_velocity_norm"] = _stats(plan_norm)
    if burden is not None:
        burden_array = np.asarray(burden, dtype=np.float64).reshape(-1)
        if len(burden_array) != len(plan):
            raise ValueError("burden must have one value per plan sample")
        result["burden"] = _stats(burden_array)
        result["correlation_with_burden"] = {
            name: _correlation(plan[:, index], burden_array)
            for index, name in enumerate(PLAN_CHANNEL_NAMES)
        }
    if boundary_time_s is not None and "sim_time" in data:
        time = np.asarray(data["sim_time"], dtype=np.float64).reshape(-1)
        before = time <= float(boundary_time_s)
        after = time > float(boundary_time_s)
        result["before_boundary"] = {
            "sample_count": int(before.sum()),
            "channel_means": {
                name: float(np.mean(plan[before, index])) if before.any() else float("nan")
                for index, name in enumerate(PLAN_CHANNEL_NAMES)
            },
        }
        result["after_boundary"] = {
            "sample_count": int(after.sum()),
            "channel_means": {
                name: float(np.mean(plan[after, index])) if after.any() else float("nan")
                for index, name in enumerate(PLAN_CHANNEL_NAMES)
            },
        }
    return result
