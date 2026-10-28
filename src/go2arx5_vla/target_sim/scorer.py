"""Independent GO2_PLACE_V1 scoring; never reads teacher phases or commands."""

from __future__ import annotations

import itertools
from typing import Mapping

import numpy as np

BOOL_FIELDS = (
    "object_present", "contact_valid", "gripper_state_valid", "native_grasp",
    "left_object_contact", "right_object_contact", "safety_valid", "native_abort",
    "illegal_contact", "joint_limit_penetration", "reset", "takeover",
)


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite array {shape}")
    return result.copy()


def pose_matrix(value, name="pose"):
    """Require genuine SE(3), including the homogeneous last row."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape[-2:] != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name}: expected finite SE(3) matrix")
    rotation = pose[..., :3, :3]
    if (not np.allclose(pose[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name}: invalid SE(3)")
    return pose.copy()


def boolean_array(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind != "b":
        raise ValueError(f"{name}: expected boolean array {shape}")
    return result.copy()


def physical_grasp(data: Mapping):
    """Native calibrated grasp AND valid object-specific bilateral contacts.

    ``native_grasp`` must come from a calibrated physical sensor predicate,
    not the issued close command. Without object/contact/DOF validity this is
    false, including no-object, unilateral contact, and release controls.
    """
    values = [np.asarray(data[name]) for name in (
        "object_present", "contact_valid", "gripper_state_valid", "native_grasp",
        "left_object_contact", "right_object_contact",
    )]
    if any(value.dtype.kind != "b" for value in values):
        raise ValueError("grasp evidence must be boolean physical predicates")
    if len({value.shape for value in values}) != 1:
        raise ValueError("grasp evidence shapes disagree")
    return np.logical_and.reduce(values)


def sustained(times, flags, duration_s=.30):
    """Missing 20Hz samples interrupt a run, rather than proving duration."""
    start = None
    previous = None
    for timestamp, flag in zip(times, flags):
        if not flag or (previous is not None and timestamp - previous > .05 + 1e-8):
            start = None
        if flag:
            if start is None:
                start = timestamp
            if timestamp - start >= duration_s - 1e-8:
                return True
        previous = timestamp
    return False


def score_episode(data: Mapping, *, goal_center, initial_object_pose,
                  table_height_m, complete=True, evidence_source="UNVERIFIED"):
    """Score saved 20Hz physical arrays. CPU fixtures are not physical gates.

    Times are task elapsed seconds: include t=0 and t=32. The numerical
    1e-8 epsilon only handles floating point arithmetic, not clock drift or
    relaxation of physical thresholds. Native high-rate streams must retain
    their original file; pass an explicitly synchronized 20Hz stream here.
    """
    times = np.asarray(data["time_s"], dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be finite, nonempty and strictly increasing")
    n = len(times)
    obj = pose_matrix(finite_array(data["object_pose"], (n, 4, 4), "object_pose"))
    tcp = pose_matrix(finite_array(data["tcp_world"], (n, 4, 4), "tcp_world"))
    initial = pose_matrix(finite_array(initial_object_pose, (4, 4), "initial_object_pose"))
    goal = finite_array(goal_center, (3,), "goal_center")
    if not isinstance(complete, (bool, np.bool_)):
        raise ValueError("complete must be an explicit boolean")
    if not np.allclose(obj[0], initial, atol=1e-8, rtol=0):
        raise ValueError("initial_object_pose must match the actual t=0 physical state")
    if np.linalg.norm(goal[:2] - initial[:2, 3]) < .15:
        raise ValueError("source-goal separation must be at least 0.15 m")
    table = float(table_height_m)
    if not np.isfinite(table) or abs(goal[2] - (table + .02)) > 1e-8:
        raise ValueError("goal center must be at table height plus cube half edge")
    linear = finite_array(data["object_linear_velocity"], (n, 3), "object_linear_velocity")
    angular = finite_array(data["object_angular_velocity"], (n, 3), "object_angular_velocity")
    flags = {name: boolean_array(data[name], (n,), name) for name in BOOL_FIELDS}
    grasp = physical_grasp(flags)
    xyz = obj[:, :3, 3]
    offsets = np.asarray(list(itertools.product((-.02, .02), repeat=3)))
    corners = np.einsum("nij,kj->nki", obj[:, :3, :3], offsets) + xyz[:, None, :]
    predicates = {
        "zone_corners": (np.abs(corners[:, :, :2] - goal[:2]) <= .07).all(axis=(1, 2)),
        "table_height": np.abs(xyz[:, 2] - (table + .02)) <= .005,
        "linear_speed": np.linalg.norm(linear, axis=1) <= .02,
        "angular_speed": np.linalg.norm(angular, axis=1) <= .10,
        "released": ~grasp & ~flags["native_grasp"],
        "tcp_clear": np.linalg.norm(tcp[:, :3, 3] - xyz, axis=1) >= .08,
        "object_and_contacts_valid": flags["object_present"] & flags["contact_valid"] & flags["gripper_state_valid"],
    }
    hold_valid = np.logical_and.reduce(list(predicates.values()))
    expected_hold = np.arange(560, 601, dtype=np.float64) / 20
    hold_indices = []
    for timestamp in expected_hold:
        matches = np.flatnonzero(np.abs(times - timestamp) <= 1e-8)
        if len(matches) == 1:
            hold_indices.append(int(matches[0]))
    clock_valid = bool(n == 641 and np.all(np.abs(times - np.arange(641) / 20) <= 1e-8))
    safe = bool(flags["safety_valid"].all() and not any(flags[key].any() for key in (
        "native_abort", "illegal_contact", "joint_limit_penetration", "reset", "takeover")))
    lift = (xyz[:, 2] - initial[2, 3] >= .02) & grasp
    carry = (np.linalg.norm(xyz[:, :2] - initial[:2, 3], axis=1) >= .10) & grasp
    history = {"sustained_grasp": sustained(times, grasp),
               "sustained_lift": sustained(times, lift), "horizontal_carry": bool(carry.any())}
    full = bool(complete and clock_valid and safe)
    passed = int(hold_valid[hold_indices].sum())
    success = bool(full and all(history.values()) and len(hold_indices) == 41 and passed == 41)
    return {"task_id": "GO2_PLACE_V1", "evidence_source": str(evidence_source),
            "complete_32s": full, "clock_valid": clock_valid, "safety_valid": safe,
            "history": history, "hold_observed": len(hold_indices), "hold_pass": passed,
            "hold_subpredicates": {key: int(value[hold_indices].sum()) for key, value in predicates.items()},
            "persistent_success": success, "selected": "NONE", "deployment": "NONE"}
