"""Object-relative source expert geometry, never a deployed student adapter.

These references require contact-physics execution and the existing data audit.
They are neither successful demonstrations nor actually committed EEF10 labels.
No robot joint command, WBC compensation or physics stepping occurs here.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np


def pose(value):
    p = np.array(value, dtype=float, copy=True)
    if (p.shape != (4, 4) or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(p[:3, :3])-1) > 1e-8):
        raise ValueError('FINITE_SE3_REQUIRED')
    return p


def inverse(value):
    p = pose(value)
    result = np.eye(4)
    result[:3, :3] = p[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ p[:3, 3]
    return result


def axis_rotation(axis, angle):
    a = np.asarray(axis, dtype=float)
    if (a.shape != (3,) or not np.isfinite(a).all() or not math.isfinite(angle)
            or abs(np.linalg.norm(a)-1) > 1e-8):
        raise ValueError('EXPLICIT_UNIT_HINGE_AXIS_AND_FINITE_ANGLE')
    x, y, z = a
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
    return result


@dataclass(frozen=True)
class DoorTurnReference:
    times_s: np.ndarray
    tcp_world: np.ndarray
    hinge_angles_rad: np.ndarray
    role: str = 'PRIVILEGED_SOURCE_GEOMETRY_UNEXECUTED'


def door_turn(*, hinge_world, axis_hinge, actual_tcp_world, actual_handle_world,
              actual_angle_rad, target_angle_rad, actual_grasp_valid,
              angle_limits_rad, max_linear_m_s=.06, max_angular_rad_s=.3):
    """Continue an actual handle grasp around its hinge from the current state.

    Recompute from measured handle/TCP/angle after any slip or recovery. The
    fixed actual tool-to-handle relation determines every future tool pose;
    neither an old demonstration timestamp nor an ideal grasp offset is used.
    Limits are geometry/reference limits, not contact or WBC safety proof.
    """
    if actual_grasp_valid is not True:
        raise ValueError('TURN_REQUIRES_ACTUAL_VALID_GRASP_REAPPROACH_AFTER_SLIP')
    h, tool, handle = map(pose, (hinge_world, actual_tcp_world, actual_handle_world))
    limits = np.asarray(angle_limits_rad, dtype=float)
    if (limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]
            or not all(math.isfinite(v) and limits[0] <= v <= limits[1]
                       for v in (actual_angle_rad, target_angle_rad))
            or not 0 < max_linear_m_s <= .06 or not 0 < max_angular_rad_s <= .3):
        raise ValueError('DECLARED_HINGE_LIMITS_AND_REFERENCE_SPEEDS_REQUIRED')
    # X_H_handle at hinge zero is derived from the current actual door angle.
    handle_zero = inverse(axis_rotation(axis_hinge, actual_angle_rad)) @ inverse(h) @ handle
    tool_to_handle = inverse(tool) @ handle
    tool_zero = handle_zero @ inverse(tool_to_handle)
    a = np.asarray(axis_hinge)
    radius = np.linalg.norm(tool_zero[:3, 3]-a*np.dot(a, tool_zero[:3, 3]))
    delta = target_angle_rad-actual_angle_rad
    duration = max(abs(delta)/max_angular_rad_s, radius*abs(delta)/max_linear_m_s)
    ticks = max(1, math.ceil(duration/.05))
    if ticks > 640:
        raise ValueError('TURN_REFERENCE_EXCEEDS_32_SECONDS')
    angles = np.linspace(actual_angle_rad, target_angle_rad, ticks+1)
    future = np.stack([h @ axis_rotation(axis_hinge, v) @ tool_zero for v in angles])
    if not np.allclose(future[0], tool, atol=1e-8, rtol=0):
        raise ValueError('SOURCE_REFERENCE_MUST_START_AT_ACTUAL_TCP')
    return DoorTurnReference(np.arange(ticks+1)*.05, future, angles)


@dataclass(frozen=True)
class BookInsertionTargets:
    preinsert_tcp_world: np.ndarray
    insert_tcp_world: np.ndarray
    actual_tool_to_book: np.ndarray
    final_corner_clearance_m: np.ndarray
    role: str = 'PRIVILEGED_SOURCE_GEOMETRY_UNEXECUTED'


def book_insertion(*, actual_tcp_world, actual_book_world, target_book_world,
                   shelf_world, book_size_m, cavity_bounds_shelf_m,
                   actual_grasp_valid, approach_clearance_m=.08):
    """Solve tool targets for a real, possibly off-center book grasp.

    Shelf +X is the insertion direction; X=0 is the opening plane. Bounds
    describe the empty rectangular cavity in that frame. All eight final
    book corners must fit. This checks geometry, not swept/contact feasibility.
    A source motion planner must connect the actual TCP to preinsert safely.
    """
    if actual_grasp_valid is not True:
        raise ValueError('INSERTION_REQUIRES_ACTUAL_VALID_GRASP')
    tool, book, goal, shelf = map(pose, (actual_tcp_world, actual_book_world,
                                       target_book_world, shelf_world))
    size, bounds = np.asarray(book_size_m, dtype=float), np.asarray(cavity_bounds_shelf_m, dtype=float)
    if (size.shape != (3,) or bounds.shape != (3, 2) or not np.isfinite(size).all()
            or not np.isfinite(bounds).all() or np.any(size <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0]) or abs(bounds[0, 0]) > 1e-9
            or not math.isfinite(approach_clearance_m) or not .02 <= approach_clearance_m <= .3):
        raise ValueError('FINITE_BOOK_AND_CAVITY_GEOMETRY_REQUIRED')
    corners = np.array(list(product((-1., 1.), repeat=3)))*size/2
    goal_shelf = inverse(shelf) @ goal
    final = corners @ goal_shelf[:3, :3].T+goal_shelf[:3, 3]
    margins = np.stack([final.min(axis=0)-bounds[:, 0], bounds[:, 1]-final.max(axis=0)])
    if margins.min() < -1e-9:
        raise ValueError('BOOK_DOES_NOT_FIT_DECLARED_TARGET_SLOT')
    tool_book = inverse(tool) @ book
    pre_shelf = goal_shelf.copy()
    # The leading corner starts outside the opening, including for tilted books.
    pre_shelf[0, 3] -= final[:, 0].max()+approach_clearance_m
    return BookInsertionTargets(shelf @ pre_shelf @ inverse(tool_book),
        goal @ inverse(tool_book), tool_book, margins)
