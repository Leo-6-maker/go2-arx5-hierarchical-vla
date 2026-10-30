"""Reference from an actual safe soft-joint boundary, never a state reset."""
import math
import numpy as np
from .object_relations import DoorTurnReference,axis_rotation,door_turn,pose

VERSION='NATIVE_SAFE_HINGE_ANCHOR_FUTURE_IN_ORIGINAL_RANGE_V1'


def from_native_boundary(*,hinge_world,axis_hinge,actual_tcp_world,actual_handle_world,
        actual_angle_rad,target_angle_rad,actual_grasp_valid,native_safety_valid,
        angle_limits_rad,max_linear_m_s=.06,max_angular_rad_s=.3):
    """Keep the measured anchor; every subsequent reference stays in range.

    Recenter the reference angle at the observed physical joint position.
    This does not clamp that position or change native limits/score. If one
    normal20Hz reference interval cannot enter the original range, reject.
    Original speed/horizon/actual-grasp checks still run in door_turn.
    """
    if native_safety_valid is not True:
        raise ValueError('NATIVE_SAFE_BOUNDARY_REFERENCE_REQUIRES_ACTUAL_SAFETY')
    limits=np.asarray(angle_limits_rad,dtype=float)
    if (limits.shape!=(2,)or not np.isfinite(limits).all()or limits[0]>=limits[1]
            or not math.isfinite(actual_angle_rad)or not math.isfinite(target_angle_rad)
            or not limits[0]<=target_angle_rad<=limits[1]):
        raise ValueError('ORIGINAL_FINITE_HINGE_RANGE_AND_TARGET_REQUIRED')
    common=dict(axis_hinge=axis_hinge,actual_tcp_world=actual_tcp_world,actual_handle_world=actual_handle_world,
        actual_grasp_valid=actual_grasp_valid,max_linear_m_s=max_linear_m_s,max_angular_rad_s=max_angular_rad_s)
    if limits[0]<=actual_angle_rad<=limits[1]:
        return door_turn(hinge_world=hinge_world,actual_angle_rad=actual_angle_rad,
            target_angle_rad=target_angle_rad,angle_limits_rad=angle_limits_rad,**common)
    # Local zero is the measured hinge frame, not a substituted physical qpos.
    delta=target_angle_rad-actual_angle_rad
    anchor=pose(hinge_world)@axis_rotation(axis_hinge,actual_angle_rad)
    local=door_turn(hinge_world=anchor,actual_angle_rad=0.,target_angle_rad=delta,
        angle_limits_rad=[min(0.,delta),max(0.,delta)],**common)
    angles=local.hinge_angles_rad+actual_angle_rad
    if not np.all((angles[1:]>=limits[0])&(angles[1:]<=limits[1])):
        raise ValueError('EVERY_FUTURE_HINGE_REFERENCE_MUST_ENTER_ORIGINAL_RANGE')
    if angles[0]!=actual_angle_rad:
        raise ValueError('MEASURED_HINGE_ANCHOR_MUST_NOT_BE_CLAMPED')
    return DoorTurnReference(local.times_s,local.tcp_world,angles)
