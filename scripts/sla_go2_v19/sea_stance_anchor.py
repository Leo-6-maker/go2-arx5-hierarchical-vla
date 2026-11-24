"""One measured constant BASE reference, elected before the standing HOLD task."""

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M, WORKING_OPEN_M
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import document

from demonstrated_grip import require
from observation_stream import rigid

PROFILE = "ONE_MEASURED_MPC_BASE_HOLD_BEFORE_TASK"
SETTLING_TICKS = 500


def reference(old_base, held_eef, measured_mpc_mount):
    """Keep the old accepted EEF. Anchor BASE at one measured estimator pose."""
    old = rigid(old_base)
    ee = rigid(held_eef)
    mount = rigid(measured_mpc_mount)
    action = np.repeat(pose10(ee, WORKING_OPEN_M / MECHANICAL_TRAVEL_M)[None], 40, axis=0)
    base = np.repeat(mount[None], 41, axis=0)
    angle = Rotation.from_matrix(mount[:3, :3] @ old[:3, :3].T).magnitude()
    return action, base, dict(
        profile=PROFILE,
        old_base_world=old.tolist(),
        measured_MPC_mount_world=mount.tolist(),
        old_EEF_accepted_world=ee.tolist(),
        base_reference_shift_mm=(1000*(mount[:3, 3] - old[:3, 3])).tolist(),
        base_reference_rotation_deg=float(np.degrees(angle)),
    )


def check_document(action, base):
    """Check the existing M_ASSIST wire without involving the planner or physics."""
    wire = document(action, mode="M_ASSIST", base_poses=base)
    hold = wire["base"]["waypoints"]
    arm = wire["ee"]["waypoints"]
    t = np.arange(41)*.05
    for points, target in ((arm, matrix10(action[0])), (hold, base[0])):
        require(len(points) == 41 and np.allclose([p["time"] for p in points], t, rtol=0, atol=1e-10)
                and np.allclose([p["position"] for p in points], target[:3, 3], rtol=0, atol=1e-10)
                and np.allclose(Rotation.from_quat([p["quaternion_xyzw"] for p in points]).as_matrix(),
                                target[:3, :3], rtol=0, atol=1e-10),
                "whole time grid must hold the original EEF or elected BASE")
    return wire
