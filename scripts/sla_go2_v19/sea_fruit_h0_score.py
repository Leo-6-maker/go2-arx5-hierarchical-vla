"""Read actual paused native packets for a stationary two-object HOLD."""

from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import require
from fault_reference import floor_feet
from go2arx5_vla.world_model.multitask_score_v6 import HOLD_SECONDS, LIFT, Score
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState
from go2arx5_vla.world_model.native_replay import forbid_integration

FEET = ("FL", "FR", "RL", "RR")
# Existing stationary Reach stance limits. Apply them throughout HOLD.
BASE_TRANSLATION_M = .020
BASE_ANGLE_DEG = 5.
FOOT_TRANSLATION_M = .005


class StationaryHoldScore:
    def __init__(self, model, geometry, *, stance_origin=None):
        self.state = TwoObjectState(model, "red")
        self.geometry = geometry
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        self.feet = [model.geom(f).id for f in FEET]
        self.base = model.body("base_link").id
        self.start = None
        self.last = None
        self.score = None
        self.initial = stance_origin
        self.success = False
        self.first_violation = None

    def update_packet(self, packet):
        import mujoco
        from scipy.spatial.transform import Rotation

        with forbid_integration(mujoco) as attempted:
            s = self.state.update(packet)
        require(not attempted, "stationary scorer attempted native integration")
        d = self.state.data
        pose = (d.xpos[self.base].copy(), d.xmat[self.base].reshape(3, 3).copy(),
                d.geom_xpos[self.feet].copy())
        t = s["native_step"]
        if self.start is None:
            if self.initial is None:
                self.initial = pose
            self.start = t
            self.score = Score("HOLD-red", s, self.geometry)
        origin, orientation, stance = self.initial
        drift = np.linalg.norm(pose[0] - origin)
        tilt = np.degrees(Rotation.from_matrix(pose[1] @ orientation.T).magnitude())
        foot = np.max(np.linalg.norm(pose[2] - stance, axis=1))
        support = set(floor_feet(self.names, packet)) == set(FEET)
        safe = bool(s["safety_valid"] and s["gripper_state_valid"] and support
                    and drift <= BASE_TRANSLATION_M and tilt <= BASE_ANGLE_DEG
                    and foot <= FOOT_TRANSLATION_M)
        if not safe and self.first_violation is None:
            self.first_violation = dict(tick=t, base_drift_m=float(drift),
                base_angle_deg=float(tilt), foot_drift_m=float(foot),
                all_feet_supported=support, contact_valid=s["safety_valid"],
                gripper_state_valid=s["gripper_state_valid"])
        row = self.score.update(dict(s, safety_valid=safe))
        self.last = t
        self.success = bool(safe and self.score.unsafe_states == 0 and self.score.gaps == 0
                            and not (self.score.wrong_grasp or self.score.wrong_contact)
                            and row["target_grasp"] and row["lift_m"] >= LIFT
                            and self.score.hold_since is not None
                            and s["time_s"] - self.score.hold_since >= HOLD_SECONDS - 1e-10)
        return self.success

    def result(self):
        require(self.score is not None, "missing stationary HOLD denominator")
        raw = self.score.result()
        return dict(
            raw, schema="SEA_STATIONARY_HOLD_NATIVE_SCORE", task_success=self.success,
            first_stance_or_contact_violation=self.first_violation,
            maximum_stance_drift_m=BASE_TRANSLATION_M,
            maximum_stance_angle_deg=BASE_ANGLE_DEG,
            maximum_foot_drift_m=FOOT_TRANSLATION_M,
        )
