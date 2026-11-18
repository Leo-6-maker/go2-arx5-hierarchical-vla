"""Sample distal and carried-object geometry on the Teacher's measured state."""
import numpy as np

from .multitask_contact_v6 import OBJECT_GEOMS
from .native_replay import forbid_integration

DISTAL_CARRY_STEP_M = .004
DISTAL_TURN_STEP_RAD = np.deg2rad(5.)


def world_yaw(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


class DistalCarryGeometry:
    def __init__(self, model, selected):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.target = model.geom(OBJECT_GEOMS[selected]).id
        self.other = model.geom(OBJECT_GEOMS['blue' if selected == 'red' else 'red']).id
        self.table = model.geom('task_table').id
        wrist = model.body('x5_link6').id
        bodies = {wrist}
        for body in range(wrist + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        self.distal = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies
                       and (model.geom_contype[i] or model.geom_conaffinity[i])]
        if not {model.geom('task_left_pad').id, model.geom('task_right_pad').id}.issubset(self.distal):
            raise ValueError('Exact wrist and both finger pad colliders required')
        self.pieces = self.distal + [self.target]
        self.distance_buffer = np.zeros(6)

    def _minimum(self, start, displacement, pairs):
        import mujoco

        samples = max(1, int(np.ceil(np.linalg.norm(displacement) / DISTAL_CARRY_STEP_M)))
        distance = float('inf')
        for i in range(samples + 1):
            self.data.geom_xpos[self.pieces] = start + displacement * (i / samples)
            for part, obstacle in pairs:
                gap = mujoco.mj_geomDistance(self.model, self.data, part, obstacle, 5., self.distance_buffer)
                distance = min(distance, float(gap))
        return distance

    def clearance(self, sample, xy, *, support_z=None, yaw_rad=0.):
        import mujoco

        with forbid_integration(mujoco):
            self.data.qpos[:] = sample['qpos']
            self.data.qvel[:] = sample['qvel']
            mujoco.mj_fwdPosition(self.model, self.data)
            start = self.data.geom_xpos[self.pieces].copy()
            target = start[-1]
            planar = np.r_[np.asarray(xy) - target[:2], 0.]
            against_other = [(piece, self.other) for piece in self.pieces]
            if support_z is None:
                against_table = [(piece, self.table) for piece in self.pieces]
                pairs = against_other + against_table
                if not yaw_rad:
                    return self._minimum(start, planar, pairs)
                initial_rotations = self.data.geom_xmat[self.pieces].reshape((-1, 3, 3)).copy()
                samples = int(np.ceil(abs(yaw_rad) / DISTAL_TURN_STEP_RAD))
                distance = float('inf')
                for angle in np.linspace(0., yaw_rad, samples + 1):
                    turn = world_yaw(angle)
                    turned = target + (start - target) @ turn.T
                    self.data.geom_xmat[self.pieces] = (turn @ initial_rotations).reshape((-1, 9))
                    distance = min(distance, self._minimum(turned, planar, pairs))
                return distance

            # Selected-object support contact is the intended end of lowering.
            against_table = [(piece, self.table) for piece in self.distal]
            return self._minimum(start, planar + [0., 0., support_z - target[2]],
                                 against_other + against_table)
