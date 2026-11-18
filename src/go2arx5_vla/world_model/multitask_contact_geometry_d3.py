"""Read-only geometry and state evidence for the D3 Teacher."""

from dataclasses import dataclass

import numpy as np

from .mobile_continuous_teacher_v2 import rotation_vector
from .multitask_placement_geometry_d2 import ANGLE_STEP_RAD, POSITION_STEP_M, PlacementGeometry
from .multitask_score_v6 import STABLE_SPEED, TABLE_TOLERANCE

SUPPORT_CONFIRM_FRAMES = 40
SUPPORT_WAIT_TICKS = 1000
# Existing continuation IK convergence bounds, in metres and radians.
POSE_SOLVE_M = .00001
POSE_SOLVE_RAD = .0001


def pose_error(actual, expected, radius):
    displacement = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
    rotation = np.linalg.norm(rotation_vector(actual[:3, :3] @ expected[:3, :3].T))
    return float(displacement + radius * rotation)


@dataclass(frozen=True)
class AttachmentWitness:
    mount: np.ndarray
    tcp: np.ndarray
    object_pose: np.ndarray
    fingers: np.ndarray

    @classmethod
    def measured(cls, sample):
        return cls(*(np.asarray(sample[key]) for key in ('base_pose', 'tcp', 'object_pose', 'finger_q')))

    def relative_residual(self, observed, forecast, radius):
        nominal_mount_tcp = np.linalg.solve(self.mount, forecast)
        measured_mount_tcp = np.linalg.solve(observed.mount, observed.tcp)
        old_attachment = np.linalg.solve(self.tcp, self.object_pose)
        new_attachment = np.linalg.solve(observed.tcp, observed.object_pose)
        return (pose_error(measured_mount_tcp, nominal_mount_tcp, radius),
                pose_error(new_attachment, old_attachment, radius),
                float(np.max(np.abs(observed.fingers - self.fingers))))


def relative_margin(sample_margin, history, radius):
    worst = max((sum(entry) for entry in history), default=0.)
    return sample_margin + worst + POSE_SOLVE_M + radius * POSE_SOLVE_RAD


def sampling_margin(sample):
    radius = np.linalg.norm(sample['tcp'][:3, 3] - sample['object_pose'][:3, 3]) + .025
    return POSITION_STEP_M / 2. + radius * ANGLE_STEP_RAD / 2., radius


class SupportClock:
    """Use uninterrupted real packets, with the native force and object geometry."""

    def __init__(self, color, geometry):
        self.color = color
        self.half_size = np.asarray(geometry['object_half_sizes'][color])
        self.table_z = geometry['table_top_world_z']
        self.last_tick = None
        self.streak = 0
        self.supported = False
        self.maximum_streak = 0

    def observe(self, sample):
        tick = sample['native_step']
        target = sample['objects'][self.color]
        pose = np.asarray(target['pose'])
        bottom = pose[2, 3] - np.abs(pose[2, :3]) @ self.half_size
        self.supported = bool(target['table_contact']
                              and abs(bottom - self.table_z) <= TABLE_TOLERANCE
                              and np.linalg.norm(target['linear_velocity']) <= STABLE_SPEED)
        contiguous = self.last_tick is not None and tick == self.last_tick + 1
        self.streak = (self.streak + 1 if contiguous else 1) if self.supported else 0
        self.maximum_streak = max(self.maximum_streak, self.streak)
        self.last_tick = tick

    def confirmed(self, tick):
        return tick == self.last_tick and self.streak >= SUPPORT_CONFIRM_FRAMES


class ContactGeometry(PlacementGeometry):
    """Apply the relative margin to wrist/held-object pairs only."""

    def __init__(self, model, selected):
        super().__init__(model, selected)
        distal = {model.body('x5_link6').id}
        for body in range(1, model.nbody):
            if int(model.body_parentid[body]) in distal:
                distal.add(body)
        self.internal_pairs = tuple((a, b) for a, b in self.pairs
                                    if b == self.target and int(model.geom_bodyid[a]) in distal)

    def check_relative(self, sample, knots, external_margin, internal_margin):
        if internal_margin is None:
            return self.check(sample, knots, external_margin)
        all_pairs = self.pairs
        try:
            self.pairs = [pair for pair in all_pairs if pair not in self.internal_pairs]
            exterior = self.check(sample, knots, external_margin)
            if not exterior['passed']:
                return dict(exterior, pair_scope='EXTERNAL')
            self.pairs = self.internal_pairs
            interior = self.check(sample, knots, internal_margin)
        finally:
            self.pairs = all_pairs
        return dict(interior, pair_scope='HELD_WRIST', unchanged_external_clearance=exterior)
