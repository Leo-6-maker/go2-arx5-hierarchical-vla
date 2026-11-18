"""Kinematic Teacher hypotheses over the frozen native scene, without integration."""

from dataclasses import dataclass

import numpy as np

from ..trajectory.so3 import matrix_to_quat, quat_to_matrix, quat_slerp
from .mobile_continuous_teacher_v2 import rotation_vector
from .multitask_contact_v6 import OBJECT_GEOMS, PENETRATION_LIMIT, preview_permitted
from .multitask_scene_v6 import OBJECTS
from .native_replay import forbid_integration

POSITION_STEP_M = .004
ANGLE_STEP_RAD = np.deg2rad(3.)
FINGER_STEP_M = .004
OPEN_QPOS_M = .040


@dataclass(frozen=True)
class Knot:
    tcp: np.ndarray
    object_pose: np.ndarray
    finger_q: np.ndarray
    root_xy: np.ndarray
    stage: str


def transform(rotation, position):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def translate(pose, offset):
    result = np.asarray(pose).copy()
    result[:3, 3] += offset
    return result


def interpolate_pose(a, b, amount):
    rotation = quat_to_matrix(quat_slerp(matrix_to_quat(a[:3, :3]),
                                         matrix_to_quat(b[:3, :3]), amount))
    return transform(rotation, a[:3, 3]*(1.-amount)+b[:3, 3]*amount)


def landing_tcp(tcp_to_object, xy, rotation, half_size, table_z):
    """Preserve the grasp attachment and use the rotated object's true bottom."""
    object_rotation = rotation @ tcp_to_object[:3, :3]
    center = np.r_[xy, table_z + np.abs(object_rotation[2]) @ half_size]
    return transform(object_rotation, center) @ np.linalg.inv(tcp_to_object)


def remaining_placement(start, object_pose, landing, fingers, root_xy, *, supported,
                        withdrawal, open_qpos_m=OPEN_QPOS_M):
    """Rebuild down/open/exit from current measured attachment and jaw state."""
    attachment = np.linalg.solve(start, object_pose)
    supported_object = object_pose if supported else landing @ attachment
    finish = start if supported else landing
    return (
        Knot(start, object_pose, fingers, root_xy, 'lower'),
        Knot(finish, supported_object, fingers, root_xy, 'lower'),
        Knot(finish, supported_object, np.full(2, open_qpos_m), root_xy, 'open'),
        Knot(translate(finish, withdrawal), supported_object,
             np.full(2, open_qpos_m), root_xy, 'withdraw'),
    )


class PlacementGeometry:
    """Continuation IK and pairwise collision proximity at bounded path samples."""

    def __init__(self, model, selected):
        import mujoco

        self.model, self.selected = model, selected
        self.data = mujoco.MjData(model)
        self.site = model.site('x5_ee').id
        joints = [model.joint(f'x5_joint{i}').id for i in range(1, 7)]
        self.qa = model.jnt_qposadr[joints]
        self.va = model.jnt_dofadr[joints]
        self.range = model.jnt_range[joints]
        self.fingers = [model.joint(f'task_{side}_slide').qposadr[0] for side in ('left', 'right')]
        self.pads = {model.geom(f'task_{side}_pad').id for side in ('left', 'right')}
        self.target = model.geom(OBJECT_GEOMS[selected]).id
        self.other = model.geom(OBJECT_GEOMS['blue' if selected == 'red' else 'red']).id
        self.table = model.geom('task_table').id
        self.floor = model.geom('floor').id
        self.object_q = int(model.joint(OBJECTS[selected]['joint']).qposadr[0])
        arm = {model.body('x5_base_link').id}
        for body in range(1, model.nbody):
            if int(model.body_parentid[body]) in arm:
                arm.add(body)
        self.arm = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.pairs = [(g, o) for g in self.arm for o in (self.table, self.other, self.target, self.floor)]
        self.pairs.append((self.target, self.other))
        self.names = [model.geom(g).name for g in range(model.ngeom)]
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.buffer = np.zeros(6)

    def solve(self, target):
        import mujoco

        m, d = self.model, self.data
        for i in range(6):
            mujoco.mj_kinematics(m, d)
            ep = target[:3, 3]-d.site_xpos[self.site]
            er = rotation_vector(target[:3, :3]@d.site_xmat[self.site].reshape(3,3).T)
            if np.linalg.norm(ep) <= 1e-5 and np.linalg.norm(er) <= 1e-4:
                return bool(np.all((d.qpos[self.qa] >= self.range[:,0])
                                   & (d.qpos[self.qa] <= self.range[:,1])))
            if i == 5:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            d.qpos[self.qa] += np.linalg.lstsq(np.vstack((self.jp[:,self.va], self.jr[:,self.va])),
                                               np.r_[ep, er], rcond=1e-10)[0]
        return False

    def sample(self, tcp, object_pose, fingers, root_xy, stage):
        import mujoco

        m, d = self.model, self.data
        if len(root_xy) == 7:
            d.qpos[:7] = root_xy
        else:
            d.qpos[:2] = root_xy
        d.qpos[self.fingers] = fingers
        d.qpos[self.object_q:self.object_q+3] = object_pose[:3, 3]
        q = matrix_to_quat(object_pose[:3,:3])
        d.qpos[self.object_q+3:self.object_q+7] = np.r_[q[3], q[:3]]
        if not self.solve(tcp):
            return None, 'KINEMATIC_LOCAL_IK_OR_JOINT_RANGE', None
        mujoco.mj_fwdPosition(m, d)
        for contact in d.contact:
            if self.arm_involved(contact.geom1, contact.geom2):
                names = (self.names[contact.geom1], self.names[contact.geom2])
                if not preview_permitted(*names, contact.dist, stage, self.selected):
                    return float(contact.dist), 'KINEMATIC_FORBIDDEN_CONTACT', names

        gap, pair = float('inf'), None
        for a, b in self.pairs:
            distance = mujoco.mj_geomDistance(m, d, a, b, 5., self.buffer)
            if a in self.pads and b == self.target and stage not in ('approach', 'align', 'withdraw'):
                # The declared pads may touch the selected object during
                # grasp/release. Their penetration limit remains in the oracle.
                if distance < -PENETRATION_LIMIT:
                    return float(distance), 'KINEMATIC_GRASP_PENETRATION', (self.names[a], self.names[b])
                continue
            if distance < gap:
                gap, pair = float(distance), (self.names[a], self.names[b])
        return gap, None, pair

    def arm_involved(self, a, b):
        return a in self.arm or b in self.arm or {a,b} == {self.target, self.other}

    def check(self, sample, knots, margin):
        import mujoco

        if len(knots) < 2 or not np.isfinite(margin) or margin < 0.:
            raise ValueError('FINITE_MULTISEGMENT_TEACHER_HYPOTHESIS_REQUIRED')
        d = self.data
        d.qpos[:], d.qvel[:], d.ctrl[:] = sample['qpos'], sample['qvel'], sample['ctrl']
        minimum = float('inf')
        pair = None
        count = 0
        with forbid_integration(mujoco):
            for segment, (a, b) in enumerate(zip(knots, knots[1:])):
                translation = np.linalg.norm(b.tcp[:3,3]-a.tcp[:3,3])
                object_travel = np.linalg.norm(b.object_pose[:3,3]-a.object_pose[:3,3])
                angle = np.linalg.norm(rotation_vector(b.tcp[:3,:3]@a.tcp[:3,:3].T))
                jaw = np.max(np.abs(b.finger_q-a.finger_q))
                left_root, right_root = a.root_xy, b.root_xy
                root = np.linalg.norm(right_root[:3] - left_root[:3])
                full_root = len(left_root) == 7
                root_angle = 0.
                if full_root:
                    left_quat = left_root[[4, 5, 6, 3]]
                    right_quat = right_root[[4, 5, 6, 3]]
                    root_angle = 2. * np.arccos(np.clip(abs(left_quat @ right_quat), 0., 1.))
                segments = int(np.ceil(max(1., translation/POSITION_STEP_M, object_travel/POSITION_STEP_M,
                                           angle/ANGLE_STEP_RAD, jaw/FINGER_STEP_M, root/POSITION_STEP_M,
                                           root_angle/ANGLE_STEP_RAD)))
                for index in range(segments+1):
                    u = index/segments
                    tcp = interpolate_pose(a.tcp, b.tcp, u)
                    if b.stage in ('carry', 'lift', 'lower'):
                        # A rigid in-hand object follows full SE(3), including
                        # the curved motion induced by changing tool rotation.
                        obj = tcp @ np.linalg.solve(a.tcp, a.object_pose)
                    else:
                        obj = interpolate_pose(a.object_pose, b.object_pose, u)
                    root_pose = (1.-u)*left_root + u*right_root
                    if full_root:
                        root_pose[3:7] = quat_slerp(left_quat, right_quat, u)[[3, 0, 1, 2]]
                    gap, reason, names = self.sample(tcp, obj, (1.-u)*a.finger_q+u*b.finger_q,
                                                    root_pose, b.stage)
                    count += 1
                    if gap is not None and gap < minimum:
                        minimum, pair = gap, names
                    if reason or gap < margin:
                        return dict(passed=False, reason=reason or 'TEACHER_CLEARANCE_MARGIN',
                                    phase=b.stage, sample_count=count, gap_m=gap, pair=names, margin_m=margin,
                                    segment=segment, segment_fraction=u)
        return dict(passed=True, phase=knots[-1].stage, sample_count=count, gap_m=minimum,
                    pair=pair, margin_m=margin)
