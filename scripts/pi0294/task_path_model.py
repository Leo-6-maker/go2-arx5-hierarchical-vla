"""Private kinematic hypotheses and the unchanged native reference-rate bounds.

Only cloned MjData is changed. These checks neither integrate physics nor
replace actual feedback/object motion, controller actions or training labels.
"""
import numpy as np

from go2arx5_vla.trajectory.so3 import matrix_to_quat, quat_to_matrix, quat_slerp
from .persistent_approach_draft import pose
from .task_path_reference import rates
from .mp_client import contact_permitted


class PlanningObjectMotion:
    def __init__(self, sample, *, mode='STATIC', push_direction=None):
        if mode not in ('STATIC', 'ACTUAL_GRASP_ATTACHMENT', 'ACTUAL_CONTACT_PUSH_TRANSLATION'):
            raise ValueError('DECLARED_PRIVATE_OBJECT_HYPOTHESIS_REQUIRED')
        if mode == 'ACTUAL_GRASP_ATTACHMENT' and sample.get('native_grasp') is not True:
            raise ValueError('ACTUAL_NATIVE_GRASP_REQUIRED_FOR_ATTACHMENT_HYPOTHESIS')
        if mode == 'ACTUAL_CONTACT_PUSH_TRANSLATION' and (
                sample.get('robot_object_contact') is not True or sample.get('native_grasp') is not False):
            raise ValueError('ACTUAL_UNGRASPED_PAD_CONTACT_REQUIRED_FOR_PUSH_HYPOTHESIS')
        self.mode, self.tcp, self.obj = mode, pose(sample['tcp']), pose(sample['object_pose'])
        self.relative = np.linalg.inv(self.tcp) @ self.obj
        self.direction = None
        if mode == 'ACTUAL_CONTACT_PUSH_TRANSLATION':
            d = np.asarray(push_direction, dtype=float)
            if d.shape != (2,) or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1) > 1e-8:
                raise ValueError('DECLARED_UNIT_PLANAR_PUSH_DIRECTION')
            self.direction = d.copy()

    def __call__(self, model, data):
        if self.mode == 'STATIC': return
        import mujoco
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'x5_ee')
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'task_cube_free')
        if min(site, joint) < 0: raise ValueError('FULL_NATIVE_TCP_AND_FREE_OBJECT_REQUIRED')
        p = np.eye(4); p[:3, :3] = data.site_xmat[site].reshape(3, 3); p[:3, 3] = data.site_xpos[site]
        future = self.obj.copy()
        if self.mode == 'ACTUAL_GRASP_ATTACHMENT': future = p @ self.relative
        else:
            travel = max(0., float((p[:2, 3]-self.tcp[:2, 3]) @ self.direction))
            future[:2, 3] += self.direction*travel
        address = int(model.jnt_qposadr[joint])
        data.qpos[address:address+3] = future[:3, 3]
        xyzw = matrix_to_quat(future[:3, :3])
        data.qpos[address+3:address+7] = np.r_[xyzw[3], xyzw[:3]]

    def evidence(self):
        return dict(mode=self.mode, role='PLANNED_OBJECT_HYPOTHESIS_NOT_OBSERVED_MOTION',
                    actual_physics_or_labels_modified=False)


def certify_chunk(model, sample, poses, *, stage, history=(), planning_object_motion=None):
    """500Hz SE3 interpolation/local IK, 20Hz .7rad/s and 1rad/s² bounds.

    The two most recently completed reference points are re-solved in the
    current actual root frame, as in TF-R3. They are command history, not old
    actual joint values. No values are repaired, clipped or slowed on failure.
    """
    import mujoco
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (40, 4, 4) or len(history) not in (0, 2):
        raise ValueError('H40_AND_ZERO_OR_TWO_EXECUTED_REFERENCE_POINTS')
    active = np.stack([pose(p) for p in poses])
    previous = np.stack([pose(r['pose']) for r in history]) if history else np.empty((0, 4, 4))
    if history and (history[-1]['end_native_step'] != sample['native_step']
                    or history[-2]['end_native_step'] != sample['native_step']-25):
        raise ValueError('REFERENCE_HISTORY_MUST_BE_LAST_TWO_PHYSICAL_INTERVALS')
    full = np.concatenate([previous, active]); prefix = len(history)
    linear, angular = rates(full, np.arange(len(full))*.05)
    if linear > .06+1e-10 or angular > .3+1e-10:
        raise ValueError('ORIGINAL_CARTESIAN_RATE_REJECT_AT_ACTUAL_REFERENCE_JOIN')
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = sample['qpos'], sample['qvel'], sample['ctrl']
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'x5_joint'+str(i)) for i in range(1, 7)])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'x5_ee')
    if min([site, *joints]) < 0: raise ValueError('EXACT_NATIVE_CHAIN_REQUIRED')
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
    extended = np.concatenate([full, full[-1:]])
    qs, iterations, points = [], 0, 0
    for tick in range(len(full)*25+1):
        at = tick*.002; i = min(int(at/.05), len(full)-1); fraction = (at-i*.05)/.05
        target = np.eye(4)
        target[:3, 3] = (1-fraction)*extended[i, :3, 3]+fraction*extended[i+1, :3, 3]
        target[:3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(extended[i, :3, :3]),
                                                matrix_to_quat(extended[i+1, :3, :3]), fraction))
        points += 1
        for iteration in range(5):
            iterations += 1; mujoco.mj_forward(model, data)
            ep = target[:3, 3]-data.site_xpos[site]
            delta = matrix_to_quat(target[:3, :3] @ data.site_xmat[site].reshape(3, 3).T)
            if delta[3] < 0: delta = -delta
            length = np.linalg.norm(delta[:3])
            er = delta[:3]*(2*np.arctan2(length, delta[3])/max(length, 1e-15))
            if iteration and np.linalg.norm(ep) <= 1e-5 and np.linalg.norm(er) <= 1e-4: break
            mujoco.mj_jacSite(model, data, jp, jr, site)
            change = np.linalg.lstsq(np.vstack([jp[:, va], jr[:, va]]), np.r_[ep, er], rcond=1e-10)[0]
            if not np.isfinite(change).all(): raise ValueError('ORIGINAL_NONFINITE_LOCAL_IK_REJECT')
            data.qpos[qa] += change
        else: raise ValueError('ORIGINAL_FIVE_ITERATION_NATIVE_IK_REJECT')
        if np.any(data.qpos[qa] < model.jnt_range[joints, 0]) or np.any(data.qpos[qa] > model.jnt_range[joints, 1]):
            raise ValueError('ORIGINAL_NATIVE_JOINT_LIMIT_REJECT_NO_CLIP')
        if planning_object_motion is not None: planning_object_motion(model, data)
        mujoco.mj_forward(model, data)
        for c in data.contact if tick >= prefix*25 else ():
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (c.geom1, c.geom2)]
            if not contact_permitted(names, c.dist, stage): raise ValueError('ORIGINAL_FULL_NATIVE_CONTACT_REJECT:'+str(names))
        qs.append(data.qpos[qa].copy())
    command_q = np.asarray(qs)[np.arange(len(full))*25]
    velocity = np.diff(command_q, axis=0)/.05
    acceleration = np.diff(velocity, axis=0)/.05
    if np.max(np.abs(velocity)) > .7*(1+1e-10): raise ValueError('ORIGINAL_JOINT_SPEED_REJECT')
    if np.max(np.abs(acceleration)) > 1.*(1+1e-10): raise ValueError('ORIGINAL_JOINT_ACCELERATION_REJECT')
    return dict(status='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF',
        points=points, local_IK_iterations=iterations, physics_integrations=0,
        actual_state_modified=False, max_joint_speed_rad_s=float(np.abs(velocity).max()),
        max_joint_acceleration_rad_s2=float(np.abs(acceleration).max()),
        executed_reference_history_points=prefix, Cartesian_reference_values_changed=False)
