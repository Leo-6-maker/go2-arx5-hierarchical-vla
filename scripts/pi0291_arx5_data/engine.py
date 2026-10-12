from __future__ import annotations

import itertools
import os
import numpy as np
import mujoco
try:
    from .config import PARAMETERS as P
    from .scene import ARM_NAMES, FINGER_NAMES, TCP_OFFSET, DOWNWARD_ROTATION, PAD_INNER_GAP_AT_ZERO, xml_scene
except ImportError:
    from config import PARAMETERS as P
    from scene import ARM_NAMES, FINGER_NAMES, TCP_OFFSET, DOWNWARD_ROTATION, PAD_INNER_GAP_AT_ZERO, xml_scene

POINTS = np.asarray(list(itertools.product((-.02, .02), repeat=3)))


def rotation_error(target, current):
    delta = target @ current.T
    angle = np.arccos(np.clip((np.trace(delta)-1)/2, -1., 1.))
    vector = .5*np.array([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0], delta[1, 0]-delta[0, 1]])
    if angle < 1e-6:
        return vector
    if abs(np.sin(angle)) < 1e-5:
        values, vectors = np.linalg.eigh((delta+delta.T)/2)
        return angle*vectors[:, np.argmax(values)]
    return vector*(angle/np.sin(angle))


class Engine:
    def __init__(self, card, mode='normal', asset_root=None):
        self.card, self.mode = card, mode
        asset_root = asset_root or card.get('asset_root') or os.environ.get('PI0291_NATIVE_ASSET_ROOT')
        if not asset_root:
            raise ValueError('Exact native MJCF asset_root must be privately bound')
        scene_card = dict(card, mode=mode)
        self.xml = xml_scene(scene_card, asset_root, P)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.ids = {name: self._id(mujoco.mjtObj.mjOBJ_BODY, actual) for name, actual in
                    [('tool', 'x5_link6'), ('cube', 'cube'), ('left_finger', 'x5_link7'), ('right_finger', 'x5_link8')]}
        self.geoms = {name: self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in
                      ('cube_geom', 'left_pad', 'right_pad', 'table', 'floor')}
        self.tcp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, 'x5_ee')
        self.joints = {name: self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_NAMES+FINGER_NAMES+('cube_free',)}
        self.arm_q = np.array([self.model.jnt_qposadr[self.joints[x]] for x in ARM_NAMES], dtype=int)
        self.arm_v = np.array([self.model.jnt_dofadr[self.joints[x]] for x in ARM_NAMES], dtype=int)
        self.finger_q = np.array([self.model.jnt_qposadr[self.joints[x]] for x in FINGER_NAMES], dtype=int)
        self.finger_v = np.array([self.model.jnt_dofadr[self.joints[x]] for x in FINGER_NAMES], dtype=int)
        self.arm_ctrl = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in ARM_NAMES])
        self.finger_ctrl = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in FINGER_NAMES])
        self.arm_bounds = self.model.jnt_range[[self.joints[x] for x in ARM_NAMES]].copy()
        self.arm_bounds[:, 0] += .005
        self.arm_bounds[:, 1] -= .005
        self.data.qpos[self.finger_q] = P['finger_travel']
        initial = np.eye(4)
        initial[:3, :3] = DOWNWARD_ROTATION
        initial[:3, 3] = [*card['source_xy'], P['table_top']+.10]
        initial = np.asarray(card.get('initial_tcp', initial), dtype=float)
        self.ik_data.qpos[:] = self.data.qpos
        solution, error = self.solve_ik(initial, np.array([0., 2.13, 1.73, -1.17, 0., 0.]), iterations=150)
        if error[0] > .0002 or error[1] > .002:
            raise ValueError(f'Initial native IK unreachable: {error}')
        self.data.qpos[self.arm_q] = solution
        self.arm_targets = solution.copy()
        self.finger_targets = np.full(2, P['finger_travel'])
        self.steps, self.safety, self.renderer = 0, [], None
        self.last_ik_error = error
        assert self.model.neq == 0 and self.model.nmocap == 0 and self.model.nu == 8
        mujoco.mj_forward(self.model, self.data)

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f'Missing named native scene item: {name}')
        return result

    def solve_ik(self, target, initial=None, iterations=6):
        data = self.ik_data
        data.qpos[:] = self.data.qpos
        data.qpos[self.arm_q] = self.arm_targets if initial is None else initial
        data.qvel[:] = 0.
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            mujoco.mj_forward(self.model, data)
            current_rotation = data.site_xmat[self.tcp_site].reshape(3, 3)
            translation = target[:3, 3] - data.site_xpos[self.tcp_site]
            angular = rotation_error(target[:3, :3], current_rotation)
            if np.linalg.norm(translation) < 1e-6 and np.linalg.norm(angular) < 1e-5:
                break
            mujoco.mj_jacSite(self.model, data, jac_position, jac_rotation, self.tcp_site)
            jac = np.r_[jac_position[:, self.arm_v], .15*jac_rotation[:, self.arm_v]]
            error = np.r_[translation, .15*angular]
            update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-7, error)
            data.qpos[self.arm_q] = np.clip(data.qpos[self.arm_q]+np.clip(update, -.15, .15),
                                            self.arm_bounds[:, 0], self.arm_bounds[:, 1])
        mujoco.mj_forward(self.model, data)
        error = [float(np.linalg.norm(target[:3, 3]-data.site_xpos[self.tcp_site])),
                 float(np.linalg.norm(rotation_error(target[:3, :3], data.site_xmat[self.tcp_site].reshape(3, 3))))]
        return data.qpos[self.arm_q].copy(), error

    def pose(self, name):
        result = np.eye(4)
        if name == 'tool':
            result[:3, :3] = self.data.site_xmat[self.tcp_site].reshape(3, 3)
            result[:3, 3] = self.data.site_xpos[self.tcp_site]
        else:
            body = self.ids[name]
            result[:3, :3] = self.data.xmat[body].reshape(3, 3)
            result[:3, 3] = self.data.xpos[body]
        return result

    def velocity(self, name):
        result = np.zeros(6)
        kind = mujoco.mjtObj.mjOBJ_SITE if name == 'tool' else mujoco.mjtObj.mjOBJ_BODY
        identifier = self.tcp_site if name == 'tool' else self.ids[name]
        mujoco.mj_objectVelocity(self.model, self.data, kind, identifier, result, 0)
        return result[3:].copy(), result[:3].copy()

    def contacts(self):
        sides, entries, illegal = {'left': [], 'right': []}, [], False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g0, g1 = int(contact.geom1), int(contact.geom2)
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = contact.frame[:3].copy()
            entries.append(dict(geom0=g0, geom1=g1, distance_m=float(contact.dist),
                                normal=normal.tolist(), normal_force_N=float(force[0])))
            for side in ('left', 'right'):
                if {g0, g1} == {self.geoms[f'{side}_pad'], self.geoms['cube_geom']} and force[0] >= P['minimum_contact_force_N']:
                    sides[side].append(normal if g1 == self.geoms['cube_geom'] else -normal)
            bodies = [int(self.model.geom_bodyid[g]) for g in (g0, g1)]
            arm_contact = any(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body).startswith('x5_') for body in bodies)
            if arm_contact and (self.geoms['table'] in (g0, g1) or self.geoms['floor'] in (g0, g1)):
                illegal = True
            if contact.dist < -P['penetration_stop_m']:
                illegal = True
        opposing = bool(sides['left'] and sides['right'] and any(np.dot(a, b) <= P['opposing_contact_dot_max']
                                                                 for a in sides['left'] for b in sides['right']))
        return bool(sides['left']), bool(sides['right']), opposing, illegal, entries

    def snapshot(self):
        mujoco.mj_forward(self.model, self.data)
        tcp, obj = self.pose('tool'), self.pose('cube')
        linear, angular = self.velocity('cube')
        tcp_linear, tcp_angular = self.velocity('tool')
        left, right, opposing, illegal, contacts = self.contacts()
        q, qd = self.data.qpos[self.finger_q].copy(), self.data.qvel[self.finger_v].copy()
        relative = bool(np.linalg.norm(linear-tcp_linear) <= P['relative_grasp_speed_max_m_s'])
        valid = bool(np.isfinite(q).all() and (q >= -.001).all() and (q <= P['finger_travel']+.001).all())
        arm_valid = bool((self.data.qpos[self.arm_q] >= self.arm_bounds[:, 0]-.006).all() and
                         (self.data.qpos[self.arm_q] <= self.arm_bounds[:, 1]+.006).all())
        return dict(time_s=float(self.data.time), tcp=tcp, object_pose=obj,
                    object_linear_velocity=linear, object_angular_velocity=angular,
                    tool_linear_velocity=tcp_linear, tool_angular_velocity=tcp_angular,
                    finger_q=q, finger_qd=qd, gripper_fraction=float(np.mean(q)/P['finger_travel']),
                    gripper_speed=float(np.mean(qd)/P['finger_travel']),
                    gripper_gap_m=float(PAD_INNER_GAP_AT_ZERO+np.sum(q)),
                    left_object_contact=left, right_object_contact=right, opposing_contact=opposing,
                    relative_motion_valid=relative, native_grasp=bool(left and right and opposing and relative and valid),
                    object_present=True, contact_valid=True, gripper_state_valid=valid,
                    safety_valid=not bool(self.safety) and not illegal and arm_valid,
                    native_abort=bool(self.safety), illegal_contact=illegal,
                    joint_limit_penetration=not (valid and arm_valid), reset=False, takeover=False,
                    contacts=contacts, actuator_force=self.data.actuator_force.copy(),
                    qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                    arm_q=self.data.qpos[self.arm_q].copy(), arm_qd=self.data.qvel[self.arm_v].copy(),
                    object_points=obj[:3, 3]+POINTS @ obj[:3, :3].T,
                    table_top_m=float(P['table_top']), last_ik_error=self.last_ik_error)

    def step(self, target, opening):
        solution, self.last_ik_error = self.solve_ik(target)
        slew = P.get('arm_target_speed_rad_s', .7)*P['dt']
        self.arm_targets += np.clip(solution-self.arm_targets, -slew, slew)
        kp = np.asarray(P.get('arm_kp', [90., 120., 100., 40., 30., 25.]))
        torques = kp*(self.arm_targets-self.data.qpos[self.arm_q])+self.data.qfrc_bias[self.arm_v]
        self.data.ctrl[self.arm_ctrl] = np.clip(torques, self.model.actuator_ctrlrange[self.arm_ctrl, 0],
                                               self.model.actuator_ctrlrange[self.arm_ctrl, 1])
        desired = np.full(2, float(opening)*P['finger_travel'])
        if self.mode == 'single_side':
            desired[1] = P['finger_travel']
        self.finger_targets += np.clip(desired-self.finger_targets,
                                       -P['finger_target_speed_m_s']*P['dt'], P['finger_target_speed_m_s']*P['dt'])
        finger_force = P['finger_kp']*(self.finger_targets-self.data.qpos[self.finger_q])
        self.data.ctrl[self.finger_ctrl] = np.clip(finger_force, -P['finger_force_max_N'], P['finger_force_max_N'])
        self.data.xfrc_applied[:] = 0.
        self.data.qfrc_applied[:] = 0.
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        mujoco.mj_forward(self.model, self.data)
        snapshot = self.snapshot()
        if snapshot['illegal_contact']:
            self.safety.append('ILLEGAL_CONTACT_OR_PENETRATION')
        if snapshot['joint_limit_penetration']:
            self.safety.append('JOINT_LIMIT')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.safety.append('NONFINITE')
        if any(w.number for w in self.data.warning):
            self.safety.append('MUJOCO_WARNING')
        if np.max(np.abs(self.data.qvel[self.finger_v])) > P['finger_speed_stop_m_s']:
            self.safety.append('FINGER_VELOCITY')
        if np.any(np.abs(self.data.qvel[self.arm_v]) > np.array([5.5, 5.5, 5.5, 20., 20., 20.])):
            self.safety.append('ARM_VELOCITY')
        if np.linalg.norm(snapshot['tool_linear_velocity']) > P['tool_speed_stop_m_s']:
            self.safety.append('TCP_VELOCITY')
        if self.safety:
            raise RuntimeError(self.safety[-1])

    def render(self, snapshot):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, P['pixels'], P['pixels'])
        images, geometry = {}, {}
        for name in ('body_rgb', 'wrist_rgb'):
            self.renderer.update_scene(self.data, camera=name)
            images[name] = self.renderer.render().copy()
            camera = self._id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            rotation = self.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, self.data.cam_xpos[camera]
            focal = P['pixels']/(2*np.tan(np.deg2rad(self.model.cam_fovy[camera])/2))
            intrinsic = np.array([[focal, 0., P['pixels']/2], [0., focal, P['pixels']/2], [0., 0., 1.]])
            points = (snapshot['object_points']-transform[:3, 3]) @ rotation
            projected = points @ intrinsic.T
            uv = projected[:, :2]/projected[:, 2, None]
            visible = []
            for point, pixel, depth in zip(snapshot['object_points'], uv, points[:, 2]):
                direction = point-transform[:3, 3]
                direction /= np.linalg.norm(direction)
                geom = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(self.model, self.data, transform[:3, 3], direction, None, 1, -1, geom)
                visible.append(bool(depth > 0 and (pixel >= 0).all() and (pixel < P['pixels']).all()
                                    and geom[0] == self.geoms['cube_geom'] and distance >= 0))
            geometry[name] = dict(K=intrinsic.tolist(), T_world_optical=transform.tolist(),
                                 parent='world_fixed_rig' if name == 'body_rgb' else 'x5_link6',
                                 optical_axis_convention='OpenCV +X right +Y down +Z forward; MJ camera +X right +Y up -Z forward',
                                 object_point_ids=list(range(8)), object_points_uv=uv.tolist(),
                                 object_point_visible=visible, timestamp_s=float(self.data.time),
                                 snapshot_id=f"{self.card['id']}:{self.steps}")
        return images, geometry

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
