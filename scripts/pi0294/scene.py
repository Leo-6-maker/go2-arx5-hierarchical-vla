"""PI0294 visuals around the unchanged PI0291 native physical mechanism."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from scripts.pi0291_arx5_data.scene import (
    ARM_NAMES, FINGER_NAMES, TCP_OFFSET, DOWNWARD_ROTATION, xml_scene as base_xml_scene,
)


def _rpy(angles):
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def xml_scene(card, asset_root, parameters=P):
    root = ET.fromstring(base_xml_scene(card, asset_root, parameters))
    root.set('model', 'PI0294_FIXED_BASE_ARX5_MULTITASK')
    visual = card.get('visual', {})
    for name in ('table', 'floor'):
        if name+'_rgba' in visual:
            root.find(f".//geom[@name='{name}']").set('rgba', ' '.join(map(str, visual[name+'_rgba'])))
    for light in root.findall('.//light'):
        diffuse = np.fromstring(light.get('diffuse', '1 1 1'), sep=' ')*visual.get('light_scale', 1.)
        light.set('diffuse', ' '.join(map(str, diffuse)))
    for name in ('body', 'wrist'):
        camera = root.find(f".//camera[@name='{name}_rgb']")
        position = np.fromstring(camera.get('pos'), sep=' ')+visual.get(name+'_camera_translation', [0, 0, 0])
        axes = np.fromstring(camera.get('xyaxes'), sep=' ').reshape(2, 3)
        axes = axes @ _rpy(visual.get(name+'_camera_rpy', [0, 0, 0])).T
        camera.set('pos', ' '.join(map(str, position)))
        camera.set('xyaxes', ' '.join(map(str, axes.ravel())))
    return ET.tostring(root, encoding='unicode')


def make_engine(card, asset_root):
    """Import MuJoCo only in the independently spawned physics process."""
    import mujoco
    from scripts.pi0291_arx5_data.engine import Engine as BaseEngine

    class Engine(BaseEngine):
        def __init__(self):
            self.card, self.mode = card, 'normal'
            self.xml = xml_scene(card, asset_root)
            self.model = mujoco.MjModel.from_xml_string(self.xml)
            self.data, self.ik_data = mujoco.MjData(self.model), mujoco.MjData(self.model)
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
            initial[:3, :3], initial[:3, 3] = DOWNWARD_ROTATION, [*card['source_xy'], P['table_top']+.10]
            initial = np.asarray(card.get('initial_tcp', initial), dtype=float)
            solution, error = self.solve_ik(initial, np.array([0., 2.13, 1.73, -1.17, 0., 0.]), iterations=150)
            if error[0] > .0002 or error[1] > .002:
                raise ValueError(f'Initial native IK unreachable: {error}')
            self.data.qpos[self.arm_q] = solution
            self.arm_targets, self.finger_targets = solution.copy(), np.full(2, P['finger_travel'])
            self.steps, self.safety, self.renderer, self.last_ik_error = 0, [], None, error
            self.reset_ik_cache()
            assert self.model.neq == 0 and self.model.nmocap == 0 and self.model.nu == 8
            assert self.model.opt.timestep == .002
            mujoco.mj_forward(self.model, self.data)

    return Engine()


def push_geometry(engine, direction):
    """Measure native pad boxes relative to TCP, including actual body frames."""
    tcp = engine.pose('tool')
    relative = []
    for side in ('left', 'right'):
        gid = engine.geoms[side+'_pad']
        center = tcp[:3, :3].T @ (engine.data.geom_xpos[gid]-tcp[:3, 3])
        center[1] -= (1 if side == 'left' else -1)*engine.data.qpos[engine.finger_q[0 if side == 'left' else 1]]
        axes = tcp[:3, :3].T @ engine.data.geom_xmat[gid].reshape(3, 3)
        relative.append(dict(side=side, center_tcp=center.tolist(), axes_tcp=axes.tolist(),
                             half_size=engine.model.geom_size[gid].tolist()))
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    yaw = np.arctan2(d[1], d[0])
    rotation = _rpy([0, 0, yaw]) @ DOWNWARD_ROTATION
    front = max(float((rotation @ np.asarray(p['center_tcp']))[:2] @ d
                      + np.abs(d @ (rotation @ np.asarray(p['axes_tcp']))[:2]) @ np.asarray(p['half_size'])) for p in relative)
    bottom = min(float((rotation @ np.asarray(p['center_tcp']))[2]
                       - np.abs((rotation @ np.asarray(p['axes_tcp']))[2]) @ np.asarray(p['half_size'])) for p in relative)
    return dict(pads=relative, direction=d.tolist(), rotation=rotation.tolist(), front_support_m=front,
                bottom_offset_m=bottom, tcp_height_m=P['table_top']-bottom+.004,
                collision_basis='MEASURED_NATIVE_PAD_GEOMS_AT_ZERO_FINGER_TRAVEL')


def qualify_static(card, asset_root, render=False):
    """Zero integration IK and collision sweep. Root controls invocation."""
    import mujoco
    from .teacher import make_teacher, PUSH_STAGES
    from scripts.pi0291_arx5_data.teacher import STAGES
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, quat_slerp, quat_to_matrix
    engine = make_engine(card, asset_root)
    rows = []
    first = engine.snapshot()
    teacher = make_teacher(card, first['object_pose'], engine)
    previous = first['tcp'].copy()
    initial_q = first['qpos'].copy()
    try:
        for stage in STAGES if card['task'] == 'T1' else PUSH_STAGES:
            teacher.stage = stage
            if stage == 'release':
                teacher.release_tcp = previous.copy()
            if stage in ('detach', 'withdraw', 'hold'):
                if getattr(teacher, 'final_tcp', None) is None:
                    teacher.final_tcp = previous.copy()
                    if card['task'] == 'T1':
                        teacher.final_tcp[2, 3] += P['withdraw_clearance']
            target, opening = teacher.target(first)
            for fraction in np.linspace(0., 1., 9):
                pose = previous.copy()
                pose[:3, 3] = previous[:3, 3]+fraction*(target[:3, 3]-previous[:3, 3])
                pose[:3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(previous[:3, :3]), matrix_to_quat(target[:3, :3]), fraction))
                q, error = engine.solve_ik(pose, initial=engine.arm_targets, iterations=150)
                engine.data.qpos[engine.arm_q] = q
                engine.data.qpos[engine.finger_q] = opening*P['finger_travel']
                mujoco.mj_forward(engine.model, engine.data)
                state = engine.snapshot()
                # A closed grasp target overlaps the unmoved cube in static FK.
                # Real object response and pad contact require live qualification.
                prohibited = []
                for contact in state['contacts']:
                    geoms = (contact['geom0'], contact['geom1'])
                    if engine.geoms['cube_geom'] in geoms:
                        continue
                    bodies = [int(engine.model.geom_bodyid[g]) for g in geoms]
                    arm = any(mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_BODY, b).startswith('x5_') for b in bodies)
                    if contact['distance_m'] < -P['penetration_stop_m'] or arm and any(engine.geoms[n] in geoms for n in ('table', 'floor')):
                        prohibited.append(contact)
                rows.append(dict(stage=stage, fraction=float(fraction), ik_error=error, arm_q=q.tolist(),
                                 illegal_contact=bool(prohibited), contacts=state['contacts'],
                                 robot_object_static_overlap='PENDING_LIVE_CONTACT_QUALIFICATION'))
            previous = target.copy()
        engine.data.qpos[:] = initial_q
        mujoco.mj_forward(engine.model, engine.data)
        camera = None
        if render:
            _, camera = engine.render(engine.snapshot())
        passed = all(r['ik_error'][0] <= .0002 and r['ik_error'][1] <= .002 and not r['illegal_contact'] for r in rows)
        return dict(status='STATIC_PASS_PENDING_LIVE_CONTACT' if passed else 'STATIC_FAIL', integration_steps=0,
                    rendered_pairs=int(render), initial_safety_valid=first['safety_valid'], targets=rows,
                    cameras=camera, task_success_claim=False)
    finally:
        engine.close()
