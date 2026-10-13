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
    engine = make_engine(card, asset_root)
    try:
        result = qualify_engine(engine, render=render)
        result['engine_starts'] = 1
        return result
    finally:
        engine.close()


def rendered_field_of_view(images, geometry, card):
    """Inspect actual camera pixels and calibrated target projection at t=0."""
    details = {}
    goal = np.array([[card['goal_xy'][0]+x, card['goal_xy'][1]+y, P['table_top']+.0002]
                     for x in (-.082, .082) for y in (-.082, .082)])
    for camera in ('body_rgb', 'wrist_rgb'):
        pixels = np.asarray(images[camera])
        if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
            raise ValueError('Expected actual uint8 double RGB')
        red = int(np.count_nonzero((pixels[..., 0] > 80) & (pixels[..., 0].astype(float) > 1.5*pixels[..., 1])
                                  & (pixels[..., 0].astype(float) > 1.5*pixels[..., 2])))
        blue = int(np.count_nonzero((pixels[..., 2] > 80) & (pixels[..., 2].astype(float) > 1.5*pixels[..., 0])
                                   & (pixels[..., 2].astype(float) > 1.5*pixels[..., 1])))
        calibration = geometry[camera]
        if abs(float(calibration['timestamp_s'])) > 1e-8:
            raise ValueError('Initial field of view must use the actual t=0 pair')
        transform = np.asarray(calibration['T_world_optical'])
        intrinsic = np.asarray(calibration['K'])
        if transform.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError('Missing exact camera calibration')
        camera_points = (goal-transform[:3, 3]) @ transform[:3, :3]
        projected = camera_points @ intrinsic.T
        uv = projected[:, :2]/projected[:, 2, None]
        goal_in_frame = bool((camera_points[:, 2] > 0).all() and (uv >= 0).all() and (uv < 256).all())
        details[camera] = dict(red_pixels=red, blue_pixels=blue, pixel_standard_deviation=float(pixels.std()),
            cube_visible_corners=int(sum(calibration['object_point_visible'])), goal_corners_uv=uv.tolist(),
            full_target_in_frame=goal_in_frame, timestamp_s=calibration['timestamp_s'],
            K=calibration['K'], T_world_optical=calibration['T_world_optical'])
    passed = bool(all(v['red_pixels'] >= 20 and v['pixel_standard_deviation'] > 8 and v['cube_visible_corners'] > 0
                      for v in details.values()) and details['body_rgb']['full_target_in_frame']
                  and details['body_rgb']['blue_pixels'] >= 20)
    return dict(passed=passed, cameras=details,
                contract='BOTH_INITIAL_CUBE_VISIBLE_BODY_FULL_TARGET_AND_BLUE_MARKER_VISIBLE',
                live_path_visibility='PENDING_REAL_EPISODE_AUDIT')


def qualify_engine(engine, *, render=False, rendered=None, guard=None):
    """Qualify a fresh existing engine without constructing another model.

    Pass the episode's t=0 (images, geometry) pair to avoid extra observations.
    No object pose is changed and no simulation time is integrated.
    """
    import mujoco
    from .teacher import make_teacher, PUSH_STAGES
    from scripts.pi0291_arx5_data.teacher import STAGES
    from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix
    if engine.steps != 0 or engine.data.time != 0:
        raise ValueError('Static qualifier requires the already-created initial engine')
    card = engine.card
    rows = []
    first = engine.snapshot()
    teacher = make_teacher(card, first['object_pose'], engine)
    previous = first['tcp'].copy()
    previous_opening = first['gripper_fraction']
    ik_seed = first['arm_q'].copy()
    saved = {name: getattr(engine.data, name).copy() for name in
             ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied')}
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
            segments = max(1, int(np.ceil(np.linalg.norm(target[:3, 3]-previous[:3, 3])/.002)),
                           int(np.ceil(float(geodesic_angle(previous[:3, :3], target[:3, :3]))/.02)),
                           int(np.ceil(abs(opening-previous_opening)*P['finger_travel']/.002)))
            fractions = np.linspace(0., 1., segments+1).tolist()
            index, previous_fraction = 0, 0.
            while index < len(fractions):
                fraction = fractions[index]
                if guard is not None:
                    guard()
                pose = previous.copy()
                pose[:3, 3] = previous[:3, 3]+fraction*(target[:3, 3]-previous[:3, 3])
                pose[:3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(previous[:3, :3]), matrix_to_quat(target[:3, :3]), fraction))
                q, error = engine.solve_ik(pose, initial=ik_seed, iterations=150)
                joint_gap = float(np.max(np.abs(q-ik_seed)))
                if joint_gap > .02 and fraction-previous_fraction > 1e-5:
                    if len(fractions) >= 10000:
                        raise RuntimeError('STATIC_JOINT_SWEEP_REFINEMENT_EXHAUSTED')
                    fractions.insert(index, (previous_fraction+fraction)/2)
                    continue
                ik_seed = q.copy()
                engine.data.qpos[engine.arm_q] = q
                engine.data.qpos[engine.finger_q] = (previous_opening+fraction*(opening-previous_opening))*P['finger_travel']
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
                                 max_joint_sample_gap_rad=joint_gap,
                                 robot_object_static_overlap='PENDING_LIVE_CONTACT_QUALIFICATION'))
                previous_fraction = fraction
                index += 1
            previous = target.copy()
            previous_opening = opening
        for name, value in saved.items():
            getattr(engine.data, name)[:] = value
        mujoco.mj_forward(engine.model, engine.data)
        rendered_pairs = 0
        if render:
            if rendered is not None:
                raise ValueError('Use existing rendered pair or request one new pair')
            rendered = engine.render(engine.snapshot())
            rendered_pairs = 1
        visibility = None if rendered is None else rendered_field_of_view(*rendered, card)
        geometry_passed = bool(first['safety_valid'] and all(r['ik_error'][0] <= .0002 and r['ik_error'][1] <= .002
                                                            and not r['illegal_contact'] and r['max_joint_sample_gap_rad'] <= .02 for r in rows))
        passed = bool(geometry_passed and visibility is not None and visibility['passed'])
        status = 'STATIC_PASS_PENDING_LIVE_CONTACT' if passed else 'STATIC_FAIL' if not geometry_passed or visibility is not None else 'STATIC_GEOMETRY_ONLY_FOV_PENDING'
        return dict(status=status, geometry_passed=geometry_passed, integration_steps=0, engine_starts=0,
                    rendered_pairs=rendered_pairs, reused_observation_pairs=int(rendered is not None and not render),
                    initial_safety_valid=first['safety_valid'], targets=rows, field_of_view=visibility,
                    cameras=None if rendered is None else rendered[1], task_success_claim=False,
                    object_pose_modified=False, full_swept_volume='ADAPTIVE_ALL_TEACHER_SEGMENTS_DISCRETE_CONTACT_SWEEP',
                    sweep_sampling_bounds=dict(tcp_translation_m=.002, tcp_rotation_rad=.02, finger_translation_m=.002,
                                               maximum_joint_gap_rad=.02), continuous_collision_certificate=False)
    finally:
        for name, value in saved.items():
            getattr(engine.data, name)[:] = value
        mujoco.mj_forward(engine.model, engine.data)
        engine.data.qacc_warmstart[:] = saved['qacc_warmstart']
        assert engine.steps == 0 and engine.data.time == 0
