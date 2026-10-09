"""Synthetic CPU checks only; no MuJoCo load, physics, GPU or hardware."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from go2arx5_vla.target_mujoco.mapping import (BASE_JOINT, CAMERAS, FINGER_JOINTS, GRIPPER_ACTUATOR,
                                             NATIVE_ACTUATORS, NATIVE_JOINTS, OBJECT_JOINT, build_named_map)
from go2arx5_vla.target_mujoco.scene import (build_task_overlay, camera_calibration, grasp_contact_evidence, gripper_spec_from_urdf,
                                           independent_gripper_force, validate_rgb_pair)


def synthetic_model():
    names = [OBJECT_JOINT, *NATIVE_JOINTS[::2], FINGER_JOINTS[1], BASE_JOINT, *NATIVE_JOINTS[1::2], FINGER_JOINTS[0]]
    ids = {n: i for i, n in enumerate(names)}
    types = [0 if n in (BASE_JOINT, OBJECT_JOINT) else 2 if n in FINGER_JOINTS else 3 for n in names]
    qadr, vadr, nq, nv = [], [], 2, 1
    for kind in types:
        qadr.append(nq); vadr.append(nv)
        nq += (7 if kind == 0 else 1) + 1
        nv += (6 if kind == 0 else 1) + 2
    controls = {n: i for i, n in enumerate(reversed((*NATIVE_ACTUATORS, GRIPPER_ACTUATOR)))}
    trnid = np.zeros((20, 2), dtype=int)
    for n, joint in zip(NATIVE_ACTUATORS, NATIVE_JOINTS):
        trnid[controls[n], 0] = ids[joint]
    trnid[controls[GRIPPER_ACTUATOR], 0] = ids[FINGER_JOINTS[0]]
    sensors, adr, dims, offset = {}, [], [], 0
    for j in NATIVE_JOINTS:
        stem = j.removesuffix('_joint') if j in NATIVE_JOINTS[:12] else j
        for suffix in ('pos', 'vel', 'torque'):
            sensors[f'{stem}_{suffix}'] = len(adr); adr.append(offset); dims.append(1); offset += 2
    for name, dim in [('imu_quat', 4), ('imu_gyro', 3), ('imu_acc', 3), ('frame_pos', 3), ('frame_vel', 3)]:
        sensors[name] = len(adr); adr.append(offset); dims.append(dim); offset += dim + 1
    for joint in FINGER_JOINTS:
        for suffix in ('pos', 'vel'):
            sensors[f'{joint}_{suffix}'] = len(adr); adr.append(offset); dims.append(1); offset += 2
    model = SimpleNamespace(nq=nq, nv=nv, nu=20, nsensordata=offset,
                            jnt_type=types, jnt_qposadr=qadr, jnt_dofadr=vadr, actuator_trnid=trnid,
                            actuator_trntype=np.zeros(20), sensor_adr=adr, sensor_dim=dims)
    model.joint = lambda n: SimpleNamespace(id=ids[n])
    model.actuator = lambda n: SimpleNamespace(id=controls[n])
    model.sensor = lambda n: SimpleNamespace(id=sensors[n])
    model.camera = lambda n: SimpleNamespace(id=5 if n == CAMERAS[0] else 2)
    model.site = lambda n: SimpleNamespace(id=3)
    return model


def test_noncontiguous_named_state_and_isolated_controls():
    model = synthetic_model()
    mapping = build_named_map(model)
    q, v = np.arange(model.nq, dtype=float), np.arange(model.nv, dtype=float)
    for name in (BASE_JOINT, OBJECT_JOINT):
        j = mapping.joints[name]
        q[j.qposadr + 3:j.qposadr + 7] = [.5, .5, -.5, .5]
    state = mapping.state(q, v)
    assert state['native_qpos'].tolist() == [q[mapping.joints[n].qposadr] for n in NATIVE_JOINTS]
    assert state['native_qvel'].tolist() == [v[mapping.joints[n].dofadr] for n in NATIVE_JOINTS]
    np.testing.assert_equal(state['base_quaternion_xyzw'], [.5, -.5, .5, .5])
    ctrl = np.full(model.nu, -999.)
    mapping.dispatch(ctrl, native_torques=np.arange(18), gripper_force=7.)
    assert ctrl[19] == -999.
    assert ctrl[mapping.actuators[GRIPPER_ACTUATOR]] == 7.
    saved = ctrl.copy()
    mapping.dispatch(ctrl, gripper_force=-3.)
    for name in NATIVE_ACTUATORS:
        assert ctrl[mapping.actuators[name]] == saved[mapping.actuators[name]]
    assert len(mapping.sensor_values(np.arange(model.nsensordata))) == 63
    q[mapping.joints[BASE_JOINT].qposadr + 3] = 2.
    with pytest.raises(ValueError, match='quaternion'):
        mapping.state(q, v)
    model.actuator_trnid[mapping.actuators[GRIPPER_ACTUATOR], 0] = mapping.joints[NATIVE_JOINTS[0]].id
    with pytest.raises(ValueError, match='transmission'):
        build_named_map(model)


def private_like_inputs(tmp_path):
    # Deliberately synthetic dimensions/masses, never deployment specifications.
    robot = ET.Element('mujoco'); ET.SubElement(robot, 'compiler', meshdir='assets')
    world = ET.SubElement(robot, 'worldbody'); base = ET.SubElement(world, 'body', name='base_link')
    ET.SubElement(base, 'freejoint'); ET.SubElement(base, 'inertial', mass='10', pos='0 0 0', diaginertia='1 1 1')
    for name in NATIVE_JOINTS:
        body = ET.SubElement(base, 'body', name=f'body_{name}')
        ET.SubElement(body, 'joint', name=name, type='hinge')
    tool = ET.SubElement(base, 'body', name='x5_link6')
    ET.SubElement(tool, 'site', name='x5_ee', pos='.12 0 0')
    urdf = ET.Element('robot')
    tcp = ET.SubElement(urdf, 'joint', name='gripper_center_fixed', type='fixed')
    ET.SubElement(tcp, 'parent', link='link6'); ET.SubElement(tcp, 'origin', xyz='.12 0 0')
    for index, side, joint_name, axis in [(7, 'left', 'gripper_joint', '0 1 0'), (8, 'right', 'joint8', '0 -1 0')]:
        pos = '.07 .03 0' if index == 7 else '.07 -.03 0'
        body = ET.SubElement(tool, 'body', name=f'x5_link{index}', pos=pos)
        ET.SubElement(body, 'inertial', mass='.1', pos='0 0 0', fullinertia='.001 .001 .001 0 0 0')
        link = ET.SubElement(urdf, 'link', name=f'link{index}')
        inertial = ET.SubElement(link, 'inertial'); ET.SubElement(inertial, 'mass', value='.1')
        ET.SubElement(inertial, 'origin', xyz='0 0 0')
        ET.SubElement(inertial, 'inertia', ixx='.001', iyy='.001', izz='.001', ixy='0', ixz='0', iyz='0')
        collision = ET.SubElement(link, 'collision'); ET.SubElement(collision, 'origin', xyz='.02 0 0')
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size='.06 .01 .02')
        joint = ET.SubElement(urdf, 'joint', name=joint_name, type='prismatic')
        ET.SubElement(joint, 'parent', link='link6'); ET.SubElement(joint, 'origin', xyz=pos)
        ET.SubElement(joint, 'axis', xyz=axis); ET.SubElement(joint, 'limit', lower='0', upper='.03', effort='10', velocity='.1')
        if index == 8:
            ET.SubElement(joint, 'mimic', joint='gripper_joint')
        pad = ET.SubElement(urdf, 'joint', name=f'{side}_pad_joint', type='fixed')
        ET.SubElement(pad, 'origin', xyz='.04 0 0')
    actuator = ET.SubElement(robot, 'actuator')
    for a, j in zip(NATIVE_ACTUATORS, NATIVE_JOINTS):
        ET.SubElement(actuator, 'motor', name=a, joint=j)
    keyframe = ET.SubElement(robot, 'keyframe')
    ET.SubElement(keyframe, 'key', name='home', qpos='0 0 .4 1 0 0 0 ' + ' '.join(str(i) for i in range(18)), ctrl=' '.join('0' for _ in range(18)))
    robot_path, urdf_path, scene_path = (tmp_path / n for n in ('robot.xml', 'model.urdf', 'scene.xml'))
    robot_path.write_text(ET.tostring(robot, encoding='unicode')); urdf_path.write_text(ET.tostring(urdf, encoding='unicode'))
    scene_path.write_text('<mujoco><include file="robot.xml"/><worldbody><geom name="floor" type="plane" size="0 0 .1"/></worldbody></mujoco>')
    return robot_path, scene_path, gripper_spec_from_urdf(urdf_path)


def test_overlay_keeps_native_mass_and_declares_provisional(tmp_path):
    robot, scene, spec = private_like_inputs(tmp_path)
    layout = dict(table_center_xyz=[.5, 0, .25], table_halfsize=[.4, .4, .05], cube_xy=[.4, 0], zone_xy=[.6, 0], cube_yaw=.3, selection_source='SYNTHETIC_CPU_TEST')
    xml, manifest = build_task_overlay(robot, scene, layout=layout, contact_spec=dict(friction=[.5, .01, .001], source='SYNTHETIC_CPU_TEST'), gripper_spec=spec)
    parsed = ET.fromstring(xml)
    assert manifest['formal_data_gate'] == 'BLOCKED_GRIPPER_SPEC'
    assert manifest['original_inertials_unchanged'] and manifest['additional_tool_mass_kg'] == 0
    assert len(parsed.find('actuator')) == 19
    assert len(list(parsed.iter('camera'))) == 2
    assert len(parsed.find('keyframe/key').get('qpos').split()) == 34
    assert len(parsed.find('keyframe/key').get('ctrl').split()) == 19
    assert len(list(parsed.iter('inertial'))) == 3
    assert parsed.find('equality/joint').get('joint2') == FINGER_JOINTS[0]
    assert parsed.find('.//freejoint[@name="pi0288_base_free"]') is not None
    assert not parsed.findall('.//weld')
    assert len(ET.parse(robot).getroot().find('actuator')) == 18
    spec['fingers'][0]['mass'] = .2
    with pytest.raises(ValueError, match='mass differs'):
        build_task_overlay(robot, scene, layout=layout, contact_spec=dict(friction=[.5, .01, .001], source='test'), gripper_spec=spec)


def test_gripper_limits_and_same_snapshot_rgb():
    command = dict(q_m=0., velocity_m_s=0., requested_m=.2, previous_setpoint_m=0., dt_s=.05,
                   stroke_m=.03, speed_m_s=.1, force_N=10., stiffness_N_m=100., calibration_id='SYNTHETIC_ONLY')
    row = independent_gripper_force(**command)
    assert row['accepted_m'] == pytest.approx(.005) and row['clamped']
    assert row['force_N'] == pytest.approx(.5)
    command['velocity_m_s'] = .1
    assert independent_gripper_force(**command)['force_N'] == 0.
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    args = dict(body_rgb=image, wrist_rgb=image.copy(), snapshot_id='s1', body_snapshot_id='s1', wrist_snapshot_id='s1',
                state_sim_s=1., body_sim_s=1., wrist_sim_s=1., episode_id='episode1', frame_index=0, ros_s=1., wall_monotonic_s=9.)
    pair = validate_rgb_pair(**args)
    assert pair['body_rgb'].flags.writeable is False
    args['wrist_sim_s'] = 1.001
    with pytest.raises(ValueError, match='future RGB'):
        validate_rgb_pair(**args)
    args['wrist_sim_s'] = 1.; args['wrist_snapshot_id'] = 'old_episode_snapshot'
    with pytest.raises(ValueError, match='snapshot'):
        validate_rgb_pair(**args)


def test_no_object_single_pad_two_pad_and_release_contact_evidence():
    args = dict(left_geom_ids={1}, right_geom_ids={2}, object_geom_ids={3}, object_in_tcp_xyz=[.01, 0., 0.],
                previous_object_in_tcp_xyz=[.01, 0., 0.], dt_s=.05, max_relative_speed_m_s=.001)
    left = dict(geom1=1, geom2=3, normal_force_N=2., normal_geom1_to_geom2=[0., 1., 0.])
    # Reverse geom order deliberately: its inward pad->object normal is -Y.
    right = dict(geom1=3, geom2=2, normal_force_N=2., normal_geom1_to_geom2=[0., 1., 0.])
    assert not grasp_contact_evidence(contacts=[], **args)['opposing_pad_contact']
    assert not grasp_contact_evidence(contacts=[left], **args)['opposing_pad_contact']
    two = grasp_contact_evidence(contacts=[left, right], **args)
    assert two['opposing_pad_contact'] and two['relative_motion_valid']
    args['object_in_tcp_xyz'] = [.02, 0., 0.]
    assert not grasp_contact_evidence(contacts=[left, right], **args)['relative_motion_valid']
    right['normal_force_N'] = 0.
    assert not grasp_contact_evidence(contacts=[left, right], **args)['opposing_pad_contact']


def test_actual_camera_transform_readout_and_native_anonymous_base():
    model = synthetic_model()
    original_joint_accessor = model.joint
    base_id = original_joint_accessor(BASE_JOINT).id
    def native_joint(name):
        if name == BASE_JOINT:
            raise KeyError(name)
        return original_joint_accessor(name)
    model.joint = native_joint
    model.body = lambda n: SimpleNamespace(id=2)
    model.body_jntadr = [0, 0, base_id]
    native = build_named_map(model, equipped=False)
    assert native.joints[BASE_JOINT].id == base_id and len(native.actuators) == 18
    model.cam_fovy = np.full(6, 60.); model.cam_bodyid = np.arange(6)
    model.cam_pos = np.zeros((6, 3)); model.cam_quat = np.tile([1., 0., 0., 0.], (6, 1))
    data = SimpleNamespace(cam_xpos=np.zeros((6, 3)), cam_xmat=np.tile(np.eye(3).reshape(9), (6, 1)))
    calibration = camera_calibration(model, data, dict(zip(CAMERAS, (5, 2))))
    assert calibration[CAMERAS[0]]['K'][0][0] == pytest.approx(128 / np.tan(np.pi / 6))
    assert calibration[CAMERAS[1]]['parent_body_id'] == 2
