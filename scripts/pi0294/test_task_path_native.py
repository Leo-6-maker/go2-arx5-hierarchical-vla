"""New-stage sequence and private object/reference checks; no physics stepping."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from .task_path_reference import PersistentPathReference
from .task_gripper_contract import teacher_physical_fraction
from .task_path_model import PlanningObjectMotion, certify_chunk
from .task_path_runtime import cards


def test_new_stage_keeps_global_ros_sequence_and_completed_intervals():
    poses = np.repeat(np.eye(4)[None], 20, axis=0); poses[:, 0, 3] = np.arange(20)*.001
    opening = teacher_physical_fraction(np.ones(20))
    r = PersistentPathReference(episode_id='SYNTHETIC', poses_world=poses,
        opening=opening, initial_native_step=2500, sequence_offset=2)
    for i in range(9):
        r.observe(native_step=2500+25*i, raw_time_s=5+i*.05, actual_tcp=poses[i], native_safe=True)
        if i == 0:
            c = r.chunk(); assert c['sequence'] == 2
            r.acknowledge_install(ref_id=c['ref_id'], native_step=2500)
    new = np.repeat(poses[8:9], 20, axis=0); new[:, 1, 3] = np.arange(20)*.001
    r.replace_stage_path(poses_world=new, opening=opening)
    c = r.chunk()
    assert len(r.intervals) == 8 and c['sequence'] == 3 and c['start_index'] == 8
    assert json.loads(c['wire'])['_pi0294']['sequence'] == 3
    np.testing.assert_array_equal(c['action_EEF10'][0, :3], poses[8, :3, 3])
    assert not c['BASE_supervision']
    with pytest.raises(ValueError): r.replace_stage_path(poses_world=new, opening=opening)


def test_four_fixed_attempts_exactly_exhaust_the_human_start_step_and_rgb_ceiling():
    rows = cards()
    assert len(rows) == 4
    assert sum(round((5+r['duration_s'])/.002)+3 for r in rows) == 62012
    assert sum(round(r['duration_s']/.05)+1 for r in rows) == 2084
    assert {r['role'] for r in rows} == {'SHORT', 'CONFIRM'}
    assert {tuple(r['source_xy']) for r in rows if r['role'] == 'SHORT'}.isdisjoint(
        {tuple(r['source_xy']) for r in rows if r['role'] == 'CONFIRM'})


def test_motion_hypotheses_require_actual_grasp_or_ungrasped_contact():
    s = dict(tcp=np.eye(4), object_pose=np.eye(4), native_grasp=False, robot_object_contact=False)
    with pytest.raises(ValueError, match='ACTUAL_NATIVE_GRASP'):
        PlanningObjectMotion(s, mode='ACTUAL_GRASP_ATTACHMENT')
    with pytest.raises(ValueError, match='ACTUAL_UNGRASPED_PAD_CONTACT'):
        PlanningObjectMotion(s, mode='ACTUAL_CONTACT_PUSH_TRANSLATION', push_direction=[1, 0])
    s.update(native_grasp=True, robot_object_contact=True)
    with pytest.raises(ValueError): PlanningObjectMotion(s, mode='ACTUAL_CONTACT_PUSH_TRANSLATION', push_direction=[1, 0])
    assert not PlanningObjectMotion(s, mode='ACTUAL_GRASP_ATTACHMENT').evidence()['actual_physics_or_labels_modified']


def native_fixture():
    mujoco = pytest.importorskip('mujoco')
    body = '<body name="a0" pos="0 0 1">'
    for i in range(1, 7):
        body += f'<body name="a{i}"><joint name="x5_joint{i}" type="slide" axis="1 0 0" range="-1 1"/>'
        body += '<geom type="sphere" size=".001" mass=".1" contype="0" conaffinity="0"/>'
    body += '<site name="x5_ee"/>'+'</body>'*7
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>'+body+
        '<body name="cube" pos="0 0 2"><freejoint name="task_cube_free"/>'
        '<geom name="task_cube_geom" type="sphere" size=".01" mass=".05"/></body></worldbody></mujoco>')
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    tcp = np.eye(4); tcp[:3, 3] = d.site_xpos[0]
    obj = np.eye(4); obj[2, 3] = 2.
    sample = dict(tcp=tcp, object_pose=obj, native_grasp=True, robot_object_contact=True,
        qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), native_step=2500)
    return mujoco, model, d, sample


def test_object_motion_changes_only_the_private_data_and_restores_no_actual_values():
    mujoco, model, data, sample = native_fixture()
    actual = sample['qpos'].copy()
    clone = mujoco.MjData(model); clone.qpos[:] = actual
    clone.qpos[0] += .05; mujoco.mj_forward(model, clone)
    motion = PlanningObjectMotion(sample, mode='ACTUAL_GRASP_ATTACHMENT'); motion(model, clone)
    assert clone.qpos[6] == pytest.approx(.05)
    np.testing.assert_array_equal(data.qpos, actual)
    np.testing.assert_array_equal(sample['qpos'], actual)


def test_dense_hold_is_checked_on_clone_and_discontinuous_join_is_rejected():
    _, model, data, sample = native_fixture()
    poses = np.repeat(sample['tcp'][None], 40, axis=0)
    proof = certify_chunk(model, sample, poses, stage='hold')
    assert proof['points'] == 1001 and proof['physics_integrations'] == 0
    np.testing.assert_array_equal(data.qpos, sample['qpos'])
    a = sample['tcp'].copy(); a[0, 3] -= .1
    history = [dict(pose=a, end_native_step=2475), dict(pose=sample['tcp'], end_native_step=2500)]
    with pytest.raises(ValueError, match='CARTESIAN_RATE'):
        certify_chunk(model, sample, poses, stage='hold', history=history)
