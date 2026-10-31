from copy import deepcopy
import json,os
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask import cabinet_braked_turn as braked,cabinet_edge_hinge as previous,source_stage_bundle as bundle
from scripts.pi0294.multitask.cabinet_repair02 import checked_card,attempt_operation
from scripts.pi0294.multitask.native_hinge_reference import from_native_boundary
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.object_relations import axis_rotation


def test_only_time_law_changes_and_original_grasp_failure_remains_a_stop():
    old=previous.cards();new=braked.cards();r=new['right'];expected=deepcopy(old['right'])
    expected.update(id=r['id'],port_revision=braked.VERSION,turn_time_profile=braked.PROFILE)
    assert r==expected and new['left']==old['left']and previous.cards()==old
    assert checked_card(r,'right')and attempt_operation(r,'right')=='ROBOCASA_RIGHT_PORT08_BRAKED_TURN'
    assert bundle.hashlib.sha256(bundle.raw(dict(BOOK=books(),ROBOCASA_CABINET=new))).hexdigest()==bundle.PORT08_CARD_SHA
    bad=deepcopy(r);bad['turn_reference_limits']['max_linear_m_s']*=2
    with pytest.raises(PermissionError):checked_card(bad,'right')
    m=dict(task_kind='ROBOCASA_CABINET',condition_side='right',cabinet_port_revision=braked.VERSION,
        predeclared_tasks=[dict(task_kind='ROBOCASA_CABINET',side='right')],attempts_this_chain_max=1)
    assert bundle.checked_manifest(m)[1]==m['predeclared_tasks']
    for side in('left',None):
        with pytest.raises(PermissionError):bundle.checked_manifest(dict(m,condition_side=side))


@pytest.mark.parametrize('direction',[1,-1])
def test_same_arc_peak_speed_and_limit_with_continuous_braking(direction):
    h=np.eye(4);tool=np.eye(4);tool[:3,3]=[.2,0,0]
    start=-direction*.00017225154761722712;goal=direction*.95*1.57
    tool=axis_rotation([0,0,1],start)@tool
    limits=[0,1.57]if direction==1 else[-1.57,0]
    original=from_native_boundary(hinge_world=h,axis_hinge=[0,0,1],actual_tcp_world=tool,actual_handle_world=tool,
        actual_angle_rad=start,target_angle_rad=goal,actual_grasp_valid=True,native_safety_valid=True,
        angle_limits_rad=limits,max_linear_m_s=.03,max_angular_rad_s=.15)
    changed=braked.brake(original,h);vel=np.diff(changed.hinge_angles_rad)/.05
    np.testing.assert_allclose(changed.tcp_world[[0,-1]],original.tcp_world[[0,-1]],atol=1e-12)
    assert changed.hinge_angles_rad[0]==start and changed.hinge_angles_rad[-1]==goal
    assert np.all(changed.hinge_angles_rad[1:]>=limits[0])and np.all(changed.hinge_angles_rad[1:]<=limits[1])
    assert np.max(np.abs(vel))<=.15+1e-12
    assert np.max(np.linalg.norm(np.diff(changed.tcp_world[:,:3,3],axis=0),axis=1))/.05<=.03+1e-12
    assert .4-1e-12<=changed.times_s[-1]-original.times_s[-1]<=.45+1e-12
    assert np.all(np.diff(np.abs(vel[-16:]))<0)and abs(vel[-1])<.001
    assert np.max(np.abs(np.diff(vel[-17:])))/.05<.3


def test_actual_safe_grasp_boundary_uses_new_profile_without_weakening_contact_gate():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit saved PORT07 actual boundary')
    raw=json.loads((Path(directory)/'CAB_EDGE07_SAVED_CONTACT_TRACE.private.json').read_bytes())
    s=next(x['state']for x in raw['timeline']if x['metadata'].get('index')==136)
    teacher=SourceSkillTeacher(braked.cards()['right']);teacher.stage='turn';teacher.observe(s);teacher.plan()
    assert len(teacher.path)>220
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    velocity=np.array([geodesic_angle(a[:3,:3],b[:3,:3])/.05 for a,b in zip(teacher.path[:-1],teacher.path[1:])])
    assert velocity.max()<=.15+1e-12 and velocity[-1]<.001
    assert np.max(np.linalg.norm(np.diff(teacher.path[:,:3,3],axis=0),axis=1))/.05<=.03+1e-12
    failure=SourceSkillTeacher(braked.cards()['right']);failure.stage='turn'
    with pytest.raises(PermissionError,match='ACTUAL_GRASP_LOST'):failure.observe(dict(s,native_grasp=False))


def test_complete_braked_route_and_native_sampled_contacts_are_actual_static_only():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit saved complete route/native forward evidence')
    p=Path(directory);route=json.loads((p/'CABINET_BRAKED08_FULL_ROUTE.private.json').read_bytes())
    native=json.loads((p/'CABINET_BRAKED08_NATIVE_STATIC.private.json').read_bytes())['result']
    card=braked.cards()['right'];candidate=route['candidate_card']
    for key in('cabinet_world','size_m','orientation'):np.testing.assert_equal(card['cabinet'][key],candidate['cabinet'][key])
    for key in('hinge_world','object_to_tcp_grasp'):
        np.testing.assert_allclose(card['door'][key],candidate['door'][key],atol=1e-15,rtol=0)
    assert route['all_Knot_IK_and_joint_speed_pass']and route['all_full_robot_fixture_knots_clear']
    assert route['with_original_verify_hold_minimum_s']<32
    assert native['source_knots']==477 and native['native_forward_calls']==11901 and native['physics_integrations']==0
    assert not native['native_penetration_or_joint_or_arm_table_violation']and native['strict_unexpected_robot_contact_samples']==0
    assert not native['qualification']and native['task_fixture_internal_max_penetration_m']<1e-12
