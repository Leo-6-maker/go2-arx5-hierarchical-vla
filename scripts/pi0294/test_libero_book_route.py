"""Synthetic actual-observation feedback tests; no MuJoCo or native GO."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.pi0294.multitask.libero_book_cards import cards
from scripts.pi0294.multitask.libero_book_teacher import (
    LiberoBookTeacher,RoutedLiberoBookTeacher,attempt_operation,checked_route_card,
    route_card,teacher_for_card)
from scripts.pi0294.multitask.libero_book_teacher import MARGIN_REVISION,DESCENT_REVISION,RELEASE_REVISION
from scripts.pi0294.multitask.feedback_teacher import CLOSED,OPEN
from scripts.pi0294.multitask.night_book_chain import checked_plan,planned_tasks
from scripts.pi0294.test_multitask_contact import state


def object_pose(z=.5071709548372156):
    B=np.eye(4);B[:3,:3]=np.diag([-1.,1.,-1.]);B[:3,3]=[.35953,-.11215,z]
    return B


def next_boundary(teacher,tcp,B,**kw):
    end=((teacher.index//8)+1)*8
    for i in range(teacher.index+1,end+1):
        teacher.observe(state('BOOK',i*.05,tcp=tcp,object_pose=B,native_grasp=True,**kw))


def carrying():
    card=route_card();teacher=teacher_for_card(card);X=np.array(card['book']['object_to_tcp_grasp'])
    B=object_pose();B[2,3]+=.075;tcp=B@X
    teacher.observe(state('BOOK',tcp=tcp,object_pose=B,native_grasp=True))
    teacher.stage='lift';teacher.goal=tcp.copy();teacher.lift_initial_object_z=B[2,3]-.075
    next_boundary(teacher,tcp,B)
    assert teacher.stage=='preinsert'
    teacher.plan()
    return teacher,X,B


def test_only_the_exact_left_static_route_changes_and_old_teacher_stays_selected():
    c=route_card();old=cards()['left'];expected=deepcopy(c)
    expected.pop('book_route_revision');expected.pop('book_route');expected['id']=old['id']
    expected['book']['object_to_tcp_grasp']=old['book']['object_to_tcp_grasp']
    assert expected==old and c['task_duration_s']==32. and c['warmup_s']==5.
    assert type(teacher_for_card(old))is LiberoBookTeacher
    assert type(teacher_for_card(c))is RoutedLiberoBookTeacher
    assert attempt_operation(c,'left')=='LIBERO_LEFT_PORT03_TOP_EDGE_ROUTE'
    assert attempt_operation(old,'left')=='LIBERO_LEFT_FIRST01'
    for key,value in [('task_duration_s',31.),('book_route_revision','UNKNOWN')]:
        bad=deepcopy(c);bad[key]=value
        with pytest.raises(PermissionError):checked_route_card(bad)
    with pytest.raises(PermissionError):checked_route_card(c,'right')


def test_nominal_grasp_matches_the_static_reference_and_keeps_the_real_grasp_relation():
    c=route_card();B=object_pose();X=np.array(c['book']['object_to_tcp_grasp']);G=B@X
    np.testing.assert_allclose(G[:3,3]-B[:3,3],[.001786,.025,.05],atol=1e-14)
    teacher=teacher_for_card(c)
    actual=X.copy();actual[1,3]+=.002
    high=B.copy();high[2,3]+=.075
    teacher.state=state('BOOK',tcp=high@actual,object_pose=high,native_grasp=True)
    teacher.lift_initial_object_z=B[2,3]
    lower,insert=teacher._book_targets()
    target=np.array(c['book']['target_book_world'])
    np.testing.assert_allclose(np.linalg.inv(target)@insert,actual,atol=1e-14)
    cross_book=teacher.waypoints['preinsert'][1]@np.linalg.inv(actual)
    assert cross_book[0,3]==pytest.approx(.454214)
    assert cross_book[2,3]==pytest.approx(high[2,3])
    assert lower[2,3]==pytest.approx(insert[2,3])


def test_carry_path_end_or_TCP_alone_cannot_cross_the_actual_book_barriers():
    teacher,X,B=carrying();initial=teacher.state['tcp'].copy()
    for _ in range(20):
        next_boundary(teacher,initial,B)
        plan=teacher.plan();np.testing.assert_array_equal(plan['opening'],np.full(40,CLOSED))
    assert teacher.stage=='preinsert'and teacher.waypoint_index['preinsert']==0
    lateral=teacher.goal.copy();lateral_book=lateral@np.linalg.inv(X)
    next_boundary(teacher,lateral,lateral_book);teacher.plan()
    assert teacher.waypoint_index['preinsert']==1 and teacher.stage=='preinsert'
    cross=teacher.goal.copy();cross_book=cross@np.linalg.inv(X)
    still_outside=cross_book.copy();still_outside[0,3]=.450
    next_boundary(teacher,cross,still_outside);teacher.plan()
    assert teacher.waypoint_index['preinsert']==1
    next_boundary(teacher,cross,cross_book);teacher.plan()
    assert teacher.waypoint_index['preinsert']==2 and teacher.stage=='preinsert'
    lower=teacher.goal.copy();next_boundary(teacher,lower,lower@np.linalg.inv(X))
    assert teacher.stage=='insert'


def test_grasp_loss_in_the_route_stops_on_an_observation_before_the_next_prefix():
    teacher,X,B=carrying();i=teacher.index+1
    with pytest.raises(PermissionError,match='ACTUAL_GRASP_LOST'):
        teacher.observe(state('BOOK',i*.05,tcp=teacher.state['tcp'],object_pose=B,native_grasp=False))
    with pytest.raises(PermissionError):teacher.plan()


def test_withdraw_checks_actual_clearance_and_preserves_the_original_two_second_hold():
    teacher=teacher_for_card(route_card());B=np.array(teacher.spec['target_book_world']);X=teacher.grasp_offset
    teacher.observe(state('BOOK',tcp=B@X,object_pose=B,native_grasp=False))
    teacher.stage='withdraw';teacher.goal=None;plan=teacher.plan()
    np.testing.assert_array_equal(plan['opening'],np.full(40,OPEN))
    for contact in [True,False]:
        end=((teacher.index//8)+1)*8;goal=teacher.goal.copy()
        for i in range(teacher.index+1,end+1):
            teacher.observe(state('BOOK',i*.05,tcp=goal,object_pose=B,native_grasp=False,
                                  robot_object_contact=contact))
        if contact:assert teacher.waypoint_index['withdraw']==0
        teacher.plan()
    assert teacher.stage=='withdraw'and teacher.waypoint_index['withdraw']==1
    goal=teacher.goal.copy();end=((teacher.index//8)+1)*8
    for i in range(teacher.index+1,end+1):
        teacher.observe(state('BOOK',i*.05,tcp=goal,object_pose=B,native_grasp=False,
                              native_libero_goal=True,object_supported_by_shelf=True))
    assert teacher.stage=='hold'and not teacher.candidate_complete
    start=teacher.index
    for i in range(start+1,start+41):
        teacher.observe(state('BOOK',i*.05,tcp=goal,object_pose=B,native_grasp=False,
                              native_libero_goal=True,object_supported_by_shelf=True))
    assert teacher.candidate_complete


def test_chain_selects_one_predeclared_left_condition_and_never_automatically_runs_right(tmp_path):
    fixed={'BOOK':dict(left=route_card(),right=cards()['right'])}
    (tmp_path/'FIXED_TASK_CARDS.json').write_text(json.dumps(fixed))
    assert checked_plan(tmp_path,'BOOK',side='left')==(('BOOK','left'),)
    with pytest.raises(PermissionError,match='ONE_STATIC_QUALIFIED'):checked_plan(tmp_path,'BOOK')
    assert planned_tasks('BOOK')==(('BOOK','left'),('BOOK','right'))
    with pytest.raises(ValueError):planned_tasks('ALL_PINNED',side='left')


def test_tracking_margin_changes_only_front_crossing_and_keeps_old_version_immutable():
    old=route_card();new=route_card(MARGIN_REVISION);same=deepcopy(new)
    same['id']=old['id'];same['book_route_revision']=old['book_route_revision']
    same['book_route']['front_cross_book_x_m']-=.01
    assert same==old
    assert checked_route_card(new)
    assert attempt_operation(new,'left')=='LIBERO_LEFT_PORT04_TRACKING_MARGIN'
    teacher=teacher_for_card(new);B=object_pose();X=teacher.grasp_offset
    B[2,3]+=.075;teacher.state=state('BOOK',tcp=B@X,object_pose=B,native_grasp=True)
    teacher.lift_initial_object_z=B[2,3]-.075;teacher._book_targets()
    cross=teacher.waypoints['preinsert'][1]@np.linalg.inv(X)
    assert cross[0,3]==pytest.approx(.464214)


def test_supported_descent_slows_only_lowering_without_changing_grip_or_goals():
    margin=route_card(MARGIN_REVISION);slow=route_card(DESCENT_REVISION);same=deepcopy(slow)
    same['id']=margin['id'];same['book_route_revision']=margin['book_route_revision']
    assert same['book_route'].pop('lower_speed_m_s')==.01
    assert same==margin
    teacher=teacher_for_card(slow);B=object_pose();B[2,3]+=.075;X=teacher.grasp_offset
    teacher.state=state('BOOK',tcp=B@X,object_pose=B,native_grasp=True)
    teacher.lift_initial_object_z=B[2,3]-.075;teacher._book_targets()
    cross=teacher.waypoints['preinsert'][1].copy();teacher.state['tcp']=cross
    teacher.stage='preinsert';teacher.waypoint_index['preinsert']=2;teacher.index=0
    teacher._build_stage_path()
    assert np.max(np.linalg.norm(np.diff(teacher.path[:,:3,3],axis=0),axis=1))/.05<=.01+1e-12
    np.testing.assert_allclose(teacher.path[-1],teacher.waypoints['preinsert'][2],atol=1e-14,rtol=0)
    assert slow['task_duration_s']==32. and attempt_operation(slow,'left')=='LIBERO_LEFT_PORT05_SLOW_DESCENT'


def test_supported_release_keeps_the_old_carry_geometry_and_does_not_command_a_floor_drag():
    old=route_card(DESCENT_REVISION);new=route_card(RELEASE_REVISION);same=deepcopy(new)
    same['id']=old['id'];same['book_route_revision']=old['book_route_revision'];assert same==old
    teacher=teacher_for_card(new);B=np.array(new['book']['target_book_world']);B[0,3]=new['book_route']['front_cross_book_x_m']
    tcp=B@teacher.grasp_offset
    teacher.observe(state('BOOK',tcp=tcp,object_pose=B,native_grasp=True))
    teacher.stage='preinsert';teacher.waypoints['preinsert']=[tcp.copy()for _ in range(3)]
    teacher.waypoint_index['preinsert']=2;teacher.goal=tcp.copy()
    teacher.plan()
    next_boundary(teacher,tcp,B,native_libero_goal=True,object_supported_by_shelf=True)
    assert teacher.stage=='release'
    plan=teacher.plan();np.testing.assert_allclose(plan['poses'],np.repeat(tcp[None],40,axis=0),atol=1e-14)
    np.testing.assert_array_equal(plan['opening'],np.full(40,OPEN))
    assert new['task_duration_s']==32. and new['independent_demo_conditions']==old['independent_demo_conditions']
    assert attempt_operation(new,'left')=='LIBERO_LEFT_PORT06_SUPPORTED_RELEASE'


@pytest.mark.parametrize('changed',[
    {'native_libero_goal':False},{'object_supported_by_shelf':False},
    {'object_linear_velocity':[.021,0,0]},{'object_angular_velocity':[0,0,.101]}])
def test_supported_release_waits_for_each_real_placement_condition(changed):
    teacher=teacher_for_card(route_card(RELEASE_REVISION));B=np.array(teacher.spec['target_book_world'])
    B[0,3]=teacher.route['front_cross_book_x_m'];tcp=B@teacher.grasp_offset
    teacher.observe(state('BOOK',tcp=tcp,object_pose=B,native_grasp=True))
    teacher.stage='preinsert';teacher.waypoints['preinsert']=[tcp.copy()for _ in range(3)]
    teacher.waypoint_index['preinsert']=2;teacher.goal=tcp.copy();teacher.plan()
    conditions=dict(native_libero_goal=True,object_supported_by_shelf=True);conditions.update(changed)
    next_boundary(teacher,tcp,B,**conditions)
    assert teacher.stage=='preinsert'
    np.testing.assert_array_equal(teacher.plan()['opening'],np.full(40,CLOSED))


def test_saved_actual_port05_prefix_is_unchanged_until_supported_release():
    path=os.environ.get('BOOK_PORT05_CLOSED_ACTUAL_REPLAY')
    if not path:pytest.skip('Explicit closed native PORT05 observation/plan capture required')
    from go2arx5_vla.task_data.codec import pose10
    packet=json.loads(Path(path).read_bytes())
    assert packet['raw_sha256']=='7a3b31968ccb538536f33ab0e868ad44bc0491354e7fe7c2e2aed5fd7fca96aa'
    old=teacher_for_card(route_card(DESCENT_REVISION));new=teacher_for_card(route_card(RELEASE_REVISION))
    plans={p['observation_index']:p for p in packet['plans']};changed=[]
    for row in packet['observations']:
        i=row['index'];s=row['state'];old.observe(s);new.observe(s)
        if i%8:continue
        original=old.plan();reference=new.plan();actual=np.array(plans[i]['rows10'])
        original_rows=np.array([pose10(p,g)for p,g in zip(original['poses'],original['opening'])])
        new_rows=np.array([pose10(p,g)for p,g in zip(reference['poses'],reference['opening'])])
        np.testing.assert_allclose(original_rows,actual,atol=1e-12,rtol=0)
        if i<384:np.testing.assert_allclose(new_rows,actual,atol=1e-12,rtol=0)
        else:
            changed.append(i)
            assert new.stage=='release'and old.stage=='insert'
            assert s['native_grasp']and s['native_libero_goal']and s['object_supported_by_shelf']
            np.testing.assert_allclose(reference['poses'],np.repeat(np.array(s['tcp'])[None],40,axis=0),atol=1e-12)
            np.testing.assert_array_equal(reference['opening'],np.full(40,OPEN))
    assert changed==[384]and len(plans)==49
