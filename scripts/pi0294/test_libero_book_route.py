"""Synthetic actual-observation feedback tests; no MuJoCo or native GO."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.pi0294.multitask.libero_book_cards import cards
from scripts.pi0294.multitask.libero_book_teacher import (
    LiberoBookTeacher,RoutedLiberoBookTeacher,attempt_operation,checked_route_card,
    route_card,teacher_for_card)
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
