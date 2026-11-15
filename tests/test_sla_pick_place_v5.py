import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import sla_pick_place_v5 as place,task_learning_grasp_v1 as grasp


def trajectory():
    samples=[];standing=[]
    for i in range(111):
        obj=np.eye(4);obj[:3,3]=[.26,0.,.62];tcp=np.eye(4);tcp[:3,3]=[.26,0.,.74]
        closed=i<50
        if 10<=i<40:obj[2,3]=.646
        if i>=20:obj[1,3]=min(.15,(i-20)*.15/19)
        samples.append(dict(native_step=5703+25*i,time_s=(5703+25*i)*.002,object_pose=obj,tcp=tcp,
            gripper_fraction=.05 if closed else .95,native_grasp=closed,
            left_object_contact=closed,right_object_contact=closed,robot_object_contact=closed,
            object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),
            place_actual_table_support_force_N=.5 if i>=40 else 0.,
            safety_valid=True,contact_valid=True,gripper_state_valid=True,object_present=True,
            native_abort=False,illegal_contact=False,joint_limit_penetration=False,reset=False,takeover=False))
        standing.append(dict(native_step=5703+25*i,contract=grasp.standing_contract(),passes=True,
            base_displacement_m=0.,base_rotation_rad=0.,maximum_foot_planar_displacement_m=0.,
            supported_feet=['FL','FR','RL','RR'],old_short_task_pose_bound_passes=True))
    return samples,standing


def test_requires_full_airborne_transfer_and_actual_released_supported_terminal():
    s,w=trajectory();score=place.score(s,.62,standing=w,goal=[.26,.15,.62])
    assert score['task_success']and score['maximum_grasped_carry_xy_m']==.15
    assert score['longest_contiguous_terminal_success_s']>=2.
    # Tail evidence proves terminal state, not the preceding manipulation.
    assert place.terminal_score(s[60:],.62,standing=w[60:],goal=[.26,.15,.62])['task_success']
    assert not place.score(s[60:],.62,standing=w[60:],goal=[.26,.15,.62])['task_success']


@pytest.mark.parametrize('failure',['support','withdrawal','release','lift','carry','safety','clock'])
def test_incomplete_or_unsafe_place_is_never_success(failure):
    s,w=trajectory()
    if failure=='support':
        for row in s:row['place_actual_table_support_force_N']=0.
    if failure=='withdrawal':
        for row in s:row['tcp'][:3,3]=row['object_pose'][:3,3]+[0,0,.03]
    if failure=='release':
        for row in s:row['gripper_fraction']=.05
    if failure=='lift':
        for row in s:row['object_pose'][2,3]=.62
    if failure=='carry':
        for row in s:row['object_pose'][1,3]=0.
    if failure=='safety':s[0]['illegal_contact']=True
    if failure=='clock':
        s[-1]['native_step']+=25;w[-1]['native_step']+=25
        with pytest.raises(ValueError):place.score(s,.62,standing=w,goal=[.26,.15,.62])
        return
    assert not place.score(s,.62,standing=w,goal=[.26,.15,.62])['task_success']


def test_hold_success_does_not_grant_place_success_and_grasp_teacher_stays_frozen():
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert teacher_fingerprint()==FROZEN_TEACHER
    s,w=trajectory()
    for row in s:
        row['native_grasp']=row['left_object_contact']=row['right_object_contact']=row['robot_object_contact']=True
        row['object_pose'][:3,3]=[.26,0.,.646];row['gripper_fraction']=.05
    assert grasp.score(s,.62,standing=w)['task_success']
    assert not place.score(s,.62,standing=w,goal=[.26,.15,.62])['task_success']


def test_old_protocol_and_place_declared_clocks_are_separate():
    from go2arx5_vla.world_model.corelab_expert_semantics_v3 import native_protocol,native_endpoint
    old=grasp.protocol(grasp.CONSTRUCTION_ROLE);assert old['task_command']==grasp.COMMAND
    new=dict(sla_pick_place_v5=dict(version=place.VERSION),task_learning_role=place.ROLE,
             task_learning_teacher_mode='NORMAL_TEACHER_V7')
    assert native_protocol(new)==place.protocol()and native_endpoint(new)==36703
    assert place.protocol()['policy_queries']==120 and old['policy_queries']==15


def test_postgrasp_reference_is_bounded_without_shortening_the_task_goal():
    s,_=trajectory();initial=s[0];teacher=place.Teacher(initial)
    teacher.stage='TRANSFER';teacher.grip=0.;teacher.lift_TCP_z=float(initial['tcp'][2,3])
    program,_=teacher.decide(initial,initial['tcp'],np.array([0,0,0,1,0,0,0.]))
    assert np.linalg.norm(program.decoded_H40[-1,:3]-initial['tcp'][:3,3])<=.004+1e-12
    np.testing.assert_allclose(teacher.goal_center-initial['object_pose'][:3,3],[0,.15,0],rtol=0,atol=1e-15)
    assert place.score_contract()['minimum_carried_xy_m']==.1
