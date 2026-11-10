import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_teacher_v8 import Teacher,RecoveryUnavailable,consumed_grip,GRASP_Z
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER

BASE=np.array([0.,0.,.3,1.,0.,0.,0.])


def sample(step=task.B):
    cube=np.eye(4);cube[:3,3]=[.26,-.014,.62]
    tcp=np.eye(4);tcp[:3,3]=cube[:3,3]+[-.012,-.0017,GRASP_Z]
    return dict(native_step=step,time_s=.002*step,tcp=tcp,object_pose=cube,native_grasp=False,
        gripper_fraction=.9545,left_object_contact=False,right_object_contact=False,
        safety_valid=True,contact_valid=True,gripper_state_valid=True,object_present=True,object_linear_velocity=np.zeros(3))


def consumed(s,g):
    raw=np.r_[s['tcp'][:3,3],[1.,0.,0.,0.,1.,0.],g];installed=raw.copy();installed[9]=(.002+.04*g)/.044
    return dict(native_end_step=s['native_step'],policy_EEF10=raw,installed_consumed_EEF10=installed,
        solver_gripper_confirmed=True)


def test_bounded_close_uses_consumed_command_not_measured_opening():
    s=sample();s['gripper_fraction']=.43;t=Teacher(s);t.stage='CLOSE'
    p,v=t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,1.),supported=True)
    assert p.decoded_H40[0,9]==1. and p.decoded_H40[-1,9]==pytest.approx(.3)
    speed=np.abs(np.diff(p.reference_path[:,9]))*.044/.05
    assert speed.max()<=.03
    assert v['physical_margin_to_original_finger_speed_m_s']>=.01


def test_supported_single_pad_is_released_before_lateral_recenter():
    t=Teacher(sample(),recovery_entry=True)
    s=sample();s['tcp'][1,3]+=.0045;s['left_object_contact']=True
    angle=.01;s['tcp'][:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    first,v=t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,.55),supported=True)
    assert v['recovery_substage']=='SUPPORTED_RELEASE'
    assert np.array_equal(first.decoded_H40[:,:3],np.repeat(s['tcp'][None,:3,3],40,axis=0))
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    assert np.allclose(rot6d_rows_to_matrix(first.decoded_H40[:,3:9]),s['tcp'][:3,:3])
    assert first.decoded_H40[-1,9]==pytest.approx(.8)
    s=copy.deepcopy(s);s.update(native_step=task.B+1000,time_s=.002*(task.B+1000),left_object_contact=False)
    second,v=t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,.8),supported=True)
    assert v['recovery_substage']=='SUPPORTED_ALIGN'
    assert np.all(second.decoded_H40[:,9]==.8)
    assert second.decoded_H40[-1,1]==pytest.approx(t.grasp_goal(s)[1])


def test_unattached_airborne_object_has_explicit_no_command_exit():
    s=sample();t=Teacher(s,recovery_entry=True)
    with pytest.raises(RecoveryUnavailable,match='UNSUPPORTED_NOT_GRASPED_NO_AIR_OPENING'):
        t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,.2),supported=False)
    assert t.previous_decision_step is None


def test_current_cube_replans_but_lift_origin_and_task_score_are_not_rebased():
    initial=sample();t=Teacher(initial,recovery_entry=True);s=copy.deepcopy(initial);s['object_pose'][0,3]+=.006
    p,v=t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,.8),supported=True)
    assert v['privileged_goal'][0]==pytest.approx(t.grasp_goal(initial)[0]+.006)
    s=copy.deepcopy(initial);s.update(native_step=task.B+1000,time_s=.002*(task.B+1000),native_grasp=True)
    s['object_pose'][2,3]+=.01;s['tcp'][2,3]+=.01
    p,v=t.decide(s,s['tcp'],BASE,consumed_reference=consumed(s,.3),supported=False)
    assert v['privileged_goal'][2]-s['tcp'][2,3]==pytest.approx(.016)
    assert t.cube[2,3]==initial['object_pose'][2,3]
    assert p.decoded_H40[-1,9]<=p.decoded_H40[0,9]


def test_consumption_clock_and_mechanical_mapping_are_mandatory():
    s=sample();bad=consumed(s,.55);bad['native_end_step']-=25
    with pytest.raises(ValueError,match='ACTUALLY_CONSUMED'):consumed_grip(s,bad)
    bad=consumed(s,.55);bad['installed_consumed_EEF10'][9]=s['gripper_fraction']
    with pytest.raises(ValueError,match='WORKING_MAPPING_ONCE'):consumed_grip(s,bad)


def test_current_object_rotation_updates_grasp_frame_and_large_rotation_has_typed_exit():
    s=sample();t=Teacher(s,recovery_entry=True);rotated=copy.deepcopy(s);angle=.01
    rotated['object_pose'][:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    assert np.allclose(t.target_rotation(rotated),rotated['object_pose'][:3,:3])
    assert not np.array_equal(t.grasp_goal(rotated),t.grasp_goal(s))
    angle=.1;rotated['object_pose'][:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    with pytest.raises(RecoveryUnavailable,match='ORIENTATION_OUTSIDE'):
        t.decide(rotated,rotated['tcp'],BASE,consumed_reference=consumed(rotated,.8),supported=True)


def test_existing_V7_source_fingerprint_is_unchanged():
    assert teacher_fingerprint()==FROZEN_TEACHER
