import numpy as np

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_recovery_v1 import (
    UnderCloseRecoveryTeacher,EXPERT,INTERVENTION,contract)
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER


def fixture():
    cube=np.eye(4);cube[:3,3]=[.26,-.014,.62]
    tcp=np.eye(4);tcp[:3,3]=cube[:3,3]+[0,-.0017,.080]
    s=dict(tcp=tcp,object_pose=cube,native_grasp=False)
    return s,np.r_[0.,0.,.3,1.,0.,0.,0.]


def at_close(teacher,s,base):
    teacher.decide(s,s['tcp'],base)
    close=dict(s,tcp=s['tcp'].copy())
    close['tcp'][:3,3]=teacher.grasp_goal()
    return close


def test_intervention_is_excluded_and_recovery_needs_two_later_real_observations():
    s,base=fixture();t=UnderCloseRecoveryTeacher(s);close=at_close(t,s,base)
    p,m=t.decide(close,close['tcp'],base)
    assert m['label_role']==INTERVENTION
    assert p.decoded_H40[0,9]==1 and p.decoded_H40[-1,9]==.55
    assert not m['recovery_evidence']['actual_under_close_failed']
    assert not m['recovery_evidence']['native_grasp_after_correction']
    p,m=t.decide(close,close['tcp'],base)
    assert m['label_role']==EXPERT and p.decoded_H40[0,9]==.55 and p.decoded_H40[-1,9]==0
    assert m['recovery_evidence']['actual_under_close_failed']
    assert not m['recovery_evidence']['native_grasp_after_correction']
    grasped=dict(close,native_grasp=True)
    p,m=t.decide(grasped,grasped['tcp'],base)
    assert t.stage=='LIFT' and m['label_role']==EXPERT
    assert m['recovery_evidence']['native_grasp_after_correction']
    assert p.decoded_H40[-1,2]>p.decoded_H40[0,2]


def test_partial_close_that_already_grasped_cannot_count_as_recovery():
    s,base=fixture();t=UnderCloseRecoveryTeacher(s);close=at_close(t,s,base)
    t.decide(close,close['tcp'],base)
    grasped=dict(close,native_grasp=True)
    _,m=t.decide(grasped,grasped['tcp'],base)
    assert t.stage=='LIFT'
    assert not m['recovery_evidence']['actual_under_close_failed']
    assert not m['recovery_evidence']['native_grasp_after_correction']


def test_intervention_changes_only_grip_and_preserves_frozen_teacher():
    s,base=fixture();t=UnderCloseRecoveryTeacher(s);normal=task.Teacher(s)
    close=at_close(t,s,base);at_close(normal,s,base)
    altered,_=t.decide(close,close['tcp'],base);original,_=normal.decide(close,close['tcp'],base)
    assert np.array_equal(altered.decoded_H40[:,:9],original.decoded_H40[:,:9])
    assert teacher_fingerprint()==FROZEN_TEACHER
    assert contract()['native_runtime_qualified']is False
    assert contract()['partial_close_is_expert_label']is False
