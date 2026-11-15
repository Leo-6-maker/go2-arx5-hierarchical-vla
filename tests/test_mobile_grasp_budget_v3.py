import numpy as np
import pytest

from go2arx5_vla.world_model.mobile_grasp_budget_v3 import BudgetTeacher,StageClock,STAGE_SECONDS,grasp_raise


def sample():
    tcp=np.eye(4);tcp[:3,3]=[.26,0,.72]
    obj=np.eye(4);obj[:3,3]=[.2565,0,.62]
    return dict(tcp=tcp,object_pose=obj,native_step=5703,native_grasp=False)


def test_separate_lateral_free_geometry_arm_preserves_other_three_targets():
    s=sample()
    for family in('F1','F2'):
        for mode in('M_FREE','M_ASSIST'):
            c=dict(family=family,mode=mode);t=BudgetTeacher(s,c)
            expected=.008 if(family,mode)==('F2','M_FREE')else 0.
            assert grasp_raise(c)==expected
            assert np.allclose(t.grasp_goal(),s['object_pose'][:3,3]+[-.012,-.0017,.012+expected])
            t.stage='DESCEND';s2=dict(s,tcp=s['tcp'].copy());s2['tcp'][:3,3]=t.grasp_goal()+[.0041,0,0]
            t.goal(s2);assert t.stage=='DESCEND'
            s2['tcp'][:3,3]=t.grasp_goal()+[.0039,0,0];t.goal(s2);assert t.stage=='CLOSE'


def test_expired_stage_does_not_force_next_phase_and_native_clock_is_not_wall_time():
    c=StageClock(5703);assert not c.observe('PREGRASP',5703)['exhausted']
    assert c.observe('PREGRASP',15703)['exhausted']
    assert c.stage=='PREGRASP'
    event=c.observe('DESCEND',15703)
    assert not event['exhausted']and event['stage_elapsed_s']==0 and event['task_elapsed_s']==20
    assert sum(STAGE_SECONDS.values())==59.
    with pytest.raises(ValueError,match='P10_STAGE_CLOCK'):c.observe('DESCEND',15704)


def test_original_complete_h40_with_raised_target_and_committed_boundary():
    s=sample();t=BudgetTeacher(s,dict(family='F2',mode='M_FREE'));t.stage='DESCEND';reference=s['tcp'].copy()
    p,e=t.decide(s,reference,[0,0,.4,0,0,0,1]);p.validate_envelope()
    assert e['grasp_goal_raise_m']==.008
    assert e['experimental_arm']=='BUDGET_AND_F2_FREE_PAD_CLEARANCE'
    assert p.decoded_H40.shape==(40,10)and np.linalg.norm(p.decoded_H40[-1,:3]-p.decoded_H40[0,:3])<=.015
    assert np.allclose(t.commit(p,5953)[:3,3],p.reference_path[10,:3])


def test_qualification_cannot_promote_task_flag_through_native_failure():
    from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record
    r=dict(completed=True,unsafe=False,technical_failure=False,shutdown_clean=True,
        detail=dict(task_success=True,terminal_native_step=20000),actual_native_integrations=20000,
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS'),condition=dict(family='F1',mode='M_FREE'))
    assert qualification_record(r)['qualified']
    for key in('unsafe','technical_failure'):
        bad=dict(r,**{key:True});assert not qualification_record(bad)['qualified']
    assert not qualification_record(dict(r,trace_audit=dict(error='MISSING_TRACE')))['qualified']
    assert not qualification_record(dict(r,trace_audit=None))['qualified']
