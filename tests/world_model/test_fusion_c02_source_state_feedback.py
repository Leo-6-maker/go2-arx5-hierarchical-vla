"""Synthetic finite feedback boundaries, never native qualification."""
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_c02_source_state_feedback_v2 as c
from go2arx5_vla.world_model import target_sequential as target
from go2arx5_vla.world_model.fusion_task_phase_pose_feedback_v6 import _exp_so3


def program():
    start=np.eye(4);start[:3,3]=[.35,-.14,.70]
    goal=start.copy();goal[2,3]=.58
    cube=goal.copy();cube[2,3]-=.08
    return c.FeedbackProgram(start,start,goal,[0.,0.,0.,1.,0.,0.,0.],cube),goal


def test_all_predeclared_slots_with_bounded_causal_measured_error():
    p,goal=program();desired=p.desired_path.copy();original=p.reference_path.copy()
    for i,step in enumerate(c.STEPS):
        actual=goal.copy();actual[0,3]-=.1;actual[:3,:3]=_exp_so3([.03,0.,0.])
        path,proof=p.prepare_feedback(step,actual);atom=(step-c.DECISION)//25
        assert proof['applied_delta_norm_m']==pytest.approx(.008)
        assert proof['applied_rotation_norm_rad']==pytest.approx(.01)
        np.testing.assert_array_equal(path[:atom+1],p.reference_path[:atom+1])
        np.testing.assert_array_equal(path[:,9],original[:,9])
        p.commit(path,proof,'a'*64)
    assert p.sum_m==pytest.approx(.128) and p.sum_rad==pytest.approx(.16)
    np.testing.assert_array_equal(p.desired_path,desired)
    assert len(p.decisions)==16 and c.E==44703
    assert target.native_task_bucket(c.PHASE,True)=='96'
    with pytest.raises(ValueError,match='EXACT_NEXT_FINITE'):
        p.prepare_feedback(c.STEPS[-1],goal)


def test_no_stale_or_reordered_measurement_and_no_unsaved_commit():
    p,goal=program()
    with pytest.raises(ValueError,match='EXACT_NEXT_FINITE'):
        p.prepare_feedback(c.STEPS[1],goal)
    path,proof=p.prepare_feedback(c.STEPS[0],goal)
    with pytest.raises(ValueError,match='SAVED_DECISION'):
        p.commit(path,proof,'')
    assert not p.decisions


def test_zero_error_is_hold_and_achieved_pose_is_never_reanchored():
    p,goal=program();path,proof=p.prepare_feedback(c.STEPS[0],goal)
    np.testing.assert_allclose(path,p.reference_path,atol=1e-15)
    assert proof['applied_delta_norm_m']==proof['applied_rotation_norm_rad']==0
    assert not proof['actual_pose_is_reference_anchor']


def test_install_requires_all_due_saved_feedback_and_exact_full40():
    p,goal=program();step=c.STEPS[0];a=SimpleNamespace(spec=dict(phase=c.PHASE),trajectory_program=p,
        trajectory_initial_base=p.base_pose_hold,observe=lambda:dict(step=step))
    with pytest.raises(ValueError,match='ALL_DUE_FEEDBACK'):
        c.validate_install(a,p.preview((step-c.DECISION)//25))
    path,proof=p.prepare_feedback(step,goal);p.commit(path,proof,'b'*64)
    preview=p.preview((step-c.DECISION)//25);c.validate_install(a,preview)
    preview[-1,0]+=.001
    with pytest.raises(ValueError,match='COMPLETE_ACTUAL'):
        c.validate_install(a,preview)


def test_modified_endpoint_or_learned_selector_refused_before_loading():
    with pytest.raises(ValueError,match='EXACT_FINITE'):
        c.config(dict(phase=c.PHASE,kind=c.KIND,max_ticks=c.E+25))
    assert c.episode('/fresh')['old_fixed_goal_failed_result_sha256']==c.FAILED_RESULT
    assert len(c.episode('/fresh')['feedback_steps'])==16
