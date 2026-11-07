"""Preview exposure and causal label masks; no inference or physical grant."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_live_causal_features_v2 as feature
from test_fusion_live_causal_features_v1 import fixture


def rows():
    history,path=fixture();plan=feature.plan_from_complete_reference_path(path)
    actual=[];end=history['actual_history_native_steps'][-1]
    for atom in range(16):
        state=history['actual_state_history'][-1].copy();state[87]+=.0001*(atom+1)
        actual.append(dict(native_start_step=end+25*atom,native_end_step=end+25*(atom+1),actual_ticks=25,
            stage='ACTUAL_CONSUMPTION_CONFIRMED',solver_policies_with_exact_installed_target=[{'confirmed':True}],
            gripper=dict(command_consumed=True),full_final_installed_H40=plan[atom//7].copy(),
            actual_installed_consumed_EEF10_BASE7=np.r_[plan[0,atom],history['past_consumed_EEF10_BASE7'][-1,10:]],
            clock=dict(source_mpc_s=(end+25*(7*(atom//7)))*.002,native_start_s=(end+25*atom)*.002),
            actual_state_vector=state))
    return history,plan,actual


def test_full_schedule_inputs_are_pre_effect_and_exclude_future_truth():
    h,p,r=rows();kw=dict(past_source_age_s=np.zeros(8),planned_H40=p,first_install_source_age_s=0.)
    a=feature.causal_inputs(h,**kw);h.update(future_response=np.ones((16,106))*9e99,future_success=True)
    b=feature.causal_inputs(h,**kw)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    assert a['planned_published_H40'].shape==(3,40,7)
    np.testing.assert_allclose(a['planned_install_offsets_s'],[0.,.35,.7])
    result=feature.response_targets(h,planned_H40=p,first_install_source_age_s=0.,observed_windows=r)
    assert result['observed_horizon_mask'].all() and result['valid_observed_atoms']==16


@pytest.mark.parametrize('fault',['unconsumed-suffix','late-clock','wrong-BASE','missing-solver','cancel','gap'])
def test_identical_first_points_do_not_hide_changed_exposure(fault):
    h,p,r=rows()
    if fault=='unconsumed-suffix':r[7]['full_final_installed_H40'][39,0]+=.001
    elif fault=='late-clock':r[7]['clock']['source_mpc_s']-=.01
    elif fault=='wrong-BASE':r[7]['actual_installed_consumed_EEF10_BASE7'][10]+=.001
    elif fault=='missing-solver':r[7]['solver_policies_with_exact_installed_target']=[]
    elif fault=='cancel':r[7]['stage']='CANCELLED'
    elif fault=='gap':r[7]['native_start_step']+=25
    result=feature.response_targets(h,planned_H40=p,first_install_source_age_s=0.,observed_windows=r)
    assert result['valid_observed_atoms']==7 and result['first_exposure_mismatch']['observed_atom']==7
    np.testing.assert_array_equal(result['observed_horizon_mask'],[True]*7+[False]*9)
    np.testing.assert_array_equal(result['TCP_SE3_and_finger_displacement'][7:],0.)


def test_missing_confirmation_at_first_atom_cannot_create_dynamics_labels():
    h,p,r=rows();r[0]['gripper']['command_consumed']=False
    result=feature.response_targets(h,planned_H40=p,first_install_source_age_s=0.,observed_windows=r)
    assert result['valid_observed_atoms']==0 and not result['observed_horizon_mask'].any()


def test_changed_planned_overlap_is_not_another_candidate_with_same_contract():
    h,p,r=rows();p[1,0,0]+=.001
    with pytest.raises(ValueError,match='SAME_FIXED_PROGRAM'):
        feature.causal_inputs(h,past_source_age_s=np.zeros(8),planned_H40=p,first_install_source_age_s=0.)


def test_complete_short_program_preserves_terminal_hold_in_all_future_previews():
    h,path=fixture();plan=feature.plan_from_complete_reference_path(path[:2])
    np.testing.assert_array_equal(plan[0,2:],np.tile(path[1],(38,1)))
    np.testing.assert_array_equal(plan[2],np.tile(path[1],(40,1)))
