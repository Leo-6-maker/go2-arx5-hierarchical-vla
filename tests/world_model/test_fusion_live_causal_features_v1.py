"""Causal feature/label isolation; no simulation, fitting or admission."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_live_causal_features_v1 as feature


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def fixture():
    r=rotation(np.pi/2);state=np.zeros((9,106));state[:,3]=1.
    state[:,25:27]=.042;state[:,86]=.2+np.arange(9)*.0001;state[:,88]=.4
    state[:,89:95]=np.r_[r[:,0],r[:,1]];state[:,104:106]=.042
    past=np.zeros((8,17));past[:,:3]=state[:8,86:89]+[0.,.001,0.]
    past[:,3:9]=r[:2].reshape(-1);past[:,9]=.95
    candidate=np.zeros((40,10));candidate[:,:3]=state[-1,86:89]
    candidate[:,1]+=.001*np.arange(40)/39
    for i in range(40):candidate[i,3:9]=(r@rotation(.01*i/39))[:2].reshape(-1)
    candidate[:,9]=.95
    history=dict(actual_state_history=state,actual_history_native_steps=list(range(34103,34304,25)),
        past_consumed_EEF10_BASE7=past,past_reference_native_end_steps=list(range(34128,34304,25)),
        history_provenance='ACTUAL_SAMPLES_AND_EXACT_INSTALLED_SOLVER_JOIN_PER_ATOM',
        full_MPC_hidden_state_restored=False)
    return history,candidate


def inputs(history,candidate,age=.01):
    return feature.causal_inputs(history,past_source_age_s=[0.,.05,.1,.15,.2,.25,.3,0.],
        candidate_H40=candidate,proposal_source_age_s=age)


def test_column_state_rotation_row_reference_rotation_and_units_are_explicit():
    history,candidate=fixture();value=inputs(history,candidate)
    assert value['robot_history'].shape==(9,70)
    assert value['measured_EEF_history'].shape==(9,8)
    assert value['planned_complete_H40'].shape==(40,7)
    np.testing.assert_allclose(value['planned_complete_H40'][-1,:6],[.001,0,0,0,0,.01],atol=1e-12)
    np.testing.assert_allclose(value['past_consumed_reference_error'][:,:3],np.tile([.001,0,0],(8,1)),atol=1e-12)
    np.testing.assert_allclose(value['prospective_install_clock'],[.01,.39],atol=1e-12)


def test_object_gt_unqualified_base_future_labels_and_ids_cannot_enter_features():
    history,candidate=fixture();expected=inputs(history,candidate);poison=copy.deepcopy(history)
    state=poison['actual_state_history'];state[:,27:34]=np.nan;state[:,60:66]=np.inf
    state[:,95:104]=-1e200;state[:,:3]=9e99
    poison['past_consumed_EEF10_BASE7'][:,10:]=np.nan
    poison.update(parent='DO_NOT_USE_ID',run_id='DO_NOT_USE_RUN',future_success=True,future_state=np.full((16,106),1e100))
    actual=inputs(poison,candidate)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_world_translation_and_absolute_step_shift_do_not_become_features():
    history,candidate=fixture();expected=inputs(history,candidate);changed=copy.deepcopy(history)
    shift=np.array([2.,-3.,4.]);changed['actual_state_history'][:,86:89]+=shift
    changed['past_consumed_EEF10_BASE7'][:,:3]+=shift;candidate=candidate.copy();candidate[:,:3]+=shift
    changed['actual_history_native_steps']=[s+1000 for s in changed['actual_history_native_steps']]
    changed['past_reference_native_end_steps']=[s+1000 for s in changed['past_reference_native_end_steps']]
    actual=inputs(changed,candidate)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-12)


@pytest.mark.parametrize('fault',['future-clock','expired-past','gap','bad-rotation','bad-grip'])
def test_bad_clocks_sequences_or_command_geometry_refuse_features(fault):
    h,c=fixture();kw=dict(past_source_age_s=np.zeros(8),candidate_H40=c,proposal_source_age_s=.01)
    if fault=='future-clock':kw['proposal_source_age_s']=.03
    elif fault=='expired-past':kw['past_source_age_s'][4]=.35
    elif fault=='gap':h['actual_history_native_steps'][3]+=25
    elif fault=='bad-rotation':c[39,3:9]=0.
    elif fault=='bad-grip':c[39,9]=1.1
    with pytest.raises((ValueError,PermissionError)):feature.causal_inputs(h,**kw)


def test_partial_real_response_labels_keep_mask_and_actual_not_desired_motion():
    h,c=fixture();future=np.repeat(h['actual_state_history'][-1:],3,axis=0)
    future[:,87]+=.0001*np.arange(1,4);future[:,104:106]-=.001
    label=feature.response_targets(h,candidate_H40=c,future_states=future,
        future_native_steps=34303+25*np.arange(1,4),future_consumed_EEF10=c[:3])
    np.testing.assert_array_equal(label['observed_horizon_mask'],[True]*3+[False]*13)
    np.testing.assert_allclose(label['TCP_SE3_and_finger_displacement'][:3,0],[.0001,.0002,.0003],atol=1e-12)
    np.testing.assert_allclose(label['TCP_SE3_and_finger_displacement'][:3,-2:],-.001,atol=1e-12)
    assert not np.allclose(label['TCP_SE3_and_finger_displacement'][0,0],c[0,1]-h['actual_state_history'][-1,87])


def test_cancel_or_changed_consumption_cannot_label_the_old_candidate():
    h,c=fixture();future=np.repeat(h['actual_state_history'][-1:],3,axis=0);consumed=c[:3].copy()
    consumed[2,0]+=.001
    with pytest.raises(ValueError,match='CHANGED_CANCELLED_OR_UNCONSUMED'):
        feature.response_targets(h,candidate_H40=c,future_states=future,
            future_native_steps=34303+25*np.arange(1,4),future_consumed_EEF10=consumed)
