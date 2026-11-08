"""Original-collector join fixtures; no physical qualification or data auth."""
import copy
import json
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_live_program_replay_v1 as replay
from go2arx5_vla.world_model.fusion_trajectory_adapter import _identity
from test_fusion_live_causal_features_v1 import fixture
from test_fusion_live_causal_features_v2 import rows


def artifacts():
    h,plan,future=rows();_,path=fixture();past=[]
    for i,end in enumerate(h['actual_history_native_steps']):
        state=h['actual_state_history'][i];tcp=np.eye(4);tcp[:3,3]=state[86:89]
        tcp[:3,0]=state[89:92];tcp[:3,1]=state[92:95];tcp[:3,2]=np.cross(tcp[:3,0],tcp[:3,1])
        past.append(dict(native_start_step=end-25,native_end_step=end,
            # The actual prefix ABI has before-atom histories and endpoint TCP,
            # but does not store actual_state_vector in each prefix row.
            actual_history_before_atom=dict(actual_state_history=[h['actual_state_history'][max(0,i-1)]]),
            actual_TCP=tcp,gripper=dict(actual_finger_q_m=state[104:106]),
            actual_installed_consumed_EEF10_BASE7=h['past_consumed_EEF10_BASE7'][max(0,i-1)],
            clock=dict(native_start_s=(end-25)*.002,source_mpc_s=(end-25)*.002)))
    # Populate the future collector's genuine rolling histories; unlike the
    # prefix, its confirmed rows additionally retain each actual state vector.
    states_by_end=dict(zip(h['actual_history_native_steps'],h['actual_state_history']))
    for i,row in enumerate(future):
        prior=past+future[:i];last=prior[-9:]
        states=[states_by_end[r['native_end_step']]for r in last]
        row['actual_history_before_atom']=dict(actual_state_history=states,
            actual_history_native_steps=[r['native_end_step']for r in last],
            past_consumed_EEF10_BASE7=[r['actual_installed_consumed_EEF10_BASE7']for r in last[1:]],
            past_reference_native_end_steps=[r['native_end_step']for r in last[1:]],
            history_provenance=h['history_provenance'],full_MPC_hidden_state_restored=False)
        state=row['actual_state_vector'];tcp=np.eye(4);tcp[:3,3]=state[86:89]
        tcp[:3,0]=state[89:92];tcp[:3,1]=state[92:95];tcp[:3,2]=np.cross(tcp[:3,0],tcp[:3,1])
        row['actual_TCP']=tcp;row['gripper']['actual_finger_q_m']=state[104:106]
        states_by_end[row['native_end_step']]=state
    spec=dict(phase=replay.PHASE,live_skill_start_step=34303,max_ticks=66503,physics_dt_s=.002,WBC_changes=False)
    prefix=dict(native_endpoint_step=34303,windows=past)
    adapt=dict(reference_path=_identity(path),clock=dict(end_atom=len(path)-1))
    decision=dict(start_step=34303,end_step=66503,future_response_observed=False,Processor_decodes=0,
        adaptation=adapt,nominal_reference_path=path,raw_H40=path,actual_history=h)
    segment=dict(start_step=34303,native_endpoint_step=66503,phase=replay.PHASE,
        adaptation=adapt,original_raw_H40=path,windows=future,all_40_knots_consumed=False)
    return json.loads(json.dumps([spec,prefix,decision,segment],default=lambda a:a.tolist()))


def test_one_real_window_shape_and_diagnostic_role_do_not_become_training_or_expert():
    record=replay.extract(*artifacts());x,names=replay.flatten_features(record)
    assert x.shape==(1614,) and record['labels']['observed_horizon_mask'].all()
    assert record['assignment']['split']=='DIAGNOSTIC' and not record['executed_expert_labels']
    assert not record['complete_H40_consumed'] and not record['full_MPC_hidden_state_restored']
    assert 'parent'not in names and 'split'not in names


def test_future_response_changes_labels_without_changing_any_causal_feature():
    raw=artifacts();a=replay.extract(*raw)
    for row in raw[3]['windows']:row['actual_state_vector'][86]+=.01
    b=replay.extract(*raw)
    np.testing.assert_array_equal(replay.flatten_features(a)[0],replay.flatten_features(b)[0])
    assert not np.array_equal(a['labels']['TCP_SE3_and_finger_displacement'],b['labels']['TCP_SE3_and_finger_displacement'])


def test_only_an_assignment_from_the_authenticated_original_spec_sets_data_role():
    raw=artifacts();raw[2]['split']='TRAIN';raw[3]['split']='TRAIN'
    assert replay.extract(*raw)['assignment']['split']=='DIAGNOSTIC'
    raw[0]['response_learning_assignment']=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',split='TRAIN',parent='NEW_TRAIN_PARENT',panel_sha256='a'*64)
    assert replay.extract(*raw)['assignment']['split']=='TRAIN'
    raw[0]['response_learning_assignment']['panel_sha256']='NOT_AUTHENTICATED_PANEL'
    with pytest.raises(ValueError,match='ORIGINAL_REGISTERED_SPEC'):replay.extract(*raw)


@pytest.mark.parametrize('fault',['state-join','past-reference','path-hash','future-decision','feature-leak'])
def test_changed_join_or_future_feature_injection_refuses_the_record(fault):
    raw=artifacts()
    if fault=='state-join':raw[1]['windows'][-1]['actual_history_before_atom']['actual_state_history'][-1][86]+=.001
    elif fault=='past-reference':raw[1]['windows'][-1]['actual_installed_consumed_EEF10_BASE7'][0]+=.001
    elif fault=='path-hash':raw[2]['nominal_reference_path'][-1][0]+=.001
    elif fault=='future-decision':raw[2]['future_response_observed']=True
    if fault=='feature-leak':
        record=replay.extract(*raw);record['features']['future_success']=np.array([1.])
        with pytest.raises(ValueError,match='FEATURE_WHITELIST'):replay.flatten_features(record)
    else:
        with pytest.raises(ValueError):replay.extract(*raw)


def test_zero_confirmed_post_decision_response_stays_masked_not_a_zero_motion_label():
    raw=artifacts();raw[3]['windows']=[];record=replay.extract(*raw)
    assert not record['labels']['observed_horizon_mask'].any()
    assert not record['first_install_clock_observed_before_effect']


def test_raw_mjdata_roundoff_uses_original_ros_nanoseconds_without_refreshing_age():
    raw=artifacts();expected=replay.extract(*raw)
    for row in raw[1]['windows']+raw[3]['windows']:
        row['clock']['native_start_s']-=8e-12
    actual=replay.extract(*raw)
    np.testing.assert_array_equal(replay.flatten_features(actual)[0],replay.flatten_features(expected)[0])
    np.testing.assert_array_equal(actual['labels']['observed_horizon_mask'],expected['labels']['observed_horizon_mask'])


def test_later_install_uses_only_its_actual_past_and_keeps_parent_grouping():
    raw=artifacts();past=(raw[1]['windows']+raw[3]['windows'][:7])[-9:]
    # Original future rows already carry their correctly rolled actual history.
    history=raw[3]['windows'][7]['actual_history_before_atom']
    assert history['actual_history_native_steps'][-1]==34478
    history=dict(history,actual_state_history=history['actual_state_history'],
        actual_history_native_steps=[r['native_end_step']for r in past],
        past_consumed_EEF10_BASE7=[r['actual_installed_consumed_EEF10_BASE7']for r in past[1:]],
        past_reference_native_end_steps=[r['native_end_step']for r in past[1:]],
        history_provenance='ACTUAL_SAMPLES_AND_EXACT_INSTALLED_SOLVER_JOIN_PER_ATOM',
        full_MPC_hidden_state_restored=False)
    raw[3]['windows'][7]['actual_history_before_atom']=history
    value=replay.extract(*raw,install_atom=7)
    assert value['forecast_native_step']==34478 and value['program_elapsed_atom']==7
    assert value['labels']['valid_observed_atoms']==9
    assert not value['overlapping_anchors_are_independent_physical_parents']
    assert value['assignment']==replay.extract(*raw)['assignment']
    assert replay.flatten_features(value)[0].shape==(1614,)
    with pytest.raises(ValueError,match='PUBLICATION_GRID'):replay.extract(*raw,install_atom=1)
    with pytest.raises(ValueError,match='NO_UNOBSERVED'):replay.extract(*raw,install_atom=21)
