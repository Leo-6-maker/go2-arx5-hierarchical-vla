"""Hermetic record tests: no model, renderer, FK, clock advance or file reads."""
import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest
from scripts.pi0294.base_intent import eef_document
from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
from go2arx5_vla.world_model.target_reference import measured_base_hold, require_live_window, trajectory_identity
from go2arx5_vla.world_model import fusion_physics_feedback as f


def identity():
    return dict(run_id='case-01',native_run_id='native-01',native_spec_sha256='1'*64,
        executor_version='2'*64,scene_sha256='3'*64,frame_id='NATIVE_TARGET_WORLD',
        tcp_semantic_id='ARX5_NATIVE_SITE_x5_ee')


def state(step,x=.3):
    qpos=np.zeros(34);qpos[2]=.4;qpos[3]=1.
    tcp=np.eye(4);tcp[:3,3]=[x,.1,.5]
    obj=np.eye(4);obj[:3,3]=[.5,.1,.46]
    return dict(native_step=step,time_s=step*.002,qpos=qpos.tolist(),qvel=[0.]*32,
        ctrl=[0.]*20,tcp=tcp.tolist(),object_pose=obj.tolist(),finger_q=[.042,.042],
        robot_object_contact=False,left_object_contact=False,right_object_contact=False,
        native_grasp=False,safety_valid=True,native_abort=False,reset=False,takeover=False)


def action(offset=0):
    a=np.tile([.3,.1,.5,1.,0.,0.,0.,1.,0.,float(teacher_physical_fraction(1))],(40,1))
    a[:,0]+=(np.arange(40)+offset)*.001
    return a


def consumption(before,after,*,offset=0,source_lag=0,program='4'*64,wire_sequence=0):
    a=action(offset);start=before['native_step'];end=after['native_step'];source=start*.002-source_lag
    base=[0.,0.,.4,0.,0.,0.,1.]
    doc=measured_base_hold(eef_document(a),base)
    doc['_pi0294']=dict(episode='native-01',sequence=wire_sequence,operation='install',
        ttl_s=.4,source_mpc_s=source)
    wire=json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False)
    sha=hashlib.sha256(wire.encode()).hexdigest()
    installed=dict(kind='installed',ref_id=sha,bound_start=source,observation_time_s=start*.002,monotonic_s=1.,
        times=[source+i*.05 for i in range(41)],
        poses=[[*b['position'],*b['quaternion_xyzw'],*e['position'],*e['quaternion_xyzw']]
            for b,e in zip(doc['base']['waypoints'],doc['ee']['waypoints'])],inputs=[[] for _ in range(41)])
    target_sha=trajectory_identity(installed['times'],installed['poses'],installed['inputs'])
    receiver_events=[dict(kind='accept',ref_id=sha,raw=wire,source_mpc_s=source,deadline=source+.4,
        receiver_mpc_s=start*.002),installed]
    sent=dict(sequence=wire_sequence,opening=float(a[0,9]),source_mpc_s=source,
        expiry=source+.4,native_start_step=start)
    w=dict(native_start_step=start,native_end_step=end,actual_ticks=25,
        reference_sha256=sha,target_trajectory_sha256=target_sha,
        clock=require_live_window(source,start*.002,25),
        solver_policies_with_exact_installed_target=[dict(trajectory_sha256=target_sha,
            receiver_send_monotonic_s=1.,received_monotonic_s=2.)],
        full_final_installed_H40=a.tolist(),actual_installed_consumed_EEF10_BASE7=np.r_[a[0],base].tolist(),
        gripper=dict(command_consumed=True,submitted=sent,
            consumed_wrapper=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',pid=12345,library_sha256='8'*64,values=dict(
                initialized=True,has_command=True,steps=end,sequence=wire_sequence,
                opening=sent['opening'],expiry=sent['expiry']))))
    return dict(window=w,proposed=dict(wire=wire,wire_sha256=sha,native_step=start,
        source_mpc_s=source,action_EEF10=a.tolist(),receiver_events=receiver_events),
        program=dict(sha256=program,start_native_step=3503),
        state_before_sha256=f.state_sha256(before),state_after_sha256=f.state_sha256(after))


def rewire(c,edit):
    """Change actual serialized wire and every direct byte pin, not its outcome."""
    doc=json.loads(c['proposed']['wire']);edit(doc)
    wire=json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False)
    sha=hashlib.sha256(wire.encode()).hexdigest()
    c['proposed']['wire']=wire;c['proposed']['wire_sha256']=sha;c['window']['reference_sha256']=sha
    for e in c['proposed']['receiver_events']:
        e['ref_id']=sha
        if e['kind']=='accept':e['raw']=wire


def saved_prediction(before,after,records,*,domain='UPSTREAM_INSTALLED_REFERENCE'):
    predicted=copy.deepcopy(after);predicted['tcp'][0][3]-=.0005
    record=dict(schema=f.PREDICTION_SCHEMA,identity=identity(),model_sha256='6'*64,
        issue_state_sha256=f.state_sha256(before),history_sha256='7'*64,
        history_span=[before['native_step']-200,before['native_step']],future_response_observed=False,
        issue_native_step=before['native_step'],target_native_step=after['native_step'],
        issue_time_s=before['time_s'],target_time_s=after['time_s'],
        conditioning_command_domain=domain,
        conditioning_commands=[f.command_condition(c,identity=identity()) for c in records],
        predicted={name:(f._base(predicted).tolist() if name=='base_pose' else predicted[name]) for name in f.FIELDS})
    return seal_prediction(record)


def seal_prediction(record):
    raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    sha=hashlib.sha256(raw).hexdigest()
    return dict(bytes=raw,sha256=sha,persist_receipt=dict(prediction_sha256=sha,
        durability='ORIGINAL_FSYNC_COMPLETED_BEFORE_EFFECT',producer_source_sha256='8'*64,
        native_step_before=record['issue_native_step'],native_step_after=record['issue_native_step'],
        persisted_monotonic_ns=100))


def timing(before,after):
    return dict(trace_sha256='9'*64,interval_start_native_step=before['native_step'],
        observation_native_step=after['native_step'],first_effect_monotonic_ns=200,observation_monotonic_ns=300)


def lifecycle(before,after,events=()):
    return dict(native_interval=[before['native_step'],after['native_step']],artifact_sha256='a'*64,events=list(events))


def packet(before=None,after=None,c=None,prediction=None,records=None,**kwargs):
    before=state(3503) if before is None else before
    after=state(3528,.302) if after is None else after
    c=consumption(before,after) if c is None else c
    return f.make_packet(identity=identity(),state_before=before,state_after=after,consumption=c,
        saved_prediction=prediction,consumed_since_prediction=[c] if records is None else records,
        lifecycle_evidence=lifecycle(before,after),causal_timing=timing(before,after),**kwargs)


def forecast(after):
    a=action();certificate=dict(status='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF',
        physics_integrations=0,actual_state_modified=False,executed_reference_history_points=2,
        points=1051,Cartesian_reference_values_changed=False,local_IK_iterations=1051,
        max_joint_speed_rad_s=.1,max_joint_acceleration_rad_s2=.2)
    return dict(candidate_id='candidate-not-command',identity=identity(),issue_native_step=after['native_step'],
        target_native_step=after['native_step']+400,model_sha256='b'*64,full_candidate_H40=a.tolist(),
        predicted=dict(tcp=after['tcp'],base_pose=f._base(after).tolist(),finger_q=after['finger_q'],
            object_pose=after['object_pose'],event_probabilities=dict(contact=.8,grasp=.6,release=.1)),
        candidate_qualification=dict(native_qualification_sha256='c'*64,actual_state_sha256=f.state_sha256(after),
            native_step=after['native_step'],full_preview_sha256=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest(),
            certificate=certificate,certificate_sha256=digest(certificate)),
        support_domain_flag='IN_DOMAIN_WITH_EXTERNAL_QUALIFICATION',model_qualification_sha256='d'*64,
        uncertainty=dict(empirical_tcp_error_interval_m=[[-.01,.01]]*3,ensemble_tcp_dispersion_m=[.001]*3))


def test_tracking_and_innovation_are_different_quantities_and_no_permissions():
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    p=packet(before,after,c,saved_prediction(before,after,[c]))
    assert p['measured']['values']['tracking_error']['tcp']['translation_world_m']==pytest.approx([.001,0,0])
    assert p['innovation']['values']['tcp']['translation_world_m']==pytest.approx([.0005,0,0])
    assert p['innovation']['valid'] and p['innovation']['partial_observability_remains']
    assert p['consumed_reference']['command_domain_evidence']==dict(UPSTREAM_INSTALLED_REFERENCE=True,
        SOLVER_TARGET_CONFIRMED=True,CONTROLLER_SELECTED_POLICY=False)
    assert p['WM_checkpoint_sha256']==['6'*64]
    for key in ('native_execution_authorized','online_deployment_qualified','current_proprioception_modified',
                'Processor_padding_modified','language_modified'):
        assert p[key] is False
    assert [p[k] for k in ('GR00T_updates','WM_updates','Actor_updates','forecast_action_installations','new_queries','new_integrations')]==[0]*6
    assert json.loads(json.dumps(p,allow_nan=False))==p


def test_missing_prediction_or_command_is_masked_null_not_zero():
    p=f.make_packet(identity=identity(),state_before=state(3503),state_after=state(3528,.302))
    assert p['consumed_reference'] is None and p['innovation']['valid'] is False
    assert p['innovation']['invalid_reasons']==['NO_SAVED_PREDICTION']
    assert all(x is None for x in p['innovation']['values'].values())
    assert p['measured']['values']['tracking_error']['tcp'] is None
    assert p['measured']['validity_mask']['tcp'] is True
    assert p['measured']['validity_mask']['tracking_tcp'] is False
    assert p['forecast']==[]


def test_observation_time_uses_wire_interpolation_and_gripper_first_row_hold():
    before,after=state(3503),state(3528,.302);c=consumption(before,after,source_lag=.01)
    c['proposed']['action_EEF10'][1][9]=float(teacher_physical_fraction(0))
    c['window']['full_final_installed_H40']=copy.deepcopy(c['proposed']['action_EEF10'])
    p=packet(before,after,c)
    assert p['measured']['values']['tracking_error']['tcp']['translation_world_m'][0]==pytest.approx(.0008)
    assert p['measured']['values']['tracking_error']['finger_error_m']==pytest.approx([0,0])
    assert p['consumed_reference']['reference_age_s']==pytest.approx(.06)


@pytest.mark.parametrize('domain,solver,valid',[
    ('UPSTREAM_INSTALLED_REFERENCE',False,True),('SOLVER_TARGET_CONFIRMED',False,False),
    ('SOLVER_TARGET_CONFIRMED',True,True),('CONTROLLER_SELECTED_POLICY',True,False),('UNKNOWN',True,False)])
def test_explicit_command_domains_do_not_invent_selected_wbc_policy(domain,solver,valid):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    if not solver:c['window'].pop('solver_policies_with_exact_installed_target')
    p=packet(before,after,c,saved_prediction(before,after,[c],domain=domain))
    assert p['innovation']['valid'] is valid
    assert p['measured']['validity_mask']['tracking_tcp'] is True
    assert p['innovation']['controller_selected_policy_id_proven'] is False
    assert p['measured']['command_domain_evidence']['SOLVER_TARGET_CONFIRMED'] is solver


def test_same_program_renewal_changes_wire_and_local_index_without_invalidating_global_condition():
    s0,s1,s2=state(3503),state(3528,.302),state(3553,.303)
    c0=consumption(s0,s1);c1=consumption(s1,s2,offset=1,wire_sequence=1)
    pred=saved_prediction(s0,s2,[c0,c1])
    p=f.make_packet(identity=identity(),state_before=s1,state_after=s2,consumption=c1,saved_prediction=pred,
        consumed_since_prediction=[c0,c1],lifecycle_evidence=lifecycle(s0,s2,[dict(native_step=3528,kind='SAME_PROGRAM_RENEWAL')]),
        causal_timing=timing(s0,s2))
    assert p['innovation']['valid']
    assert c0['proposed']['wire_sha256']!=c1['proposed']['wire_sha256']
    assert p['consumed_reference']['actual_consumed_index']==0
    assert p['consumed_reference']['condition']['global_atom_index']==1


def test_unrelated_solver_hash_masks_solver_domain_without_masking_upstream_domain():
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    c['window']['solver_policies_with_exact_installed_target'][0]['trajectory_sha256']='e'*64
    for domain,valid in [('SOLVER_TARGET_CONFIRMED',False),('UPSTREAM_INSTALLED_REFERENCE',True)]:
        p=packet(before,after,c,saved_prediction(before,after,[c],domain=domain))
        assert p['innovation']['valid'] is valid
        assert not p['measured']['command_domain_evidence']['SOLVER_TARGET_CONFIRMED']
        assert p['measured']['command_domain_evidence']['UPSTREAM_INSTALLED_REFERENCE']


def test_same_reference_rolling_target_is_joined_from_its_actual_receiver_event():
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    rolling=copy.deepcopy(c['proposed']['receiver_events'][1]);rolling['times']=[x+.002 for x in rolling['times']]
    sha=trajectory_identity(rolling['times'],rolling['poses'],rolling['inputs'])
    c['window']['solver_policies_with_exact_installed_target'][0]['trajectory_sha256']=sha
    c['reference_receiver_events']=c['proposed']['receiver_events']+[rolling]
    p=packet(before,after,c,saved_prediction(before,after,[c],domain='SOLVER_TARGET_CONFIRMED'))
    assert p['innovation']['valid'] and p['measured']['command_domain_evidence']['SOLVER_TARGET_CONFIRMED']


@pytest.mark.parametrize('change',['wrong_send_time','missing_send_time'])
def test_solver_send_time_must_belong_to_exact_receiver_target_event(change):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    if change=='wrong_send_time':c['window']['solver_policies_with_exact_installed_target'][0]['receiver_send_monotonic_s']=.9
    else:c['proposed']['receiver_events'][1].pop('monotonic_s')
    p=packet(before,after,c,saved_prediction(before,after,[c],domain='SOLVER_TARGET_CONFIRMED'))
    assert not p['innovation']['valid']
    assert p['measured']['validity_mask']['tracking_tcp']
    assert not p['measured']['command_domain_evidence']['SOLVER_TARGET_CONFIRMED']


@pytest.mark.parametrize('change',['lookahead','source_phase'])
def test_changed_full_h40_lookahead_or_source_phase_is_a_changed_prediction_condition(change):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    prediction=saved_prediction(before,after,[c])
    changed=consumption(before,after,source_lag=.01 if change=='source_phase' else 0)
    if change=='lookahead':
        changed['proposed']['action_EEF10'][20][0]+=.001
        changed['window']['full_final_installed_H40']=copy.deepcopy(changed['proposed']['action_EEF10'])
        rewire(changed,lambda d:d['ee']['waypoints'][20]['position'].__setitem__(0,
            changed['proposed']['action_EEF10'][20][0]))
    p=packet(before,after,changed,prediction,records=[changed])
    assert p['consumption_invalid_reason'] is None
    assert p['innovation']['valid'] is False
    assert 'EVERY_ACTUAL_PROGRAM_SUFFIX' in p['innovation']['invalid_reasons'][0]


@pytest.mark.parametrize('kind',sorted(f.DISCONTINUITIES)+['UNKNOWN_EVENT'])
def test_any_discontinuity_or_unknown_event_invalidates_old_prediction(kind):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    p=f.make_packet(identity=identity(),state_before=before,state_after=after,consumption=c,
        saved_prediction=saved_prediction(before,after,[c]),consumed_since_prediction=[c],
        lifecycle_evidence=lifecycle(before,after,[dict(native_step=3510,kind=kind)]),causal_timing=timing(before,after))
    assert not p['innovation']['valid'] and p['innovation']['invalid_reasons']
    assert p['measured']['validity_mask']['tracking_tcp']


@pytest.mark.parametrize('change',[
    'late_save','wrong_target','future_history','future_response','wrong_identity','missing_domain',
    'wrong_issue_state','wrong_program','missing_atom','wrong_chain','wrong_lifecycle','no_timing','bad_bytes',
    'reset','takeover','native_abort','missing_reset'])
def test_causal_or_command_evidence_failure_masks_innovation(change):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    prediction=saved_prediction(before,after,[c]);record=json.loads(prediction['bytes'])
    consumed=[copy.deepcopy(c)];life=lifecycle(before,after);time=timing(before,after)
    if change=='wrong_target':record['target_native_step']+=25
    if change=='future_history':record['history_span'][1]+=25
    if change=='future_response':record['future_response_observed']=True
    if change=='wrong_identity':record['identity']['run_id']='another-restart'
    if change=='missing_domain':record.pop('conditioning_command_domain')
    if change=='wrong_issue_state':record['issue_state_sha256']='e'*64
    if change=='wrong_program':consumed[0]['program']['sha256']='e'*64
    if change=='missing_atom':consumed=[]
    if change=='wrong_chain':consumed[0]['state_after_sha256']='e'*64
    if change=='wrong_lifecycle':life['native_interval'][1]+=25
    if change=='no_timing':time={}
    if change in ('reset','takeover','native_abort'):after[change]=True
    if change=='missing_reset':after.pop('reset')
    prediction=seal_prediction(record)
    if change=='late_save':prediction['persist_receipt']['persisted_monotonic_ns']=200
    if change=='bad_bytes':prediction['bytes']+=b' '
    p=f.make_packet(identity=identity(),state_before=before,state_after=after,consumption=c,
        saved_prediction=prediction,consumed_since_prediction=consumed,lifecycle_evidence=life,causal_timing=time)
    assert not p['innovation']['valid'] and p['innovation']['invalid_reasons']
    assert all(v is False for v in p['innovation']['validity_mask'].values())


@pytest.mark.parametrize('change',['wrong_wire','missing_ack','wrong_installed_target','wrong_ack_source',
    'wrong_h40','wrong_eef_wire','wrong_base','wrong_grip_seq','stale_grip',
    'wrong_grip_expiry','grip_not_consumed','wrong_target_time','raw_grip','wrong_actual_hash','expired_pose'])
def test_bad_actual_consumption_never_becomes_tracking_or_innovation(change):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    if change=='wrong_wire':c['proposed']['wire']+=' '
    if change=='missing_ack':c['proposed']['receiver_events'].pop(0)
    if change=='wrong_installed_target':c['proposed']['receiver_events'][1]['poses'][0][0]+=.1
    if change=='wrong_ack_source':c['proposed']['receiver_events'][0]['source_mpc_s']-=.01
    if change=='wrong_h40':c['window']['full_final_installed_H40'][3][0]+=.001
    if change=='wrong_eef_wire':rewire(c,lambda d:d['ee']['waypoints'][3]['position'].__setitem__(0,.9))
    if change=='wrong_base':rewire(c,lambda d:d['base']['waypoints'][0]['position'].__setitem__(0,.1))
    if change=='wrong_grip_seq':c['window']['gripper']['consumed_wrapper']['values']['sequence']+=1
    if change=='stale_grip':
        c['window']['gripper']['submitted']['source_mpc_s']-=.1
        c['window']['gripper']['submitted']['expiry']-=.1
        c['window']['gripper']['consumed_wrapper']['values']['expiry']-=.1
    if change=='wrong_grip_expiry':c['window']['gripper']['submitted']['expiry']+=.1
    if change=='grip_not_consumed':c['window']['gripper']['command_consumed']=False
    if change=='wrong_target_time':c['window']['clock']['native_end_s']+=.002
    if change=='raw_grip':c['proposed']['action_EEF10'][0][9]=1.;c['window']['full_final_installed_H40'][0][9]=1.
    if change=='wrong_actual_hash':c['state_after_sha256']='e'*64
    if change=='expired_pose':c['proposed']['source_mpc_s']-=.4
    p=packet(before,after,c)
    assert p['consumption_invalid_reason'] and p['consumed_reference'] is None
    assert p['measured']['validity_mask']['tracking_tcp'] is False
    assert p['measured']['validity_mask']['tcp'] is True


def test_held_old_gripper_sequence_is_not_new_per_atom_consumption():
    s0,s1,s2=state(3503),state(3528,.302),state(3553,.303)
    c0=consumption(s0,s1);c1=consumption(s1,s2,offset=1,wire_sequence=0)
    p=f.make_packet(identity=identity(),state_before=s1,state_after=s2,consumption=c1,
        saved_prediction=saved_prediction(s0,s2,[c0,c1]),consumed_since_prediction=[c0,c1],
        lifecycle_evidence=lifecycle(s0,s2),causal_timing=timing(s0,s2))
    assert p['innovation']['valid'] is False
    assert 'CONTIGUOUS_PER_ATOM_GRIPPER_SEQUENCE' in p['innovation']['invalid_reasons'][0]
    assert p['measured']['validity_mask']['tracking_tcp']


def test_gripper_child_or_library_change_invalidates_cross_interval_prediction():
    s0,s1,s2=state(3503),state(3528,.302),state(3553,.303)
    c0=consumption(s0,s1);c1=consumption(s1,s2,offset=1,wire_sequence=1)
    pred=saved_prediction(s0,s2,[c0,c1])
    c1['window']['gripper']['consumed_wrapper']['pid']+=1
    p=f.make_packet(identity=identity(),state_before=s1,state_after=s2,consumption=c1,
        saved_prediction=pred,consumed_since_prediction=[c0,c1],
        lifecycle_evidence=lifecycle(s0,s2),causal_timing=timing(s0,s2))
    assert p['innovation']['valid'] is False
    assert 'SAME_ACTUAL_OWNED_GRIPPER_CHILD' in p['innovation']['invalid_reasons'][0]


def test_packet_assembly_does_not_read_artifacts_or_spawn_processes(monkeypatch):
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    pred=saved_prediction(before,after,[c]);candidate=forecast(after)
    originals=copy.deepcopy((before,after,c,pred,candidate))
    def forbidden(*args,**kwargs):raise AssertionError('Offline packet attempted an effect')
    monkeypatch.setattr('builtins.open',forbidden)
    monkeypatch.setattr('subprocess.Popen',forbidden)
    p=packet(before,after,c,pred,forecast_candidates=[candidate])
    assert p['innovation']['valid']
    assert (before,after,c,pred,candidate)==originals


def test_forecasts_are_separate_and_uncalibrated_uncertainty_or_events_masked():
    after=state(3528,.302);candidate=forecast(after);original=copy.deepcopy(candidate)
    p=packet(after=after,forecast_candidates=[candidate]);row=p['forecast'][0]
    assert candidate==original and row['record_consistent']
    assert row['validity_mask']['tcp'] and row['field_availability_mask']['event_probabilities']['contact']
    assert not row['validity_mask']['event_probabilities']['contact']
    assert row['uncertainty']['values']['ensemble_tcp_dispersion_m']==[.001]*3
    assert row['uncertainty']['validity_mask']['ensemble_tcp_dispersion_m'] is False
    assert row['usable_for_execution'] is False and row['model_or_candidate_qualified_by_packet'] is False
    assert row['forecast_is_current_proprioception'] is False and row['forecast_is_command'] is False
    assert row['forecast_horizon_seconds']==pytest.approx(.8)
    candidate['event_calibration_sha256']='e'*64;candidate['uncertainty']['calibration_sha256']='f'*64
    row=packet(after=after,forecast_candidates=[candidate])['forecast'][0]
    assert row['validity_mask']['event_probabilities']['contact']
    assert row['uncertainty']['validity_mask']['ensemble_tcp_dispersion_m']
    assert row['uncertainty_calibration_verified'] is False


@pytest.mark.parametrize('change',['unknown_wm','missing_model_proof','wrong_state','wrong_preview_hash',
    'bad_certificate','physics_certificate','modified_state_certificate','wrong_clock','raw_grip',
    'incomplete_h40','bad_predicted','bad_events','incomplete_certificate','repaired_reference_certificate'])
def test_forecast_support_and_legal_candidate_missing_proof_is_retained_but_masked(change):
    after=state(3528,.302);candidate=forecast(after);proof=candidate['candidate_qualification']
    if change=='unknown_wm':candidate['support_domain_flag']='UNKNOWN'
    if change=='missing_model_proof':candidate.pop('model_qualification_sha256')
    if change=='wrong_state':proof['actual_state_sha256']='e'*64
    if change=='wrong_preview_hash':proof['full_preview_sha256']='e'*64
    if change=='bad_certificate':proof['certificate_sha256']='e'*64
    if change=='physics_certificate':proof['certificate']['physics_integrations']=1
    if change=='modified_state_certificate':proof['certificate']['actual_state_modified']=True
    if change=='wrong_clock':candidate['target_native_step']+=1
    if change=='raw_grip':candidate['full_candidate_H40'][0][9]=1.
    if change=='incomplete_h40':candidate['full_candidate_H40'].pop()
    if change=='bad_predicted':candidate['predicted']=None
    if change=='bad_events':candidate['predicted']['event_probabilities']=None
    if change=='incomplete_certificate':
        proof['certificate'].pop('points');proof['certificate_sha256']=digest(proof['certificate'])
    if change=='repaired_reference_certificate':
        proof['certificate']['Cartesian_reference_values_changed']=True;proof['certificate_sha256']=digest(proof['certificate'])
    row=packet(after=after,forecast_candidates=[candidate])['forecast'][0]
    assert not row['record_consistent'] and row['invalid_reasons']
    assert not any(row['validity_mask'][key] for key in f.FIELDS)
    assert row['usable_for_execution'] is False


def test_se3_pi_rotation_is_finite_and_base_wxyz_conversion_is_explicit():
    after=state(3528,.302);after['tcp'][1][1]=-1;after['tcp'][2][2]=-1
    p=packet(after=after)
    e=p['measured']['values']['tracking_error']['tcp']
    assert e['rotation_error_rad']==pytest.approx(np.pi)
    assert np.linalg.norm(e['rotation_reference_frame_rad'])==pytest.approx(np.pi)
    assert e['convention']=='INV_REFERENCE_TIMES_ACTUAL_NO_EULER_SUBTRACTION'
    assert np.asarray(p['measured']['values']['base_pose'])[:3,:3]==pytest.approx(np.eye(3))
    assert json.loads(json.dumps(p,allow_nan=False))==p


def test_invalid_forecast_identity_still_has_explicit_null_values_and_false_masks():
    c=forecast(state(3528,.302));c['identity']['frame_id']='different-frame'
    row=packet(forecast_candidates=[c])['forecast'][0]
    assert not row['record_consistent']
    assert all(row['values'][name] is None and row['validity_mask'][name] is False for name in f.FIELDS)
    assert all(x is None for x in row['values']['event_probabilities'].values())


def test_invalid_metadata_and_nonfinite_saved_prediction_remain_json_safe_masked_records():
    before,after=state(3503),state(3528,.302);c=consumption(before,after)
    pred=saved_prediction(before,after,[c]);raw=pred['bytes'].replace(b'"future_response_observed":false',
        b'"future_response_observed":false,"invalid_extra":NaN')
    pred['bytes']=raw;pred['sha256']=hashlib.sha256(raw).hexdigest()
    candidate=forecast(after);candidate['candidate_id']=b'not-text';candidate['model_sha256']=b'x'*64
    candidate['support_domain_flag']=b'not-text';candidate['uncertainty']['calibration_sha256']=b'x'*64
    p=packet(before,after,c,pred,forecast_candidates=[candidate])
    assert p['innovation']['valid'] is False and p['forecast'][0]['record_consistent'] is False
    assert p['forecast'][0]['candidate_id'] is None and p['forecast'][0]['support_domain_flag']=='UNKNOWN'
    assert json.loads(json.dumps(p,allow_nan=False))==p


def test_missing_actual_fields_mask_independently_and_forecast_does_not_fill_them():
    before,after=state(3503),state(3528,.302);candidate=forecast(after)
    for key in ('tcp','object_pose','finger_q','qpos','robot_object_contact'):after.pop(key)
    p=f.make_packet(identity=identity(),state_before=before,state_after=after,forecast_candidates=[candidate])
    assert all(p['measured']['values'][key] is None for key in f.FIELDS)
    assert all(p['measured']['validity_mask'][key] is False for key in f.FIELDS)
    assert p['measured']['values']['events']['robot_object_contact'] is None
    assert p['forecast'][0]['field_availability_mask']['tcp']
    assert not p['forecast'][0]['record_consistent']


@pytest.mark.parametrize('change',['time','gap','boolstep','frame_missing','executor_missing','nan'])
def test_actual_clock_and_explicit_identity_fail_closed(change):
    before,after=state(3503),state(3528,.302);ident=identity()
    if change=='time':after['time_s']+=.002
    if change=='gap':after=state(3553)
    if change=='boolstep':after['native_step']=True
    if change=='frame_missing':ident.pop('frame_id')
    if change=='executor_missing':ident['executor_version']=''
    if change=='nan':after['time_s']=float('nan')
    with pytest.raises(ValueError):f.make_packet(identity=ident,state_before=before,state_after=after)
