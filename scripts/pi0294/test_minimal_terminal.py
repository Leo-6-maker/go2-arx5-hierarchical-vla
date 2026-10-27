"""Actual historical ledger input; all future main terminal/model data are fixtures."""
from copy import deepcopy
import pytest
from scripts.pi0294.test_minimal_execution import actual,request,closed_CPU_fixture
from scripts.pi0294.generalization import minimal_terminal as t,minimal_execution as e,minimal_effects as effects
from scripts.pi0294.pv1 import mix_accounting as a


@pytest.fixture
def bound(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap)
    v=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)['proposed_global']
    return e.bind(v,r['request_id'],'12345.stdct-mgmt-02',now_epoch=1793131102.)


def captured_CPU_terminal(g,*,success):
    r=e.rows(g)[0];common=dict(request_id=r['request_id'],request_sha256=r['request_sha256'],job_id=r['job_id'],source_sha=r['source_sha'])
    slices={s['id']:dict(status='COMPLETE'if success else'NOT_STARTED',optimizer_updates=s['optimizer_limit']if success else 0,
        generation_queries=s['query_limit']if success else 0,model_loads=loads if success else 0,
        **(dict(terminal_receipt_sha256='b'*64)if success else{}))for s,loads in zip(e.native_sessions(),(1,3,3,2,2))}
    if not success:slices[e.native_sessions()[0]['id']]['status']='ACTIVE'
    ef=dict(schema=effects.SUMMARY,**common,pending_updates=[],unknown_query_intents=[],global_budget_is_only_external_reservation=True,
        new_physics=0,new_episodes=0,optimizer_updates_committed=1028 if success else 0,query_attempts=147 if success else 0,
        model_load_attempts=11 if success else 0,sessions=slices,active_session=None if success else e.native_sessions()[0]['id'],
        all_five_sessions_runtime_complete=success)
    native=dict(schema=t.NATIVE,**common,job_state='F',submission_nonce=r['submission_nonce'],Exit_status=0 if success else 1,
        all_native_workers_terminal=True,terminal_inventory_stable=True,raw_native_sha256='a'*64,process_final_sha256='b'*64,
        wrapper_final_sha256='c'*64,inventory_sha256='d'*64,conservative_GPU_s=3600,conservative_CPU_s=1900,
        new_native_conservative_CPU_s=1000,prior_readonly_preparation_CPU_upper_s=900,
        finished_epoch=1793131103.,observed_epoch=1793131104.)
    audit=dict(status=t.AUDIT if success else t.PARTIAL,**common,effect_summary_sha256=e.sha(e.raw(ef)),inventory_sha256='d'*64,
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0)
    if success:
        audit.update(audited_session_ids=[s['id']for s in e.native_sessions()],same15_zero_step_raw_wrapped_arms_exact=True,
            two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact=True,official_full_model_reconstruction_output_pairs_exact=True,
            frozen_parameters_and_buffers_unchanged=True,endpoints={arm:dict(global_step=512,cursor=16384,scheduler_horizon=1024,
                raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,full_resume=True,qualification_only=False,
                TRAIN_probes=12,language_probes=24,checkpoint_manifest_sha256='e'*64,official_inference_manifest_sha256='f'*64,
                full_streamed_members_sha256='a'*64)for arm in e.training.ARMS})
    return dict(native=native,effects=ef,audit=audit,now_epoch=1793131105.)


@pytest.mark.parametrize('success',[True,False])
def test_original_global_once_known_terminal_keeps_all_old_costs_and_remaining_pack(bound,success):
    before=deepcopy(bound);r=e.rows(bound)[0];args=captured_CPU_terminal(bound,success=success)
    v=t.settle(bound,r['request_id'],**args);assert bound==before
    assert v['used']['PBS_submissions']==bound['used']['PBS_submissions']
    assert v['used']['H200_s']==bound['used']['H200_s']+3600
    assert v['CPU_s']==bound['CPU_s']+1900
    assert v['optimizer_updates']==bound['optimizer_updates']+(1028 if success else 0)
    assert v['generation_queries']==bound['generation_queries']+(147 if success else 0)
    assert v['model_loads']==bound['model_loads']+(11 if success else 0)
    assert v['PBS_submissions'][:-1]==bound['PBS_submissions'][:-1]
    assert v[a.PACKAGE]==bound[a.PACKAGE]and v['pending_updates']==v['unknown_effects']==[]
    assert v[e.package.FIELD]['remaining_resources']==bound[e.package.FIELD]['remaining_resources']
    assert e.forecast_remaining(v)==v[e.package.FIELD]['remaining_resources']
    assert e.rows(v)[0]['scientific_endpoint_complete']is success
    assert v['new_storage_current_bytes']==bound['new_storage_current_bytes']+e.native_resources()['Hopper_storage_bytes']
    with pytest.raises(PermissionError):t.settle(v,r['request_id'],**args)


@pytest.mark.parametrize('change',['notF','still_live','wrong_job','wrong_source','old_schema','unknown_query','pending',
    '900_refund','900_double_charge','negative_new_CPU','too_many_updates','missing_qualification_calls','future_effect',
    'skipped_session','missing_restore','CPU_flags_only','wrong_horizon','wrong_cursor','inference_not_rebuilt','wrong_R0','unknown_global'])
def test_unproven_or_unknown_terminal_cannot_advance_or_erase_effects(bound,change):
    args=captured_CPU_terminal(bound,success=True);n,x,u=args['native'],args['effects'],args['audit'];ids=list(x['sessions'])
    if change=='notF':n['job_state']='R'
    elif change=='still_live':n['all_native_workers_terminal']=False
    elif change=='wrong_job':n['job_id']='1497959.stdct-mgmt-02'
    elif change=='wrong_source':x['source_sha']='0'*40
    elif change=='old_schema':x['schema']='V5_CD_ACTUAL_NATIVE_EFFECT_SUMMARY_V1'
    elif change=='unknown_query':x['unknown_query_intents']=[146]
    elif change=='pending':x['pending_updates']=[{'effect':'UNKNOWN_KEEP'}]
    elif change=='900_refund':n['prior_readonly_preparation_CPU_upper_s']=0
    elif change=='900_double_charge':n['conservative_CPU_s']+=900
    elif change=='negative_new_CPU':n['new_native_conservative_CPU_s']=-1;n['conservative_CPU_s']=899
    elif change=='too_many_updates':x['optimizer_updates_committed']=1029
    elif change=='missing_qualification_calls':x['optimizer_updates_committed']=1024
    elif change=='future_effect':x['sessions'][ids[1]]['status']='NOT_STARTED'
    elif change=='skipped_session':x['sessions'][ids[1]]['status']='ACTIVE'
    elif change=='missing_restore':x['sessions'][ids[1]]['model_loads']=1;x['model_load_attempts']=9
    elif change=='CPU_flags_only':u['status']='PASS_CPU_SMALL_MODEL_FIXTURE'
    elif change=='wrong_horizon':u['endpoints'][e.training.ARMS[0]]['scheduler_horizon']=512
    elif change=='wrong_cursor':u['endpoints'][e.training.ARMS[0]]['cursor']=32768
    elif change=='inference_not_rebuilt':u['official_full_model_reconstruction_output_pairs_exact']=False
    elif change=='wrong_R0':u['statistics_sha256']='f'*64
    else:bound['unknown_effects']=[{'intent':'KEEP_UNKNOWN'}]
    u['effect_summary_sha256']=e.sha(e.raw(x));before=deepcopy(bound)
    with pytest.raises(PermissionError):t.settle(bound,e.rows(bound)[0]['request_id'],**args)
    assert bound==before


def test_runtime_complete_and_counts_cannot_stand_in_for_independent_full_model_audit(bound):
    args=captured_CPU_terminal(bound,success=True);args['audit']['status']=t.PARTIAL
    v=t.settle(bound,e.rows(bound)[0]['request_id'],**args)
    assert e.rows(v)[0]['scientific_endpoint_complete']is False
    assert v[e.package.FIELD]['status']=='MAIN_NATIVE_TERMINAL_INCOMPLETE'


def test_current_terminal_row_cannot_be_faked_after_original_once_settle(bound):
    args=captured_CPU_terminal(bound,success=True);v=t.settle(bound,e.rows(bound)[0]['request_id'],**args)
    v['PBS_submissions'][-1]['committed_optimizer_updates']=1024
    with pytest.raises(PermissionError):e.validate(v)


def test_legal_late_accounting_does_not_extend_native_deadline(bound):
    args=captured_CPU_terminal(bound,success=False);args['now_epoch']=e.rows(bound)[0]['deadline_epoch']+30
    v=t.settle(bound,e.rows(bound)[0]['request_id'],**args)
    assert e.rows(v)[0]['settled_epoch']==args['now_epoch']
    assert e.rows(v)[0]['deadline_epoch']==e.rows(bound)[0]['deadline_epoch']
    args['native']['finished_epoch']=e.rows(bound)[0]['deadline_epoch']+1
    args['native']['observed_epoch']=args['native']['finished_epoch']+1
    with pytest.raises(PermissionError):t.settle(bound,e.rows(bound)[0]['request_id'],**args)


@pytest.mark.parametrize('success',[True,False])
def test_all_fourteen_future_dimensions_keep_old_work_after_actual_cost_projection(bound,success):
    import base64,gzip,json,os
    from pathlib import Path
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    m=Path(os.environ['ARX5_FIRST512_MEASUREMENT_CAPTURE']).parent
    old=json.loads((m/'CORE_RETRY12_CONTINUATION_INDEPENDENT_DUAL_READBACK.private.json').read_bytes())['local']
    local=gzip.decompress(base64.b64decode(old['LOCAL_gzip_base64']))
    before=first512_forecast(e.raw(bound),local)
    v=t.settle(bound,e.rows(bound)[0]['request_id'],**captured_CPU_terminal(bound,success=success))
    after=first512_forecast(e.raw(v),local)
    assert len(before['dimensions'])==len(after['dimensions'])==14
    assert before['cap_changes_required']==after['cap_changes_required']=={}
    assert all(after['dimensions'][k]['cap']==before['dimensions'][k]['cap']for k in before['dimensions'])
    assert all(after['dimensions'][k]['projected']<=before['dimensions'][k]['projected']for k in before['dimensions'])
    assert v[e.package.FIELD]['remaining_resources']==bound[e.package.FIELD]['remaining_resources']
    assert after['existing_C_D_GEN_Evals_and_old_costs_retained']and after['matrix3_and_final20GPUh40GiB_retained']
