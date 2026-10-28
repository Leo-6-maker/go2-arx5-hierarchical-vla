"""Actual closed GLOBAL/prefix bytes; every NEW request/native/audit is fixture.

No register/bind/settle proposal is persisted. basic() is an identity layer,
not a successful complete physical/native admission or full-budget forecast.
"""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_minimal_remaining_contract import actual_prefix
from scripts.pi0294.generalization import minimal_remaining_execution as r,minimal_remaining_contract as c
from scripts.pi0294.generalization import minimal_execution as e,minimal_terminal as t,minimal_remaining_effects as effects
from scripts.pi0294.generalization import minimal_remaining_concurrency as concurrency,minimal_remaining_dispatch as dispatch


@pytest.fixture(scope='module')
def packet(actual_prefix):
    folder=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not folder:pytest.skip('actual same64 closed canonical capture required')
    capture=json.loads((Path(folder)/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())
    g=json.loads(gzip.decompress(base64.b64decode(capture['GLOBAL_gzip_base64'])))
    old=next(x for x in g['PBS_submissions']if x['request_id']==r.authority.REQUEST)
    previous=g[r.retry.ENTRY_FIELD];now=1793219000.
    physical=dict(schema=r.PHYSICAL_SCHEMA,prior_physical_custody_sha256=old['physical_custody_sha256'],
        old_metadata_roots=deepcopy(previous['physical_custody']['old_roots']),
        old_metadata_portable_sha256=previous['physical_custody']['portable_inventory']['portable_sha256'],
        old_metadata_identity_sha256=previous['terminal']['physical']['portable_identity_sha256'])
    request=dict(r.fixed_request_fields(),request_id='00000000-0000-4000-8000-000000000001',
        submission_nonce='00000000-0000-4000-8000-000000000002',source_sha='a'*40,
        run_manifest_sha256='b'*64,cold_input_audit_sha256='c'*64,raw_parent_audit_plan_sha256='d'*64,
        frozen_input_packet_sha256='e'*64,Core_measurement_capture_sha256=g[e.FIELD]['Core_measurement_proof']['capture_sha256'],
        physical_custody_sha256=e.sha(e.raw(physical)),effective_queue_proof_sha256='f'*64,
        created_epoch=now-1,submit_count=39,**{k:g[e.authority.FIELD][k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    r.request(request);row=dict(request,request_sha256=e.sha(e.raw(request)),status=e.STATES[1],
        job_id='123456.stdct-mgmt-02',submitted_epoch=now,native_reply_epoch=now+1)
    before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads')}
    before.update(PBS_submissions=38,H200_s=g['used']['H200_s'])
    g['PBS_submissions'].append(row);g['used']['PBS_submissions']+=1
    g[r.FIELD]=dict(schema=r.SCHEMA,version=r.VERSION,request_id=row['request_id'],request_sha256=row['request_sha256'],
        request_base64=base64.b64encode(e.raw(request)).decode(),source_sha=row['source_sha'],run_manifest_sha256=row['run_manifest_sha256'],
        prefix_result_base64=base64.b64encode(actual_prefix[0]).decode(),prefix_closure_base64=base64.b64encode(actual_prefix[1]).decode(),
        prefix_binding=c.saved_prefix(*actual_prefix),completion_scope_grant_sha256=r.authority.GRANT_SHA,
        completion_authority_sha256=e.sha(e.raw(g[r.authority.FIELD])),original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),
        native_hold=r.native_resources(),control_paid_here=e._vector({}),old_paid_gross_and_history_unchanged=True,
        before=before,registered_epoch=now,status='REGISTERED',completed_sessions=[],
        physical_custody=physical,physical_custody_sha256=request['physical_custody_sha256'])
    return g,request,row


def test_actual_closed_prefix_identity_does_not_imply_physical_or_native_admission(packet):
    g,request,row=packet;m,selected=r.basic(g)
    assert selected==row and m['prefix_binding']['old_paid_effects']==c.OLD_PAID
    assert r.process_CPU_limits(request)['prior_readonly_preparation_CPU_upper_s']==0
    assert r.native_resources()['Hopper_storage_bytes']==0 and request['paid_output_allowance_bytes']==144294967296
    assert len(concurrency.cold_Core_rows(g))==1
    # The deliberately incomplete future physical fixture must not become GO.
    with pytest.raises((KeyError,PermissionError)):r.validate(g)


@pytest.mark.parametrize('key,value',[('optimizer_update_limit',1024),('query_limit',147),('model_load_limit',11),
    ('profile',e.PROFILE),('prior_readonly_preparation_CPU_upper_s',900),('additional_output_upper_bytes',1)])
def test_new_request_cannot_alias_old_full_replay_or_add_output_or_old_CPU_again(packet,key,value):
    request=deepcopy(packet[1]);request[key]=value
    with pytest.raises(PermissionError):r.request(request)


def terminal_packet(packet):
    g,_,row=packet;common={k:row[k]for k in('request_id','request_sha256','source_sha','job_id')}
    sessions={s['id']:dict(status='COMPLETE',optimizer_updates=s['own_optimizer_limit'],generation_queries=s['own_query_limit'],
        model_loads=s['own_model_load_limit'],terminal_receipt_sha256='a'*64)for s in c.sessions()}
    ef=dict(schema=effects.SUMMARY,**common,pending_updates=[],unknown_query_intents=[],global_budget_is_only_external_reservation=True,
        new_physics=0,new_episodes=0,optimizer_updates_committed=1026,query_attempts=76,model_load_attempts=7,
        sessions=sessions,active_session=None,all_remaining_sessions_runtime_complete=True,retained_old_native=c.OLD_PAID,
        audited_prefix_result_sha256=c.RESULT,audited_prefix_closure_sha256=c.CLOSED,
        carried_saved_E2_zero_queries=15,carried_queries_added_to_new_links=0)
    native=dict(schema=t.NATIVE,**common,job_state='F',submission_nonce=row['submission_nonce'],Exit_status=0,
        all_native_workers_terminal=True,terminal_inventory_stable=True,raw_native_sha256='a'*64,
        process_final_sha256='b'*64,wrapper_final_sha256='c'*64,inventory_sha256='d'*64,
        conservative_GPU_s=241,conservative_CPU_s=100,new_native_conservative_CPU_s=100,
        prior_readonly_preparation_CPU_upper_s=0,finished_epoch=row['submitted_epoch']+100,observed_epoch=row['submitted_epoch']+101)
    audit=dict(status=t.REMAINING_AUDIT,**common,saved_prefix_result_sha256=c.RESULT,saved_prefix_closure_sha256=c.CLOSED,
        effect_summary_sha256=e.sha(e.raw(ef)),inventory_sha256=native['inventory_sha256'],raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        statistics_sha256=e.R0,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0,
        audited_session_ids=list(sessions),same15_zero_step_raw_wrapped_arms_exact=True,
        two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact=True,official_full_model_reconstruction_output_pairs_exact=True,
        frozen_parameters_and_buffers_unchanged=True,endpoints={arm:dict(global_step=512,cursor=16384,scheduler_horizon=1024,
            raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,full_resume=True,qualification_only=False,
            TRAIN_probes=12,language_probes=24,checkpoint_manifest_sha256='e'*64,official_inference_manifest_sha256='f'*64,
            full_streamed_members_sha256='a'*64)for arm in e.training.ARMS})
    return native,ef,audit


@pytest.mark.parametrize('change',[None,'old_schema','old_queries','missing_qualification','missing_load','old_CPU',
    'unknown','pending','not_F','still_live','wrong_prefix','wrong_horizon','not_rebuilt'])
def test_actual_prefix_plus_future_terminal_shape_preserves_all_original_fences(packet,change):
    g,_,row=packet;n,x,a=terminal_packet(packet)
    if change=='old_schema':x['schema']='ARX5_MINIMAL_ACTUAL_FIVE_SESSION_EFFECT_SUMMARY_V1'
    elif change=='old_queries':x['query_attempts']+=15
    elif change=='missing_qualification':x['optimizer_updates_committed']=1024
    elif change=='missing_load':
        x['model_load_attempts']=6;x['sessions'][c.sessions()[0]['id']]['model_loads']=2
    elif change=='old_CPU':n['prior_readonly_preparation_CPU_upper_s']=900;n['conservative_CPU_s']+=900
    elif change=='unknown':x['unknown_query_intents']=[0]
    elif change=='pending':x['pending_updates']=[{'keep':'UNKNOWN'}]
    elif change=='not_F':n['job_state']='R'
    elif change=='still_live':n['all_native_workers_terminal']=False
    elif change=='wrong_prefix':a['saved_prefix_result_sha256']='0'*64
    elif change=='wrong_horizon':a['endpoints'][e.training.ARMS[0]]['scheduler_horizon']=512
    elif change=='not_rebuilt':a['official_full_model_reconstruction_output_pairs_exact']=False
    a['effect_summary_sha256']=e.sha(e.raw(x))
    if change:
        with pytest.raises(PermissionError):t.terminal_values(row,n,x,a,remaining_global=g)
    else:
        amounts=t.terminal_values(row,n,x,a,remaining_global=g)
        assert amounts['success']and(amounts['updates'],amounts['queries'],amounts['loads'])==(1026,76,7)
