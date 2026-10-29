"""Captured full history plus explicitly hypothetical future GPU/process inputs."""
import base64,gzip,json,os,time
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_saved512_execution as x,minimal_saved512_contract as c
from scripts.pi0294.generalization import minimal_saved512_physical as physical,minimal_execution_adapter as adapters
from scripts.pi0294.dec1 import vanda_campaign as queue


def test_complete_real_history_with_new_fixture_request_preserves_all14(tmp_path):
    where=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not where:pytest.skip('actual815f canonical capture required')
    m=Path(where);z=json.loads((m/'SAVED512_RECOVERY_CONTROL01_NATIVE_INDEPENDENT_AFTER_EXACT_STALE_ZERO.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(z['GLOBAL_gzip_base64']));g=json.loads(gb);assert e.sha(gb)==z['GLOBAL_sha256']
    local=gzip.decompress(base64.b64decode(json.loads((m/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())['LOCAL_gzip_base64']))
    actual=json.loads(sorted(m.glob('SAVED512_RECOVERY_FINITE_AUDIT03_STATUS*'))[-1].read_bytes())
    a=e.raw(actual['records']['RESULT.json']['value']);b=json.dumps(actual['records']['OS_EXIT.json']['value']).encode();binding=c.binding(a,b)
    now=time.time();p=dict(schema=x.PHYSICAL_SCHEMA,saved_binding=binding,request_id='00000000-0000-4000-8000-000000000051',
        submission_nonce='00000000-0000-4000-8000-000000000052',run_uuid=c.RUN,paid_output_ceiling=x.PAID_OUTPUT,
        new_output_bound_bytes=70000000000,additional_output_charge_bytes=0,old_closed_lease_sha256=c.scope.LEASE,
        old_science_root='/CPU_FIXTURE/arx5-minimal-raw-parent-first512-remainder-02',old_snapshot_sha256='a'*64,
        old_snapshot_identity_sha256='b'*64,prefix_initial_allocated_bytes=13945905152,old_metadata_ceiling=268435456,
        old_science_initial_allocated_bytes=51945381888,retained_cold_roots=[dict(run_uuid='arx5-minimal-saved-e1-inference-e2-fresh512-01',source_sha='ade4dbe39e3def8bec9be6c9608a29105552bbb4',manifest_sha256='0299c9e9ea1ec9fcbb6b83295113710881dd38026ed3206e9f8170b689e298ba',closed_lease_sha256='c'*64,snapshot_identity_sha256='d'*64,allocated_bytes=15000000)])
    r=dict(x.fixed_request_fields(),request_id=p['request_id'],submission_nonce=p['submission_nonce'],source_sha='a'*40,
        run_manifest_sha256='b'*64,cold_input_audit_sha256='c'*64,raw_parent_audit_plan_sha256='d'*64,
        frozen_input_packet_sha256='e'*64,Core_measurement_capture_sha256=g[e.FIELD]['Core_measurement_proof']['capture_sha256'],
        physical_custody_sha256=e.sha(e.raw(p)),effective_queue_proof_sha256='f'*64,created_epoch=now-1,submit_count=40,
        **{k:g[e.authority.FIELD][k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    x.request(r);native=dict(select='1:ngpus=1:ncpus=36:mem=240gb',ngpus=1,ncpus=36,mem='240gb',walltime='08:00:00')
    assert x.select_profile(native,r)==(36,240*1024**3)
    with pytest.raises(PermissionError):x.select_profile(dict(native,walltime='16:30:00'),r)
    proof=dict(status='ACTUAL_SAVED512_ALL_RETAINED_ROOTS_AND_NEW_OUTPUT_BOUNDED_NO_PERMISSION',observed_epoch=now,
        custody_sha256=r['physical_custody_sha256'],old_snapshot_identity_sha256=p['old_snapshot_identity_sha256'],
        prefix_allocated_bytes=13945905152,old_metadata_ceiling=268435456,old_science_allocated_bytes=51945381888,
        new_allocated_bytes=10000000,retained_cold_allocated_bytes=15000000,next_output_upper_bytes=1000000,ledger_changed=False,native_permission=False)
    full=queue.collect_queue(runner=lambda *args,**kw:SimpleNamespace(returncode=0,stdout='',stderr=''),clock=lambda:now)
    fresh=dict(complete_queue=full,observed_epoch=now,GLOBAL_sha256=e.sha(gb),complete_queue_verified=True,
        old_workers_terminal=True,old_closed_science_lease_sha256=c.scope.LEASE,independent_confirmation_base64='')
    result=x.register(gb,local,request_bytes=e.raw(r),result_bytes=a,exit_bytes=b,physical_custody=p,
        physical_proof=proof,physical_observed_epoch=now,fresh_readback=fresh,now_epoch=now)
    v=result['proposed_global'];budget=result['full_future_proof']
    assert budget['cap_changes_required']=={} and len(budget['dimensions'])==14
    assert v['PBS_submissions'][:-1]==g['PBS_submissions']and v[e.package.FIELD]==g[e.package.FIELD]
    assert v['used']['PBS_submissions']==40 and budget['dimensions']['PBS_submissions']['projected']==43
    assert v['optimizer_updates']==g['optimizer_updates']and v['generation_queries']==g['generation_queries']
    assert v['CPU_s']==g['CPU_s']and v['new_storage_current_bytes']==g['new_storage_current_bytes']
    assert v['reserved']['optimizer_updates']-g['reserved']['optimizer_updates']==512
    assert v['reserved']['neural_queries']-g['reserved']['neural_queries']==62
    assert c.scope.planned_unregistered(v)==e._vector({})and result['native_admission']is False
    assert adapters.current(v)[0]is x
    from scripts.pi0294.generalization.minimal_saved512_predispatch import validate
    validate(v,full,r,e.sha(e.raw(r)),now_epoch=now)
    bound=x.bind(v,r['request_id'],'123456.stdct-mgmt-02',now_epoch=now+1)
    with pytest.raises(PermissionError):x.bind(bound,r['request_id'],'123457.stdct-mgmt-02',now_epoch=now+2)
    for field,value in [('optimizer_update_limit',1024),('model_load_limit',7),('profile',adapters.remaining.PROFILE)]:
        bad=dict(r);bad[field]=value
        with pytest.raises(PermissionError):x.request(bad)
    badproof=dict(proof,next_output_upper_bytes=70000000000)
    with pytest.raises(PermissionError):physical.check_proof(badproof,p,r,now_epoch=now,observed_epoch=now)
    # Hypothetical native F: preserved partial E1 inference, no new training.
    from scripts.pi0294.generalization import minimal_terminal as terminal,minimal_saved512_effects as effects
    row=x.basic(bound)[1];common={k:row[k]for k in('request_id','request_sha256','source_sha','job_id')}
    sessions={s['id']:dict(status='NOT_STARTED',optimizer_updates=0,generation_queries=0,model_loads=0)for s in c.sessions()}
    sessions[c.sessions()[0]['id']].update(status='ACTIVE',model_loads=2,generation_queries=3)
    ef=dict(schema=effects.SUMMARY,**common,pending_updates=[],unknown_query_intents=[],global_budget_is_only_external_reservation=True,
        new_physics=0,new_episodes=0,optimizer_updates_committed=0,query_attempts=3,model_load_attempts=2,
        sessions=sessions,active_session=c.sessions()[0]['id'],all_saved512_sessions_runtime_complete=False,
        audited_saved512=binding,retained_old_native=c.OLD_PAID,E1_new_optimizer_calls=0,carried_queries_added_to_new_links=0)
    native=dict(schema=terminal.NATIVE,**common,job_state='F',submission_nonce=r['submission_nonce'],Exit_status=1,
        all_native_workers_terminal=True,terminal_inventory_stable=True,raw_native_sha256='a'*64,process_final_sha256='b'*64,
        wrapper_final_sha256='c'*64,inventory_sha256='d'*64,conservative_GPU_s=241,conservative_CPU_s=100,
        new_native_conservative_CPU_s=100,prior_readonly_preparation_CPU_upper_s=0,finished_epoch=now+10,observed_epoch=now+11)
    audited=dict(status=terminal.PARTIAL,**common,saved512_result_sha256=c.RESULT,saved512_OS_exit_sha256=c.OS_EXIT,E1_new_optimizer_calls=0,
        effect_summary_sha256=e.sha(e.raw(ef)),inventory_sha256=native['inventory_sha256'],raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        statistics_sha256=e.R0,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0)
    amounts=terminal.terminal_values(row,native,ef,audited,saved512_global=bound)
    assert not amounts['success']and(amounts['updates'],amounts['queries'],amounts['loads'])==(0,3,2)
    for key,value in [('E1_new_optimizer_calls',1),('pending_updates',[{'keep':'UNKNOWN'}])]:
        bad=deepcopy(ef);bad[key]=value;bad_a=dict(audited,effect_summary_sha256=e.sha(e.raw(bad)))
        with pytest.raises(PermissionError):terminal.terminal_values(row,native,bad,bad_a,saved512_global=bound)
    final=x.settle(bound,r['request_id'],native=native,effects=ef,audit=audited,physical_proof=proof,
        physical_observed_epoch=now,now_epoch=now+12)
    assert x.basic(final)[1]['scientific_endpoint_complete']is False
    assert final['used']['H200_s']==g['used']['H200_s']+241 and final['optimizer_updates']==g['optimizer_updates']
    assert final['generation_queries']==g['generation_queries']+3 and final['model_loads']==g['model_loads']+2
    assert final['new_storage_current_bytes']==g['new_storage_current_bytes']and final['new_storage_peak_bytes']==g['new_storage_peak_bytes']
    assert final['PBS_submissions'][:39]==g['PBS_submissions']and final[e.package.FIELD]==g[e.package.FIELD]
    with pytest.raises(PermissionError):x.settle(final,r['request_id'],native=native,effects=ef,audit=audited,physical_proof=proof,
        physical_observed_epoch=now,now_epoch=now+13)
