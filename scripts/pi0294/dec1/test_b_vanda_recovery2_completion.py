"""Future-handle CPU fixtures; none is a native job, audit or closure proof."""
import base64,hashlib,json
from copy import deepcopy
import pytest
from . import b_vanda_recovery2_completion as c
from .test_b_retry4096_completion import fixture as previous_fixture

@pytest.fixture(autouse=True)
def isolate_prior_grant_validators(monkeypatch):
    # Prior full grant/history validation has its own tests and captured-byte
    # checks. This small fixture isolates the new final handle/OS-exit rules.
    monkeypatch.setattr(c.recovery,'validate_global',lambda _:None)
    monkeypatch.setattr(c.recovery,'validate_adopted',lambda _h,_d:None)

def fixture():
    old,v,probe=previous_fixture();r=old['request']
    r.update(c.recovery.scope(),source_sha=c.SOURCE,run_uuid=c.RUN,training_lineage_key=c.recovery.LINEAGE,
        request_id='10000000-0000-4000-8000-000000000001',submission_nonce='10000000-0000-4000-8000-000000000002',
        run_manifest_sha256='c'*64)
    raw=(json.dumps(r,indent=2)+'\n').encode();sha=hashlib.sha256(raw).hexdigest();job='9900001.stdct-mgmt-02'
    row=dict(job_id=job,request_id=r['request_id'],submission_nonce=r['submission_nonce'],source_sha=c.SOURCE,
        request_sha256=sha,training_lineage_key=c.recovery.LINEAGE,start_native_step=2048,segment_stop_step=4096,
        status='SUBMITTED_PENDING_NATIVE',optimizer_update_limit=2048,query_limit=29)
    h=dict(PBS_submissions=[deepcopy(row)],optimizer_updates=9666,generation_queries=1214,model_loads=8,
        training_runs={c.recovery.LINEAGE:dict(last_native_step=2048)})
    e=dict(request=r,request_bytes_base64=base64.b64encode(raw).decode(),request_sha256=sha,job_id=job,
        registration=dict(root='/CPU_FIXTURE/'+c.RUN,manifest_sha256='c'*64,source_sha=c.SOURCE),
        bound_submission=row,bound_ledger=h)
    for d in (v['effects'],v['process_final'],v['wrapper_final'],v['native_proof']):
        if 'job_id' in d:d['job_id']=job
        if 'request_id' in d:d['request_id']=r['request_id']
        if 'request_sha256' in d:d['request_sha256']=sha
        if 'source_sha' in d:d['source_sha']=c.SOURCE
    v['process_final']['effects']=deepcopy(v['effects'])
    v['raw_native']['Variable_List']=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce'])
    process=dict(pid=123456,proc_start_ticks='654321',uid=22182,source_sha=c.SOURCE,maximum_qsub=1,agent_connection_required=False)
    launch=dict(pid=123455,source_sha=c.SOURCE,process_kind='ORIGINAL_FOREGROUND_EXIT_CAPTURE_WRAPPER')
    packet=dict(OS_exit=dict(schema='VANDA_ORIGINAL_FOREGROUND_OWNER_OS_EXIT_V1',returncode=0,
        foreground_wait_completed=True,source_sha=c.SOURCE,wrapper_pid=123455,owner_process=deepcopy(process)),
        owner_process=process,launch=launch,preserved_stop_exists=False,
        current_process=dict(pid=123456,present_same_instance=False,physical_platform='VANDA_A40',observed_epoch=100,proc_start_ticks=None))
    add_queries(v,e)
    return e,v,probe,packet

def encoded(v):return base64.b64encode(json.dumps(v).encode()).decode()

def add_queries(v,e):
    result=dict(status='PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS',job_id=e['job_id'],
        request_sha256=e['request_sha256'],parent_checkpoint_manifest_sha256=c.a40_allocation.PARENT,
        statistics_sha256=c.a40_allocation.R0,new_physics=0,new_tasks=0,new_optimizer_updates=0,records=[])
    ledger=dict(job_id=e['job_id'],request_sha256=e['request_sha256'],source_sha=c.SOURCE,run_uuid=c.RUN,
        generation_queries=17,model_loads=2,optimizer_updates=4096,query_intent_links=[])
    v.update(query_intent_bytes_base64={},compatibility_array_sha256={})
    for i in range(17):
        message='probe-A40-COMPATIBILITY-draw-'+str(i);path='reservations/probe-generations/'+message+'.json'
        meta=dict(arrays_sha256=hashlib.sha256(str(i).encode()).hexdigest(),inference_precision={'CPU_fixture':True})
        row=dict(run_uuid=c.RUN,message_id=message,payload_sha256='d'*64,status='COMPLETE',result=meta,result_sha256=c.digest(meta))
        ledger['query_intent_links'].append(dict(index=i,message_id=message,intent_relative=path,payload_sha256='d'*64))
        result['records'].append(dict(index=i,seed=29400+i,finite_shapes_verified=True,**meta))
        v['query_intent_bytes_base64'][path]=encoded(row)
        v['compatibility_array_sha256']['A40_COMPATIBILITY/query-'+str(i).zfill(2)+'.npz']=meta['arrays_sha256']
    v.update(native_effect_ledger_base64=encoded(ledger),compatibility_result_base64=encoded(result))

def test_actual_bound_future_request_and_kernel_owner_return():
    e,v,_,packet=fixture();c.expected(e);c.owner(packet,e,now_epoch=101);c.native(v['raw_native'],e);c.effects(v['effects'],e)

@pytest.mark.parametrize('fault',['old_job','old_source','old_lineage','manifest','request_bytes','nonce','bound_job','baseline'])
def test_old_request_or_changed_binding_never_becomes_new_completion(fault):
    e,_,_,_=fixture()
    if fault=='old_job':e['job_id']=c.recovery.JOB
    elif fault=='old_source':e['request']['source_sha']=c.recovery.SOURCE
    elif fault=='old_lineage':e['request']['training_lineage_key']=c.b.LINEAGE
    elif fault=='manifest':e['registration']['manifest_sha256']='0'*64
    elif fault=='request_bytes':e['request_bytes_base64']=base64.b64encode(b'{}').decode()
    elif fault=='nonce':e['bound_submission']['submission_nonce']='other'
    elif fault=='bound_job':e['bound_submission']['job_id']=c.recovery.JOB
    else:e['bound_ledger']['optimizer_updates']=8744
    if fault in ('old_source','old_lineage'):
        raw=json.dumps(e['request']).encode();e['request_bytes_base64']=base64.b64encode(raw).decode()
        e['request_sha256']=hashlib.sha256(raw).hexdigest()
        e['bound_submission']['request_sha256']=e['bound_ledger']['PBS_submissions'][0]['request_sha256']=e['request_sha256']
    with pytest.raises(PermissionError):c.expected(e)

@pytest.mark.parametrize('fault',['nonzero','missing_wait','live','stale','unknown','different_PID','source','stop','PID_reuse_same_start','bool_exit'])
def test_real_owner_exit_is_not_inferred_from_a_file_or_observation_timeout(fault):
    e,_,_,p=fixture()
    if fault=='nonzero':p['OS_exit']['returncode']=7
    elif fault=='missing_wait':p['OS_exit']['foreground_wait_completed']=False
    elif fault=='live':p['current_process']['present_same_instance']=True
    elif fault=='stale':p['current_process']['observed_epoch']=-30
    elif fault=='unknown':p['current_process']['present_same_instance']=None
    elif fault=='different_PID':p['current_process']['pid']+=1
    elif fault=='source':p['owner_process']['source_sha']=c.recovery.SOURCE
    elif fault=='stop':p['preserved_stop_exists']=True
    elif fault=='bool_exit':p['OS_exit']['returncode']=False
    else:p['current_process']['proc_start_ticks']=p['owner_process']['proc_start_ticks']
    with pytest.raises(PermissionError):c.owner(p,e,now_epoch=101)

def test_actual_query_and_load_counts_are_charged_within_the_existing_bound(monkeypatch):
    e,v,probe,_=fixture();monkeypatch.setattr(c.b,'PROBE_SHA',probe)
    assert c.audit(v,e) is v
    for queries,loads in ((17,2),(18,3),(29,4)):
        z=deepcopy(v);z['effects'].update(query_attempts=queries,model_load_attempts=loads);z['process_final']['effects']=deepcopy(z['effects'])
        c.effects(z['effects'],e)  # Financial bounds cannot relabel extra draws as scientific completion.
        if queries!=17:
            with pytest.raises(PermissionError):c.audit(z,e)
    for name,value in [('query_attempts',16),('query_attempts',30),('query_attempts',True),('model_load_attempts',0),
            ('model_load_attempts',5),('optimizer_updates_committed',1024),('pending_updates',[{}])]:
        z=deepcopy(v);z['effects'][name]=value
        with pytest.raises(PermissionError):c.audit(z,e)

@pytest.mark.parametrize('fault',['other_job','other_request','old_R0','missing_array','changed_array',
    'duplicate_draw','replayed_probe','unknown_intent','changed_result','changed_seed','nonfinite_shape'])
def test_complete_actual_query_census_rejects_replay_and_mismatched_arrays(fault):
    e,v,_,_=fixture();result=json.loads(base64.b64decode(v['compatibility_result_base64']))
    ledger=json.loads(base64.b64decode(v['native_effect_ledger_base64']));path=ledger['query_intent_links'][0]['intent_relative']
    row=json.loads(base64.b64decode(v['query_intent_bytes_base64'][path]))
    if fault=='other_job':result['job_id']=c.recovery.JOB
    elif fault=='other_request':result['request_sha256']='0'*64
    elif fault=='old_R0':result['statistics_sha256']='0'*64
    elif fault=='missing_array':v['compatibility_array_sha256'].pop('A40_COMPATIBILITY/query-00.npz')
    elif fault=='changed_array':v['compatibility_array_sha256']['A40_COMPATIBILITY/query-00.npz']='0'*64
    elif fault=='duplicate_draw':ledger['query_intent_links'][1]=deepcopy(ledger['query_intent_links'][0])
    elif fault=='replayed_probe':row['message_id']='probe-B_COVERAGE-1024-original'
    elif fault=='unknown_intent':row['status']='INTENT'
    elif fault=='changed_result':row['result']['arrays_sha256']='0'*64
    elif fault=='changed_seed':result['records'][0]['seed']+=1
    else:result['records'][0]['finite_shapes_verified']=False
    v.update(native_effect_ledger_base64=encoded(ledger),compatibility_result_base64=encoded(result))
    v['query_intent_bytes_base64'][path]=encoded(row)
    with pytest.raises(PermissionError):c.query_provenance(v,e)

@pytest.mark.parametrize('fault',['optimizer','RNG','sampler','processor','model','cursor','original_probe','wrapper','bool_native'])
def test_complete_saved_states_and_original_probe_bytes_are_still_required(monkeypatch,fault):
    e,v,probe,_=fixture();monkeypatch.setattr(c.b,'PROBE_SHA',probe)
    if fault=='optimizer':v['checkpoint_members']['4096'].pop('optimizer.pt')
    elif fault=='RNG':v['checkpoint_members']['2048'].pop('rng_state.pth')
    elif fault=='sampler':v['checkpoint_members']['1024'].pop('SAMPLER_STATE.json')
    elif fault=='processor':v['checkpoint_members']['4096'].pop('processor/tokenizer.json')
    elif fault=='model':v['checkpoint_members']['4096']['model-00002-of-00002.safetensors']='invalid'
    elif fault=='cursor':v['checkpoints']['4096']['actual_consumed_cursor']=65536
    elif fault=='original_probe':v['original_probe_result_base64']=base64.b64encode(b'changed').decode()
    elif fault=='wrapper':v['wrapper_final']['returncode']=1
    else:v['raw_native']['Exit_status']=False
    with pytest.raises(PermissionError):c.audit(v,e)

def settlement_fixture():
    e,v,_,_=fixture();h=deepcopy(e['bound_ledger']);row=h['PBS_submissions'][0]
    row.update(status='NATIVE_TERMINAL',native_exit_status=0,committed_optimizer_updates=2048,query_attempts=17,
        reserved_optimizer_updates=0,reserved_neural_queries=0)
    h.update(optimizer_updates=11714,generation_queries=1231,model_loads=10,unknown_effects=[],pending_updates=[],
        reserved={},CPU_reserved_s=0,storage_reserved_bytes=0)
    h['training_runs']={c.recovery.LINEAGE:dict(last_native_step=4096),c.b.LINEAGE:dict(last_native_step=2970),
        'B_COVERAGE':dict(last_native_step=1829),c.b.zero.HR.LINEAGE:dict(last_native_step=4096)}
    d=dict(optimizer_updates=11714,unknown_effects=[])
    settled=dict(job_id=e['job_id'],request_id=e['request']['request_id'],ledger=h,DEC1=d,updates=11714,queries=1231,unknown_effects=[])
    return e,v['effects'],settled

def test_exact_cumulative_costs_and_both_paid_failed_lineages_survive():
    e,effects,s=settlement_fixture();assert c.settled(s,e,effects)['query_attempts']==17
    for fault in ('old_total','old2970','old1829','reserved','query_reservation_as_actual','model_load_refund','unknown'):
        z=deepcopy(s)
        if fault=='old_total':z['updates']=10792
        elif fault=='old2970':z['ledger']['training_runs'][c.b.LINEAGE]['last_native_step']=2048
        elif fault=='old1829':z['ledger']['training_runs']['B_COVERAGE']['last_native_step']=1024
        elif fault=='reserved':z['ledger']['reserved']['H200_s']=1
        elif fault=='query_reservation_as_actual':z['queries']=1243
        elif fault=='model_load_refund':z['ledger']['model_loads']=8
        else:z['DEC1']['unknown_effects']=['unknown']
        with pytest.raises(PermissionError):c.settled(z,e,effects)
