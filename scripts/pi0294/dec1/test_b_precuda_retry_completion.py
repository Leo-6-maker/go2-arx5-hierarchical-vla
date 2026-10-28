"""CPU fixtures only; the live attempt is never made terminal by these tests."""
import base64,hashlib,json
from copy import deepcopy
import pytest
from . import b_precuda_retry_completion as c
from . import b_recovery2_completion_contract as core
from .test_b_vanda_recovery2_completion import fixture as previous_fixture,encoded

@pytest.fixture
def packet(monkeypatch):
    # Isolate the new identity/exit contract; actual complete history is checked
    # separately using private captured full native request and bound-ledger bytes.
    monkeypatch.setattr(c.authority,'validate_global',lambda _:None)
    monkeypatch.setattr(c.authority,'validate_adopted',lambda _h,_d:None)
    e,v,probe,owner=previous_fixture();r=e['request']
    old_job=e['job_id'];old_request=r['request_id'];old_sha=e['request_sha256']
    r.update(c.authority.scope(),source_sha=c.SOURCE,run_uuid=c.RUN,request_id=c.REQUEST,
        submission_nonce=c.NONCE,run_manifest_sha256=c.GPU_MANIFEST)
    r.setdefault('sealed_inputs',{})[c.authority.PLAN]=c.authority.GRANT_SHA
    raw=(json.dumps(r,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(c,'REQUEST_SHA',digest)  # Explicit synthetic request fixture, not actual native bytes.
    e.update(job_id=c.JOB,request_bytes_base64=base64.b64encode(raw).decode(),request_sha256=digest,
        control_manifest_sha256=c.CPU_MANIFEST,registration=dict(root='/CPU_FIXTURE/'+c.RUN,
            manifest_sha256=c.GPU_MANIFEST,source_sha=c.SOURCE))
    row=e['bound_submission'];row.update(job_id=c.JOB,request_id=c.REQUEST,submission_nonce=c.NONCE,
        source_sha=c.SOURCE,request_sha256=digest)
    e['bound_ledger']['PBS_submissions']=[deepcopy(row)]
    def walk(value):
        if isinstance(value,dict):
            return {k:(c.JOB if k=='job_id' and x==old_job else c.REQUEST if k=='request_id' and x==old_request
                else digest if k=='request_sha256' and x==old_sha else c.SOURCE if k=='source_sha' else walk(x)) for k,x in value.items()}
        if isinstance(value,list):return [walk(x) for x in value]
        return value
    v=walk(v);v['process_final']['effects']=deepcopy(v['effects'])
    v['raw_native']['Variable_List']=dict(PV1_REQUEST_ID=c.REQUEST,PV1_SUBMISSION_NONCE=c.NONCE)
    ledger=json.loads(base64.b64decode(v['native_effect_ledger_base64']))
    ledger.update(job_id=c.JOB,request_sha256=digest,source_sha=c.SOURCE,run_uuid=c.RUN)
    result=json.loads(base64.b64decode(v['compatibility_result_base64']));result.update(job_id=c.JOB,request_sha256=digest)
    v.update(native_effect_ledger_base64=encoded(ledger),compatibility_result_base64=encoded(result))
    for path,raw in v['query_intent_bytes_base64'].items():
        z=json.loads(base64.b64decode(raw));z['run_uuid']=c.RUN;v['query_intent_bytes_base64'][path]=encoded(z)
    owner=walk(owner);owner['owner_process'].update(pid=c.OWNER_PID,proc_start_ticks=c.OWNER_START)
    owner['launch'].update(pid=c.WRAPPER_PID,proc_start_ticks=c.WRAPPER_START)
    owner['OS_exit'].update(owner_process=deepcopy(owner['owner_process']),wrapper_pid=c.WRAPPER_PID)
    owner['current_process'].update(pid=c.OWNER_PID)
    return e,v,probe,owner

def test_exact_new_attempt_full_shared_audit_and_original_owner_exit(packet,monkeypatch):
    e,v,probe,owner=packet;monkeypatch.setattr(core.b,'PROBE_SHA',probe)
    c.expected(e);c.owner(owner,e,now_epoch=101);assert c.audit(v,e) is v

@pytest.mark.parametrize('fault',['old_job','old_request','request_bytes','nonce','GPU_manifest','CPU_manifest','grant','scope','lineage'])
def test_old_or_partial_binding_never_becomes_current_completion(packet,fault):
    e,_,_,_=packet
    if fault=='old_job':e['job_id']=c.authority.JOB
    elif fault=='old_request':e['request']['request_id']=c.authority.REQUEST
    elif fault=='request_bytes':e['request_bytes_base64']=encoded({})
    elif fault=='nonce':e['bound_submission']['submission_nonce']='other'
    elif fault=='GPU_manifest':e['registration']['manifest_sha256']='a'*64
    elif fault=='CPU_manifest':e['control_manifest_sha256']='a'*64
    elif fault=='grant':e['request']['sealed_inputs'][c.authority.PLAN]='a'*64
    elif fault=='scope':e['request'].pop(c.authority.SCOPE_KEYS[1])
    else:e['request']['training_lineage_key']=core.b.LINEAGE
    with pytest.raises(PermissionError):c.expected(e)

@pytest.mark.parametrize('fault',['owner','owner_start','wrapper','wrapper_start','exit1','live','unknown','stale','STOP','missing_wait'])
def test_live_failed_different_or_unobserved_owner_cannot_pass(packet,fault):
    e,_,_,p=packet
    if fault=='owner':p['owner_process']['pid']+=1
    elif fault=='owner_start':p['owner_process']['proc_start_ticks']='1'
    elif fault=='wrapper':p['launch']['pid']+=1
    elif fault=='wrapper_start':p['launch']['proc_start_ticks']='1'
    elif fault=='exit1':p['OS_exit']['returncode']=1
    elif fault=='live':p['current_process']['present_same_instance']=True
    elif fault=='unknown':p['current_process']['present_same_instance']=None
    elif fault=='stale':p['current_process']['observed_epoch']=-30
    elif fault=='STOP':p['preserved_stop_exists']=True
    else:p['OS_exit']['foreground_wait_completed']=False
    with pytest.raises(PermissionError):c.owner(p,e,now_epoch=101)

@pytest.mark.parametrize('fault',['partial','optimizer','scheduler','RNG','cursor','probe','query29','extra_query','query_array','old_array','native_exit1'])
def test_all_saved_state_query_and_native_terminal_gates_survive_shared_extraction(packet,monkeypatch,fault):
    e,v,probe,_=packet;monkeypatch.setattr(core.b,'PROBE_SHA',probe)
    if fault=='partial':v['effects']['last_native_step']=2970
    elif fault in ('optimizer','scheduler','RNG'):v['checkpoint_members']['4096'].pop(dict(optimizer='optimizer.pt',scheduler='scheduler.pt',RNG='rng_state.pth')[fault])
    elif fault=='cursor':v['checkpoints']['4096']['actual_consumed_cursor']=65536
    elif fault=='probe':v['original_probe_result_base64']=encoded({})
    elif fault=='query29':v['effects']['query_attempts']=29
    elif fault=='extra_query':v['query_intent_bytes_base64']['extra.json']=encoded({})
    elif fault=='query_array':v['compatibility_array_sha256']['A40_COMPATIBILITY/query-00.npz']='a'*64
    elif fault=='old_array':
        x=json.loads(base64.b64decode(v['compatibility_result_base64']));x['job_id']=c.authority.JOB;v['compatibility_result_base64']=encoded(x)
    else:v['raw_native']['Exit_status']=1
    with pytest.raises(PermissionError):c.audit(v,e)
