"""Pure proposals and temporary files; never native qsub, loads or settlement."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import evaluation_accounting as a
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.pv1.evaluation_ledger import EvaluationLedger,for_request
from scripts.pi0294.pv1.neural_ledger import check,Ledger
from scripts.pi0294.test_pv1_neural_ledger import value
from scripts.pi0294.test_pv1_a40_evaluation_accounting import row

NOW=e.T0+100010


def state(bound=True):
    v=value();r=row()
    v.update(campaign_run_uuid='20261018T182956Z-localpc-pv1',CPU_reserved_s=80000,
        storage_reserved_bytes=r['additional_output_upper_bytes'],new_storage_current_bytes=0,new_storage_peak_bytes=0)
    if not bound:r.update(job_id=None,status=a.STATES[0])
    v['PBS_submissions']=[r];v['used']['PBS_submissions']=1
    v['reserved'].update(H200_s=28860,neural_queries=r['query_limit'])
    check(v);return v


def debit(v,kind,count,identifier):
    r=v['PBS_submissions'][-1]
    return a.charge(v,request_id=r['request_id'],job_id=r['job_id'],kind=kind,count=count,effect_id=identifier)


def final_inputs(v):
    r=v['PBS_submissions'][-1]
    n=dict(schema='PV1_NATIVE_A40_DEV_TERMINAL_V1',job_id=r['job_id'],job_state='F',project=r['project'],
        job_name=r['job_name'],owner='e1192548',ngpus=1,request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,terminal_inventory_stable=True,Exit_status=1,observed_epoch=NOW,
        conservative_GPU_s=200,conservative_CPU_s=20,new_resident_allocated_upper_bytes=1024)
    f=dict(schema='PV1_A40_NATIVE_DEV_EFFECT_SUMMARY_V1',job_id=r['job_id'],request_id=r['request_id'],
        request_sha256=r['request_sha256'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],
        evaluation_candidate=r['evaluation_candidate'],checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],
        statistics_sha256=e.R0,optimizer_updates_committed=0,new_physics=0,new_episodes=0,
        effect_journal_complete=True,query_attempts=3,model_load_attempts=1,unknown_query_intents=[])
    return n,f


def settled(v,n=None,f=None,now=NOW+1):
    native,effects=final_inputs(v)
    return a.settle(v,request_id=v['PBS_submissions'][-1]['request_id'],native_proof=native if n is None else n,
        effects=effects if f is None else f,now_epoch=now)


def test_one_known_reply_binds_without_resubmission_and_never_rebinds():
    v=state(False);old=deepcopy(v);r=v['PBS_submissions'][-1]
    b=a.bind_job(v,request_id=r['request_id'],job_id='123.stdct-mgmt-02',now_epoch=NOW)
    assert v==old and b['PBS_submissions'][-1]['job_id']=='123.stdct-mgmt-02'
    assert b['used']==v['used'] and b['reserved']==v['reserved']
    with pytest.raises(PermissionError):a.bind_job(b,request_id=r['request_id'],job_id='123.stdct-mgmt-02',now_epoch=NOW)
    with pytest.raises(PermissionError):a.bind_job(v,request_id=r['request_id'],job_id='123.hopper-m-02',now_epoch=NOW)


def test_debits_and_terminal_deltas_charge_every_cost_once_and_preserve_input():
    original=state();v=debit(original,'model_loads',1,'actor/load')
    v=debit(v,'generation_queries',1,'actor/query0');v=debit(v,'generation_queries',1,'actor/query1')
    v=debit(v,'CPU_s',7,'actor/cpu');before=deepcopy(v);done=settled(v);r=done['PBS_submissions'][-1]
    assert v==before and original['generation_queries']==0
    assert done['generation_queries']==done['used']['neural_queries']==3
    assert done['model_loads']==1 and done['CPU_s']==20 and done['used']['H200_s']==200
    assert done['optimizer_updates']==done['used']['optimizer_updates']==0
    assert done['used']['PBS_submissions']==1 and done['reserved']['H200_s']==0
    assert done['reserved']['neural_queries']==done['CPU_reserved_s']==done['storage_reserved_bytes']==0
    assert done['new_storage_current_bytes']==done['new_storage_peak_bytes']==1024
    assert r['settlement_financial_deltas']==dict(GPU_s=200,CPU_s=13,queries=1,model_loads=0)
    assert r['native_exit_status']==1 and r['charges_DEC1_subcaps'] is False
    with pytest.raises(PermissionError):settled(done)


def test_exact_unknown_queries_keep_marker_and_hold_without_erasing_older_unknowns():
    v=debit(state(),'model_loads',1,'load');v=debit(v,'generation_queries',1,'q0')
    v['unknown_effects']=['OLD_UNKNOWN_PRESERVED'];v=debit(v,'CPU_s',5,'cpu')
    n,f=final_inputs(v);f['unknown_query_intents']=[1]
    done=settled(v,n,f)
    assert done['reserved']['neural_queries']==1 and done['unknown_effects'][0]=='OLD_UNKNOWN_PRESERVED'
    assert done['unknown_effects'][1]['unknown_query_intents']==[1]
    assert done['PBS_submissions'][-1]['reserved_neural_queries']==1
    assert done['generation_queries']==3 and done['CPU_s']==20
    with pytest.raises(PermissionError):debit(done,'generation_queries',1,'retry')


@pytest.mark.parametrize('target,key,value',[
    ('n','job_state','R'),('n','job_id','other.stdct-mgmt-02'),('n','recorded_process_terminal',False),
    ('n','terminal_inventory_stable',False),('n','submission_nonce','other'),('n','observed_epoch',NOW-121),
    ('n','conservative_GPU_s',28861),('n','conservative_CPU_s',80001),('n','Exit_status',True),
    ('f','effect_journal_complete',False),('f','optimizer_updates_committed',1),('f','new_physics',1),
    ('f','new_episodes',1),('f','source_sha','f'*40),('f','statistics_sha256','f'*64),
    ('f','checkpoint_manifest_sha256','f'*64),('f','query_attempts',999999),
    ('f','model_load_attempts',3),('f','unknown_query_intents',[1,1]),('f','unknown_query_intents',[3])])
def test_unknown_finality_wrong_identity_or_overrun_leaves_entire_input_escrow(target,key,value):
    v=state();before=deepcopy(v);n,f=final_inputs(v)
    (n if target=='n' else f)[key]=value
    with pytest.raises(PermissionError):settled(v,n,f)
    assert v==before and v['reserved']['H200_s']==28860


def test_terminal_journal_cannot_roll_back_already_committed_costs():
    v=debit(state(),'model_loads',1,'load');v=debit(v,'generation_queries',1,'query');v=debit(v,'CPU_s',7,'cpu')
    for field,low in [('conservative_CPU_s',6),('query_attempts',0),('model_load_attempts',0)]:
        n,f=final_inputs(v)
        (n if field=='conservative_CPU_s' else f)[field]=low
        with pytest.raises(PermissionError):settled(v,n,f)


def test_cpu_and_query_charge_ids_are_once_and_unknown_fences_only_new_neural_effects():
    v=state()
    with pytest.raises(PermissionError):debit(v,'generation_queries',1,'before-load')
    v=debit(v,'model_loads',1,'load')
    with pytest.raises(PermissionError):debit(v,'model_loads',1,'load')
    v['unknown_effects']=['PRESERVED_UNKNOWN'];old=deepcopy(v)
    with pytest.raises(PermissionError):debit(v,'generation_queries',1,'query')
    assert v==old
    v=debit(v,'CPU_s',10,'closeout-CPU')
    assert v['CPU_s']==10 and v['CPU_reserved_s']==79990 and v['unknown_effects']==['PRESERVED_UNKNOWN']
    with pytest.raises(PermissionError):debit(v,'CPU_s',10,'closeout-CPU')


def test_financial_closeout_after_hard_close_does_not_open_new_work():
    v=state();n,f=final_inputs(v);n['observed_epoch']=e.HARD_CLOSE+1
    done=settled(v,n,f,now=e.HARD_CLOSE+2)
    assert done['PBS_submissions'][-1]['status']=='NATIVE_TERMINAL'
    assert done['PBS_submissions'][-1]['hard_close_epoch']==e.HARD_CLOSE


def put(path,value,readonly=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value));path.chmod(0o400 if readonly else 0o600)


@pytest.fixture
def registered(tmp_path):
    v=state();r=v['PBS_submissions'][-1];root=tmp_path/v['campaign_run_uuid']/'children'/r['child_run_uuid']
    m=dict(platform_evaluation_version=e.VERSION,scope=e.ROLE,run_uuid=root.name,
        owner='EXISTING_LOCAL_PC',owner_user='e1192548',code=dict(vla_commit=r['source_sha']),
        execution=dict(profile=e.PROFILE,uid=os.getuid()))
    put(root/'RUN_MANIFEST.json',m,True)
    request=dict(platform_evaluation_version=e.VERSION,submit_count=1,run_uuid=root.name,
        source_sha=r['source_sha'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),evaluation_candidate=r['evaluation_candidate'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=e.R0,stage=r['stage'],project=r['project'])
    p=root/'pbs/alloc-01/REQUEST.json';put(p,request,True)
    r.update(request_sha256=sha(p),run_manifest_sha256=sha(root/'RUN_MANIFEST.json'))
    put(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json',v)
    put(root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(run_uuid=v['campaign_run_uuid'],
        T0_epoch=e.T0,stop_new_physics_epoch=e.STOP_PHYSICS,hard_closeout_epoch=e.HARD_CLOSE))
    lease=root/'lease.json';put(lease,dict(status='ACTIVE',run_uuid=root.name,authority_release=e.AUTHORITY,deadline_epoch=e.STOP_PHYSICS))
    put(root/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
    return root,p,r


def test_mature_ledger_adapter_keeps_one_lock_and_transfers_escrow(registered):
    root,p,r=registered
    with patch.dict(os.environ,dict(PBS_JOBID=r['job_id'],USER='e1192548')),\
         patch('scripts.pi0294.pv1.neural_ledger.time.time',return_value=NOW):
        ledger=for_request(root,p);assert isinstance(ledger,EvaluationLedger)
        ledger.charge('model_loads',1);ledger.charge('generation_queries',1);ledger.cpu(7)
        v=json.loads(ledger.path.read_text());row=v['PBS_submissions'][-1]
        assert v['revision']==3 and len(row['charged_effects'])==3
        assert v['generation_queries']==1 and v['reserved']['neural_queries']==row['query_limit']-1
        assert v['CPU_s']==7 and v['CPU_reserved_s']==79993
        put(root/'STOP.json',dict(preserve=True));before=ledger.path.read_bytes()
        with pytest.raises(PermissionError):ledger.charge('generation_queries',1)
        assert before==ledger.path.read_bytes()
        ledger.cpu(2);v=json.loads(ledger.path.read_text());assert v['CPU_s']==9
        done=settled(v);assert done['CPU_s']==20 and done['generation_queries']==3


@pytest.mark.parametrize('fault',['wrong_job','wrong_user','writable_request','manifest_changed','request_changed','wrong_child','final'])
def test_adapter_rejects_wrong_native_or_sealed_identity_before_mutating_global(registered,fault):
    root,p,r=registered;job=r['job_id'];user='e1192548'
    before=(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes()
    if fault=='wrong_job':job='wrong.stdct-mgmt-02'
    if fault=='wrong_user':user='other'
    if fault=='writable_request':p.chmod(0o600)
    if fault=='manifest_changed':put(root/'RUN_MANIFEST.json',dict(foreign=True),True)
    if fault=='request_changed':v=json.loads(p.read_text());v['submission_nonce']='other';put(p,v,True)
    if fault=='wrong_child':
        q=root.parent/'other-child';q.mkdir();root=q
    with patch.dict(os.environ,dict(PBS_JOBID=job,USER=user)),pytest.raises((PermissionError,FileNotFoundError,KeyError)):
        for_request(root,p,final=fault=='final')
    assert before==(p.parents[4]/'HOPPER_RESOURCE_LEDGER.json').read_bytes()


def test_original_h200_ledger_factory_stays_original_and_unknown_journal_is_not_resent(registered,tmp_path):
    root,p,r=registered
    m=json.loads((root/'RUN_MANIFEST.json').read_text());m.pop('platform_evaluation_version')
    m.update(scope='PI0294_PV1_SHARED_POLICY_HOPPER_V1');m['execution']['profile']='hopper-neural';put(root/'RUN_MANIFEST.json',m,True)
    request=json.loads(p.read_text());request.pop('platform_evaluation_version');put(p,request,True)
    assert type(for_request(root,p)) is Ledger
    from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
    from scripts.pi0294.dec1.source_cpu_once_io import commit,reconcile
    target=tmp_path/'only-fixture-ledger.json';target.write_bytes(b'{}\n');old=hashlib.sha256(target.read_bytes()).hexdigest()
    intent=tmp_path/'once-intent.json';receipt=tmp_path/'once-receipt.json';lock=tmp_path/'once.lock'
    journal=Journal(tmp_path/'rpc','fixture');calls=[]
    def operation():
        calls.append(1)
        commit(target,expected_before_sha256=old,proposed_bytes=b'{"known_effect":1}\n',request_id='fixture',
            intent_path=intent,receipt_path=receipt,lock_path=lock)
        raise ConnectionError('AFTER_COMMIT_REPLY_UNKNOWN')
    with pytest.raises(ConnectionError):journal.once('effect',dict(scope='fixture'),operation)
    with pytest.raises(MutationOutcomeUnknown):journal.once('effect',dict(scope='fixture'),operation)
    assert calls==[1] and json.loads(journal.path('effect').read_text())['status']=='INTENT'
    result=reconcile(target,intent_path=intent,receipt_path=receipt)
    assert result['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    assert result['read_only'] is True and result['automatic_retry_allowed'] is False
    assert json.loads(journal.path('effect').read_text())['status']=='INTENT'
