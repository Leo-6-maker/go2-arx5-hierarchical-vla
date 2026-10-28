"""Isolated CPU fixtures: no actual PBS, GPU, model or resource permission."""
from copy import deepcopy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.pv1 import launcher,neural
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure
from scripts.pi0294.test_pv1_a40_evaluation_identity import case,proof,put
from scripts.pi0294.test_pv1_a40_evaluation_lifecycle import state


def bound(c):
    v=state();row=v['PBS_submissions'][0];r=c['r']
    old=c['directory'];new=old.parent/'alloc-01';old.rename(new);c['directory']=new
    r['submit_count']=1
    for key in ('noise_bank','remaining_conditions','prior_outcomes_sha256','planned_panel_sha256',
            'cold_input_audit_sha256','checkpoint_reuse_proof_sha256','prequeue_future_admission_sha256',
            'CPU_upper_s','CPU_envelope_profile','declared_process_CPU_limit_s','additional_output_upper_bytes'):
        r[key]=row[key]
    put(new/'REQUEST.json',r)
    row.update(child_run_uuid=c['root'].name,job_id=c['a']['job_id'],request_sha256=sha(new/'REQUEST.json'))
    for key in ('source_sha','run_manifest_sha256','created_epoch'):
        row[key]=r[key]
    return v


def bootstrap(c,v,p=None):
    return e.bootstrap_identity(c['m'],c['r'],v,run_uuid=c['root'].name,
        manifest_sha256=sha(c['root']/'RUN_MANIFEST.json'),request_sha256=sha(c['directory']/'REQUEST.json'),
        now=c['now'],user='e1192548',hostname=c['a']['hostname'],job_id=c['a']['job_id'],
        queue_proof=proof(c) if p is None else p)


def test_bootstrap_uses_real_job_fields_without_constructing_an_admission(case):
    c=case;v=bound(c);before=deepcopy(v);facts=bootstrap(c,v)
    assert v==before and 'status' not in facts and 'model_loaded' not in facts
    assert facts['job_id']==c['a']['job_id'] and facts['native_allocated_ncpus']==4
    assert set(facts)=={'job_id','hostname','allocation_start_epoch','allocation_end_epoch',
        'native_select','native_allocated_ncpus','native_allocated_memory_bytes'}


@pytest.mark.parametrize('fault',[
    'GPU_escrow','CPU_escrow','output_escrow','query_escrow','unknown','pending',
    'unbound','request_hash','source','nonce','child','ordinal','prior','checkpoint',
    'another_Q','unregistered_campaign','partial_queue','stale','not_running','training'])
def test_bootstrap_preserves_input_on_bad_escrow_identity_or_complete_queue(case,fault):
    c=case;v=bound(c);row=v['PBS_submissions'][0];p=proof(c)
    if fault=='GPU_escrow':v['reserved']['H200_s']-=1
    if fault=='CPU_escrow':v['CPU_reserved_s']-=1
    if fault=='output_escrow':v['storage_reserved_bytes']-=1
    if fault=='query_escrow':v['reserved']['neural_queries']-=1
    if fault=='unknown':v['unknown_effects']=['PRESERVED_UNKNOWN']
    if fault=='pending':v['pending_updates']=['PRESERVED_PENDING']
    if fault=='unbound':row.update(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',job_id=None)
    if fault=='request_hash':row['request_sha256']='f'*64
    if fault=='source':row['source_sha']='f'*40
    if fault=='nonce':row['submission_nonce']='22345678-1234-4234-8234-123456789abd'
    if fault=='child':row['child_run_uuid']='other'
    if fault=='ordinal':c['r']['submit_count']=2
    if fault=='prior':c['r']['prior_outcomes_sha256']='f'*64
    if fault=='checkpoint':c['r']['checkpoint_manifest_sha256']=e.ENDPOINTS['B_COVERAGE']
    if fault in ('another_Q','unregistered_campaign'):
        jid='124.stdct-mgmt-02';j=deepcopy(c['j']);j.update(job_state='Q')
        if fault=='unregistered_campaign':j['Job_Name']='pi0294_unregistered'
        p['qselect']['stdout']+=jid+'\n';p['qstat']['argv'].append(jid)
        p['qstat']['stdout']=json.dumps(dict(Jobs={c['a']['job_id']:c['j'],jid:j}))
        if fault=='another_Q':
            row2=deepcopy(row);row2.update(job_id=jid,request_id='22345678-1234-4234-8234-123456789abc',
                submission_nonce='22345678-1234-4234-8234-123456789abd')
            v['PBS_submissions'].append(row2);v['used']['PBS_submissions']=2
    if fault=='partial_queue':p['qstat']['stdout']='{"Jobs":{}}'
    if fault=='stale':p['observed_epoch']-=121
    if fault=='not_running':
        j=deepcopy(c['j']);j['job_state']='Q';p['qstat']['stdout']=json.dumps(dict(Jobs={c['a']['job_id']:j}))
    if fault=='training':c['r']['platform_training_version']='old-train'
    before=deepcopy(v)
    with pytest.raises(PermissionError):bootstrap(c,v,p)
    assert v==before


def test_launch_bootstrap_selects_vanda_complete_queue_and_never_hopper(case,tmp_path):
    c=case;v=bound(c)
    # Give the CPU fixture the same parent/children geometry as native roots.
    original=c['root'];target=tmp_path/'campaign/children'/original.name
    target.parent.mkdir(parents=True);original.rename(target);c['root']=target;c['directory']=target/'pbs/alloc-01'
    put(target.parent.parent/'HOPPER_RESOURCE_LEDGER.json',v)
    adoption=dict(run_uuid=c['r']['campaign_run_uuid'])
    with patch.dict(os.environ,dict(PBS_JOBID=c['a']['job_id'],USER='e1192548',CUDA_VISIBLE_DEVICES='0')),\
         patch.object(launcher,'validate',return_value=(c['m'],{},{},{},adoption)),\
         patch.object(launcher.socket,'gethostname',return_value=c['a']['hostname']),\
         patch.object(launcher.time,'time',return_value=c['now']),\
         patch.object(e,'collect_queue',return_value=proof(c)) as queue,\
         patch.object(launcher,'owned_jobs',side_effect=AssertionError('NO_HOPPER_QUERY')):
        answer=launcher.native_bootstrap(target,c['directory']/'REQUEST.json')
        assert answer[3:]==(c['a']['job_id'],c['a']['hostname']) and queue.call_count==1
        assert not (c['directory']/'PBS_READY.json').exists()
        queue.side_effect=ReadOnlyFailure('CURRENT_QUEUE_UNKNOWN')
        with pytest.raises(ReadOnlyFailure):launcher.native_bootstrap(target,c['directory']/'REQUEST.json')
        assert not (c['directory']/'PBS_READY.json').exists()
        queue.side_effect=None;queue.reset_mock()
        c['r']['stage']='P2_PRIMARY';put(c['directory']/'REQUEST.json',c['r'])
        with pytest.raises(PermissionError):launcher.native_bootstrap(target,c['directory']/'REQUEST.json')
        queue.assert_not_called()


def test_original_h200_bootstrap_keeps_its_own_native_select_path(case,tmp_path):
    c=case;v=bound(c);r=c['r'];m=c['m']
    m.pop('platform_evaluation_version');m['scope']='PI0294_PV1_SHARED_POLICY_HOPPER_V1'
    m['execution']['profile']='hopper-neural'
    original=c['root'];target=tmp_path/'campaign/children'/original.name
    target.parent.mkdir(parents=True);original.rename(target)
    put(target/'RUN_MANIFEST.json',m);r['run_manifest_sha256']=sha(target/'RUN_MANIFEST.json')
    r.pop('platform_evaluation_version');r['ncpus']=4
    request=target/'pbs/alloc-01/REQUEST.json';put(request,r)
    put(target.parent.parent/'HOPPER_RESOURCE_LEDGER.json',v)
    job=deepcopy(c['j']);job['Resource_List']['select']+=':gpu_model=H200'
    with patch.dict(os.environ,dict(PBS_JOBID=c['a']['job_id'],USER='e1192548',CUDA_VISIBLE_DEVICES='0')),\
         patch.object(launcher,'validate',return_value=(m,{},{},{},dict(run_uuid=r['campaign_run_uuid']))),\
         patch.object(launcher.socket,'gethostname',return_value=c['a']['hostname']),\
         patch.object(launcher,'owned_jobs',return_value={c['a']['job_id']:job}) as hopper,\
         patch.object(e,'collect_queue',side_effect=AssertionError('NO_VANDA_ALIAS')):
        assert launcher.native_bootstrap(target,request)[3]==c['a']['job_id']
        assert hopper.call_count==1
        job['Resource_List']['select']='1:ncpus=4:mem=32gb:ngpus=1:gpu_model=A40'
        with pytest.raises(PermissionError,match='BOOTSTRAP_FRESH_OWN'):
            launcher.native_bootstrap(target,request)


@pytest.mark.parametrize('name',['A40','NVIDIA A40'])
def test_visible_device_is_exact_a40_for_typed_admitted_dev(case,name):
    c=case;torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:name))
    with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='0')):
        assert neural.require_native_device(c['r'],c['a'],torch) is None


@pytest.mark.parametrize('fault',[
    'H200','A100','A40X','hidden','zero','two','bool_count','training','H200_receipt',
    'wrong_checkpoint','wrong_stats','optimizer','typed_missing_on_request','typed_missing_on_receipt'])
def test_wrong_device_or_admission_cannot_choose_legacy_fallback(case,fault):
    c=case;name='NVIDIA A40';count=1;visible='0'
    if fault in ('H200','A100','A40X'):name='NVIDIA '+fault
    if fault=='hidden':visible=''
    if fault=='zero':count=0
    if fault=='two':count=2
    if fault=='bool_count':count=True
    if fault=='training':c['r']['platform_training_version']='old'
    if fault=='H200_receipt':c['a']['status']='PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
    if fault=='wrong_checkpoint':c['r']['checkpoint_manifest_sha256']=e.ENDPOINTS['B_COVERAGE']
    if fault=='wrong_stats':c['a']['statistics_sha256']='f'*64
    if fault=='optimizer':c['r']['optimizer_updates']=1
    if fault=='typed_missing_on_request':c['r'].pop('platform_evaluation_version')
    if fault=='typed_missing_on_receipt':c['a'].pop('platform_evaluation_version')
    torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count,get_device_name=lambda i:name))
    with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES=visible)),pytest.raises(PermissionError):
        neural.require_native_device(c['r'],c['a'],torch)


def test_untyped_original_device_gate_still_requires_h200():
    for name,accepted in [('NVIDIA H200',True),('NVIDIA A40',False)]:
        torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:name))
        with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='0')):
            if accepted:assert neural.require_native_device({}, {},torch) is None
            else:
                with pytest.raises(PermissionError,match='H200_ONLY'):neural.require_native_device({}, {},torch)


@pytest.mark.parametrize('remaining,expected',[(100000,79760),(79760,79760),(40000,40000),(0,0)])
def test_strict_process_cpu_is_not_allocated_capacity_or_a_larger_global_cap(case,remaining,expected):
    c=case;bound(c)
    assert e.process_cpu_limit(c['r'],remaining)==expected


@pytest.mark.parametrize('key,value',[('CPU_upper_s',80001),('declared_process_CPU_limit_s',80000),
    ('CPU_envelope_profile',None),('walltime_s',22500),('platform_evaluation_version','old')])
def test_cpu_upper_must_match_the_full_accounting_row(case,key,value):
    c=case;bound(c);c['r'][key]=value
    with pytest.raises(PermissionError):e.process_cpu_limit(c['r'],100000)


def test_new_device_branch_does_not_replace_full_native_admission(case):
    c=case
    with patch.object(neural,'admit',side_effect=PermissionError('FULL_NATIVE_ADMISSION_REQUIRED')),\
         patch.object(neural.model,'load') as load,patch.object(neural,'qualify_transport') as qualification:
        with pytest.raises(PermissionError,match='FULL_NATIVE_ADMISSION_REQUIRED'):
            neural.run(c['root'],c['directory']/'REQUEST.json')
    load.assert_not_called();qualification.assert_not_called()
