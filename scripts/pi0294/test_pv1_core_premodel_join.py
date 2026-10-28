"""CPU receipts only; raw device metadata does not create native admission."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.pi0294.pv1 import evaluation_core_identity as n,evaluation_core_accounting as c
from scripts.pi0294.pv1 import heartbeat_capture as h,pre_model_qualification as q
from scripts.pi0294.test_pv1_core_evaluation_admission import captured_inputs
from scripts.pi0294.test_pv1_core_evaluation_accounting import NOW,BOOT,process
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE,GPU
from scripts.pi0294.test_mp1_transport_fixture import put


def admission(inputs):
    m=json.loads(inputs['manifest_bytes']);r=json.loads(inputs['request_bytes'])
    p=process(dict(r,request_sha256=hashlib.sha256(inputs['request_bytes']).hexdigest()))
    handle=c.process_handle(BOOT,p['pid'],p['proc_start'])
    p.update(native_handle=handle,native_model_permission=False,physics_authorized=False,
        kernel_limits=n.kernel_limits('400000 100000','51539607552','32'),
        native_unit_properties=dict(MainPID=str(p['pid']),ActiveState='active',
            RuntimeMaxUSec='8h',ControlGroup='/CPU-test-fixture'),
        native_GPU_raw=dict(devices=dict(returncode=0,stdout=DEVICE,stderr=''),
            processes=dict(returncode=0,stdout='',stderr='')),
        device=n.parse_GPU_rows(DEVICE,'',GPU_uuid=GPU))
    a={k:r[k] for k in ('source_sha','request_id','submission_nonce','stage','evaluation_candidate',
        'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number')}
    a.update(status=n.STATUS,platform_evaluation_version=c.VERSION,run_uuid=m['run_uuid'],
        manifest_sha256=hashlib.sha256(inputs['manifest_bytes']).hexdigest(),
        request_sha256=hashlib.sha256(inputs['request_bytes']).hexdigest(),job_id=handle,native_handle=handle,
        hostname=p['hostname'],allocation_start_epoch=p['started_epoch'],
        allocation_end_epoch=p['started_epoch']+28800,model_loaded=False,new_optimizer_updates=0,
        native_process=p,native_process_sha256=c.digest(p))
    return m,r,a


def joined(m,r,a):
    return n.sealed_identity(m,r,a,run_uuid=m['run_uuid'],
        manifest_sha256=r['run_manifest_sha256'],request_sha256=a.get('request_sha256','f'*64))


def test_full_native_receipt_join_reuses_actual_request_and_unchanged_budget(captured_inputs):
    inputs,_=captured_inputs;m,r,a=admission(inputs);before=deepcopy(a)
    assert joined(m,r,a)==before and a==before
    assert a['job_id'].startswith('coregpu:') and 'PBS_JOBID' not in a
    assert a['native_process']['native_model_permission'] is False
    # Neither a status-only stub nor the previous Vanda admission is sufficient.
    for stub in [dict(status=n.STATUS),dict(a,status=c.old.STATUS),dict(a,native_process=None)]:
        with pytest.raises(PermissionError):joined(m,r,stub)


@pytest.mark.parametrize('fault',[
    'other_checkpoint','other_nonce','other_manifest','PBS_alias','old_start','different_boot',
    'same_PID_new_start','missing_nvidia_read','busy_device','wrong_device','unbounded_CPU',
    'unbounded_memory','inactive_unit','other_main_PID','missing_raw_digest','extended_wall',
    'model_already_loaded','physics_permission','training','partial_Core_profile'])
def test_raw_identity_and_platform_mutations_refused(captured_inputs,fault):
    inputs,_=captured_inputs;m,r,a=admission(inputs);p=a['native_process']
    if fault=='other_checkpoint':a['checkpoint_manifest_sha256']='f'*64
    if fault=='other_nonce':p['submission_nonce']='42345678-1234-4234-8234-123456789abc'
    if fault=='other_manifest':p['run_manifest_sha256']='f'*64
    if fault=='PBS_alias':a['PBS_JOBID']=a['job_id']
    if fault=='old_start':p['started_epoch']=r['created_epoch']-2
    if fault=='different_boot':p['boot_id']='42345678-1234-4234-8234-123456789abc'
    if fault=='same_PID_new_start':p['proc_start']+=1
    if fault=='missing_nvidia_read':p['native_GPU_raw']['devices']['returncode']=1
    if fault=='busy_device':p['native_GPU_raw']['processes']['stdout']=GPU+', 12345, python, 12\n'
    if fault=='wrong_device':p['native_GPU_raw']['devices']['stdout']=DEVICE.replace('RTX 4080 SUPER','A40')
    if fault=='unbounded_CPU':p['kernel_limits']['CPU_quota']=500000
    if fault=='unbounded_memory':p['kernel_limits']['memory_max_bytes']=64*1024**3
    if fault=='inactive_unit':p['native_unit_properties']['ActiveState']='inactive'
    if fault=='other_main_PID':p['native_unit_properties']['MainPID']='1'
    if fault=='extended_wall':a['allocation_end_epoch']+=1
    if fault=='model_already_loaded':a['model_loaded']=True
    if fault=='physics_permission':p['physics_authorized']=True
    if fault=='training':r['optimizer_updates']=1
    if fault=='partial_Core_profile':m['execution']['profile']='hopper-neural'
    if fault!='missing_raw_digest':a['native_process_sha256']=c.digest(p)
    else:a['native_process_sha256']='f'*64
    with pytest.raises(PermissionError):joined(m,r,a)


def test_same_native_handle_still_requires_all_600_real_seconds_and_raw_journal(captured_inputs,tmp_path):
    inputs,_=captured_inputs;m,r,a=admission(inputs);root=tmp_path/m['run_uuid'];d=root/'pbs/alloc-01'
    d.mkdir(parents=True)
    for path,raw in [(root/'RUN_MANIFEST.json',inputs['manifest_bytes']),(d/'REQUEST.json',inputs['request_bytes'])]:
        path.write_bytes(raw);path.chmod(0o400)
    assert hashlib.sha256((root/'RUN_MANIFEST.json').read_bytes()).hexdigest()==r['run_manifest_sha256']
    rows=[]
    for i in range(81):
        t=NOW+7.5*i
        rows.append(dict(sequence=i,monotonic_s=7.5*i,proof=dict(server_revalidation=True,
            cached_clock_epoch=t,revalidated_clock_epoch=t,clock_sampled_after_read=True,
            heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid=root.name,observed_epoch=t-10))))
    journal=d/'heartbeat-capture/READS.jsonl';journal.parent.mkdir(parents=True)
    journal.write_text(''.join(json.dumps(x)+'\n' for x in rows));journal.chmod(0o400)
    value=dict(schema='PI0294_PV1_NATIVE_HEARTBEAT_CAPTURE_V1',status='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY',
        authority_release=c.AUTHORITY,run_uuid=root.name,job_id=a['job_id'],hostname=a['hostname'],
        manifest_sha256=r['run_manifest_sha256'],request_sha256=a['request_sha256'],source_sha=r['source_sha'],
        clock_source=h.CLOCK,rows=rows,raw_journal_relative=str(journal.relative_to(root)),
        raw_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),CPU_s=.1,models_loaded=0,
        new_model_queries=0,new_physics_integrations=0,heartbeat_written=False,transport_qualified=False)
    put(d/h.FILENAME,value,True)
    assert h.verify(root,d,a)==value
    # Heartbeat fragments do not stand in for the missing eighty-cycle proof.
    with pytest.raises(PermissionError,match='NOT_RUN'):q.require(root,d,a)
    for fault in ('short','expired','before_read','wrong_run'):
        v=deepcopy(value)
        if fault=='short':v['rows']=v['rows'][:-1]
        if fault=='expired':v['rows'][20]['proof']['heartbeat']['observed_epoch']-=301
        if fault=='before_read':v['rows'][20]['proof']['clock_sampled_after_read']=False
        if fault=='wrong_run':v['rows'][20]['proof']['heartbeat']['run_uuid']='other'
        journal.chmod(0o600);journal.write_text(''.join(json.dumps(x)+'\n' for x in v['rows']));journal.chmod(0o400)
        v['raw_journal_sha256']=hashlib.sha256(journal.read_bytes()).hexdigest();put(d/h.FILENAME,v,True)
        with pytest.raises(PermissionError):h.verify(root,d,a)
