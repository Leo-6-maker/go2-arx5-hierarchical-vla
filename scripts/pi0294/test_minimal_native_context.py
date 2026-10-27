"""Captured full resources, future registered process/identity CPU fixtures.

Real temporary files exercise changed GLOBAL/STOP and original owner handling;
none of these tests is a native registration, CUDA or GR00T qualification.
"""
import base64,json,time,os,socket
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_minimal_native_admission import actual,packet
from scripts.pi0294.generalization import minimal_native_context as c,minimal_native_runtime as runtime,minimal_execution as e
from scripts.pi0292.durable import write


def identity(packet):
    r=json.loads(packet['request_bytes']);m=json.loads(packet['manifest_bytes'])
    root=Path(m['output_root']);p=root/'pbs'/('alloc-'+str(r['submit_count']).zfill(2))/'REQUEST.json'
    b=dict(run_root=str(root),uid=22182,run_manifest_sha256=r['run_manifest_sha256'])
    return dict(root=root,request_path=p,m=m,b=b,r=r,g=json.loads(packet['global_bytes']),
        uid=22182,pbs_job=packet['job_id'],now=packet['now_epoch'],require_bound=True)


def test_own_standard_registered_context_joins_original_clock_and_new_scope(packet):
    args=identity(packet);before=deepcopy(args);assert c.registered_identity(**args)is None
    assert args==before


@pytest.mark.parametrize('change',['wrong_uid','other_job','old_scope','mutable_other_root','other_source',
    'different_manifest','expired','request_alias','wrong_profile','agent_liveness'])
def test_context_identity_does_not_impersonate_old_training_or_another_owner(packet,change):
    p=identity(packet)
    if change=='wrong_uid':p['uid']=2009
    elif change=='other_job':p['pbs_job']='98765.stdct-mgmt-02'
    elif change=='old_scope':p['m']['platform_training_version']='V5_CD_NATIVE'
    elif change=='mutable_other_root':p['m']['registered_runs_root']='/OTHER'
    elif change=='other_source':p['m']['code']['vla_commit']='a'*40
    elif change=='different_manifest':p['b']['run_manifest_sha256']='a'*64
    elif change=='expired':p['now']=p['r']['deadline_epoch']
    elif change=='request_alias':p['request_path']=p['root']/'INPUT_REQUEST.json'
    elif change=='wrong_profile':p['m']['execution']['profile']='vanda-neural-training'
    else:p['m']['execution']['agent_liveness_required']=True
    with pytest.raises((PermissionError,ValueError)):c.registered_identity(**p)


def save_immutable(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(e.raw(value));path.chmod(0o400)


def test_actual_temp_guard_reads_changed_unknown_and_STOP_even_with_cached_full_budget(packet,tmp_path):
    root=tmp_path/'children'/'main';root.mkdir(parents=True);lease=tmp_path/'LEASE.json'
    m=json.loads(packet['manifest_bytes']);r=json.loads(packet['request_bytes'])
    save_immutable(root/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
    save_immutable(lease,dict(status='ACTIVE',run_uuid=root.name,manifest_sha256=r['run_manifest_sha256']))
    save_immutable(root/c.LOCAL,dict(source='ACTUAL_CANONICAL_CORE_LOCAL_READONLY',capture_is_fresh_native_authority=False,
        LOCAL_base64=base64.b64encode(packet['local_bytes']).decode(),LOCAL_sha256=e.sha(packet['local_bytes'])))
    global_path=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';global_path.write_bytes(packet['global_bytes'])
    allocation=dict(job_id=packet['job_id'],request_sha256=e.sha(packet['request_bytes']),source_sha=r['source_sha'],
        run_manifest_sha256=r['run_manifest_sha256'],allocation_start_epoch=r['T_CD'],
        allocation_end_epoch=r['deadline_epoch'],deadline_epoch=r['deadline_epoch'])
    beat=dict(run_uuid=root.name,scope=e.VERSION,grant_sha256=e.authority.GRANT_SHA,job_id=allocation['job_id'],
        request_sha256=allocation['request_sha256'],source_sha=r['source_sha'],global_unknown_effects=[],
        native_job_state='R',observed_epoch=time.time())
    save_immutable(root/'control/OWNER_HEARTBEAT.json',beat)
    guard=c.Guard(root,allocation);proof=guard();assert proof['cap_changes_required']=={}
    assert guard()is proof  # Exact current bytes may reuse only pure budget work.
    (root/'STOP.json').write_text('{}')
    with pytest.raises(PermissionError,match='CURRENT_STOP'):guard()
    (root/'STOP.json').unlink();g=json.loads(packet['global_bytes']);g['unknown_effects']=['PRESERVED']
    global_path.write_bytes(e.raw(g))
    with pytest.raises(PermissionError,match='UNCLEARED_GLOBAL_FENCES'):guard()
    assert json.loads(global_path.read_bytes())['unknown_effects']==['PRESERVED']


def test_first_Q_wait_preserves180s_and_does_not_swallow_identity_error():
    calls=[]
    def guard(**kw):
        calls.append(kw)
        if len(calls)==1:raise c.StartupOwnerStillQueued()
    runtime.await_running_owner(guard,clock=lambda:0,sleep=lambda _:None)
    assert calls==[dict(startup=True)]*2
    def wrong(**kw):raise PermissionError('WRONG_IDENTITY')
    with pytest.raises(PermissionError,match='WRONG_IDENTITY'):
        runtime.await_running_owner(wrong,clock=lambda:0,sleep=lambda _:None)
    clock=iter((0,181))
    def queued(**kw):raise c.StartupOwnerStillQueued()
    with pytest.raises(TimeoutError,match='180S'):
        runtime.await_running_owner(queued,clock=lambda:next(clock),sleep=lambda _:None)


def test_actual_CPU_kernel_process_identity_does_not_expire_after600s():
    now=time.time();job='CPU-test-only-not-a-PBS-qualification';host=socket.gethostname()
    row=dict(job_id=job,hostname=host,pid=os.getpid(),
        proc_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),observed_epoch=now-700)
    ready=dict(row,observed_epoch=now)
    assert c.phase_process(ready,row,job_id=job,hostname=host)is None
    for key in('pid','proc_start','boot_id'):
        bad=dict(ready);bad[key]=bad[key]+1 if type(bad[key])is int else'OTHER_BOOT'
        with pytest.raises(PermissionError):c.phase_process(bad,row,job_id=job,hostname=host)
