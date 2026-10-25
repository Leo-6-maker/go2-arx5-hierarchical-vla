"""Cold CPU joins of the original runtime; no native/model/physics GO.

The owner tests read this test process's real boot/stat/cgroup. Core UID, host,
unit replies and all C50 terminal endpoints remain explicit fixtures.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pi0294.test_v5_core_native_identity import request, allocation
from scripts.pi0294.test_v5_core_full_admission import full, captured_global
from scripts.pi0294.pv1 import mix_accounting as a, mix_core_identity as n, successor_cd as s
from scripts.pi0294.pv1 import mix_core_context as context, evaluation_core_native as native
from scripts.pi0294.pv1 import evaluation_core_runtime as runtime, pre_model_qualification as qualification


def put(path, value, *, writable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_bytes(value if isinstance(value, bytes) else a.raw(value))
    path.chmod(0o600 if writable else 0o400)


@pytest.fixture
def owner(tmp_path, monkeypatch):
    m, r = request(); parent = tmp_path/'campaign'; root = parent/'neural-children'/r['run_uuid']
    control = parent/'control'/'v5-cd-core-c50-owner-fixture'
    registry = parent/'registry.json'; lease = parent/'lease.json'
    ob = dict(run_root=str(control),run_uuid=control.name,uid=2009,role=n.OWNER_ROLE+'__'+control.name,
        registered_parent=str(control.parent),registry=str(registry),lease=str(lease))
    om = dict(run_uuid=control.name,output_root=str(control),scope=n.OWNER_ROLE,code=m['code'],
        authority=m['authority'],limits=m['limits'],execution=dict(profile='4080-cpu-data',
            output_binding_sha256=a.sha(a.raw(ob)),private_path_role=ob['role']))
    binding = dict(owner_root=str(control),owner_manifest_sha256=a.sha(a.raw(om)))
    m['execution']['sealed_inputs'] = {context.LOCAL_BINDING:a.sha(a.raw(binding))}
    r['run_manifest_sha256'] = a.sha(a.raw(m))
    pid=os.getpid(); stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    proc=dict(schema='V5_CD_CORE_RESIDENT_OWNER_NATIVE_PROCESS_V1',hostname='corelab-4080s',uid=2009,
        run_uuid=control.name,source_sha=r['source_sha'],request_id=r['request_id'],request_sha256=a.sha(a.raw(r)),
        manifest_sha256=binding['owner_manifest_sha256'],native_unit=control.name+'.service',task=s.TASK,
        scope_grant_sha256=s.GRANT_SHA,T_CD=r['T_CD'],deadline_epoch=r['deadline_epoch'],pid=pid,
        proc_start=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_epoch=1200.)
    for p,v in [(root/'RUN_MANIFEST.json',m),(root/context.LOCAL_BINDING,binding),
                (root/'pbs/alloc-01/REQUEST.json',r),(control/'RUN_MANIFEST.json',om),
                (control/'OUTPUT_BINDING.private.json',ob),(control/n.OWNER_STARTED,proc),
                (parent/'OUTPUT_BINDING.private.json',dict(registry=str(registry)))]:put(p,v)
    put(registry,dict(entries=[dict(role=ob['role'],path=str(control),run_uuid=control.name)]),writable=True)
    put(lease,dict(status='ACTIVE',run_uuid=control.name,authority_release=s.TASK_SOURCE,
        deadline_epoch=r['deadline_epoch']),writable=True)
    original_stat=Path.stat
    # Preserve nanosecond stat members: os.stat_result construction loses them.
    def owned_stat(path,*args,**kw):
        actual=original_stat(path,*args,**kw)
        class Owned:
            st_uid=2009
            def __getattr__(self,name):return getattr(actual,name)
        return Owned()
    monkeypatch.setattr(Path,'stat',owned_stat)
    monkeypatch.setattr(n.os,'getuid',lambda:2009)
    monkeypatch.setattr(n.os,'uname',lambda:SimpleNamespace(nodename='corelab-4080s'))
    monkeypatch.setattr(n.time,'time',lambda:1205.)
    cg=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
    reply=SimpleNamespace(returncode=0,stdout=f'MainPID={pid}\nActiveState=active\nControlGroup={cg}\n')
    calls=[]
    def unit(*args,**kw):calls.append((args,kw));return reply
    monkeypatch.setattr(n.subprocess,'run',unit)
    return SimpleNamespace(root=root,r=r,proc=proc,control=control,registry=registry,lease=lease,reply=reply,calls=calls)


def check_owner(v):return n.current_owner(v.root,v.r,request_sha256=a.sha(a.raw(v.r)))


def test_new_owner_reads_real_boot_process_group_and_bounded_unit_without_old_queue(owner):
    assert check_owner(owner)==owner.proc
    assert len(owner.calls)==1 and owner.calls[0][1]['timeout']==8
    assert not (owner.control/'NATIVE_CORE_QUEUE_ACCEPTED.json').exists()


@pytest.mark.parametrize('case',range(10))
def test_stale_wrong_or_unregistered_owner_never_allows_startup(owner,case):
    if case==0:owner.proc['proc_start']+=1
    if case==1:owner.proc['boot_id']='old-boot'
    if case==2:owner.proc['request_sha256']='0'*64
    if case==3:owner.proc['source_sha']='0'*40
    if case==4:owner.proc['native_unit']='old-v4a-owner.service'
    if case==5:owner.reply.returncode=1
    if case==6:owner.reply.stdout='MainPID=1\nActiveState=active\nControlGroup=/other\n'
    if case==7:put(owner.registry,dict(entries=[]),writable=True)
    if case==8:put(owner.lease,dict(status='CLOSED',run_uuid=owner.control.name,
        authority_release=s.TASK_SOURCE,deadline_epoch=owner.r['deadline_epoch']),writable=True)
    if case==9:owner.proc['deadline_epoch']+=1
    put(owner.control/n.OWNER_STARTED,owner.proc)
    with pytest.raises(PermissionError):check_owner(owner)


def test_phase_wait_uses_only_adopted_CD_clock_and_keeps_legacy_clock():
    _,r=request();assert runtime.phase_deadline(r)==s.scope(r['T_CD'])['deadline_epoch']
    assert runtime.phase_deadline({})==runtime.c.old.STOP_PHYSICS
    for change in [dict(evaluation_candidate='B_COVERAGE'),dict(stage='P4_DEV'),dict(deadline_epoch=1.)]:
        with pytest.raises(PermissionError):runtime.phase_deadline(dict(r,**change))


def test_original_dispatcher_routes_CD_without_an_old_AB_alias(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import neural
    _,r=request();rp=tmp_path/'REQUEST.json';put(rp,r);called=[]
    monkeypatch.setattr(runtime,'run',lambda *args:called.append(args)or 'CD_COLD_FIXTURE')
    assert neural.run(tmp_path,rp)=='CD_COLD_FIXTURE' and called==[(tmp_path,rp)]


def test_original_runtime_wait_checks_current_CD_owner_before_admit_or_native_import(owner,monkeypatch):
    m=json.loads((owner.root/'RUN_MANIFEST.json').read_bytes());rp=owner.root/'pbs/alloc-01/REQUEST.json'
    monkeypatch.setattr(runtime,'validate',lambda *args:(m,))
    _,_,v=allocation();process=v['native_process']
    monkeypatch.setattr(runtime.identity,'observe_current',lambda *args:process)
    monkeypatch.setenv(runtime.identity.HANDLE_ENV,'')
    monkeypatch.setattr(runtime.resource,'setrlimit',lambda *args:None)
    monkeypatch.setattr(runtime.native,'admit',lambda *args:pytest.fail('OWNER_NOT_YET_VERIFIED'))
    def refuse(*args,**kw):raise PermissionError('CURRENT_CD_OWNER_REFUSED')
    monkeypatch.setattr(n,'current_owner',refuse)
    with pytest.raises(PermissionError,match='CURRENT_CD_OWNER_REFUSED'):runtime.run(owner.root,rp)
    final=json.loads((rp.parent/'NATIVE_PROCESS_FINAL.json').read_bytes())
    assert final['status']=='FAILED_PRESERVED' and final['new_optimizer_updates']==0
    assert not (rp.parent/'MODEL_LOAD_INTENTS').exists()


def test_heartbeat_fragment_revalidates_CD_identity_before_accepting_saved_rows(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import heartbeat_capture as heartbeat
    m,r,v=allocation();root=tmp_path/r['run_uuid'];directory=root/'pbs/alloc-01'
    put(root/'RUN_MANIFEST.json',m);put(directory/'REQUEST.json',r);put(directory/heartbeat.FILENAME,{})
    def refuse(*args,**kw):raise PermissionError('CD_SEALED_IDENTITY_REQUIRED')
    monkeypatch.setattr(heartbeat.Core_native,'sealed_identity',refuse)
    with pytest.raises(PermissionError,match='CD_SEALED_IDENTITY_REQUIRED'):heartbeat.verify(root,directory,v)


def test_native_collector_reads_exact_dual_bytes_and_rejects_changed_canonical_binding(tmp_path,full,monkeypatch):
    parent=tmp_path/'campaign';m=json.loads(full['manifest_bytes']);r=json.loads(full['request_bytes'])
    root=parent/'neural-children'/r['run_uuid'];directory=root/'pbs/alloc-01'
    local=parent/'control'/'existing'/'LOCAL_RESOURCE_LEDGER.private.json'
    braw=a.raw(dict(canonical_LOCAL_ledger=str(local)))
    m['execution']['sealed_inputs'][context.LOCAL_BINDING]=a.sha(braw)
    put(root/'RUN_MANIFEST.json',m);put(directory/'REQUEST.json',full['request_bytes'])
    names={context.LOCAL_BINDING:braw,context.CAPTURE:full['training_capture_bytes'],
        native.original.FRESH_SNAPSHOT:full['snapshot_bytes'],
        'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json':full['population_bytes'],
        'plans/SOURCE_EVALUATION_PLAN.json':full['plan_bytes'],'plans/PRIOR_EVALUATION_OUTCOMES.json':full['prior_bytes']}
    for name,raw in names.items():put(root/name,raw)
    put(local,full['local_bytes'],writable=True)
    ledger=SimpleNamespace(raw_state_bytes=None)
    def state():ledger.raw_state_bytes=full['global_bytes'];return json.loads(ledger.raw_state_bytes)
    ledger.state=state
    process=json.loads(full['native_facts_bytes'])
    monkeypatch.setattr(native.time,'time',lambda:full['now_epoch'])
    v=native.resource_inputs(root,directory/'REQUEST.json',ledger,process)
    for key in ('global_bytes','local_bytes','snapshot_bytes','training_capture_bytes','population_bytes','plan_bytes','prior_bytes'):
        assert v[key]==full[key]
    put(root/context.LOCAL_BINDING,dict(canonical_LOCAL_ledger=str(tmp_path/'other')))
    with pytest.raises(PermissionError):native.resource_inputs(root,directory/'REQUEST.json',ledger,process)


def test_budget_now_runs_full_CD_join_on_original_process_reader(full,monkeypatch,tmp_path):
    rp=tmp_path/'REQUEST.json';put(rp,full['request_bytes']);calls=[]
    monkeypatch.setattr(native.identity,'observe_current',lambda *args,**kw:calls.append(args)or json.loads(full['native_facts_bytes']))
    monkeypatch.setattr(native,'resource_inputs',lambda *args:full)
    proof,process=native.budget_now(tmp_path,rp,object())
    assert len(calls)==1 and proof['all_remaining_costs_fit']and not proof['model_permission']
    assert proof['allocation']['native_handle']==process['native_handle']
    full['snapshot_bytes']=a.raw(dict(json.loads(full['snapshot_bytes']),GLOBAL_sha256='0'*64))
    with pytest.raises(PermissionError):native.budget_now(tmp_path,rp,object())


def test_CD_status_reaches_existing_exact_qualification_directory(tmp_path,monkeypatch):
    m,r,v=allocation();root=tmp_path/v['run_uuid'];directory=root/'pbs/alloc-01'
    put(root/'RUN_MANIFEST.json',m);put(directory/'REQUEST.json',r)
    v['request_sha256']=a.sha(a.raw(r));put(directory/'ALLOCATION_ADMISSION.json',v)
    monkeypatch.setattr(qualification.Core_native,'current_handle',lambda:v['job_id'])
    monkeypatch.setattr(qualification.socket,'gethostname',lambda:v['hostname'])
    monkeypatch.setattr(qualification.time,'time',lambda:v['allocation_start_epoch']+5)
    monkeypatch.setattr(qualification,'namespace_submission',lambda *args:None)
    called=[];monkeypatch.setattr(qualification.Core_native,'current_identity',lambda *args,**kw:called.append(kw))
    assert qualification.current_directory(root,v)==directory
    assert called==[dict(qualified_model_readback=True)]
    put(directory/'NATIVE_PROCESS_FINAL.json',dict(status='FAILED'))
    with pytest.raises(PermissionError):qualification.current_directory(root,v)
