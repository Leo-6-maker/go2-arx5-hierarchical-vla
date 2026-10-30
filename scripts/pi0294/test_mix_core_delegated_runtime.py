"""Original runtime interfaces with actual temp onceIO, no neural/native GO.

Core account/device/unit/qualification/model are explicit fixtures. The local
file store, immutable bytes, original onceIO receipts and process PID/boot are
real CPU-side observations; no connection, CUDA, model or physics is used.
"""
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import pytest

from scripts.pi0294.dec1.owner_transport import UnknownMutation
from scripts.pi0294.test_v5_core_full_admission import full,captured_global
from scripts.pi0294.test_mix_core_delegation import activated,terminal_inputs
from scripts.pi0294.pv1 import mix_accounting as a,mix_core_accounting as c
from scripts.pi0294.pv1 import mix_core_delegation as d,mix_core_delegation_store as store
from scripts.pi0294.pv1 import mix_core_delegated_ledger as local,mix_core_delegated_control as control
from scripts.pi0294.pv1 import evaluation_core_ledger as old,evaluation_core_native as native,evaluation_core_runtime as runtime
from scripts.pi0294.pv1 import mix_core_identity as identity,successor_cd as successor


def put(path,value,*,mutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_bytes(value if type(value)is bytes else a.raw(value));path.chmod(0o600 if mutable else 0o400)


@pytest.fixture
def running(full,tmp_path,monkeypatch,request):
    v=deepcopy(full);m,r,p,g,l,snap=(json.loads(v[k]) for k in ('manifest_bytes','request_bytes','native_facts_bytes','global_bytes','local_bytes','snapshot_bytes'))
    parent=tmp_path/'campaign';root=parent/'neural-children'/r['run_uuid'];physical=parent/'children'/r['run_uuid']
    localpath=parent/'control'/'resident'/'LOCAL_RESOURCE_LEDGER.private.json'
    bindings={'plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json':dict(physical_peer_root=str(physical)),
              local.LOCAL_BINDING:dict(canonical_LOCAL_ledger=str(localpath))}
    for name,value in bindings.items():m['execution']['sealed_inputs'][name]=a.sha(a.raw(value))
    m['output_root']=str(root);r['run_manifest_sha256']=a.sha(a.raw(m))
    start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    native_pid=os.getpid();native_start=start
    if getattr(request,'param',None)=='separate-native':
        # Separate real CPU process identities; GPU/unit readings remain fixtures.
        child=subprocess.Popen([sys.executable,'-B','-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
        def close_child():
            child.stdin.close();child.wait(timeout=3)
        request.addfinalizer(close_child)
        native_pid=child.pid;native_start=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    p.update(pid=native_pid,proc_start=native_start,boot_id=boot,native_handle=c.process_handle(boot,native_pid,native_start),
             request_sha256=a.sha(a.raw(r)),run_manifest_sha256=a.sha(a.raw(m)))
    p['native_unit_properties']['MainPID']=str(p['pid'])
    for obj in (g,l):
        row=obj[c.FIELD][r['request_id']]
        row.update({k:p[k] for k in ('pid','proc_start','boot_id','native_handle')})
        row.update(request_sha256=p['request_sha256'],run_manifest_sha256=p['run_manifest_sha256'],native_bind_proof_sha256=c.digest(p))
    snap.update(GLOBAL_sha256=a.sha(a.raw(g)),LOCAL_sha256=a.sha(a.raw(l)))
    for key,value in [('manifest_bytes',m),('request_bytes',r),('native_facts_bytes',p),('global_bytes',g),('local_bytes',l),('snapshot_bytes',snap)]:v[key]=a.raw(value)
    result=activated(v,tmp_path);body=json.loads(result['grant_bytes'])['body'];m=body['manifest'];r=body['request'];p=body['admission']['allocation']['native_process']
    rp=root/'pbs/alloc-01/REQUEST.json';put(root/'RUN_MANIFEST.json',m);put(rp,r)
    for name,value in bindings.items():put(root/name,value)
    for peer in (root,physical):
        lease=peer/'own-lease.json';put(lease,dict(status='ACTIVE',run_uuid=root.name,authority_release=successor.TASK_SOURCE,deadline_epoch=r['deadline_epoch']),mutable=True)
        put(peer/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
    put(root/'OWNER_HEARTBEAT.json',dict(run_uuid=root.name,observed_epoch=1207.,canonical_GLOBAL_is_fresh=False))
    put(localpath,result['local_bytes'],mutable=True)
    put(root/'NATIVE_CORE_PROCESS_BOUND.json',dict(request_id=r['request_id'],request_sha256=a.sha(a.raw(r)),native_handle=p['native_handle'],finite_delegation_grant_sha256=a.sha(result['grant_bytes'])))
    put(root/'evidence'/store.GRANT,result['grant_bytes']);put(root/'evidence'/store.STATE,result['state_bytes'],mutable=True)
    put(root/'evidence'/local.ACTIVATION,local.activation_bundle(result['grant_bytes'],now_epoch=1206.,**result['activation_args']))
    put(root/'evidence'/control.FILE,control.initial(result['grant_bytes'],adopted_global_bytes=result['global_bytes']),mutable=True)
    clock=[1207.];observations=[]
    monkeypatch.setattr(local,'validate',lambda *args:(m,))
    monkeypatch.setattr(identity,'current_owner',lambda *args,**kw:dict(pid=os.getpid(),proc_start=start))
    monkeypatch.setattr(local.time,'time',lambda:clock[0])
    def observe(*args):
        observations.append(clock[0]);value=deepcopy(p);value['observed_epoch']=clock[0];return value
    monkeypatch.setattr(local.process_reader,'observe',observe)
    obj=old.Ledger(root,rp)
    return SimpleNamespace(root=root,rp=rp,physical=physical,localpath=localpath,ledger=obj,clock=clock,
                           value=result,body=body,request=r,manifest=m,process=p,observations=observations)


def test_original_factory_mode_runs_model_query_candidate_and_RPC_metrics_without_remote(running,monkeypatch):
    v=running;obj=v.ledger;assert isinstance(obj,local.Ledger)
    monkeypatch.setattr(old.Ledger,'_rpc',lambda *args,**kw:pytest.fail('NO_REMOTE_RPC_ALLOWED'))
    proof,process=native.budget_now(v.root,v.rp,obj)
    assert proof['canonical_GLOBAL_is_fresh'] is False and proof['canonical_GLOBAL_observed_epoch']==1205.
    assert proof['original_admission']['dimensions']['PBS_submissions']['matrix']==3
    assert process['pid']==os.getpid() and not proof['model_permission']
    obj.charge('model_loads',1);obj.charge('generation_queries',1)
    from scripts.pi0294.pv1.neural import record_candidate
    record_candidate(obj,'C50',dict(checkpoint_manifest_sha256=v.request['checkpoint_manifest_sha256'],global_step=1024))
    from scripts.pi0294.rpc_metrics import SessionMetrics
    metrics=SessionMetrics(maximum=150000);metrics.operations=80;metrics.settle(obj,1.25)
    state=obj.state();assert state['RPC_calls']==80 and state['CPU_s']==1.25
    assert state['delegated_effects']['row']['charged_query_attempts']==1
    assert not hasattr(obj,'raw_state_bytes') and not (v.root/'evidence/ACTUAL_REMOTE_GLOBAL_SNAPSHOT.private.json').exists()
    assert len(list((v.root/'evidence').glob('DELEGATED-EFFECT-*.RECEIPT.json')))==2
    assert len(list((v.root/'evidence').glob('DELEGATED-CONTROL-*.RECEIPT.json')))==2
    with pytest.raises(PermissionError,match='NO_FAKE_FRESH_GLOBAL'):
        native.resource_inputs(v.root,v.rp,obj,process)


@pytest.mark.parametrize('kind',range(6))
def test_guard_rejects_actual_local_unknown_closedlease_stop_grant_or_orphan(running,kind):
    v=running
    if kind==0:
        current=json.loads(v.localpath.read_bytes());current['unknown_effects']=['UNKNOWN'];put(v.localpath,current,mutable=True)
    if kind==1:
        lp=v.physical/'own-lease.json';current=json.loads(lp.read_bytes());current['status']='CLOSED';put(lp,current,mutable=True)
    if kind==2:put(v.physical/'SOURCE_PHYSICS_STOP.json',{})
    if kind==3:
        gp=v.root/'evidence'/store.GRANT;value=json.loads(gp.read_bytes());value['body']['limits']['model_loads']=3;put(gp,value)
    if kind==4:put(v.root/'evidence'/('DELEGATED-EFFECT-'+'a'*64+'.INTENT.json'),{})
    if kind==5:put(v.root/'evidence'/('DELEGATED-CONTROL-'+'a'*64+'.INTENT.json'),{})
    with pytest.raises(PermissionError):v.ledger.charge('model_loads',1)
    assert json.loads((v.root/'evidence'/store.STATE).read_bytes())['sequence']==0


def test_closeout_CPU_survives_STOP_without_live_GPU_and_is_not_paid_twice(running,monkeypatch):
    v=running;put(v.physical/'STOP.json',{})
    monkeypatch.setattr(local.process_reader,'observe',lambda *args:pytest.fail('FINALLY_MUST_NOT_REQUIRE_LIVE_GPU'))
    receipt=v.ledger.cpu(12.5)
    assert receipt['canonical_CPU_charged'] is False
    assert control.read_confirmed(v.root,v.value['grant_bytes'])['final_neural_CPU_s']==12.5
    with pytest.raises(UnknownMutation):v.ledger.cpu(12.5)
    assert json.loads((v.root/'evidence'/store.STATE).read_bytes())['row'].get('charged_CPU_s',0)==0


def test_resident_readonly_observer_uses_same_fresh_proof_without_startup_heartbeat_cycle(running):
    v=running;(v.root/'OWNER_HEARTBEAT.json').unlink()
    observer=local.Ledger(v.root,v.rp,resident_observer=True)
    assert observer.live_proof()['status']=='PASS_EXACT_FINITE_CORE_DELEGATED_RESERVATION'
    with pytest.raises(PermissionError,match='ONLY_BOUND_NEURAL'):observer.charge('model_loads',1)
    with pytest.raises(PermissionError,match='OBSERVER_CANNOT'):
        with observer.transaction() as state:state['CPU_s']+=1


def resident_admission(v,monkeypatch):
    from scripts.pi0294.pv1 import pre_model_qualification as q
    admission=deepcopy(v.body['admission']['allocation'])
    put(v.rp.parent/'ALLOCATION_ADMISSION.json',admission)
    monkeypatch.setattr(q.socket,'gethostname',lambda:admission['hostname'])
    monkeypatch.delenv('PV1_CORE_NATIVE_GPU_HANDLE',raising=False)
    monkeypatch.delenv('PBS_JOBID',raising=False)
    observer=local.Ledger(v.root,v.rp,resident_observer=True)
    return q,admission,observer


@pytest.mark.parametrize('running',['separate-native'],indirect=True)
def test_resident_reads_stored_admission_after_live_observation_without_becoming_GPU(running,monkeypatch):
    v=running;q,admission,observer=resident_admission(v,monkeypatch);seen=[]
    assert admission['native_process']['pid']!=os.getpid()
    v.clock[0]+=601
    def qualified(root,directory,allocation):
        # Raw 600s/80-cycle verification is a separate CPU fixture here.
        seen.append(allocation)
        assert root==v.root and directory==v.rp.parent and allocation==admission
        assert q.Core_native.current_handle()is None
        return dict(status='CPU_RAW_QUALIFICATION_FIXTURE')
    monkeypatch.setattr(q,'require',qualified)
    assert observer.live_proof()['allocation']!=admission
    assert q.require_resident(v.root,observer)['status']=='CPU_RAW_QUALIFICATION_FIXTURE'
    assert seen==[admission] and observer._confirmed()['sequence']==0
    with pytest.raises(PermissionError,match='CURRENT_ALLOCATION'):
        q.require_current(v.root,admission)


@pytest.mark.parametrize('running',['separate-native'],indirect=True)
@pytest.mark.parametrize('fault',['STOP','closed_lease','dead_unit','changed_admission','terminal','foreign_owner','wrong_root'])
def test_resident_qualification_keeps_original_live_and_identity_fences(running,monkeypatch,fault):
    v=running;q,admission,observer=resident_admission(v,monkeypatch)
    monkeypatch.setattr(q,'require',lambda *args:pytest.fail('NO_QUALIFICATION_OR_RELEASE_ON_REJECTED_IDENTITY'))
    root=v.root
    if fault=='STOP':put(v.physical/'STOP.json',{})
    if fault=='closed_lease':
        p=v.physical/'own-lease.json';value=json.loads(p.read_bytes());value['status']='CLOSED';put(p,value,mutable=True)
    if fault=='dead_unit':
        p=deepcopy(v.process);p['native_unit_properties']['MainPID']='0'
        monkeypatch.setattr(local.process_reader,'observe',lambda *args:p)
    if fault=='changed_admission':
        admission['source_sha']='b'*40;put(v.rp.parent/'ALLOCATION_ADMISSION.json',admission)
    if fault=='terminal':put(v.rp.parent/'NATIVE_PROCESS_FINAL.json',{})
    if fault=='foreign_owner':monkeypatch.setattr(identity,'current_owner',lambda *args,**kw:dict(pid=os.getpid()+100,proc_start=observer.actor_start))
    if fault=='wrong_root':root=v.physical
    with pytest.raises(PermissionError):q.require_resident(root,observer)
    assert observer._confirmed()['sequence']==0


def test_control_lost_reply_keeps_actual_receipt_and_never_repeats_metrics(running,monkeypatch):
    v=running;original=control.once.commit
    def lost(*args,**kwargs):original(*args,**kwargs);raise TimeoutError('ACTUAL_CAS_SIMULATED_LOST_REPLY')
    monkeypatch.setattr(control.once,'commit',lost)
    with pytest.raises(UnknownMutation):
        with v.ledger.transaction() as state:state['CPU_s']+=1;state['RPC_calls']+=80
    confirmed=control.read_confirmed(v.root,v.value['grant_bytes'])
    assert confirmed['CPU_s']==1 and confirmed['RPC_calls']==80
    # No automatic retry: the original RPC SessionMetrics latch remains set.
    assert len(confirmed['events'])==1


def test_control_missing_receipt_blocks_next_query(running,monkeypatch):
    v=running;v.ledger.charge('model_loads',1);original=control.once._exclusive
    def fail(path,raw,mode=0o400):
        if str(path).endswith('.RECEIPT.json'):raise OSError('MISSING_RECEIPT')
        return original(path,raw,mode)
    monkeypatch.setattr(control.once,'_exclusive',fail)
    with pytest.raises(UnknownMutation):
        with v.ledger.transaction() as state:state['RPC_calls']+=80
    with pytest.raises(FileNotFoundError):v.ledger.charge('generation_queries',1)


def test_terminal_merges_original_neural_cost_and_prepaid_metrics_only_once(running):
    v=running;v.ledger.charge('model_loads',1);v.ledger.charge('generation_queries',1)
    with v.ledger.transaction() as state:state['CPU_s']+=2;state['RPC_calls']+=80
    v.ledger.cpu(12.5)
    current=(v.root/'evidence'/store.STATE).read_bytes();v.value['state_bytes']=current
    args=terminal_inputs(v.value)
    result=control.merge_terminal(v.root,v.value['grant_bytes'],current,v.value['global_bytes'],v.value['local_bytes'],**args)
    rid=v.request['request_id'];g=result['proposed_global'];l=result['proposed_LOCAL']
    assert g[c.FIELD][rid]['status']==c.STATES[3]
    assert g['V5_CD_prepaid_control_actual_metrics'][rid]==dict(CPU_s=2,RPC_calls=80)
    assert g[d.FIELD][rid]['terminal_control_sha256']==l[d.FIELD][rid]['terminal_control_sha256']
    with pytest.raises(PermissionError,match='PREVIOUSLY_MERGED'):
        control.merge_terminal(v.root,v.value['grant_bytes'],current,a.raw(g),a.raw(l),**args)


def test_exact_selector_does_not_change_old_MODE_or_accept_unknown_mode():
    assert local.selected({}, {}) is False
    with pytest.raises(PermissionError):local.selected({'execution':{'global_accounting_mode':'OTHER'}},{})
    with pytest.raises(PermissionError):local.selected({'execution':{'global_accounting_mode':d.MODE}},{})
    assert type(old.Ledger.__new__(old.Ledger)) is old.Ledger


def test_original_runtime_crosses_two_releases_qualification_model_and_serve_with_finite_store(running,monkeypatch):
    """Run the real orchestration; GPU/model/600s observations are CPU fixtures."""
    from scripts.pi0294.pv1 import neural
    v=running;directory=v.rp.parent;order=[];label='C50-DEV-N294101'
    def release_file(name):
        put(directory/name,dict(schema='PV1_CORE_EXISTING_NATIVE_PHASE_RELEASE_V1',run_uuid=v.root.name,
            request_id=v.request['request_id'],request_sha256=a.sha(v.rp.read_bytes()),source_sha=v.request['source_sha'],
            job_id=v.process['native_handle'],phase=name,physics_authorized=False,new_optimizer_updates=0,observed_epoch=v.clock[0]))
    release_file('START_NEURAL.json')
    monkeypatch.setattr(runtime,'validate',lambda *args:(v.manifest,))
    monkeypatch.setattr(native,'validate',lambda *args:(v.manifest,))
    monkeypatch.setattr(runtime.identity,'observe_current',lambda *args,**kw:dict(v.process,observed_epoch=v.clock[0]))
    monkeypatch.setattr(runtime.resource,'setrlimit',lambda *args:None)
    def qualify(root,path,allocation,guard):
        order.append('qualification');guard();v.clock[0]+=600
        put(v.root/'OWNER_HEARTBEAT.json',dict(run_uuid=v.root.name,observed_epoch=v.clock[0],canonical_GLOBAL_is_fresh=False))
        release_file('CORE_MODEL_RELEASE.json');guard()
        return dict(status='CPU_FIXTURE_OF_ORIGINAL_600S_80CYCLE_RESULT',receipt_sha256='e'*64)
    monkeypatch.setattr(runtime,'qualify_transport',qualify)
    monkeypatch.setattr(runtime,'require_current',lambda *args:dict(status='CPU_QUALIFICATION_FIXTURE'))
    monkeypatch.setattr(runtime,'prepare_environment',lambda *args:None)
    monkeypatch.setattr(runtime,'configure_torch',lambda *args:None)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda *args:None))
    monkeypatch.setattr(neural,'require_native_device',lambda *args:order.append('device'))
    checkpoint=v.root/'model_views/C50';checkpoint.mkdir(parents=True)
    put(checkpoint/'CHECKPOINT_COMPLETE.json',dict(CPU_fixture=True))
    put(v.root/'plans/EVALUATION_CHECKPOINT.private.json',dict(full_resume_files_present=False,
        checkpoint_manifest_sha256=v.request['checkpoint_manifest_sha256'],checkpoint=str(checkpoint)))
    real_sha=runtime.sha
    monkeypatch.setattr(runtime,'sha',lambda path:v.request['checkpoint_manifest_sha256'] if Path(path)==checkpoint/'CHECKPOINT_COMPLETE.json' else real_sha(path))
    model_identity=dict(checkpoint_manifest_sha256=v.request['checkpoint_manifest_sha256'],global_step=1024)
    monkeypatch.setattr(runtime,'checkpoint_identity',lambda *args:model_identity)
    def load(*args,**kwargs):
        order.append('model');state=v.ledger.state()
        assert state['delegated_effects']['row']['charged_model_loads']==1
        return SimpleNamespace(model=object())
    monkeypatch.setattr(runtime.model,'load',load)
    monkeypatch.setattr(runtime.model,'unload',lambda *args:order.append('unload'))
    monkeypatch.setattr(runtime,'parameter_inventory',lambda *args:{})
    put(v.root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json',{})
    put(v.root/'plans/SOURCE_EVALUATION_PLAN.json',v.value['inputs']['plan_bytes'])
    put(v.root/'plans/PRIOR_EVALUATION_OUTCOMES.json',[])
    panel=json.loads(v.value['inputs']['plan_bytes'])['panels'][label]
    monkeypatch.setattr(runtime,'session_spec',lambda *args:panel)
    put(v.root/'plans/PV1-PREMODEL-ALLOCATION-01/REGISTRY.json',{})
    def serve(root,labels,instance,allocation,ledger,guard,**kwargs):
        order.append('serve');assert isinstance(ledger,local.Ledger) and kwargs['max_queries']==1920
        guard();ledger.charge('generation_queries',1)
    monkeypatch.setattr(runtime,'serve',serve)
    result=runtime.run(v.root,v.rp)
    assert order==['qualification','device','model','serve','unload']
    assert result['status']=='PV1_DEV_NATIVE_PANELS_TERMINAL_ACTUAL_OUTCOMES_AUDITED_SEPARATELY'
    proof=json.loads((directory/'POST_QUALIFICATION_NATIVE_RESOURCE_PROOF.json').read_bytes())
    assert proof['canonical_GLOBAL_observed_epoch']==1205. and proof['local_observed_epoch']==1807.
    assert proof['canonical_GLOBAL_is_fresh'] is False
    assert json.loads((directory/'NATIVE_PROCESS_FINAL.json').read_bytes())['new_optimizer_updates']==0
    assert control.read_confirmed(v.root,v.value['grant_bytes'])['final_neural_CPU_s'] is not None


def test_original_Journal_unknown_blocks_later_token_even_with_confirmed_local_receipt(running,monkeypatch):
    v=running;original=store.once.commit
    def lost(*args,**kwargs):original(*args,**kwargs);raise TimeoutError('ACTUAL_RECEIPT_LOST_RETURN')
    monkeypatch.setattr(store.once,'commit',lost)
    with pytest.raises(UnknownMutation):v.ledger.charge('model_loads',1)
    assert store.read_confirmed(v.root,expected_grant_sha256=a.sha(v.value['grant_bytes']))['state']['sequence']==1
    monkeypatch.setattr(store.once,'commit',original)
    reloaded=old.Ledger(v.root,v.rp)
    with pytest.raises(UnknownMutation,match='PRIOR_WRITE'):reloaded.charge('generation_queries',1)
    assert reloaded.state()['delegated_effects']['sequence']==1


@pytest.mark.parametrize('change',range(4))
def test_control_transaction_cannot_change_budget_or_overrun_prepaid_CPU_or_query_counts(running,change):
    v=running
    with pytest.raises((PermissionError,UnknownMutation,ValueError)):
        with v.ledger.transaction() as state:
            if change==0:state['CPU_s']=6600.01
            if change==1:state['RPC_calls']=150001
            if change==2:state['canonical_GLOBAL_is_fresh']=True
            if change==3:state['CPU_s']=-1
    assert control.read_confirmed(v.root,v.value['grant_bytes'])['CPU_s']==0
