"""Real local subprocess/Journal heartbeat with synthetic registered roots.

No native scheduler, model, train, account authentication or external dispatch.
"""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import signal
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from scripts.pi0292.durable import read,write
from . import vanda_campaign as c,vanda_resident_transport as r
from .owner_transport import Journal,UnknownMutation,ReadOnlyFailure
from .test_vanda_campaign import fixture,raw

def setup(tmp_path):
    root=tmp_path/'parent/children/fixture-run';directory=root/'pbs/alloc-01';directory.mkdir(parents=True)
    (root/'control').mkdir();control=root.parent/'new-resident-control';control.mkdir()
    source='a'*40;write(directory/'REQUEST.json',dict(source_sha=source))
    write(root/'RUN_MANIFEST.json',dict(code=dict(vla_commit=source,capsule_relative='capsule/code'),execution=dict(control_execution_platform='VANDA_A40')))
    registry=tmp_path/'registry.json';write(registry,dict(entries=[dict(role='CONTROL',path=str(control)),dict(role='GPU',path=str(root))]))
    lease=tmp_path/'GPU-lease.json';write(lease,dict(status='ACTIVE',run_uuid=root.name));write(root/'OUTPUT_BINDING.private.json',dict(lease=str(lease),uid=os.getuid(),role='GPU',registry=str(registry),run_manifest_sha256=c.sha((root/'RUN_MANIFEST.json').read_bytes())))
    m=dict(code=dict(vla_commit=source),execution=dict(physical_site='VANDA',execution_class='CPU_CONTROL_ONLY_NO_CUDA',agent_liveness_required=False))
    write(control/'RUN_MANIFEST.json',m);manifest_sha=c.sha((control/'RUN_MANIFEST.json').read_bytes())
    cl=tmp_path/'control-lease.json';write(cl,dict(status='ACTIVE',run_uuid=control.name))
    write(control/'OUTPUT_BINDING.private.json',dict(run_manifest_sha256=manifest_sha,uid=os.getuid(),lease=str(cl),registry=str(registry),role='CONTROL'))
    code=root/'capsule/code';repo=Path(__file__).resolve().parents[3]
    # Real child Python imports the exact CPU code copied to its fixture capsule.
    for folder in ('pi0291_arx5_data','pi0292','pi0294','iteration','operations'):
        shutil.copytree(repo/'scripts'/folder,code/'scripts'/folder,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts/__init__.py').touch()
    identity=dict(run_uuid=root.name,job_id='14.stdct-mgmt-02',source_sha=source,request_sha256=c.sha((directory/'REQUEST.json').read_bytes()))
    folder=tmp_path/'journal';folder.mkdir();journal=Journal(folder,identity)
    h,d,a,q=fixture();now=time.time();a['observed_epoch']=q['observed_epoch']=now
    grant=(repo/c.handoff.PLAN).read_bytes()
    view=c.observe(grant_bytes=grant,global_ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),authority_readback=a,queue_readback=q,now_epoch=now)
    state=dict(observed_epoch=now,view=view,authority=dict(global_ledger_base64=base64.b64encode(raw(h)).decode(),DEC1_ledger_base64=base64.b64encode(raw(d)).decode()))
    args=dict(journal=journal,remote_root=root,request_directory=directory,code_root=code,control_root=control,control_manifest_sha256=manifest_sha,python=sys.executable)
    return args,state,h,control

def create(args,**kw):
    with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(r.socket,'gethostname',return_value='stdct-login-01'):
        return r.VandaResidentRPC(**dict(args,**kw))

def beat(rpc):
    return rpc('vanda','IGNORED_FOR_TYPED_HEARTBEAT',dict(TARGET=rpc.root,DIRECTORY=rpc.directory,JOB=rpc.journal.identity['job_id'],VIEW=rpc.observation['view']),'source-A40-same-owner-existing-job-heartbeat')

def test_real_local_native_heartbeat_has_no_ssh_and_original_watchdog(tmp_path):
    args,state,h,control=setup(tmp_path);rpc=create(args);rpc.bind_observation(state)
    out=beat(rpc)
    assert list(rpc.argv)==['vanda'] and rpc.argv['vanda']==(sys.executable,'-')
    assert out['heartbeat']['owner']=='EXISTING_LOCAL_PC' # same logical science owner
    assert out['heartbeat']['control_execution_platform']=='VANDA_A40' and out['heartbeat']['agent_liveness_required'] is False
    assert out['campaign_view_status']==c.PASS and out['vanda_campaign_view']['Hopper_current_queue_observed'] is False
    assert read(rpc.journal.state_path)['status']=='CONFIRMED'
    c.validate_owner_heartbeat(out['heartbeat'],ledger=h,now_epoch=out['heartbeat']['observed_epoch']+300)
    with pytest.raises(PermissionError):c.validate_owner_heartbeat(out['heartbeat'],ledger=h,now_epoch=out['heartbeat']['observed_epoch']+300.001)
    for site,label in [('hopper','source-A40-current-unfiltered-native-hopper'),('vanda','pretend-fresh-hopper-view')]:
        with pytest.raises(PermissionError):rpc(site,'raise AssertionError()',{},label)

def test_local_lost_heartbeat_reply_uses_exact_readback_without_second_publication(tmp_path):
    args,state,h,_=setup(tmp_path);calls=[]
    def lost(argv,program,*,timeout_s):
        calls.append(program);result=r.local_invoke(argv,program,timeout_s=timeout_s)
        if ' import publish\n' in program:raise TimeoutError('CPU_FIXTURE_LOST_LOCAL_RETURN')
        return result
    rpc=create(args,transport=lost);rpc.bind_observation(state);out=beat(rpc)
    assert len(calls)==2 and sum(' import publish\n' in s for s in calls)==1
    assert read(rpc.journal.state_path)['status']=='CONFIRMED'
    assert len(list((Path(rpc.root)/'control/HEARTBEAT_PUBLICATION_INTENTS').glob('*.json')))==1
    c.validate_owner_heartbeat(out['heartbeat'],ledger=h,now_epoch=time.time())

def test_runtime_heartbeat_survives_real_accounting_progress_between_heartbeats(tmp_path):
    args,state,h,_=setup(tmp_path);rpc=create(args);rpc.bind_observation(state);out=beat(rpc)
    later=deepcopy(h);later['optimizer_updates']=later.get('optimizer_updates',0)+1
    later.setdefault('used',{})['optimizer_updates']=later['optimizer_updates'];later['revision']=later.get('revision',0)+1
    c.validate_owner_heartbeat(out['heartbeat'],ledger=later,now_epoch=time.time())
    later['unknown_effects']=['unresolved_effect']
    with pytest.raises(PermissionError):c.validate_owner_heartbeat(out['heartbeat'],ledger=later,now_epoch=time.time())

def test_existing_watch_survives_launcher_exit_and_returns_distinct_native_terminal(tmp_path):
    """Detached CPU fixture, original watch; no scheduler, model or auth call."""
    fixture_script=tmp_path/'detached_fixture.py'
    repo=Path(__file__).resolve().parents[3]
    fixture_script.write_text('import sys\nsys.path[:0]='+repr([str(repo),str(repo/'src')])+'\n'+'''
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1.cold_owner import watch
from scripts.pi0294.dec1 import vanda_campaign as c
from scripts.pi0294.dec1.test_vanda_campaign import fixture,terminal_fixture,raw
base=Path(sys.argv[1]);job='14.stdct-mgmt-02';ticks=0
for n,v in [('SOURCE_REQUEST.private.json',dict(request=dict(submit_count=14),directory=str(base))),('SOURCE_JOB_BOUND.private.json',dict(job_id=job)),('TRAIN_REGISTERED.private.json',dict(root=str(base)))]:write(base/n,v)
def fresh(*a,**kw):
    done=(base/'finish').exists();now=time.time()
    if done:h,d,authority,queue,terminal=terminal_fixture();terminal['observed_epoch']=now
    else:h,d,authority,queue=fixture();terminal=None
    authority.update(observed_epoch=now,global_sha256=c.sha(raw(h)),DEC1_sha256=c.sha(raw(d)));queue['observed_epoch']=now
    grant=Path(__import__('scripts.pi0294.dec1.test_vanda_campaign',fromlist=['x']).__file__).resolve().parents[3]/c.handoff.PLAN
    view=c.observe(grant_bytes=grant.read_bytes(),global_ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),authority_readback=authority,queue_readback=queue,terminal_readbacks=[terminal] if terminal else [],now_epoch=time.time())
    c.validate(view,ledger=h,DEC1_ledger=d,now_epoch=time.time())
    native=json.loads(terminal['stdout'])['Jobs'][job] if done else dict(Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007',job_state='Q')
    return dict(view=view,rows=dict(vanda=dict(jobs={} if done else {job:native},terminal_job=native if done else None,receipts={})))
def rpc(*a,**kw):
    global ticks
    ticks+=1;write(base/'heartbeat.json',dict(ticks=ticks,pid=os.getpid(),ppid=os.getppid(),CPU_fixture=True))
def record(name,value):write(base/(name+'.json'),value);return value
owner=SimpleNamespace(BASE=base,cold_rpc_bound=True,native=SimpleNamespace(T0=1792348196),read=read,record=record,update=write,
    fresh=fresh,previous=SimpleNamespace(rpc=rpc),HEARTBEAT='',auth=lambda secret:dict(CPU_fixture=True),release=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('NO_RELEASE_IN_Q_F_FIXTURE')))
result=watch(owner,None,sleep=lambda _:time.sleep(.05));write(base/'done.json',dict(result=result,CPU_fixture=True,pid=os.getpid()))
''')
    launcher=('import subprocess,sys,json; p=subprocess.Popen([sys.executable,'+repr(str(fixture_script))+','+repr(str(tmp_path))+'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=open('+repr(str(tmp_path/'fixture.stderr'))+',"w"),start_new_session=True); print(json.dumps(dict(pid=p.pid,launcher=__import__("os").getpid())))')
    isolated_env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    parent=subprocess.run([sys.executable,'-c',launcher],capture_output=True,text=True,timeout=5,check=True,env=isolated_env)
    identity=json.loads(parent.stdout);pid=identity['pid'];proc=Path('/proc')/str(pid)
    try:
        end=time.monotonic()+8;seen=None
        while time.monotonic()<end:
            path=tmp_path/'heartbeat.json'
            if path.exists():
                seen=read(path)
                if seen['ticks']>=2 and seen['ppid']!=identity['launcher']:break
            time.sleep(.02)
        assert seen and seen['ticks']>=2 and seen['ppid']!=identity['launcher'],(tmp_path/'fixture.stderr').read_text()
        assert proc.exists() and seen['pid']==pid
        (tmp_path/'finish').touch()
        while time.monotonic()<end and not(tmp_path/'done.json').exists():time.sleep(.02)
        result=read(tmp_path/'done.json')['result']
        assert result['status']=='NATIVE_TERMINAL_PRESERVED_NEEDS_ACTUAL_EFFECT_SETTLEMENT'
        assert result['native']['job_state']=='F' and result['native']['Exit_status']==0
        assert result['do_not_resubmit'] is True
    finally:
        if not(tmp_path/'done.json').exists() and proc.exists():os.kill(pid,signal.SIGTERM)

def test_typed_queue_failure_cannot_turn_effect_bearing_call_into_retry(tmp_path):
    diagnostic=dict(schema=c.QUEUE_FAILURE,queue_known=False,diagnostic=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=162,stdout='',stderr='PBS unavailable'))
    def failed(*a,**kw):return SimpleNamespace(returncode=75,stdout='',stderr=json.dumps(diagnostic))
    with pytest.raises(ReadOnlyFailure):r.local_invoke((sys.executable,'-'),'pass',timeout_s=30,runner=failed)
    (tmp_path/'typed-error').mkdir()
    journal=Journal(tmp_path/'typed-error',dict(run_uuid='fixture',job_id='1.stdct-mgmt-02',source_sha='a'*40,request_sha256='b'*64))
    calls=[]
    def invoke(*a):
        calls.append(1);return r.local_invoke((sys.executable,'-'),'pass',timeout_s=30,runner=failed)
    with pytest.raises(UnknownMutation):journal.call('MUTATION_ONCE_FIXTURE','MUTATION',invoke)
    assert len(calls)==1 and read(journal.state_path)['status']=='UNKNOWN'
    with pytest.raises(UnknownMutation):journal.call('MUTATION_ONCE_FIXTURE','MUTATION',invoke)
    assert len(calls)==1

@pytest.mark.parametrize('fault',['wrong_host','CUDA','wrong_python','expired_lease','different_source','agent_required','GPU_platform_missing','manifest_changed'])
def test_cannot_construct_resident_adapter_for_old_agent_or_unregistered_capsule(tmp_path,fault):
    args,state,h,control=setup(tmp_path)
    if fault=='wrong_host':
        with patch.object(r.socket,'gethostname',return_value='local-agent'),pytest.raises(PermissionError):r.VandaResidentRPC(**args)
        return
    if fault=='CUDA':
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(r.socket,'gethostname',return_value='stdct-login-01'),pytest.raises(PermissionError):r.VandaResidentRPC(**args)
        return
    if fault=='wrong_python':args['python']='/nonexistent/python'
    elif fault=='expired_lease':
        p=Path(read(control/'OUTPUT_BINDING.private.json')['lease']);v=read(p);v['status']='CLOSED';write(p,v)
    elif fault in ('different_source','agent_required'):
        p=control/'RUN_MANIFEST.json';m=read(p)
        if fault=='different_source':m['code']['vla_commit']='b'*40
        else:m['execution']['agent_liveness_required']=True
        write(p,m);args['control_manifest_sha256']=c.sha(p.read_bytes());b=read(control/'OUTPUT_BINDING.private.json');b['run_manifest_sha256']=args['control_manifest_sha256'];write(control/'OUTPUT_BINDING.private.json',b)
    elif fault=='GPU_platform_missing':
        p=Path(args['remote_root'])/'RUN_MANIFEST.json';m=read(p);m['execution'].clear();write(p,m)
    else:args['control_manifest_sha256']='0'*64
    with pytest.raises(PermissionError):create(args)

def test_original_owner_watch_can_only_bind_new_authorized_resident(tmp_path):
    args,state,h,_=setup(tmp_path);rpc=create(args)
    owner=SimpleNamespace(fresh=lambda:state,previous=SimpleNamespace())
    proof=dict(status='PASS_NEW_VANDA_REGISTERED_RESIDENT_OWNER',identity=args['journal'].identity,physical_control_platform='VANDA_A40',
        prior_owner_process_confirmed_absent=True,prior_GPU_lease_closed_and_financially_settled=True,
        old_Hopper_origin_fenced_not_claimed_closed=True,new_recovery_authority_adopted=True,agent_connection_required=False)
    r.bind_resident_owner(owner,rpc,validate_new_registration=lambda:proof)
    assert owner.fresh()==state and owner.previous.rpc is rpc and owner.resident_control_platform=='VANDA_A40'
    with pytest.raises(PermissionError):r.bind_resident_owner(owner,rpc,validate_new_registration=lambda:proof)

@pytest.mark.parametrize('fault',['no_view','grant','producer','dropped_migration','queue_future','authority_changed'])
def test_new_heartbeat_kind_cannot_be_spoofed_by_renaming_legacy_receipt(tmp_path,fault):
    args,state,h,_=setup(tmp_path);rpc=create(args);rpc.bind_observation(state);out=beat(rpc)
    intent=read(rpc.journal.state_path)['intent']
    if fault=='no_view':out.pop('vanda_campaign_view')
    elif fault=='grant':out['heartbeat']['grant_sha256']=c.handoff.GRANT_SHA
    elif fault=='producer':out['heartbeat']['agent_liveness_required']=True
    elif fault=='dropped_migration':out['heartbeat'].pop('migration_marker_sha256')
    elif fault=='queue_future':out['vanda_campaign_view']['queue_observed_epoch']=time.time()+60
    else:out['heartbeat']['control_authority_proof']={}
    from .owner_transport import checked_heartbeat
    with pytest.raises(PermissionError):checked_heartbeat(out,intent,now=time.time())
