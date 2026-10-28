"""Actual temporary original Journal; future owner/PBS/resource fixtures.

No native owner is registered or started, and no model/optimizer/qsub runs.
"""
import json,time
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.test_minimal_native_admission import actual,packet
from scripts.pi0294.generalization import minimal_native_owner as o,minimal_native_admission as a
from scripts.pi0294.generalization import minimal_execution as e,minimal_session_gate as gates,minimal_effects as effects
from scripts.pi0294.dec1.owner_transport import UnknownMutation
from scripts.pi0292.durable import read


@pytest.fixture
def owner(packet,tmp_path):
    p=deepcopy(packet);now=time.time();p['now_epoch']=now;p['queue_proof']['observed_epoch']=now
    site=json.loads(p['site_bytes']);site['observed_epoch']=site['quota']['observed_epoch']=now;p['site_bytes']=e.raw(site)
    root=tmp_path/json.loads(p['manifest_bytes'])['run_uuid'];base=root/'control/resident-owner';base.mkdir(parents=True)
    req=json.loads(p['request_bytes']);d=root/'pbs'/('alloc-'+str(req['submit_count']).zfill(2));d.mkdir(parents=True)
    (d/'REQUEST.json').write_bytes(p['request_bytes']);(d/'REQUEST.json').chmod(0o400)
    v=o.Owner.__new__(o.Owner);v.adapter=e;v.root=root;v.BASE=base;v.global_path=root.parent/'HOPPER_RESOURCE_LEDGER.json'
    v.global_path.write_bytes(p['global_bytes']);lease=root/'LEASE.json';lease.write_bytes(e.raw(dict(status='ACTIVE')))
    v.binding=dict(lease=str(lease));v.current=dict(rows=dict(vanda=dict(jobs={p['job_id']:json.loads(p['native_job_bytes'])})),
        view=dict(status='PASS_V5_CURRENT_UNFILTERED_QUEUE',observed_epoch=now,unknown_campaign_jobs=[],
            Hopper_current_queue_observed=False,unavailable_Hopper_is_fresh_or_empty=False,complete_account_queue=p['queue_proof']))
    o.once(base/'SOURCE_REQUEST.private.json',dict(request=req,request_sha256=e.sha(p['request_bytes']),directory=str(d)))
    o.once(base/'SOURCE_JOB_BOUND.private.json',dict(job_id=p['job_id']))
    v.running=o.mature.new_journal(base,'running-rpc',dict(run_uuid=root.name,job_id=p['job_id'],
        request_sha256=e.sha(p['request_bytes']),source_sha=req['source_sha']))
    phase=gates.plan()[0];counts=dict(optimizer_updates=0,generation_queries=0,model_loads=0)
    ready=dict(schema=gates.READY,ticket_version=gates.VERSION,**phase,job_id=p['job_id'],request_id=req['request_id'],
        submission_nonce=req['submission_nonce'],request_sha256=e.sha(p['request_bytes']),source_sha=req['source_sha'],
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,session_counters=counts,observed_epoch=now-1)
    value=dict(schema=effects.SCHEMA,platform_training_version=req['platform_training_version'],request_sha256=ready['request_sha256'],job_id=p['job_id'],source_sha=req['source_sha'],
        active_session=phase['session_id'],pending_updates=[],unknown_effects=[],sessions={phase['session_id']:counts})
    slot=gates.directory(d,phase['session_id'],phase['phase']);slot.mkdir(parents=True);o.once(slot/'READY.json',ready)
    (root/'A40_NATIVE_EFFECT_LEDGER.json').write_bytes(e.raw(value))
    calls=[]
    def native_material(**kwargs):
        calls.append(kwargs);args=dict(p,ready_bytes=(slot/'READY.json').read_bytes(),effect_bytes=e.raw(value))
        return a.validate(**args),dict(explicit_CPU_fixture_not_native=True)
    v.native_material=native_material
    return v,p,slot,calls


def test_original_Journal_releases_only_next_finite_ticket_and_never_replays(owner):
    v,p,slot,calls=owner
    assert v.release_science_session({},secret=None)is True
    saved=read(slot/'RELEASE.json');assert saved['ready_sha256']==e.sha((slot/'READY.json').read_bytes())
    assert v.running._state()['status']=='CONFIRMED' and saved['native_proof']['whole_remaining_package_shortfalls']=={}
    before={str(f):f.read_bytes()for f in slot.iterdir()if f.is_file()}
    assert v.release_science_session({},secret=None)is False and len(calls)==1
    assert before=={str(f):f.read_bytes()for f in slot.iterdir()if f.is_file()}
    assert saved['native_proof']['permission_file_created']is False


def test_actual_lost_reply_after_release_keeps_UNKNOWN_and_does_not_republish(owner,monkeypatch):
    v,p,slot,calls=owner;real=o.once
    def lost(path,value):
        out=real(path,value)
        if Path(path).name=='RELEASE.json':raise TimeoutError('CPU_FIXTURE_REPLY_LOST_AFTER_WRITE')
        return out
    monkeypatch.setattr(o,'once',lost)
    with pytest.raises(UnknownMutation):v.release_science_session({})
    assert v.running._state()['status']=='UNKNOWN' and(slot/'RELEASE.json').is_file()
    before=(slot/'RELEASE.json').read_bytes();assert v.release_science_session({})is False
    assert (slot/'RELEASE.json').read_bytes()==before and len(calls)==1
    called=[]
    with pytest.raises(UnknownMutation):v.running.call('NO_NEW_WRITE','MUTATION',lambda *a:called.append(True))
    assert not called and v.running._state()['status']=='UNKNOWN'


def test_same_original_Journal_checks_typed_current_heartbeat_not_old_CD_flags(owner):
    v,p,slot,calls=owner
    out=v.heartbeat_rpc('vanda','',dict(JOB=p['job_id'],VIEW=v.current['view']),'CPU-fixture')
    assert out['schema']==o.HEARTBEAT and out['heartbeat']['scope']==e.VERSION
    assert out['heartbeat']['native_job_state']=='R'and v.running._state()['status']=='CONFIRMED'
    saved=(v.root/'control/OWNER_HEARTBEAT.json').read_bytes()
    g=json.loads(p['global_bytes']);g['pending_updates']=[{'PRESERVED':True}];v.global_path.write_bytes(e.raw(g))
    with pytest.raises(UnknownMutation):v.heartbeat_rpc('vanda','',dict(JOB=p['job_id'],VIEW=v.current['view']),'unknown')
    assert (v.root/'control/OWNER_HEARTBEAT.json').read_bytes()==saved
    assert json.loads(v.global_path.read_bytes())['pending_updates']==[{'PRESERVED':True}]


def test_actual_worker_inventory_does_not_follow_input_alias_or_hide_partial_files(tmp_path):
    folder=tmp_path/'training/arm/checkpoint-2';folder.mkdir(parents=True)
    (folder/'optimizer.pt').write_bytes(b'CPU-partial-file-not-a-neural-state')
    a=o.worker_inventory(tmp_path);assert len(a['files'])==1 and a['logical_bytes']>0
    (folder/'partial').write_bytes(b'unfinished');b=o.worker_inventory(tmp_path)
    assert a['worker_inventory_sha256']!=b['worker_inventory_sha256']and len(b['files'])==2
    (folder/'input-alias').symlink_to(folder/'optimizer.pt')
    with pytest.raises(PermissionError):o.worker_inventory(tmp_path)


@pytest.mark.parametrize('success',[True,False])
def test_original_owner_full_terminal_uses_real_once_GLOBAL_and_own_lease_close(packet,tmp_path,monkeypatch,success):
    """PBS/five-model proofs are fixtures; CAS/Journal/files really execute."""
    from scripts.pi0294.test_minimal_terminal import captured_CPU_terminal
    from scripts.pi0294.generalization import minimal_native_audit as audit_module
    p=deepcopy(packet);g=json.loads(p['global_bytes']);req=json.loads(p['request_bytes']);job=p['job_id']
    future=captured_CPU_terminal(g,success=success);effects=future['effects']
    root=tmp_path/'CPU-main-terminal-fixture';base=root/'control/resident-owner';base.mkdir(parents=True)
    d=root/'pbs/alloc-35';d.mkdir(parents=True);(d/'REQUEST.json').write_bytes(p['request_bytes']);(d/'REQUEST.json').chmod(0o400)
    final=dict(job_id=job,request_id=req['request_id'],source_sha=req['source_sha'],request_sha256=e.sha(p['request_bytes']),
        CPU_actual_s=1000,effects=effects)
    o.once(d/'PBS_PROCESS_FINAL.json',final);o.once(d/'PBS_WRAPPER_FINAL.json',dict(job_id=job,returncode=0 if success else 1))
    v=o.Owner.__new__(o.Owner);v.adapter=e;v.root=root;v.BASE=base;v.global_path=tmp_path/'HOPPER_RESOURCE_LEDGER.json'
    v.global_path.write_bytes(p['global_bytes']);lease=tmp_path/'leases/own.json';lease.parent.mkdir()
    lease.write_bytes(e.raw(dict(status='ACTIVE',run_uuid=root.name,manifest_sha256=req['run_manifest_sha256'])))
    v.binding=dict(lease=str(lease));o.once(base/'SOURCE_REQUEST.private.json',dict(request=req,
        request_sha256=e.sha(p['request_bytes']),directory=str(d)));o.once(base/'SOURCE_JOB_BOUND.private.json',dict(job_id=job))
    v.running=o.mature.new_journal(base,'running-rpc',dict(run_uuid=root.name,job_id=job,source_sha=req['source_sha'],request_sha256=e.sha(p['request_bytes'])))
    def fixture_audit(root,request_path,*,native,inventory_sha256):
        out=deepcopy(future['audit']);out['inventory_sha256']=inventory_sha256;return out
    monkeypatch.setattr(audit_module,'audit',fixture_audit)
    native=dict(job_state='F',Exit_status=0 if success else 1,stime=1793131103,obittime=1793134703,
        resources_used=dict(walltime='01:00:00',cput='00:20:00'))
    out=v.finish(dict(native=native))
    now=read(v.global_path);row=e.rows(now)[0]
    assert row['scientific_endpoint_complete']is success and read(lease)['status']=='CLOSED'
    assert now['used']['H200_s']==g['used']['H200_s']+3720
    assert now['CPU_s']==g['CPU_s']+1440+900
    assert now['PBS_submissions'][:-1]==g['PBS_submissions'][:-1]
    before=v.global_path.read_bytes();closed=lease.read_bytes()
    with pytest.raises(FileExistsError):v.finish(dict(native=native))
    assert v.global_path.read_bytes()==before and lease.read_bytes()==closed
