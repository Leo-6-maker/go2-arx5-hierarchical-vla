"""Captured old costs with future native fixtures, real temporary Journals.

Original check/admission/phase/onceIO execute unmocked. Site/PBS/proc/physical
observations are labelled future fixtures; no network, model or native write.
"""
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pi0294.generalization import minimal_execution as e
from scripts.pi0294.generalization import minimal_retry_execution as retry
from scripts.pi0294.generalization import minimal_execution_adapter as adapters
from scripts.pi0294.generalization import minimal_native_admission as admission
from scripts.pi0294.generalization import minimal_session_gate as gates
from scripts.pi0294.generalization import minimal_effects as effects
from scripts.pi0294.generalization import minimal_concurrency as concurrency
from scripts.pi0294.generalization import minimal_native_context as context
from scripts.pi0294.generalization import minimal_native_owner as owner_module
from scripts.pi0294.test_minimal_retry_execution import captured_evidence, evidence, prepared, propose, terminal_fixture
from scripts.pi0294.test_minimal_native_admission import queue_limits
from scripts.pi0294.test_minimal_native_context import identity
from scripts.pi0294.test_minimal_native_owner import owner as old_owner
from scripts.pi0294.test_minimal_native_owner import (
    test_original_Journal_releases_only_next_finite_ticket_and_never_replays as release_once,
    test_actual_lost_reply_after_release_keeps_UNKNOWN_and_does_not_republish as release_lost)
from scripts.pi0294.test_minimal_effects import draw
from scripts.pi0294.dec1 import vanda_campaign as queue, a40_training_runtime as runtime
from scripts.pi0294.pv1 import mix_core_accounting as core
from scripts.pi0292.durable import read


@pytest.fixture
def packet(prepared):
    p=deepcopy(prepared);r=json.loads(p['request_bytes']);now=p['now_epoch']+5
    limits=e.raw(queue_limits(now-10));r['created_epoch']=now-9
    r['effective_queue_proof_sha256']=e.sha(limits)
    m=dict(scope=e.scope.SCOPE,platform_training_version=retry.VERSION,run_uuid=retry.RUN,
        output_root=p['physical_proof']['new_root'],registered_runs_root='/cpu-fixture/children',
        execution=dict(profile=retry.PROFILE,agent_liveness_required=False,sealed_inputs={
            'plans/MINIMAL_EFFECTIVE_QUEUE_LIMITS.private.json':e.sha(limits)}),
        code=dict(vla_commit=r['source_sha']))
    r['run_manifest_sha256']=e.sha(e.raw(m));p['request_bytes']=e.raw(r)
    g=propose(p);job='987654.stdct-mgmt-02';g=retry.bind(g,r['request_id'],job,now_epoch=now-3)
    native=dict(job_state='R',queue='batch_gpu',Rerunable=False,run_count=1,project=r['project'],
        Job_Name=r['job_name'],Job_Owner='e1192548@stdct-login-01',exec_host='GN-A40-022/0*36',
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(ncpus=36,ngpus=1,mem='240gb',walltime='16:30:00',select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40'),
        stime=time.strftime('%a %b %d %H:%M:%S %Y',time.localtime(now-2)))
    q=dict(schema=queue.QUEUE,user='e1192548',observed_epoch=now,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout=job+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',job],returncode=0,stdout=json.dumps(dict(Jobs={job:native})),stderr=''))
    site=dict(schema=admission.SITE,source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
        run_uuid=r['run_uuid'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        request_sha256=e.sha(e.raw(r)),source_sha=r['source_sha'],job_id=job,observed_epoch=now,
        project=dict(name=r['project'],active=True,users=['e1192548'],cluster=['stdct-mgmt-02'],start_date='2026-03-12',end_date='2027-03-12'),
        balance_GPU_hours=1000.,balance_CPU_hours=1000.,
        quota=dict(observed_epoch=now,user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',available_bytes=1024**5))
    return dict(global_bytes=e.raw(g),local_bytes=p['local_bytes'],request_bytes=e.raw(r),manifest_bytes=e.raw(m),
        job_id=job,hostname='GN-A40-022',user='e1192548',native_job_bytes=e.raw(native),queue_proof=q,
        site_bytes=e.raw(site),effective_queue_bytes=limits,independent_job_bytes=None,now_epoch=now)


def test_original_admission_selects_new_row_and_full_future_with_old_terminal_unchanged(packet):
    before=deepcopy(packet);proof=admission.validate(**packet)
    g=json.loads(packet['global_bytes']);adapter,row,_=adapters.current(g)
    assert adapter is retry and row['request_id']==proof['request_id']
    assert e.rows(g)[0]['native_exit_status']==1 and e.rows(g)[0]!=row
    assert proof['whole_remaining_package_shortfalls']=={}
    assert proof['CPU_accounting_upper_s']==165600 and len(proof['full_remaining_proof']['dimensions'])==14
    assert not proof['permission_file_created'] and packet==before
    assert context.registered_identity(**identity(packet)) is None


@pytest.mark.parametrize('change',['Q','F','run_count','nonce','old_profile','old_request','old_manifest',
    'missing_history','missing_index','unknown','pending','stale_quota'])
def test_retry_admission_does_not_bypass_real_R_identity_or_full_history(packet,change):
    p=deepcopy(packet)
    def mutate(key,fn):
        value=json.loads(p[key]);fn(value);p[key]=e.raw(value)
    if change in ('Q','F','run_count','nonce'):
        native=json.loads(p['native_job_bytes'])
        if change in ('Q','F'):native['job_state']=change
        elif change=='run_count':native['run_count']=2
        else:native['Variable_List']['PV1_SUBMISSION_NONCE']='OTHER'
        p['native_job_bytes']=e.raw(native);p['queue_proof']['qstat']['stdout']=json.dumps(dict(Jobs={p['job_id']:native}))
    elif change=='old_profile':mutate('request_bytes',lambda r:r.update(profile=e.PROFILE))
    elif change=='old_request':p['request_bytes']=e.raw(e.rows(json.loads(p['global_bytes']))[0])
    elif change=='old_manifest':mutate('manifest_bytes',lambda m:m.update(platform_training_version=e.VERSION))
    elif change=='missing_history':mutate('global_bytes',lambda g:g[retry.FIELD].pop('history'))
    elif change=='missing_index':mutate('global_bytes',lambda g:g.pop(retry.FIELD))
    elif change in ('unknown','pending'):mutate('global_bytes',lambda g:g.update({change+'_effects' if change=='unknown' else 'pending_updates':['PRESERVED']}))
    else:mutate('site_bytes',lambda s:s['quota'].update(observed_epoch=p['now_epoch']-901))
    before=deepcopy(p)
    with pytest.raises((PermissionError,KeyError,ValueError)):admission.validate(**p)
    assert p==before


def test_all_thirteen_retry_phases_keep_original_counters_and_exact_new_effect_identity(packet):
    r=json.loads(packet['request_bytes']);rs=e.sha(packet['request_bytes'])
    for d in gates.plan():
        spec=next(v for v in e.native_sessions() if v['id']==d['session_id'])
        counts=dict(optimizer_updates=0,generation_queries=0,model_loads=0)
        if d['phase']=='FIRST_UPDATE':counts.update(model_loads=1,generation_queries=0 if spec['formal_candidate'] else 15)
        elif d['phase']=='RESTORE_FULL':counts.update(optimizer_updates=2,generation_queries=16,model_loads=1)
        elif d['phase']=='OFFICIAL_EXPORT':counts.update(optimizer_updates=512,generation_queries=36,model_loads=1)
        ready=dict(schema=gates.READY,ticket_version=gates.VERSION,**d,job_id=packet['job_id'],request_id=r['request_id'],
            submission_nonce=r['submission_nonce'],request_sha256=rs,source_sha=r['source_sha'],
            raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,session_counters=counts)
        value=dict(schema=effects.SCHEMA,platform_training_version=retry.VERSION,request_sha256=rs,
            job_id=packet['job_id'],source_sha=r['source_sha'],active_session=d['session_id'],
            pending_updates=[],unknown_effects=[],sessions={d['session_id']:counts})
        assert gates.check_ready(ready,r,request_sha256=rs,job_id=packet['job_id'],effect_value=value)==d
        for key,v in [('platform_training_version',e.VERSION),('pending_updates',['PRESERVED']),('unknown_effects',['PRESERVED'])]:
            bad=dict(value);bad[key]=v
            with pytest.raises(PermissionError):gates.check_ready(ready,r,request_sha256=rs,job_id=packet['job_id'],effect_value=bad)


def test_new_effect_journal_retains_unknown_and_never_reinitializes(packet,tmp_path):
    r=json.loads(packet['request_bytes']);p=tmp_path/'pbs/REQUEST.json';p.parent.mkdir();p.write_bytes(packet['request_bytes']);p.chmod(0o400)
    allocation=dict(job_id=packet['job_id'],request_id=r['request_id'],request_sha256=e.sha(p.read_bytes()),source_sha=r['source_sha'])
    ledger=effects.Effects(tmp_path,allocation,request_path=p,guard=lambda:None);v=ledger.initialize()
    assert v['platform_training_version']==retry.VERSION and list(v['sessions'])==[x['id'] for x in e.native_sessions()]
    with pytest.raises(PermissionError):ledger.initialize()
    ledger.start_session(e.native_sessions()[0]['id']);ledger.charge('model_loads',1)
    with pytest.raises(RuntimeError):draw(ledger,'NEW_RETRY_LOST_REPLY',fail=True)
    assert ledger.summary()['unknown_query_intents']==[0]
    from scripts.pi0294.e1.journal import MutationOutcomeUnknown
    before=ledger.path.read_bytes()
    with pytest.raises(MutationOutcomeUnknown):draw(ledger,'NEW_RETRY_LOST_REPLY')
    assert ledger.path.read_bytes()==before


@pytest.fixture
def owner(packet,tmp_path):
    value=old_owner.__wrapped__(packet,tmp_path);value[0].adapter=retry;return value


def test_original_release_Journal_uses_new_request_once(owner):release_once(owner)


def test_original_release_lost_reply_new_request_keeps_UNKNOWN(owner,monkeypatch):release_lost(owner,monkeypatch)


def test_retry_heartbeat_keeps_complete_typed_request_and_rejects_legacy_scope(owner):
    v,p,_,_=owner
    out=v.heartbeat_rpc('vanda','',dict(JOB=p['job_id'],VIEW=v.current['view']),'CPU-fixture')
    assert out['heartbeat']['scope']==retry.VERSION and out['retry_request']==json.loads(p['request_bytes'])
    bad=deepcopy(out);bad['retry_request']['profile']=e.PROFILE
    with pytest.raises(PermissionError):owner_module.validate_heartbeat_receipt(bad,{},now=time.time())


def test_retry_concurrency_binds_new_main_and_never_reuses_old_mirror_or_identity(packet):
    g=json.loads(packet['global_bytes']);before=deepcopy(g);_,row,mark=adapters.current(g)
    binding=concurrency.binding(g)
    assert binding['schema']==concurrency.RETRY_VERSION and binding['main_request_id']==row['request_id']
    d=dict(evaluation_candidate='D25',native_platform='CORELAB_RTX4080',platform_evaluation_version=core.VERSION,
        source_sha='b'*40,created_epoch=mark['registered_epoch']+1,**{concurrency.KEY:binding})
    assert concurrency.validate_pair(g,[row],[d])==binding
    d[concurrency.KEY]=concurrency.binding_identity(e.rows(g)[0])
    with pytest.raises(PermissionError):concurrency.validate_pair(g,[row],[d])
    with pytest.raises(PermissionError):concurrency.validate_LOCAL(g,json.loads(packet['local_bytes']))
    assert g==before


def test_original_entry_dispatches_retry_to_same_five_session_runtime(packet,tmp_path,monkeypatch):
    from scripts.pi0294.generalization import minimal_native_runtime
    (tmp_path/'RUN_MANIFEST.json').write_bytes(packet['manifest_bytes']);request_path=tmp_path/'REQUEST.json'
    calls=[]
    monkeypatch.setattr(minimal_native_runtime,'run',lambda *args:calls.append(args) or 'FIVE_SESSION_ENTRY')
    monkeypatch.setattr(runtime.context,'validate_root',lambda *a,**k:pytest.fail('legacy generic worker'))
    assert runtime.run(tmp_path,request_path)=='FIVE_SESSION_ENTRY' and calls==[(tmp_path,request_path)]
    m=json.loads(packet['manifest_bytes']);m['execution']['profile']=e.PROFILE
    (tmp_path/'RUN_MANIFEST.json').write_bytes(e.raw(m))
    with pytest.raises(PermissionError):runtime.run(tmp_path,request_path)
    assert len(calls)==1


def test_current_retry_guard_checks_changed_GLOBAL_and_STOP_with_real_temporary_files(packet,tmp_path):
    from scripts.pi0294.test_minimal_native_context import test_actual_temp_guard_reads_changed_unknown_and_STOP_even_with_cached_full_budget
    test_actual_temp_guard_reads_changed_unknown_and_STOP_even_with_cached_full_budget(packet,tmp_path)


@pytest.mark.parametrize('success',[False,True])
def test_original_owner_once_terminal_keeps_old_paid_cost_and_new_zero_prior(owner,monkeypatch,success):
    """Future full-state/physical observations; original temporary CAS is real."""
    from scripts.pi0294.generalization import minimal_native_audit
    v,p,_,_=owner;g=json.loads(p['global_bytes']);r=json.loads(p['request_bytes'])
    before_old=deepcopy(e.rows(g)[0]);future=terminal_fixture(g,success=success)
    saved=read(v.BASE/'SOURCE_REQUEST.private.json');d=Path(saved['directory']);job=p['job_id']
    owner_module.once(d/'PBS_PROCESS_FINAL.json',dict(job_id=job,request_id=r['request_id'],source_sha=r['source_sha'],
        request_sha256=e.sha(p['request_bytes']),CPU_actual_s=1000,effects=future['effects']))
    owner_module.once(d/'PBS_WRAPPER_FINAL.json',dict(job_id=job,returncode=0 if success else 1))
    lease=Path(v.binding['lease']);lease.write_bytes(e.raw(dict(status='ACTIVE',run_uuid=v.root.name,
        manifest_sha256=r['run_manifest_sha256'])))
    def fixture_audit(root,request_path,*,native,inventory_sha256):
        out=deepcopy(future['audit']);out['inventory_sha256']=inventory_sha256;return out
    monkeypatch.setattr(minimal_native_audit,'audit',fixture_audit)
    physical_calls=[]
    def physical_fixture(**kwargs):physical_calls.append(kwargs);return deepcopy(future['physical_proof'])
    v.retry_physical=SimpleNamespace(inspect=physical_fixture)
    start=int(json.loads(p['request_bytes'])['created_epoch'])+10
    native=dict(job_state='F',Exit_status=0 if success else 1,stime=start,obittime=start+3600,
        resources_used=dict(walltime='01:00:00',cput='00:20:00'))
    v.finish(dict(native=native));out=read(v.global_path)
    adapter,row,_=adapters.current(out)
    assert adapter is retry and row['scientific_endpoint_complete'] is success
    assert e.rows(out)[0]==before_old and out[e.FIELD]==g[e.FIELD]
    assert out['CPU_s']==g['CPU_s']+1440 and out['used']['H200_s']==g['used']['H200_s']+3720
    assert out['new_storage_current_bytes']==g['new_storage_current_bytes']
    assert read(lease)['status']=='CLOSED' and physical_calls==[dict(next_output_upper_bytes=64*1024**2)]
    settled=v.global_path.read_bytes();closed=lease.read_bytes()
    with pytest.raises(FileExistsError):v.finish(dict(native=native))
    assert v.global_path.read_bytes()==settled and lease.read_bytes()==closed and len(physical_calls)==1
