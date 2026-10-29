"""Actual closed1506193 history; all future jobs/process/model outputs are fixtures.

The captured tests use complete original GLOBAL bytes and the real direct
authority. They never call SSH/PBS/HF or load a model, and cannot grant native
permission. Existing real onceIO is exercised against disposable local files.
"""
import base64,gzip,json,os,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_e2only_scope as scope
from scripts.pi0294.generalization import minimal_e2only_execution as x,minimal_e2only_contract as c
from scripts.pi0294.generalization import minimal_e2only_physical as physical,minimal_execution_adapter as adapters
from scripts.pi0294.generalization import minimal_session_gate as gates,minimal_terminal as terminal
from scripts.pi0294.generalization.minimal_e2only_effects import E2OnlyEffects,SUMMARY,COMPLETE
from scripts.pi0294.generalization.minimal_e2only_audit import AUDIT
from scripts.pi0294.dec1 import vanda_campaign as queue


@pytest.fixture(scope='module')
def actual():
    where=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not where:pytest.skip('actual closed1506193 full canonical capture required')
    m=Path(where)
    d=json.loads((m/'SAVED512_F1506193_CANONICAL_CLOSED_CAPTURE.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(d['GLOBAL_gzip_base64']));assert e.sha(gb)==d['GLOBAL_sha256']
    l=json.loads((m/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())
    lb=gzip.decompress(base64.b64decode(l['LOCAL_gzip_base64']))
    grant=Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/ARX5_E2_ONLY_RAW512_PBS44_DIRECT_CAP_AUTHORITY_20261029.json'
    now=time.time();closed=dict(observed_epoch=now,GLOBAL_sha256=e.sha(gb),lease_sha256=scope.LEASE,
        all_complete_canonical_bytes_match=True,finite_terminal_OS_exit=dict(returncode=0),CPU_fixture_only=True)
    adopted=scope.propose(gb,lb,grant.read_bytes(),closed_readback=closed,now_epoch=now)
    g=adopted['proposed_global']
    p=dict(schema=x.PHYSICAL_SCHEMA,predecessor_binding=c.binding(),request_id='00000000-0000-4000-8000-000000000061',
        submission_nonce='00000000-0000-4000-8000-000000000062',run_uuid=c.RUN,old_closed_lease_sha256=scope.LEASE,
        previous_root='/CPU_FIXTURE/arx5-minimal-saved-e1-inference-e2-fresh512-03',previous_manifest_sha256='a'*64,
        previous_custody_sha256='b'*64,previous_snapshot_sha256='c'*64,previous_snapshot_identity_sha256='d'*64,
        paid_output_ceiling=x.PAID_OUTPUT,new_output_bound_bytes=70000000000,additional_output_charge_bytes=0,
        retained_initial_bytes=66671091712)
    from scripts.pi0294.test_minimal_native_admission import queue_limits
    limits=e.raw(queue_limits(now-1))
    manifest=dict(scope=e.scope.SCOPE,platform_training_version=x.VERSION,run_uuid=c.RUN,
        registered_runs_root='/CPU_FIXTURE/children',output_root='/CPU_FIXTURE/children/'+c.RUN,
        execution=dict(profile=x.PROFILE,agent_liveness_required=False,sealed_inputs={}),code=dict(vla_commit='a'*40))
    r=dict(x.fixed_request_fields(),request_id=p['request_id'],submission_nonce=p['submission_nonce'],source_sha='a'*40,
        run_manifest_sha256=e.sha(e.raw(manifest)),cold_input_audit_sha256='c'*64,raw_parent_audit_plan_sha256='d'*64,
        frozen_input_packet_sha256='e'*64,Core_measurement_capture_sha256=g[e.FIELD]['Core_measurement_proof']['capture_sha256'],
        physical_custody_sha256=e.sha(e.raw(p)),effective_queue_proof_sha256=e.sha(limits),created_epoch=now,
        submit_count=41,**{k:g[e.authority.FIELD][k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    proof=dict(status='ACTUAL_E2_ONLY_ALL_OLD_GROSS_AND_NEW70GB_NO_PERMISSION',observed_epoch=now,
        custody_sha256=r['physical_custody_sha256'],previous_snapshot_identity_sha256=p['previous_snapshot_identity_sha256'],
        retained_allocated_bytes=p['retained_initial_bytes'],new_allocated_bytes=10000000,next_output_upper_bytes=1000000,
        ledger_changed=False,native_permission=False)
    full=queue.collect_queue(runner=lambda *a,**kw:SimpleNamespace(returncode=0,stdout='',stderr=''),clock=lambda:now)
    fresh=dict(complete_queue=full,observed_epoch=now,GLOBAL_sha256=e.sha(e.raw(g)),complete_queue_verified=True,
        old_workers_terminal=True,old_closed_science_lease_sha256=scope.LEASE,independent_confirmation_base64='')
    proposed=x.register(e.raw(g),lb,request_bytes=e.raw(r),physical_custody=p,physical_proof=proof,
        physical_observed_epoch=now,fresh_readback=fresh,now_epoch=now)
    registered=proposed['proposed_global'];bound=x.bind(registered,r['request_id'],'123456.stdct-mgmt-02',now_epoch=now+1)
    return dict(gb=gb,lb=lb,grant=grant.read_bytes(),closed=closed,adopted=adopted,registered=registered,
        bound=bound,request=r,custody=p,physical=proof,now=now,queue=full,fresh=fresh,manifest=manifest,limits=limits,registered_future=proposed['full_future_proof'])


def test_actual_all14_and_old_history_through_registration_and_bind(actual):
    a=actual;before=json.loads(a['gb']);adopt=a['adopted']['proposed_global'];registered=a['registered']
    assert a['adopted']['full_future_proof']['cap_changes_required']=={}
    assert len(a['registered_future']['dimensions'])==14
    assert a['registered_future']['dimensions']['PBS_submissions']['projected']==44
    assert a['registered_future']['dimensions']['optimizer_updates']['projected']==15936
    assert adopt['CPU_s']==before['CPU_s']+3600
    assert adopt['new_storage_current_bytes']==registered['new_storage_current_bytes']==before['new_storage_current_bytes']
    assert registered['PBS_submissions'][:-1]==before['PBS_submissions']
    assert registered[e.package.FIELD]==before[e.package.FIELD]
    assert registered['optimizer_updates']==before['optimizer_updates']==15424
    assert registered['used']['PBS_submissions']==41
    assert scope.planned_unregistered(registered)==e._vector({})
    assert adapters.current(a['bound'])[0]is x
    from scripts.pi0294.generalization.minimal_e2only_dispatch import validate
    validate(registered,a['queue'],a['request'],e.sha(e.raw(a['request'])),now_epoch=a['now'])
    with pytest.raises(PermissionError):x.bind(a['bound'],a['request']['request_id'],'123457.stdct-mgmt-02',now_epoch=a['now']+2)


def test_periodic_save_is_committed_recovery_only_not_a_new_candidate(actual,tmp_path,monkeypatch):
    from scripts.pi0294.generalization.minimal_e2only_worker import recovery_save_headroom
    r=actual['request'];assert r['recovery_checkpoint_policy']==c.recovery_policy()
    assert c.recovery_boundary(256,8192,pending=[],unknown=[])
    assert not c.recovery_boundary(255,8160,pending=[],unknown=[])
    for pending,unknown in (([{}],[]),([],['UNKNOWN'])):
        with pytest.raises(PermissionError):c.recovery_boundary(256,8192,pending=pending,unknown=unknown)
    base=tmp_path/'training'/c.sessions()[0]['id'];base.mkdir(parents=True)
    class Guard:
        def reserve(self,n):assert n==26000000000
    monkeypatch.setattr(os,'statvfs',lambda p:SimpleNamespace(f_bavail=100000000000,f_frsize=1))
    recovery_save_headroom(tmp_path,r,256,guard=Guard())
    first=base/'checkpoint-256';first.mkdir();(first/'CHECKPOINT_COMPLETE.json').write_text('{}')
    before=(first/'CHECKPOINT_COMPLETE.json').read_bytes()
    recovery_save_headroom(tmp_path,r,512,guard=Guard())
    with pytest.raises(PermissionError):recovery_save_headroom(tmp_path,r,256,guard=Guard())
    second=base/'checkpoint-400';second.mkdir();(second/'CHECKPOINT_COMPLETE.json').write_text('{}')
    with pytest.raises(PermissionError):recovery_save_headroom(tmp_path,r,512,guard=Guard())
    assert before==(first/'CHECKPOINT_COMPLETE.json').read_bytes()and first.is_dir()and second.is_dir()


def test_prevalidated_native_CAS_keeps_original_changed_bytes_and_replay_rejections(actual,tmp_path):
    from scripts.pi0294.generalization.minimal_native_owner import Owner
    v=Owner.__new__(Owner);v.adapter=x;v.global_path=tmp_path/'HOPPER_RESOURCE_LEDGER.json'
    before=e.raw(actual['adopted']['proposed_global']);target=actual['registered']
    v.global_path.write_bytes(before)
    writer=v.prepared_global_write('REGISTER_REQUEST',before,target)
    out=writer({},40);assert out['after_sha256']==e.sha(e.raw(target))
    assert v.global_path.read_bytes()==e.raw(target)
    with pytest.raises(FileExistsError):writer({},40)
    v.global_path=tmp_path/'SECOND_DISPOSABLE_LEDGER.json';v.global_path.write_bytes(before)
    second=v.prepared_global_write('REGISTER_REQUEST',before,target)
    v.global_path.write_bytes(before+b' ')
    with pytest.raises(PermissionError,match='ACTUAL_LEDGER_CHANGED'):second({},40)
    assert v.global_path.read_bytes()==before+b' '


def test_new_E2_256_save_seals_only_own_complete_same_job_without_mutating_inputs(actual,tmp_path,monkeypatch):
    from scripts.pi0294.generalization.minimal_worker import seal_checkpoint
    r=actual['request'];sid=c.sessions()[0]['id'];cp=tmp_path/'training'/sid/'checkpoint-256';cp.mkdir(parents=True)
    payload=cp/'model.safetensors';payload.write_bytes(b'CPU_fixture_not_model_state')
    complete=dict(global_step=256,training_identity=c.sessions()[0]['arm'],full_resume=True,
        qualification_only=False,session_terminal=False,scheduler_horizon=1024,origin_job_id='CPU_PBS_FIXTURE',
        origin_source_sha=r['source_sha'],files={'model.safetensors':e.sha(payload.read_bytes())})
    (cp/'CHECKPOINT_COMPLETE.json').write_bytes(e.raw(complete));pin=e.sha(e.raw(complete))
    mode_before=payload.stat().st_mode
    monkeypatch.setenv('PBS_JOBID','WRONG_JOB')
    with pytest.raises(PermissionError):seal_checkpoint(tmp_path,sid,256,pin,guard=lambda:None,recovery_request=r)
    assert payload.stat().st_mode==mode_before
    monkeypatch.setenv('PBS_JOBID','CPU_PBS_FIXTURE')
    saved=seal_checkpoint(tmp_path,sid,256,pin,guard=lambda:None,recovery_request=r)
    assert saved['tensor_bytes_restreamed']==0 and payload.stat().st_mode&0o777==0o400
    assert payload.read_bytes()==b'CPU_fixture_not_model_state'
    with pytest.raises(PermissionError):seal_checkpoint(tmp_path,sid,256,pin,guard=lambda:None)


@pytest.mark.parametrize('change',['grant','stale','lease','unknown','retained122','old_E1'])
def test_missing_authority_or_changed_real_history_cannot_be_adopted(actual,change):
    a=actual;g=json.loads(a['gb']);closed=deepcopy(a['closed']);grant=a['grant']
    if change=='grant':grant=grant.replace(b'44',b'45')
    elif change=='stale':closed['observed_epoch']=a['now']-121
    elif change=='lease':closed['lease_sha256']='0'*64
    elif change=='unknown':g['unknown_effects']=['RETAIN']
    elif change=='retained122':g['PBS_submissions'][-1]['committed_optimizer_updates']=0
    else:g[scope.previous.FIELD]['completed_sessions']=[]
    with pytest.raises(PermissionError):scope.propose(e.raw(g),a['lb'],grant,closed_readback=closed,now_epoch=a['now'])


@pytest.mark.parametrize('field,value',[('optimizer_update_limit',390),('query_limit',62),('model_load_limit',5),
    ('profile',adapters.saved512.PROFILE),('run_uuid',adapters.saved512.RUN),('E1_new_training_or_inference',1)])
def test_no_old_alias_extra_E1_or_partial122_resume(actual,field,value):
    r=dict(actual['request']);r[field]=value
    with pytest.raises(PermissionError):x.request(r)


def test_exact_native_capacity_and_three_phase_counters(actual):
    r=actual['request'];native=dict(select='1:ngpus=1:ncpus=36:mem=240gb',ngpus=1,ncpus=36,mem='240gb',walltime='06:14:00')
    assert x.select_profile(native,r)==(36,240*1024**3)
    with pytest.raises(PermissionError):x.select_profile(dict(native,walltime='08:00:00'),r)
    assert [p['phase']for p in gates.plan(r)]==['MODEL_RAW','FIRST_UPDATE','OFFICIAL_EXPORT']
    for phase,counts in [('MODEL_RAW',(0,0,0)),('FIRST_UPDATE',(0,0,1)),('OFFICIAL_EXPORT',(512,36,1))]:
        c.check_phase(c.sessions()[0]['id'],phase,dict(zip(c.KEYS,counts)))
    with pytest.raises(PermissionError):c.check_phase(c.sessions()[0]['id'],'FIRST_UPDATE',dict(zip(c.KEYS,(122,0,1))))
    with pytest.raises(PermissionError):c.check_phase('E1_ADAPTER_ONLY_FORMAL512','MODEL_RAW',dict(zip(c.KEYS,(0,0,0))))


def test_gross_pool_cannot_hide_new_metadata_or_erase_failed_root(actual):
    a=actual;bad=dict(a['physical'],next_output_upper_bytes=70000000000)
    with pytest.raises(PermissionError):physical.check_proof(bad,a['custody'],a['request'],now_epoch=a['now'],observed_epoch=a['now'])
    bad=dict(a['physical'],retained_allocated_bytes=0)
    with pytest.raises(PermissionError):physical.check_proof(bad,a['custody'],a['request'],now_epoch=a['now'],observed_epoch=a['now'])


def terminal_packet(a,success):
    row=x.basic(a['bound'])[1];common={k:row[k]for k in('request_id','request_sha256','source_sha','job_id')}
    sid=c.sessions()[0]['id'];updates,queries,loads=(512,37,2)if success else(122,0,1)
    sessions={sid:dict(status='COMPLETE'if success else'ACTIVE',optimizer_updates=updates,generation_queries=queries,model_loads=loads)}
    if success:sessions[sid]['terminal_receipt_sha256']='a'*64
    effects=dict(schema=SUMMARY,**common,pending_updates=[],unknown_query_intents=[],global_budget_is_only_external_reservation=True,
        new_physics=0,new_episodes=0,optimizer_updates_committed=updates,query_attempts=queries,model_load_attempts=loads,
        sessions=sessions,active_session=None if success else sid,**{COMPLETE:success},closed_predecessor=c.binding(),E1_new_training_or_inference=0)
    native=dict(schema=terminal.NATIVE,**common,job_state='F',submission_nonce=row['submission_nonce'],Exit_status=0 if success else 1,
        all_native_workers_terminal=True,terminal_inventory_stable=True,raw_native_sha256='a'*64,process_final_sha256='b'*64,
        wrapper_final_sha256='c'*64,inventory_sha256='d'*64,conservative_GPU_s=241,conservative_CPU_s=100,
        new_native_conservative_CPU_s=100,prior_readonly_preparation_CPU_upper_s=0,finished_epoch=a['now']+10,observed_epoch=a['now']+11)
    audit=dict(status=AUDIT if success else terminal.PARTIAL,**common,closed_predecessor=c.binding(),E1_new_training_or_inference=0,
        effect_summary_sha256=e.sha(e.raw(effects)),inventory_sha256=native['inventory_sha256'],raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        statistics_sha256=e.R0,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0)
    if success:
        cp=dict(global_step=512,cursor=16384,scheduler_horizon=1024,raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
            statistics_sha256=e.R0,full_resume=True,qualification_only=False,TRAIN_probes=12,language_probes=24,
            checkpoint_manifest_sha256='c'*64,official_inference_manifest_sha256='d'*64,full_streamed_members_sha256='e'*64,
            original_training_job_id=row['job_id'],current_job_optimizer_updates=512)
        from scripts.pi0294.generalization.minimal_saved512_contract import RESULT
        audit.update(audited_session_ids=[sid],own_full_optimizer_scheduler_RNG_sampler_exact=True,
            original_qualification_evidence_reused_not_reexecuted=True,saved_qualification_input_audit_sha256=RESULT,
            official_full_model_reconstruction_output_pairs_exact=True,frozen_parameters_and_buffers_unchanged=True,
            endpoints={e.training.ARMS[1]:cp})
    return row,native,effects,audit


@pytest.mark.parametrize('success',[False,True])
def test_original_terminal_once_retains_all_old_costs_and_counts_only_own_E2(actual,success):
    a=actual;row,native,effects,audit=terminal_packet(a,success)
    amounts=terminal.terminal_values(row,native,effects,audit,e2only_global=a['bound'])
    assert amounts['success']is success
    out=x.settle(a['bound'],row['request_id'],native=native,effects=effects,audit=audit,
        physical_proof=a['physical'],physical_observed_epoch=a['now'],now_epoch=a['now']+12)
    assert out['PBS_submissions'][:40]==a['bound']['PBS_submissions'][:40]
    assert out['optimizer_updates']==15424+(512 if success else 122)
    assert out['generation_queries']==5972+(37 if success else 0)
    assert out['new_storage_current_bytes']==a['bound']['new_storage_current_bytes']
    assert out['new_storage_peak_bytes']==a['bound']['new_storage_peak_bytes']
    assert x.basic(out)[1]['scientific_endpoint_complete']is success
    with pytest.raises(PermissionError):x.settle(out,row['request_id'],native=native,effects=effects,audit=audit,
        physical_proof=a['physical'],physical_observed_epoch=a['now'],now_epoch=a['now']+13)


@pytest.mark.parametrize('change',['pending','old_job','E1','step122','no_stream','no_noise'])
def test_runtime_flags_cannot_promote_missing_or_wrong_own_endpoint(actual,change):
    row,native,effects,audit=terminal_packet(actual,True)
    if change=='pending':effects['pending_updates']=[{'KEEP':'UNKNOWN'}]
    elif change=='old_job':native['job_id']=scope.JOB
    elif change=='E1':effects['E1_new_training_or_inference']=1
    elif change=='step122':audit['endpoints'][e.training.ARMS[1]]['global_step']=122
    elif change=='no_stream':audit['own_full_optimizer_scheduler_RNG_sampler_exact']=False
    else:audit['official_full_model_reconstruction_output_pairs_exact']=False
    audit['effect_summary_sha256']=e.sha(e.raw(effects))
    with pytest.raises(PermissionError):terminal.terminal_values(row,native,effects,audit,e2only_global=actual['bound'])


def test_actual_temporary_effects_initialization_contains_only_E2(actual,tmp_path):
    a=actual;r=a['request'];root=tmp_path/c.RUN;d=root/'pbs/alloc-41';d.mkdir(parents=True)
    path=d/'REQUEST.json';path.write_bytes(e.raw(r))
    allocation=dict(job_id='123456.stdct-mgmt-02',request_id=r['request_id'],request_sha256=e.sha(path.read_bytes()),source_sha=r['source_sha'])
    ledger=E2OnlyEffects(root,allocation,request_path=path,guard=lambda:None);ledger.initialize()
    v=ledger.guard();assert list(v['sessions'])==[c.sessions()[0]['id']]
    assert v['optimizer_updates']==v['generation_queries']==v['model_loads']==0
    ledger.start_session(c.sessions()[0]['id']);ledger.charge('model_loads',1)
    assert ledger.guard()['sessions'][c.sessions()[0]['id']]['model_loads']==1
    with pytest.raises(PermissionError):ledger.start_session('E1_ADAPTER_ONLY_FORMAL512')
    with pytest.raises(PermissionError):ledger.initialize()


def test_new_request_reaches_original_native_admission_and_own_process_join(actual):
    from scripts.pi0294.generalization import minimal_native_admission as n,minimal_native_context as context
    a=actual;r=a['request'];now=a['now']+5;job='123456.stdct-mgmt-02'
    native=dict(job_state='R',queue='batch_gpu',Rerunable=False,run_count=1,project=r['project'],
        Job_Name=r['job_name'],Job_Owner='e1192548@stdct-login-01',exec_host='GN-A40-022/0*36',
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(ncpus=36,ngpus=1,mem='240gb',walltime='06:14:00',select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40'),
        stime=time.strftime('%a %b %d %H:%M:%S %Y',time.localtime(now-2)))
    q=dict(schema=queue.QUEUE,user='e1192548',observed_epoch=now,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout=job+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',job],returncode=0,stdout=json.dumps(dict(Jobs={job:native})),stderr=''))
    site=dict(schema=n.SITE,source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',run_uuid=r['run_uuid'],
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=e.sha(e.raw(r)),
        source_sha=r['source_sha'],job_id=job,observed_epoch=now,
        project=dict(name=r['project'],active=True,users=['e1192548'],cluster=['stdct-mgmt-02'],start_date='2026-03-12',end_date='2027-03-12'),
        balance_GPU_hours=1000.,balance_CPU_hours=1000.,
        quota=dict(observed_epoch=now,user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',available_bytes=1024**5))
    packet=dict(global_bytes=e.raw(a['bound']),local_bytes=a['lb'],request_bytes=e.raw(r),manifest_bytes=e.raw(a['manifest']),
        job_id=job,hostname='GN-A40-022',user='e1192548',native_job_bytes=e.raw(native),queue_proof=q,site_bytes=e.raw(site),
        effective_queue_bytes=a['limits'],independent_job_bytes=None,now_epoch=now)
    proof=n.validate(**packet)
    assert proof['whole_remaining_package_shortfalls']=={} and proof['active_CPU_limit']==8
    assert proof['model_loaded']is proof['permission_file_created']is False
    root=Path(a['manifest']['output_root']);binding=dict(uid=22182,run_root=str(root),run_manifest_sha256=r['run_manifest_sha256'])
    context.registered_identity(root,root/'pbs/alloc-41/REQUEST.json',a['manifest'],binding,r,a['bound'],
        uid=22182,pbs_job=job,now=now,require_bound=True)
    sid=c.sessions()[0]['id'];effect=dict(schema=E2OnlyEffects.schema,platform_training_version=x.VERSION,
        closed_predecessor=c.binding(),request_sha256=e.sha(e.raw(r)),job_id=job,source_sha=r['source_sha'],
        active_session=sid,pending_updates=[],unknown_effects=[],sessions={sid:dict(optimizer_updates=0,generation_queries=0,model_loads=0)})
    ready=dict(schema=gates.READY,ticket_version=gates.ticket_version(r),**c.binding(),**c.tickets()[0],
        job_id=job,request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=e.sha(e.raw(r)),
        source_sha=r['source_sha'],raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        session_counters=dict(optimizer_updates=0,generation_queries=0,model_loads=0))
    assert gates.check_ready(ready,r,request_sha256=e.sha(e.raw(r)),job_id=job,effect_value=effect)==c.tickets()[0]
    effect['sessions'][sid]['optimizer_updates']=122
    with pytest.raises(PermissionError):gates.check_ready(ready,r,request_sha256=e.sha(e.raw(r)),job_id=job,effect_value=effect)
