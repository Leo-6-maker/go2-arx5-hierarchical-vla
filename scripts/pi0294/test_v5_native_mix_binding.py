"""Differential C/D hooks; no models or native admission in these tests."""
from copy import deepcopy
import importlib.util,json,types,uuid
import pytest
from scripts.pi0294.pv1 import mix_training as train,mix1024,mix_accounting as a,successor_cd as s
from scripts.pi0294.scene_cards import digest


def local_effects():return dict(optimizer_updates=0,pending_updates=[],unknown_effects=[],training_runs={})
def tiny_recipe(arm='C50'):return dict(schema=mix1024.SCHEMA,arm=arm)

def test_intent_commit_never_replays_or_crosses_arm():
    v=local_effects();r=tiny_recipe();n=train.begin_update(v,arm='C50',local_step=0,request_id='one',recipe=r)
    assert v==local_effects() and n['optimizer_updates']==0
    with pytest.raises(PermissionError):train.begin_update(n,arm='C50',local_step=0,request_id='one',recipe=r)
    with pytest.raises(PermissionError):train.commit_update(n,arm='D25',local_step_after=1,request_id='one')
    c=train.commit_update(n,arm='C50',local_step_after=1,request_id='one');assert c['optimizer_updates']==1 and c['training_runs']['C50']['last_native_step']==1
    with pytest.raises(PermissionError):train.commit_update(c,arm='C50',local_step_after=1,request_id='one')


@pytest.mark.parametrize('arm',['A_CONTROL','B_COVERAGE','C100','D50'])
def test_no_legacy_alias_or_extra_arm(arm):
    with pytest.raises(PermissionError):train.begin_update(local_effects(),arm=arm,local_step=0,request_id='one',recipe=tiny_recipe(arm))


def test_1025_and_unknown_update_refused():
    v=local_effects();v['optimizer_updates']=1024
    with pytest.raises(PermissionError):train.begin_update(v,arm='C50',local_step=1024,request_id='one',recipe=tiny_recipe())
    v=local_effects();v['unknown_effects']=['unresolved']
    with pytest.raises(PermissionError):train.begin_update(v,arm='C50',local_step=0,request_id='one',recipe=tiny_recipe())


def test_stage_parts_fit_one_whole_package_with_no_extra_candidate():
    t,e=a.stage_vector('TRAIN'),a.stage_vector('EVAL');w=s.whole_package()
    assert all(2*t[k]+2*e[k]<=w[k] for k in w)
    assert t['optimizer_updates']==1024 and e['optimizer_updates']==0
    assert e['episode_starts']==26 and w['episode_starts']==52


def test_effect_class_keeps_original_query_onceIO():
    from scripts.pi0294.pv1.mix_native import Effects
    from scripts.pi0294.dec1.a40_native_ledger import Ledger
    assert Effects.charge is Ledger.charge and Effects.transaction is Ledger.transaction
    x=Effects.__new__(Effects);x.request=dict(arm='C50',source_sha='a'*40,request_id='one');x.allocation=dict(request_sha256='b'*64,job_id='1.stdct-mgmt-02')
    v=dict(schema='V5_CD_LOCAL_NATIVE_EFFECT_LEDGER_V1',platform_training_version=a.VERSION,arm='C50',source_sha='a'*40,request_sha256='b'*64,job_id='1.stdct-mgmt-02',unknown_effects=[],optimizer_updates=0,generation_queries=0,model_loads=0,query_intent_links=[],CPU_s=0.,pending_updates=[],training_runs={})
    assert x.check(v)==v
    bad=deepcopy(v);bad.update(optimizer_updates=1,training_runs={'C50':{'last_native_step':1}})
    with pytest.raises(PermissionError):x.check(bad)


def test_native_source_has_no_model_import_before_registered_context():
    import ast
    from pathlib import Path
    p=Path(__file__).parent/'pv1/mix_native.py';tree=ast.parse(p.read_text());top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any((getattr(n,'module','')or'').startswith(('torch','transformers','mujoco','rclpy')) for n in top)


def test_original_watch_uses_registered_successor_clock_without_changing_old_clock(tmp_path):
    from scripts.pi0294.dec1.cold_owner import watch
    from scripts.pi0292.durable import read,write
    from types import SimpleNamespace
    import time
    marker=dict(s.scope(time.time()-1),grant_sha256=s.GRANT_SHA,before_sha256=dict(GLOBAL='a'*64,LOCAL='b'*64),authorized_storage_caps=dict(s.NEW_STORAGE),old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
    write(tmp_path/'SOURCE_REQUEST.private.json',dict(request=dict(submit_count=26),directory=str(tmp_path)))
    write(tmp_path/'SOURCE_JOB_BOUND.private.json',dict(job_id='26.stdct-mgmt-02'))
    write(tmp_path/'TRAIN_REGISTERED.private.json',dict(root=str(tmp_path)))
    state=dict(rows=dict(vanda=dict(jobs={'26.stdct-mgmt-02':dict(Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007',job_state='F',Exit_status=0)},receipts={})),view={})
    events=[]
    owner=SimpleNamespace(BASE=tmp_path,native=SimpleNamespace(T0=1792348196),successor_scope=marker,cold_rpc_bound=True,read=read,update=write,
        record=lambda n,v:events.append((n,v)),fresh=lambda *a,**k:state)
    result=watch(owner,'',scope_binding=marker)
    assert result['native']['Exit_status']==0 and result['job_id']=='26.stdct-mgmt-02'
    assert any(n=='SOURCE_NATIVE_TERMINAL_UNSETTLED' for n,v in events)
    with pytest.raises(PermissionError):watch(owner,'',scope_binding=s.scope(time.time()))


def test_request_binds_actual_manifest_and_audit_after_sealed_template(tmp_path):
    from scripts.pi0294.pv1.mix_owner import prepare_request
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_dataset import sha
    template=dict(schema='V5_CD_ACTUAL_NATIVE_TRAINING_REQUEST_V1',platform_training_version=a.VERSION,native_platform='VANDA_A40',arm='C50',stage='V5_MATCHED_TRAIN1024',parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,optimizer_update_limit=1024,query_limit=29,CPU_upper_s=54000,reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',additional_output_upper_bytes=70000000000,source_sha='a'*40,status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',job_id=None,expected_native_start_step=0,training_block=dict(arm='C50',replica=False,stop_after_step=1024,resume_checkpoint_relative=None))
    write(tmp_path/'RUN_MANIFEST.json',dict(template=template));audit=tmp_path/'control/COLD_INPUT_AUDIT.private.json';audit.parent.mkdir();write(audit,dict(status='PASS_ACTUAL_V5_CD_COLD_INPUTS_NO_MODEL',run_manifest_sha256=sha(tmp_path/'RUN_MANIFEST.json'),source_sha='a'*40))
    r=prepare_request(tmp_path,template,source_sha='a'*40,submit_count=26,now_epoch=1)
    assert 'run_manifest_sha256' not in template and r['cold_input_audit_sha256']==sha(audit) and r['run_manifest_sha256']==sha(tmp_path/'RUN_MANIFEST.json')
    with pytest.raises(PermissionError):prepare_request(tmp_path,dict(template,request_sha256='b'*64),source_sha='a'*40,submit_count=26,now_epoch=1)
    write(tmp_path/'RUN_MANIFEST.json',dict(changed=True))
    with pytest.raises(PermissionError):prepare_request(tmp_path,template,source_sha='a'*40,submit_count=26,now_epoch=1)


def test_cd_eval_is_same_original_panel_and_keeps_valid_negatives():
    from scripts.pi0294.pv1.mix_evaluation import plan
    from scripts.pi0294.test_pv1_evaluation_plan import groups
    from scripts.pi0294.pv1.evaluation_plan import development,remaining_panel
    p=plan(groups());old=development(groups(),route='COVERAGE_FIRST',candidate='A_CONTROL',new_bank_manifest_sha256='a'*64)['panels']['A_CONTROL-DEV-N294101']
    def normalized(spec):return [{k:v for k,v in row.items()if k not in ('id','policy_candidate','pv1_model_label','pv1_session_label')}for row in spec['cards']]
    assert normalized(p['panels']['C50'])==normalized(p['panels']['D25'])==normalized(old)
    assert all(s['checkpoint_step']==1024 and len(s['cards'])==24 for s in p['panels'].values())
    spec=p['panels']['D25'];c=spec['cards'][0];row=dict(model='D25',condition_id=c['condition_id'],attempt_id=c['id'],status='VALID_TASK_FAIL',failure_stage='S1_PRE_EXECUTION_REJECT',pipeline_trustworthy=True,live_native_failures=0)
    assert len(remaining_panel([row],spec))==23
    assert p['analysis']['bootstrap_unit']=='JOINT_T1_T2_GEOMETRY_BLOCK' and p['engineering_replacements_total_max']==4


def test_owner_creates_original_journal_directories_once_for_setup_and_bound_job(tmp_path):
    from scripts.pi0294.pv1.mix_owner import new_journal
    from scripts.pi0294.dec1.owner_transport import Journal
    identity=dict(run_uuid='fixture',job_id='NO_GPU_YET',source_sha='a'*40,request_sha256='b'*64)
    setup=new_journal(tmp_path,'setup-rpc',identity)
    assert isinstance(setup,Journal) and setup.call('READ','READ_ONLY',lambda intent,limit:dict(actual='read'))==dict(actual='read')
    with pytest.raises(FileExistsError):new_journal(tmp_path,'setup-rpc',identity)
    running=new_journal(tmp_path,'running-rpc',dict(identity,job_id='26.stdct-mgmt-02'))
    assert isinstance(running,Journal) and running.identity['job_id']=='26.stdct-mgmt-02'


def test_new_heartbeat_uses_real_original_journal_and_exact_native_nonce(tmp_path):
    from scripts.pi0294.dec1.owner_transport import Journal,checked_heartbeat
    from scripts.pi0294.dec1.test_vanda_campaign import fixture
    from scripts.pi0294.pv1.mix_owner import new_journal
    import time
    h,d,auth,q=fixture();jobs=json.loads(q['qstat']['stdout'])['Jobs'];job_id='14.stdct-mgmt-02';job=jobs[job_id]
    q['observed_epoch']=1000.;job['Variable_List']={'PV1_REQUEST_ID':'request','PV1_SUBMISSION_NONCE':'nonce'};q['qstat']['stdout']=json.dumps(dict(Jobs=jobs))
    identity=dict(run_uuid='fixture',job_id=job_id,source_sha='a'*40,request_sha256='b'*64)
    marker=dict(s.scope(900.),grant_sha256=s.GRANT_SHA,before_sha256=dict(GLOBAL='a'*64,LOCAL='b'*64),authorized_storage_caps=dict(s.NEW_STORAGE),old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
    jdir=tmp_path/'rpc';jdir.mkdir();j=Journal(jdir,identity,clock=lambda:1000.)
    captured=[]
    def publish(intent,limit):
        beat=dict(identity,request_id='request',submission_nonce='nonce',scope=s.TASK,grant_sha256=s.GRANT_SHA,rpc_intent_id=intent['operation_id'],observed_epoch=1000.,native_job_state=job['job_state'],global_unknown_effects=[])
        value=dict(schema='V5_CD_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1',heartbeat=beat,scope_binding=marker,view=dict(status='PASS_V5_CURRENT_UNFILTERED_QUEUE',unknown_campaign_jobs=[],Hopper_current_queue_observed=False,unavailable_Hopper_is_fresh_or_empty=False,observed_epoch=1000.,complete_account_queue=q),lease_status='ACTIVE',run_STOP=False,parent_STOP=False,unknown_effects=[],pending_updates=[])
        captured.append((value,intent));return value
    result=j.call('HEARTBEAT','HEARTBEAT',publish);assert result==captured[0][0]
    for key in ('nonce','unknown','stop','old_kind','stale'):
        value,intent=deepcopy(captured[0])
        if key=='nonce':value['heartbeat']['submission_nonce']='different'
        if key=='unknown':value['unknown_effects']=['unresolved']
        if key=='stop':value['parent_STOP']=True
        if key=='old_kind':value.pop('schema')
        if key=='stale':value['view']['observed_epoch']=800.
        with pytest.raises((PermissionError,ValueError)):checked_heartbeat(value,intent,now=1000.)


def test_submit_cpu_fixture_exercises_all_directories_and_original_journal_once(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import mix_owner as owner
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.native_dataset import sha
    from types import SimpleNamespace
    root=tmp_path/'run';root.mkdir();base=root/'control/resident-owner';base.mkdir(parents=True)
    template=dict(schema='V5_CD_ACTUAL_NATIVE_TRAINING_REQUEST_V1',platform_training_version=a.VERSION,native_platform='VANDA_A40',arm='C50',stage='V5_MATCHED_TRAIN1024',parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,optimizer_update_limit=1024,query_limit=29,CPU_upper_s=54000,reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',additional_output_upper_bytes=70000000000,source_sha='a'*40,status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',job_id=None,expected_native_start_step=0,training_block=dict(arm='C50',replica=False,stop_after_step=1024,resume_checkpoint_relative=None))
    write(root/'RUN_MANIFEST.json',dict(code=dict(vla_commit='a'*40)))
    write(root/'control/COLD_INPUT_AUDIT.private.json',dict(status='PASS_ACTUAL_V5_CD_COLD_INPUTS_NO_MODEL',run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),source_sha='a'*40))
    gp=tmp_path/'global.json';write(gp,dict(PBS_submissions=[{}]*25,storage_reserved_bytes=1))
    o=owner.Owner.__new__(owner.Owner);o.root=root;o.BASE=base;o.global_path=gp;o.manifest=read(root/'RUN_MANIFEST.json');o.request_template=template;o.CONFIG=dict(programs=dict(SUBMIT='# CPU fixture, backend is intercepted'))
    o.auth=lambda secret:None;o.program=lambda name:dict(report_age_s=0,available_bytes_lower=10**15);o.fresh=lambda:dict(view=dict(campaign_jobs=[]));writes=[]
    o.write_global=lambda phase,fn:writes.append(phase)or dict(CPU_fixture_no_native_ledger_write=True)
    o.setup=owner.new_journal(base,'setup-rpc',dict(run_uuid='run',job_id='CPU_FIXTURE_NO_GPU',source_sha='a'*40,request_sha256='b'*64));calls=[]
    def backend(argv,**kw):
        assert argv==[__import__('sys').executable,'-'] and 'capture_output'in kw
        calls.append(kw['input']);return SimpleNamespace(returncode=0,stdout=json.dumps(dict(returncode=0,stdout='9999999.stdct-mgmt-02')),stderr='')
    monkeypatch.setattr(owner.subprocess,'run',backend)
    assert o.submit('')=='9999999.stdct-mgmt-02' and len(calls)==1 and writes==['REGISTER_REQUEST','BIND']
    assert (root/'pbs/alloc-26/REQUEST.json').is_file() and o.running.root.is_dir()
    assert read(o.setup.state_path)['status']=='CONFIRMED' and read(base/'SOURCE_JOB_BOUND.private.json')['job_id']=='9999999.stdct-mgmt-02'
    with pytest.raises(FileExistsError):owner.request_directory(root,26)
