"""Saved actual CPU bytes + declared future GPU fixtures, no model/PBS work."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_training_accounting as a,amd_learning,amd_training as t
from scripts.pi0294.test_pv1_recipe import old_recipe

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def actual():
    p=B/'GEN1_VANDA_OFFICIAL_EXPORT_GIT_REPAIR02_PARENT04_SEAL_FAILURE_CLOSED_INDEPENDENT_READBACK.private.json'
    if not p.exists():pytest.skip('actual saved-data failure close readback unavailable')
    closed=json.loads(p.read_bytes());g=json.loads(base64.b64decode(closed['GLOBAL_base64']))
    capture=json.loads((B/'GEN1_CPU1497612_SAVED_FAILURE_FULL_AUDIT_CAPTURE.private.json').read_bytes())
    packet=base64.b64decode(capture['files']['plans/GEN1_V5_PREVIEW_V2.private.json']['base64'])
    source=a.closed_saved_exports(g,capture,lease_bytes=base64.b64decode(closed['lease_base64']),packet_bytes=packet)
    local=base64.b64decode(json.loads((B/'GEN1_VANDA_OFFICIAL_EXPORT_PREPARED.private.json').read_bytes())['historical_LOCAL_base64'])
    r=dict(schema='GEN1_ACTUAL_NATIVE_TRAINING_REQUEST_V1',platform_training_version=a.VERSION,
        native_platform='VANDA_A40',arm=a.bank.IDENTITY,stage='GEN1_INDEPENDENT_TRAIN1024',
        scope_grant_sha256=a.authority.GRANT_SHA,parent_checkpoint_manifest_sha256=a.original.PARENT,
        statistics_sha256=a.original.STATS,optimizer_update_limit=1024,query_limit=29,CPU_upper_s=54000,
        reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',
        project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',additional_output_upper_bytes=70000000000,
        expected_native_start_step=0,training_block=dict(arm=a.bank.IDENTITY,replica=False,stop_after_step=1024,resume_checkpoint_relative=None),
        request_id='11111111-1111-4111-8111-111111111111',submission_nonce='22222222-2222-4222-8222-222222222222',
        source_sha='a'*40,request_sha256='b'*64,run_manifest_sha256='c'*64,cold_input_audit_sha256='d'*64,
        closed_export_sha256=a.sha(a.raw(source)),bank_manifest_sha256=source['bank_manifest_sha256'],
        CPU_bank_deadline_epoch=source['operation_deadline_epoch'],
        recipe_sha256=a.sha(a.raw(amd_learning.recipe(old_recipe()))),status=a.STATES[0],job_id=None,
        T_CD=g[a.authority.FIELD]['T_CD'],deadline_epoch=g[a.authority.FIELD]['deadline_epoch'],
        reserved_optimizer_updates=1024,reserved_neural_queries=29)
    return dict(g=g,r=r,closed=source,original_recipe=old_recipe(),historical_LOCAL_bytes=local,now_epoch=closed['epoch']+1,
        capture=capture,lease_bytes=base64.b64decode(closed['lease_base64']),packet_bytes=packet)

def register(p):return a.register(**{k:p[k]for k in ('g','r','closed','original_recipe','historical_LOCAL_bytes','now_epoch')})

def test_actual_saved_export_closure_is_only_cpu_pack_input_not_model_or_training_permission(actual):
    x=actual['closed'];assert not x['complete_bank_ready']and x['CPU_pack_required_before_model_release']
    assert not x['native_training_permission']and x['support']['missing']==[]and len(x['saved_numeric_members'])==10
    assert x['bank_manifest_sha256']=='4dd8a677a00b21ecbea6f8649ce17ebce8e68811741c3c14b5dd5bef58138f64'

def test_full_actual_forecast_registers_one_GEN_train_keeps_old_cost_package_matrix_and_final(actual):
    before=deepcopy(actual['g']);v=register(actual);g=v['proposed_global'];proof=v['full_remaining_proof']
    assert not proof['cap_changes_required']and not v['native_admission']and v['original_C_D_package_unchanged']
    assert g[a.original.PACKAGE]==before[a.original.PACKAGE]and g['used']['PBS_submissions']==33
    assert proof['dimensions']['PBS_submissions']['projected']==36
    assert all(x['projected']<=x['cap']for x in proof['dimensions'].values())
    assert proof['matrix3_and_final20GPUh40GiB_retained']
    assert g['reserved']==dict(before['reserved'],H200_s=before['reserved']['H200_s']+22560,
        optimizer_updates=1024,neural_queries=before['reserved']['neural_queries']+29)
    assert g['CPU_s']==before['CPU_s']+1800 and g['CPU_reserved_s']==before['CPU_reserved_s']+54000
    assert g['new_storage_current_bytes']==before['new_storage_current_bytes']+64*1024**2
    assert g['storage_reserved_bytes']==before['storage_reserved_bytes']+70000000000
    assert actual['g']==before
    from scripts.pi0294.pv1.neural_ledger import check
    check(g)
    with pytest.raises(PermissionError):register(dict(actual,g=g))

@pytest.mark.parametrize('bad',['C50','parent','recipe','bank','source_active','unknown','late','LOCAL','lease'])
def test_real_data_recipe_scope_and_prior_terminal_cannot_be_bypassed(actual,bad):
    p=deepcopy(actual)
    if bad=='C50':p['r']['arm']='C50'
    elif bad=='parent':p['r']['parent_checkpoint_manifest_sha256']='0'*64
    elif bad=='recipe':p['r']['recipe_sha256']='0'*64
    elif bad=='bank':p['r']['bank_manifest_sha256']='0'*64
    elif bad=='source_active':a.export.rows(p['g'])[0]['status']=a.STATES[1]
    elif bad=='unknown':p['g']['unknown_effects']=['transport_unknown']
    elif bad=='late':p['now_epoch']=p['r']['deadline_epoch']
    elif bad=='LOCAL':p['historical_LOCAL_bytes']=b'{}'
    else:
        lease=json.loads(p['lease_bytes']);lease['status']='ACTIVE'
        with pytest.raises(PermissionError):a.closed_saved_exports(p['g'],p['capture'],lease_bytes=a.raw(lease),packet_bytes=p['packet_bytes'])
        return
    with pytest.raises((PermissionError,KeyError)):register(p)

def test_effects_settle_actual_delta_once_and_full_failed_bound_no_C_D_stage_advance(actual):
    g=register(actual)['proposed_global'];r=a.rows(g)[0];g=a.bind(g,r['request_id'],'123.stdct-mgmt-02',now_epoch=actual['now_epoch']+1)
    n=dict(job_state='F',job_id='123.stdct-mgmt-02',request_id=r['request_id'],submission_nonce=r['submission_nonce'],Exit_status=0,
        conservative_GPU_s=8500,conservative_CPU_s=8200)
    e=dict(schema=a.EFFECTS,request_id=r['request_id'],request_sha256=r['request_sha256'],job_id=n['job_id'],arm=a.bank.IDENTITY,
        pending_updates=[],unknown_query_intents=[],optimizer_updates_committed=1024,query_attempts=29,model_load_attempts=3)
    audit=dict(status=a.AUDIT,arm=a.bank.IDENTITY,actual_consumed_cursor=32768,parent_checkpoint_manifest_sha256=a.original.PARENT,statistics_sha256=a.original.STATS)
    out=a.settle_training(g,r['request_id'],native=n,effects=e,audit=audit,now_epoch=actual['now_epoch']+9000)
    assert out['optimizer_updates']==14786 and out['used']['H200_s']==158843
    assert out[a.original.PACKAGE]==g[a.original.PACKAGE]and a.rows(out)[0]['scientific_endpoint_complete']
    assert out['new_storage_current_bytes']==g['new_storage_current_bytes']+70000000000
    assert out['generation_queries']==4511 and out['model_loads']==g['model_loads']+3
    with pytest.raises(PermissionError):a.settle_training(out,r['request_id'],native=n,effects=e,audit=audit,now_epoch=1)

def test_same_mature_effect_journal_and_known_pending_1024_only():
    from scripts.pi0294.generalization.amd_training_runtime import Effects
    from scripts.pi0294.dec1.a40_native_ledger import Ledger
    assert Effects.charge is Ledger.charge and Effects.transaction is Ledger.transaction
    x=Effects.__new__(Effects);x.request=dict(source_sha='a'*40,request_id='one');x.allocation=dict(request_sha256='b'*64,job_id='1.stdct-mgmt-02')
    v=dict(schema='GEN1_LOCAL_NATIVE_EFFECT_LEDGER_V1',platform_training_version=a.VERSION,arm=a.bank.IDENTITY,start_native_step=0,
        source_sha='a'*40,request_sha256='b'*64,job_id='1.stdct-mgmt-02',unknown_effects=[],optimizer_updates=0,
        generation_queries=0,model_loads=0,query_intent_links=[],CPU_s=0.,pending_updates=[],training_runs={})
    assert x.check(v)==v
    pending=t.begin_update(v,arm=a.bank.IDENTITY,local_step=0,request_id='one',recipe=amd_learning.recipe(old_recipe()))
    with pytest.raises(PermissionError):x.check(pending)
    v['generation_queries']=17;v['query_intent_links']=[{}]*17
    x.check(t.begin_update(v,arm=a.bank.IDENTITY,local_step=0,request_id='one',recipe=amd_learning.recipe(old_recipe())))

def test_saved_numeric_copy_seals_new_file_preserves_original_mode_inode_and_rejects_replay(tmp_path):
    from scripts.pi0294.generalization.amd_training_pack import copy_sealed
    from scripts.pi0294.native_dataset import sha
    source=tmp_path/'old.npz';source.write_bytes(b'real temporary CPU bytes');source.chmod(0o600)
    st=source.stat();target=tmp_path/'new/numeric.npz';r=copy_sealed(source,target,sha(source))
    assert target.stat().st_mode&0o777==0o400 and source.stat().st_mode&0o777==0o600
    assert source.stat().st_ino==st.st_ino and target.stat().st_ino!=st.st_ino and target.read_bytes()==source.read_bytes()
    with pytest.raises(FileExistsError):copy_sealed(source,target,sha(source))
    link=tmp_path/'symlink';link.symlink_to(source)
    with pytest.raises(PermissionError):copy_sealed(link,tmp_path/'wrong',sha(source))
    with pytest.raises(PermissionError):copy_sealed(source,tmp_path/'wrong','0'*64)

def test_no_module_level_model_import_and_same_official_trainer_loop():
    import ast,inspect
    from scripts.pi0294.generalization import amd_training_runtime as runtime,amd_training_owner as owner
    for mod in [runtime,owner]:
        tree=ast.parse(inspect.getsource(mod))
        assert not any((getattr(n,'module','')or'').startswith(('torch','transformers','mujoco'))for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    assert 'mix_training import run as original'in inspect.getsource(t.run)
    assert 'cold_owner.watch(owner,secret,scope_binding=owner.successor_scope)'in inspect.getsource(owner.main)

def test_original_journal_accepts_only_exact_GEN_native_heartbeat_and_resolves_lost_reply_readonly(tmp_path):
    from scripts.pi0294.dec1.owner_transport import Journal,checked_heartbeat
    from scripts.pi0294.dec1.test_vanda_campaign import fixture
    _,_,_,q=fixture();jobs=json.loads(q['qstat']['stdout'])['Jobs'];job='14.stdct-mgmt-02'
    jobs[job]['Variable_List']={'PV1_REQUEST_ID':'request','PV1_SUBMISSION_NONCE':'nonce'}
    q['observed_epoch']=1000.;q['qstat']['stdout']=json.dumps(dict(Jobs=jobs));root=tmp_path/'rpc';root.mkdir()
    identity=dict(run_uuid='fixture',job_id=job,source_sha='a'*40,request_sha256='b'*64)
    journal=Journal(root,identity,clock=lambda:1000.);published=[];writes=[]
    def publish(intent,limit):
        value=dict(schema='GEN1_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1',
            heartbeat=dict(identity,request_id='request',submission_nonce='nonce',scope=a.VERSION,grant_sha256=a.authority.GRANT_SHA,
                rpc_intent_id=intent['operation_id'],observed_epoch=1000.,native_job_state=jobs[job]['job_state'],global_unknown_effects=[]),
            view=dict(status='PASS_V5_CURRENT_UNFILTERED_QUEUE',unknown_campaign_jobs=[],Hopper_current_queue_observed=False,
                unavailable_Hopper_is_fresh_or_empty=False,observed_epoch=1000.,complete_account_queue=q),
            lease_status='ACTIVE',run_STOP=False,parent_STOP=False,unknown_effects=[],pending_updates=[])
        published.append((value,intent));writes.append(1);raise TimeoutError('lost native heartbeat reply')
    value=journal.call('HEARTBEAT','HEARTBEAT',publish,readback=lambda intent,limit:published[-1][0]);assert len(writes)==1
    for key in ('nonce','unknown','stop','scope','stale'):
        v,intent=deepcopy(published[0])
        if key=='nonce':v['heartbeat']['submission_nonce']='wrong'
        if key=='unknown':v['unknown_effects']=['unresolved']
        if key=='stop':v['run_STOP']=True
        if key=='scope':v['heartbeat']['scope']='V5_CD'
        if key=='stale':v['view']['observed_epoch']=800.
        with pytest.raises((PermissionError,ValueError)):checked_heartbeat(v,intent,now=1000.)

def test_original_submit_entry_factory_keeps_wall_CPU_mail_and_frozen_request(actual):
    from scripts.pi0294.generalization import amd_training_launch as launch
    p=json.loads((B/'D25_VANDA_FIRST_PREPARED.private.json').read_bytes());program=launch.programs(p['programs'])
    assert 'scripts.pi0294.generalization.amd_training_runtime'in program['SUBMIT']
    assert 'timeout=30'in program['SUBMIT']and 'timeout=45'not in program['SUBMIT']
    assert program['AUTH']==p['programs']['AUTH']and program['QUOTA']==p['programs']['QUOTA']
    from scripts.pi0294.test_v5_native_mix_binding import test_request_binds_actual_manifest_and_audit_after_sealed_template
    r=dict(actual['r']);r.pop('request_sha256');r['bank_manifest_sha256']='0'*64
    # A factory takes real closed identity, never the old C/D bank field.
    v=launch.template(r,source_sha='a'*40,run_uuid='fixture',request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        scope_marker=actual['g'][a.authority.FIELD],closed=actual['closed'],original_recipe=old_recipe())
    assert v['bank_manifest_sha256']==actual['closed']['bank_manifest_sha256']and v['arm']==a.bank.IDENTITY
    assert not {'run_manifest_sha256','cold_input_audit_sha256','request_sha256'}&set(v)

def test_child_CPU_guard_changes_cannot_reach_parent_interpreter_or_GPU_environment(tmp_path,monkeypatch):
    import subprocess,sys,inspect
    from scripts.pi0294.generalization import amd_training_runtime as runtime
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','fixture-allocation')
    p=subprocess.run([sys.executable,'-c',"import os;os.environ['CUDA_VISIBLE_DEVICES']='';print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        text=True,capture_output=True,check=True)
    assert p.stdout=='\n'and os.environ['CUDA_VISIBLE_DEVICES']=='fixture-allocation'
    source=inspect.getsource(runtime.run)
    assert 'subprocess.run([sys.executable'in source and 'prepare(root,'not in source
    assert source.index('CPU_BANK_PRELUDE_OS_EXIT.json')<source.index('wait_release(')<source.index('model.load(')

def test_model_release_refuses_missing_complete_bank_before_auth_or_any_publication(tmp_path):
    from scripts.pi0294.generalization.amd_training_owner import Owner
    base=tmp_path/'owner';base.mkdir();d=tmp_path/'pbs';d.mkdir();(tmp_path/'plans').mkdir()
    (base/'SOURCE_REQUEST.private.json').write_text(json.dumps(dict(request={},directory=str(d))))
    (base/'SOURCE_JOB_BOUND.private.json').write_text(json.dumps(dict(job_id='1.stdct-mgmt-02')))
    (tmp_path/'plans/GEN1_TRAIN_INPUTS.private.json').write_text(json.dumps(dict(dataset=str(tmp_path/'datasets/FROZEN_GEN1_BANK'))))
    owner=Owner.__new__(Owner);owner.root=tmp_path;owner.BASE=base
    with pytest.raises(FileNotFoundError):owner.release({},secret=None)
    assert not(d/'START_NEURAL.json').exists()
