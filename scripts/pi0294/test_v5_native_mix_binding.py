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
