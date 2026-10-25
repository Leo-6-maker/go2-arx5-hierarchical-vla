"""Original native terminal mechanics and C/D conservation; CPU fixtures only."""
from copy import deepcopy
import pytest
from scripts.pi0294.test_v5_core_stage_accounting import registered,started
from scripts.pi0294.pv1 import mix_accounting as a,mix_core_accounting as c,mix_core_settlement as t,successor_cd as s
from scripts.pi0294.pv1.successor_core_context import LEDGER_SCHEMA


@pytest.fixture
def actual_shape(monkeypatch):
    # Synthetic fixture lacks the old campaign header; full native captured
    # GLOBAL validation is independently required before deployment.
    monkeypatch.setattr(c,'checked',a.validate_global)
    g,l,r,m=registered();l.update(episode_starts=0,engine_starts=0,physics_ticks=dict(live=0,shadow=0));process=started(r)
    p=c.bind(g,l,request_id=r['request_id'],process=process,now_epoch=1205.);g,l=p['proposed_global'],p['proposed_LOCAL']
    args=dict(request_id=r['request_id'],native_handle=process['native_handle'])
    for kind,count,effect in [('model_loads',1,'model/0'),('generation_queries',1,'query/0'),('CPU_s',12.5,'CPU/0')]:g=c.charge(g,**args,kind=kind,count=count,effect_id=effect)
    phys=dict(schema=LEDGER_SCHEMA,run_uuid=r['child_run_uuid'],authority_release=s.TASK_SOURCE,
        Core_CPU_s=350.,main_starts=1,engine_starts=3,live_and_recovery_integrations=200,
        shadow_integrations=341,reservations={},unknown_effects=[])
    proof=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_TERMINAL_V1',request_id=r['request_id'],native_handle=process['native_handle'],
        hostname=r['hostname'],uid=2009,GPU_uuid=r['GPU_uuid'],boot_id=process['boot_id'],pid=process['pid'],proc_start=process['proc_start'],
        process_state='TERMINAL',request_sha256=r['request_sha256'],submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,terminal_inventory_stable=True,exit_evidence_kind='SYSTEMD_EXEC_MAIN_EXIT',
        native_exit_status=0,finished_epoch=1400.,observed_epoch=1401.,conservative_GPU_s=260,conservative_CPU_s=40,
        new_resident_allocated_upper_bytes=r['additional_output_upper_bytes'])
    effects=dict(schema='PV1_CORE_GPU_NATIVE_DEV_EFFECT_SUMMARY_V1',request_id=r['request_id'],native_handle=process['native_handle'],request_sha256=r['request_sha256'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],evaluation_candidate='C50',checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=a.STATS,optimizer_updates_committed=0,new_physics=0,new_episodes=0,effect_journal_complete=True,query_attempts=1,model_load_attempts=1,unknown_query_intents=[])
    delivery=dict(status='ACTUAL_CLOSED_C50_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED',checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],full_resume_files_present=False,new_GPU_model_queries_physics=0,HF_revision='e'*40,files={f'model-{i}.safetensors':{}for i in range(3)})
    return dict(g=g,l=l,r=r,physical_bytes=a.raw(phys),native_proof=proof,effects=effects,delivery_receipt_bytes=a.raw(delivery))


def call(v):
    return t.propose(a.raw(v['g']),a.raw(v['l']),request_id=v['r']['request_id'],now_epoch=1402.,**{k:v[k]for k in ('physical_bytes','native_proof','effects','delivery_receipt_bytes')})


def test_actual_paid_query_and_physics_projection_is_monotone_not_double_charged(actual_shape):
    v=actual_shape;g,l=v['g'],v['l'];first=t.sync_LOCAL(g,l,request_id=v['r']['request_id'],physical_bytes=v['physical_bytes'])
    again=t.sync_LOCAL(g,first,request_id=v['r']['request_id'],physical_bytes=v['physical_bytes'])
    assert again==first and first['CPU_s']['CoreLab']==l['CPU_s']['CoreLab']+362.5
    assert first['CPU_s']['Hopper']==l['CPU_s']['Hopper']+12.5
    assert first['physics_ticks']['live']==l['physics_ticks']['live']+200
    assert first[c.FIELD]==g[c.FIELD]and first[a.PACKAGE]==g[a.PACKAGE]
    assert first['Core_native_GPU_reserved']['physics_ticks']==l['Core_native_GPU_reserved']['physics_ticks']-541


def test_terminal_refunds_only_unused_current_holds_and_preserves_every_future_upper(actual_shape):
    v=actual_shape;old=deepcopy(v);result=call(v);g,l=result['proposed_global'],result['proposed_LOCAL'];r=g[c.FIELD][v['r']['request_id']]
    assert v==old and r['status']==c.STATES[3]and r['reserved_neural_queries']==0
    assert g['used']['H200_s']==v['g']['used']['H200_s']+260
    assert g['CPU_s']==v['g']['CPU_s']+27.5
    assert l['CPU_s']['CoreLab']==v['l']['CPU_s']['CoreLab']+390
    assert g[a.PACKAGE]==v['g'][a.PACKAGE]==l[a.PACKAGE]
    assert g[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024']
    assert g['PBS_submissions']==v['g']['PBS_submissions']
    assert l['old_history_fixture']==v['l']['old_history_fixture']
    assert not any(l['Core_native_GPU_reserved'].values())
    v.update(g=g,l=l)
    with pytest.raises(PermissionError):call(v)


@pytest.mark.parametrize('case',range(12))
def test_no_guessed_exit_unknown_effect_counter_rollback_or_old_model_terminal(actual_shape,case):
    v=actual_shape
    if case==0:v['native_proof']['recorded_process_terminal']=False
    if case==1:v['native_proof']['terminal_inventory_stable']=False
    if case==2:v['native_proof']['native_handle']='wrong'
    if case==3:v['native_proof']['observed_epoch']=1200.
    if case==4:v['native_proof']['conservative_GPU_s']=61
    if case==5:v['native_proof']['conservative_CPU_s']=1
    if case==6:v['effects']['unknown_query_intents']=[0]
    if case==7:v['effects']['optimizer_updates_committed']=1
    if case==8:v['effects']['checkpoint_manifest_sha256']='0'*64
    if case==9:v['effects']['query_attempts']=0
    if case==10:v['physical_bytes']=a.raw(dict(__import__('json').loads(v['physical_bytes']),reservations={'live':dict(maximum=200)}))
    if case==11:v['delivery_receipt_bytes']=a.raw(dict(__import__('json').loads(v['delivery_receipt_bytes']),checkpoint_manifest_sha256='0'*64))
    with pytest.raises((PermissionError,KeyError)):call(v)


def test_all_valid_negative_results_complete_only_after_both_real_lease_bytes(actual_shape,monkeypatch):
    import json
    from scripts.pi0294.test_v5_evaluation_history import evidence,groups,raw,ENDPOINT
    from scripts.pi0294.pv1 import baseline,dataset
    v=actual_shape;r=v['r'];r['checkpoint_manifest_sha256']=ENDPOINT
    for value in (v['g'],v['l']):
        value[c.FIELD][r['request_id']]['checkpoint_manifest_sha256']=ENDPOINT
        value[a.PACKAGE]['endpoints']['C50']['checkpoint_manifest_sha256']=ENDPOINT
    v['effects']['checkpoint_manifest_sha256']=ENDPOINT
    delivery=json.loads(v['delivery_receipt_bytes']);delivery['checkpoint_manifest_sha256']=ENDPOINT;v['delivery_receipt_bytes']=a.raw(delivery)
    proposal=call(v);g,l=proposal['proposed_global'],proposal['proposed_LOCAL'];row=g[c.FIELD][r['request_id']]
    population=raw(dict(groups=groups()));pin=a.sha(population);monkeypatch.setattr(baseline,'POPULATION',pin);monkeypatch.setattr(dataset,'POPULATION',pin)
    lease=a.raw(dict(status='CLOSED',run_uuid=r['child_run_uuid'],authority_release=s.TASK_SOURCE,terminal_native_sha256=row['terminal_native_sha256'],all_actual_workers_terminal=True,closed_epoch=1403.))
    kwargs=dict(request_id=r['request_id'],population_bytes=population,evidence=evidence('C50'),GPU_lease_bytes=lease,physical_lease_bytes=lease,now_epoch=1404.)
    out=t.complete_stage(a.raw(g),a.raw(l),**kwargs)
    assert out['proposed_global'][a.PACKAGE]['completed_stages']==list(s.ORDER[:2])
    assert out['receipt']['history']['successes']==dict(T1=0,T2=0)
    for change in [dict(evidence=kwargs['evidence'][:12]),dict(GPU_lease_bytes=a.raw(dict(json.loads(lease),status='ACTIVE'))),dict(physical_lease_bytes=a.raw(dict(json.loads(lease),terminal_native_sha256='0'*64)))]:
        with pytest.raises(PermissionError):t.complete_stage(a.raw(g),a.raw(l),**dict(kwargs,**change))
