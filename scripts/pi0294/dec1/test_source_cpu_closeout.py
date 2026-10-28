"""Terminal-bound financial-only proposals on synthetic immutable supplied bytes."""
from copy import deepcopy
import json
import pytest
from scripts.pi0294.dec1 import source_cpu_closeout as unit


def enc(v):return unit.encode(v)
def dims(**v):return dict.fromkeys(unit.intake.DIMS,0)|v


@pytest.fixture
def packet(monkeypatch):
    t0=1792348196;now=1792706200.;request='synthetic-source-financial-01';cid='SYNTHETIC_COMPONENT';run='synthetic-source-01'
    amount=dims(CoreLab_CPU_s=15540,local_CPU_s=720,episode_starts=4,physics_ticks=128000,constructors=326,
        CoreLab_output_bytes=3506438144,local_output_bytes=67108864,transfer_bytes=1073741824)
    m=dict(scope=unit.intake.ROLE,owner='EXISTING_LOCAL_PC',run_uuid=run,
        authority=dict(protocol_ref=unit.intake.PARENT_AUTHORITY),code=dict(vla_commit='b'*40),
        limits=dict(PV1_T0_epoch=t0,PV1_hard_close_epoch=unit.intake.HARD_CLOSE,operation_deadline_epoch=now+3600))
    budget=dict(schema=unit.intake.BUDGET_SCHEMA,native_complete=True,request_id=request,requested=amount,
        GR00T_queries=0,GPU_submissions=0,training_updates=0,components=dict(
            registration=dims(CoreLab_CPU_s=120),control=dims(CoreLab_CPU_s=300),audit_closeout=dims(CoreLab_CPU_s=600)))
    plan=dict(conditions=[dict(condition_id=cid,card_sha256='c'*64)])
    plan_sha=unit.sha(enc(plan));monkeypatch.setattr(unit.intake,'PLAN_SHA256',plan_sha)
    row=dict(schema='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1',request_id=request,role=unit.intake.ROLE,
        authority_release=unit.intake.AUTHORITY,unknown_effects=[],status='RESERVED',child_run_uuid=run,
        campaign_run_uuid='synthetic-campaign',child_manifest_sha256=unit.sha(enc(m)),
        budget_sha256=unit.sha(enc(budget)),component_plan_sha256=plan_sha,requested=deepcopy(amount),remaining=deepcopy(amount))
    p=dict(run_uuid='synthetic-campaign',T0_epoch=t0,hard_closeout_epoch=unit.intake.HARD_CLOSE,
        unknown_effects=[],Hopper_unknown_effects=[],CPU_s=dict(CoreLab=1000.,local_PC=2000.,Hopper=3000.),
        CPU_reserved_s=dict(CoreLab=amount['CoreLab_CPU_s'],local_PC=amount['local_CPU_s'],Hopper=0),
        episode_starts=42,engine_starts=100,physics_ticks=dict(live=40000,shadow=40000),network_transfer_accounted_bytes=30000,
        source_cpu_reservations={request:deepcopy(row)},source_cpu_reserved=deepcopy(amount),
        children=[dict(name=run,source_budget_request_id=request,CPU_reserved_s=amount['CoreLab_CPU_s'],local_CPU_reserved_s=amount['local_CPU_s'],
            episode_starts_reserved=4,physics_ticks_reserved=128000,engine_starts_reserved=326)],
        immutable_GPU_history=dict(failed_jobs=['preserved'],updates=3843,checkpoints=['all-retained']))
    c=dict(schema='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1',authority_release=unit.intake.AUTHORITY,
        campaign_run_uuid=p['run_uuid'],original_T0_epoch=t0,hard_close_epoch=unit.intake.HARD_CLOSE,unknown_effects=[],
        CPU_s=100.,episode_starts=0,physics_ticks=0,constructors=0,source_cpu_reservations={request:deepcopy(row)},
        source_cpu_reserved=deepcopy(amount),source_cpu_current_classification=dims(CoreLab_CPU_s=100.,local_CPU_s=2000.,
            CoreLab_output_bytes=1000,local_output_bytes=2000,transfer_bytes=30000),old_GPU_history=p['immutable_GPU_history'])
    child=dict(schema='PI0294_PV1_CORE_POLICY_LEDGER_V1',run_uuid=run,authority_release=unit.intake.PARENT_AUTHORITY,
        source_budget_request_id=request,Core_CPU_s=139.158001888,main_starts=1,engine_starts=6,
        live_and_recovery_integrations=0,shadow_integrations=490,
        reservations={cid:dict(kind='live_and_recovery_integrations',maximum=16000,started_epoch=now-100)},
        unknown_effects=[dict(identifier=cid,error='SourceComponentUnknown',reservation_retained=True)])
    lease=dict(status='ACTIVE',owner='EXISTING_LOCAL_PC',authority_release=unit.intake.PARENT_AUTHORITY,
        run_uuid=run,deadline_epoch=m['limits']['operation_deadline_epoch'])
    artifacts={'RESULT.json':dict(id=cid,fixture=False,source_sha=m['code']['vla_commit'],card_sha256='c'*64,
        status='FAILED_PRESERVED',physics_steps=400),
        'S1_RESULT.json':dict(episode_id=cid,engineering_unknown=True,actual_stop_event=None,
            actual_stop_state=dict(stop_acknowledged=False,unknown=True),
            execution_state=dict(unknown=dict(shadow=False)),component_meter_unknown=False,shadow_integrations=490,native_live_integrations=400,live_integrations_lower_bound=400),
        'SOURCE_COMPONENT_STOP.json':dict(episode_id=cid,stopped=True,native_frame_inflight=False,native_steps=400)}
    handles=[]
    for index,(name,rc) in enumerate((('counter',0),('component',1))):
        common=dict(pid=100+index,proc_start=str(200+index),source_sha=m['code']['vla_commit'])
        handles += [dict(relative='worker_processes/'+name+'.json',sha256='a'*64,value=common,active=False),
            dict(relative='worker_processes/'+name+'.exit.json',sha256='a'*64,
                value=dict(common,exit_status=rc,process_CPU_s=3.1 if rc==0 else 16.5,manifest_sha256=unit.sha(enc(m)),run_uuid=run),active=False)]
    f=dict(status='READ_ONLY_TERMINAL_ACTORS_AND_PRESERVED_FINANCIAL_BOUND_INPUTS',new_writes=0,new_native_integrations=0,
        manifest_sha256=unit.sha(enc(m)),returned_STOP_ACK=False,scientific_classification=unit.VERDICT,
        observed_epoch=now-10,parent_Core_ledger=deepcopy(c),parent_Core_ledger_sha256=unit.sha(enc(c)),
        child_ledger=child,child_ledger_sha256=unit.sha(enc(child)),child_lease=lease,child_lease_sha256=unit.sha(enc(lease)),
        handles=handles,highrate_shape=[400,45],native_files={n:dict(sha256=unit.sha(enc(v)),bytes=len(enc(v))) for n,v in artifacts.items()})
    f['native_files'].update({'highrate.npz':dict(sha256='d'*64,bytes=10000),'SOURCE_COMPONENT_EVENTS.jsonl':dict(sha256='e'*64,bytes=20000)})
    data=dict(parent=p,Core=c,manifest=m,budget=budget,plan=plan,terminal=f,artifacts=artifacts,now=now,request=request)
    def bind():
        # Exact latest supplied Core/child bytes are bound, never guessed.
        f['parent_Core_ledger']=deepcopy(c);f['parent_Core_ledger_sha256']=unit.sha(enc(c))
        return dict(parent_bytes=enc(p),Core_bytes=enc(c),manifest_bytes=enc(m),budget_bytes=enc(budget),plan_bytes=enc(plan),
            terminal_bytes=enc(f),artifact_bytes={n:enc(v) for n,v in artifacts.items()},settlement_id=request+'-terminal-bound-close',settled_epoch=now)
    data['bind']=bind;return data


def test_full_unknown_live_upper_charged_once_unstarted_work_not_duplicated_child_and_GPU_unchanged(packet):
    f=packet;args=f['bind']();before=deepcopy(f['terminal']['child_ledger']);r=unit.propose(**args)
    p,c,invoice=r['proposed_parent'],r['proposed_Core'],r['invoice'];amount=f['budget']['requested']
    charged=invoice['financial_charged'];assert charged['physics_ticks']==16490 and invoice['scientific_verdict']==unit.VERDICT
    assert charged['CoreLab_CPU_s']==120+3.1+16.5+900
    assert invoice['unstarted_or_known_unused']['episode_starts']==3
    assert invoice['observed']['component_attempts']==1 and invoice['financial_charged']['episode_starts']==1
    assert p['CPU_s']['CoreLab']==f['parent']['CPU_s']['CoreLab']+charged['CoreLab_CPU_s']
    assert p['CPU_s']['local_PC']==f['parent']['CPU_s']['local_PC']+amount['local_CPU_s']
    assert p['CPU_reserved_s']==dict(CoreLab=0,local_PC=0,Hopper=0)
    assert p['physics_ticks']['live']==40000 and p['physics_ticks']['shadow']==40000
    assert p['physics_ticks']['closed_source_component_financial_upper']==16490
    assert c['CPU_s']==100.+charged['CoreLab_CPU_s'] and c['physics_ticks']==16490 and c['constructors']==6
    assert p['immutable_GPU_history']==f['parent']['immutable_GPU_history'] and c['old_GPU_history']==f['Core']['old_GPU_history']
    assert f['terminal']['child_ledger']==before and before['unknown_effects'] and before['reservations']
    assert not invoice['actual_returned_STOP_ACK'] and not invoice['source_or_target_qualified'] and not r['native_dispatch_admitted']
    assert r['original_child_ledger_must_remain_unchanged']
    # Returned copies cannot mutate the caller's raw factual evidence.
    p['source_cpu_financial_closeouts'][f['request']]['original_live_reservation_and_unknowns']['unknown_effects'].clear()
    assert f['terminal']['child_ledger']['unknown_effects']


def test_settled_request_refuses_second_financial_charge(packet):
    a=packet['bind']();r=unit.propose(**a)
    a.update(parent_bytes=enc(r['proposed_parent']),Core_bytes=enc(r['proposed_Core']))
    with pytest.raises(PermissionError,match='UNSETTLED'):unit.propose(**a)


@pytest.mark.parametrize('fault',['active','missing_exit','wrong_exit','wrong_start','old_read','parent_unknown',
    'different_Core_bytes','artifact_changed','claimed_stop_ack','claimed_scientific_pass','missing_raw',
    'wrong_partial_count','unknown_erased','reservation_erased','small_CPU_hold','small_component_bound',
    'aggregate_changed','other_row_bad','other_settled_nonzero','wrong_manifest','wrong_clock','neural_budget'])
def test_unresolved_or_mismatched_effects_never_produce_a_financial_release(packet,fault):
    f=packet;t=f['terminal'];p=f['parent'];c=f['Core'];r=f['request']
    if fault=='active':t['handles'][0]['active']=True
    if fault=='missing_exit':t['handles'].pop()
    if fault=='wrong_exit':t['handles'][-1]['value']['exit_status']=0
    if fault=='wrong_start':t['handles'][-1]['value']['proc_start']='different'
    if fault=='old_read':t['observed_epoch']=f['now']-901
    if fault=='parent_unknown':p['unknown_effects']=['unknown write']
    if fault=='different_Core_bytes':t['parent_Core_ledger_sha256']='f'*64
    if fault=='artifact_changed':f['artifacts']['RESULT.json']['physics_steps']=401
    if fault=='claimed_stop_ack':t['returned_STOP_ACK']=True
    if fault=='claimed_scientific_pass':t['scientific_classification']='PASS'
    if fault=='missing_raw':t['native_files'].pop('highrate.npz')
    if fault=='wrong_partial_count':t['highrate_shape'][0]=401
    if fault=='unknown_erased':t['child_ledger']['unknown_effects']=[]
    if fault=='reservation_erased':t['child_ledger']['reservations']={}
    if fault=='small_CPU_hold':p['CPU_reserved_s']['CoreLab']=1
    if fault=='small_component_bound':t['child_ledger']['Core_CPU_s']=20000
    if fault=='aggregate_changed':p['source_cpu_reserved']['physics_ticks']+=1
    if fault=='other_row_bad':p['source_cpu_reservations']['untyped']={}
    if fault=='other_settled_nonzero':
        extra=deepcopy(p['source_cpu_reservations'][r]);extra['request_id']='other';extra['status']='SETTLED';p['source_cpu_reservations']['other']=extra
    if fault=='wrong_manifest':f['manifest']['scope']='ANOTHER_ROLE'
    if fault=='wrong_clock':p['T0_epoch']+=1
    if fault=='neural_budget':f['budget']['GR00T_queries']=1
    a=f['bind']()
    # Rebind hashes ONLY for controlled fixtures testing semantic violations.
    if fault in ('small_component_bound','unknown_erased','reservation_erased'):
        t['child_ledger_sha256']=unit.sha(enc(t['child_ledger']));a['terminal_bytes']=enc(t)
    if fault=='different_Core_bytes':
        t['parent_Core_ledger_sha256']='f'*64;a['terminal_bytes']=enc(t)
    before=deepcopy(p)
    with pytest.raises((PermissionError,KeyError)):unit.propose(**a)
    assert p==before
