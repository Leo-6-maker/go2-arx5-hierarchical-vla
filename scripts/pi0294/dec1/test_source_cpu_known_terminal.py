"""UNRUN CPU fixture source; pure copied settlement, never native evidence."""
from copy import deepcopy
import pytest
from scripts.pi0294.dec1 import source_cpu_known_terminal as unit
from scripts.pi0294.dec1.test_source_cpu_closeout import packet as legacy_packet, enc, dims


@pytest.fixture
def known(legacy_packet,monkeypatch):
    f=legacy_packet;p=f['parent'];c=f['Core'];b=f['budget'];m=f['manifest'];t=f['terminal']
    monkeypatch.setattr(unit,'SOURCE_REQUEST',f['request'])
    monkeypatch.setattr(unit,'SOURCE_MANIFEST_SHA',unit.sha(enc(m)))
    monkeypatch.setattr(unit,'SOURCE_RELEASE',m['code']['vla_commit'])
    ids=['SYNTHETIC_CONDITION_'+str(i) for i in range(4)]
    f['plan']['conditions']=[dict(condition_id=cid,card_sha256='c'*64) for cid in ids]
    plan_sha=unit.sha(enc(f['plan']));monkeypatch.setattr(unit.intake,'PLAN_SHA256',plan_sha)
    amount=b['requested'];components={n:dims() for n in unit.intake.COMPONENTS}
    components['registration']['CoreLab_CPU_s']=120
    components['control'].update(CoreLab_CPU_s=300,local_CPU_s=amount['local_CPU_s'])
    components['audit_closeout']['CoreLab_CPU_s']=600
    components['counter_qualification']['constructors']=2
    components['main'].update({k:amount[k]-sum(v[k] for v in components.values()) for k in amount})
    b['components']=components
    for ledger in (p,c):
        row=ledger['source_cpu_reservations'][f['request']]
        row.update(component_plan_sha256=plan_sha,budget_sha256=unit.sha(enc(b)))
        old=deepcopy(row);old.update(request_id=unit.history.OLD_REQUEST,child_run_uuid=unit.history.OLD_REQUEST,
            status='SETTLED',remaining=dims(),scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',
            financial_closeout_sha256='e'*64,full_unresolved_live_bound_charged=True)
        ledger['source_cpu_reservations'][unit.history.OLD_REQUEST]=old
        ledger['source_cpu_financial_closeouts']={unit.history.OLD_REQUEST:dict(
            financial_charged=dims(physics_ticks=16490),scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',
            original_live_reservation_and_unknowns=dict(reservations={'old':dict(maximum=16000)},unknown_effects=['RETAINED']))}
    invoice=p['source_cpu_financial_closeouts'][unit.history.OLD_REQUEST]
    monkeypatch.setattr(unit.history,'INVOICE_SHA',unit.sha(enc(invoice)))
    for ledger in (p,c):
        ledger['source_cpu_reservations'][unit.history.OLD_REQUEST]['financial_closeout_sha256']=unit.history.INVOICE_SHA
    p['physics_ticks']['closed_source_component_financial_upper']=16490
    c['physics_ticks']=16490;c['episode_starts']=1;c['constructors']=6
    c['source_cpu_current_classification'].update(physics_ticks=16490,episode_starts=1,constructors=6)
    statuses=['COMPONENT_COMPLETE','KNOWN_REFERENCE_OR_S1_REJECT','NATIVE_SAFETY_FAIL','NOT_RUN']
    rows=[dict(condition_id=cid,status=status,main_integrations=live,shadow_integrations=shadow)
        for cid,status,live,shadow in zip(ids,statuses,[16000,0,400,0],[16000,25,400,0])]
    # NOT_RUN has no alleged physics fields, receipt, stop or attempt.
    rows[-1]={'condition_id':ids[-1],'status':'NOT_RUN'}
    audit=dict(schema='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1',conditions=rows,
        component_counts={s:statuses.count(s) for s in set(statuses)},
        purpose='SOURCE_INTERFACE_COMPONENT_QUALIFICATION_ONLY',student_neural_ACK_qualification=False,
        target_qualification=False,core64_cells=0,new_training_data=False)
    child=t['child_ledger'];child.update(unknown_effects=[],reservations={},main_starts=3,
        live_and_recovery_integrations=16400,shadow_integrations=16425,engine_starts=86)
    t.update(schema=unit.TERMINAL_SCHEMA,status='READ_ONLY_ALL_SOURCE_ACTORS_TERMINAL',
        source_sha=m['code']['vla_commit'],all_owned_actors_terminal=True,active_workers=0,unknown_effects=[])
    t['handles']=[]
    for i,name in enumerate(['source-counter-01',*('component-'+r['condition_id'] for r in rows[:3])]):
        value=dict(pid=90000000+i,proc_start=str(200+i),source_sha=m['code']['vla_commit'])
        t['handles'] += [dict(relative='worker_processes/'+name+'.json',value=value,active=False),
            dict(relative='worker_processes/'+name+'.exit.json',value=dict(value,exit_status=0,
                process_CPU_s=3.1,manifest_sha256=unit.sha(enc(m)),run_uuid=m['run_uuid']),active=False)]
    attempt_bytes={};t['attempts']=[]
    for r in rows[:3]:
        cid=r['condition_id'];anchor=dict(schema='DEC1_SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR_V1',
            condition_id=cid,component_plan_sha256=plan_sha,manifest_sha256=unit.sha(enc(m)),
            runtime_release=m['code']['vla_commit'],owner_closeout_known=True,
            engine_sha256=unit.intake.ORIGINAL_13['scripts/pi0291_arx5_data/engine.py'])
        ar=enc(anchor);rr=dict(schema='DEC1_SOURCE_COMPONENT_ATTEMPT_V1',condition_id=cid,attempt_id=cid,
            component_plan_sha256=plan_sha,terminal_anchor_sha256=unit.sha(ar),
            native_STOP_return_and_owner_event_confirmed=True,main_integrations=r['main_integrations'],
            shadow_integrations=r['shadow_integrations'],student_neural_ACK_qualification=False,
            GR00T_queries=0,new_training_data=False)
        rr_raw=enc(rr);attempt_bytes[cid]=dict(terminal_anchor=ar,attempt=rr_raw)
        t['attempts'].append(dict(condition_id=cid,status=r['status'],manifest_sha256=unit.sha(enc(m)),
            source_sha=m['code']['vla_commit'],plan_sha256=plan_sha,terminal_anchor_sha256=unit.sha(ar),
            attempt_sha256=unit.sha(rr_raw),actual_stop_returned=True,audit_file_hashes_verified=True,unknown_effects=[]))
    def bind():
        actors={}
        for handle in t['handles']:
            raw=enc(handle['value']);handle['sha256']=unit.sha(raw);actors[handle['relative']]=raw
        t.update(plan_sha256=unit.sha(enc(f['plan'])),audit_sha256=unit.sha(enc(audit)),
            child_ledger_sha256=unit.sha(enc(child)),parent_sha256=unit.sha(enc(p)),Core_sha256=unit.sha(enc(c)))
        return dict(parent_bytes=enc(p),Core_bytes=enc(c),manifest_bytes=enc(m),budget_bytes=enc(b),
            plan_bytes=enc(f['plan']),audit_bytes=enc(audit),child_ledger_bytes=enc(child),terminal_bytes=enc(t),
            actor_bytes=actors,attempt_bytes=deepcopy(attempt_bytes),settlement_id=f['request']+'-known-terminal-close',
            settled_epoch=f['now'])
    f.update(audit=audit,attempt_bytes=attempt_bytes,known_bind=bind)
    return f


def test_known_success_reject_safety_and_notrun_charge_actual_separate_domains_once(known):
    f=known;before=deepcopy((f['parent'],f['Core'],f['terminal']['child_ledger']))
    r=unit.propose(**f['known_bind']());p=r['proposed_parent'];c=r['proposed_Core'];s=r['settlement'];q=f['request']
    assert s['actual_live_ticks']==16400 and s['actual_shadow_ticks']==16425
    assert s['financial_charged']['physics_ticks']==32825 and s['financial_charged']['episode_starts']==3
    assert s['unused_released']['episode_starts']==1 and len(s['attempts'])==3 and len(s['conditions'])==4
    assert p['physics_ticks']['live']==f['parent']['physics_ticks']['live']+16400
    assert p['physics_ticks']['shadow']==f['parent']['physics_ticks']['shadow']+16425
    assert p['physics_ticks']['closed_source_component_financial_upper']==16490
    assert c['physics_ticks']==16490+32825
    for prior,after in ((f['parent'],p),(f['Core'],c)):
        assert after['source_cpu_financial_closeouts']==prior['source_cpu_financial_closeouts']
        assert after['source_cpu_reservations'][unit.history.OLD_REQUEST]==prior['source_cpu_reservations'][unit.history.OLD_REQUEST]
        assert q not in after['source_cpu_financial_closeouts']
        assert after[unit.INDEX][q]==s and after['source_cpu_reserved']==dims()
        assert after['source_cpu_reservations'][q]['remaining']==dims()
    assert (f['parent'],f['Core'],f['terminal']['child_ledger'])==before
    assert r['native_dispatch_admitted'] is False and not s['student_neural_ACK_qualification']
    assert not s['target_qualification'] and s['core64_cells']==0


def test_known_once_binding_and_double_settlement_reject(known):
    f=known;a=f['known_bind']();r=unit.propose(**a)
    assert r['before_parent_sha256']==unit.sha(a['parent_bytes'])
    assert r['before_Core_sha256']==unit.sha(a['Core_bytes'])
    assert r['settlement_sha256']==unit.sha(enc(r['settlement']))
    a.update(parent_bytes=enc(r['proposed_parent']),Core_bytes=enc(r['proposed_Core']))
    with pytest.raises(PermissionError,match='NEVER_SETTLE'):unit.propose(**a)


def test_omitted_whole_actor_pair_cannot_release_an_attempt(known):
    f=known;f['terminal']['handles']=f['terminal']['handles'][:-2]
    with pytest.raises(PermissionError,match='EVERY_ACTUAL_ATTEMPT'):unit.propose(**f['known_bind']())


def test_failed_counter_cannot_be_claimed_as_known_component_qualification(known):
    f=known;f['terminal']['handles'][1]['value']['exit_status']=1
    with pytest.raises(PermissionError,match='COUNTER_QUALIFICATION_EXIT_ZERO'):unit.propose(**f['known_bind']())


def test_counter_only_all_notrun_is_not_charged_as_four_attempts(known):
    f=known;f['audit']['conditions']=[dict(condition_id=r['condition_id'],status='NOT_RUN') for r in f['audit']['conditions']]
    f['audit']['component_counts']={'NOT_RUN':4}
    f['terminal']['handles']=f['terminal']['handles'][:2];f['terminal']['attempts']=[];f['attempt_bytes'].clear()
    f['terminal']['child_ledger'].update(main_starts=0,live_and_recovery_integrations=0,shadow_integrations=0,engine_starts=2)
    proposal=unit.propose(**f['known_bind']());settlement=proposal['settlement']
    assert settlement['financial_charged']['episode_starts']==settlement['financial_charged']['physics_ticks']==0
    assert settlement['source_component_result_counts']=={'NOT_RUN':4}
    assert settlement['unused_released']['episode_starts']==4


@pytest.mark.parametrize('fault',['audit_unknown','meter_unknown','reserved_shadow','active_actor','missing_exit',
    'wrong_start','missing_attempt','no_stop_return','anchor_changed','audit_changed','counter_mismatch',
    'budget_overrun','duplicate_condition','NOT_RUN_charged','old_current_unknown','old_receipt_reuse'])
def test_no_known_release_for_unknown_inflight_or_unbound_evidence(known,fault):
    f=known;t=f['terminal'];child=t['child_ledger'];a=f['audit'];before=deepcopy(f['parent'])
    if fault=='audit_unknown':a['conditions'][0]['status']='ENGINEERING_INCOMPLETE_UNKNOWN'
    elif fault=='meter_unknown':child['unknown_effects']=['RETAINED']
    elif fault=='reserved_shadow':child['reservations']={'pending':dict(maximum=200)}
    elif fault=='active_actor':t['handles'][0]['active']=True
    elif fault=='missing_exit':t['handles'].pop()
    elif fault=='wrong_start':t['handles'][-1]['value']['proc_start']='not-original'
    elif fault=='missing_attempt':t['attempts'].pop()
    elif fault=='no_stop_return':t['attempts'][0]['actual_stop_returned']=False
    elif fault=='anchor_changed':
        f['attempt_bytes'][a['conditions'][0]['condition_id']]['terminal_anchor']+=b' '
    elif fault=='counter_mismatch':child['live_and_recovery_integrations']+=1
    elif fault=='budget_overrun':child['Core_CPU_s']=f['budget']['requested']['CoreLab_CPU_s']+1
    elif fault=='duplicate_condition':a['conditions'][1]=deepcopy(a['conditions'][0])
    elif fault=='NOT_RUN_charged':child['main_starts']=4
    elif fault=='old_current_unknown':f['Core']['unknown_effects']=['NEW_UNKNOWN_WRITE']
    elif fault=='old_receipt_reuse':f['parent'][unit.INDEX]={f['request']:{'existing':'retained'}}
    args=f['known_bind']()
    if fault=='audit_changed':args['audit_bytes']+=b' '
    with pytest.raises((PermissionError,KeyError)):unit.propose(**args)
    # Mutation above is controlled fixture setup, not proposal output.
    if fault not in ('old_receipt_reuse',):assert f['parent']==before
