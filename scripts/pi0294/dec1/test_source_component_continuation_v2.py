"""Synthetic aggregation fixtures only; helper did not import/run them.

No owner/native dispatch, ledger, Teacher, model or physics execution. Independent
terminal audit semantics are tested elsewhere; these cases test lineage/counts.
"""
from copy import deepcopy
from collections import Counter
import json
import pytest
from scripts.pi0294.dec1 import source_component_continuation_v2 as unit
from scripts.pi0294.dec1.test_source_commissioning_plan import population


def raw(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


def receipt(value):return raw(dict(returncode=0,read_only=True,stdout=json.dumps(value),stderr=''))


def audited(ids,statuses,source):
    rows=[dict(condition_id=cid,status=status) for cid,status in zip(ids,statuses)]
    return receipt(dict(source_sha=source,readonly=True,new_native_integrations=0,
        denied_external_native_attempts=[],result=dict(
        schema='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1',purpose=unit.PURPOSE,
        student_neural_ACK_qualification=False,target_qualification=False,core64_cells=0,new_training_data=False,
        conditions=rows,component_counts=dict(Counter(statuses)))))


@pytest.fixture
def sample(population,monkeypatch):
    data,encode=population;pop=encode(data);plan=raw(unit.plans.build(pop,owner_release='a'*40))
    monkeypatch.setattr(unit,'PLAN_SHA256',unit.sha(plan));monkeypatch.setattr(unit,'PLAN_OWNER_RELEASE','a'*40)
    ids=[c['condition_id'] for c in unit.document(plan)['conditions']]
    original=audited(ids,[unit.UNKNOWN,'NOT_RUN','NOT_RUN','NOT_RUN'],unit.OLD_SOURCE)
    closed=receipt(dict(status='INDEPENDENT_DUAL_FINANCIAL_CLOSURE_RETAINED_UNKNOWN_AND_FILES_CONFIRMED',
        source_lease_status='CLOSED',active_workers=0,original_child_ledger_unchanged=True,
        unknown_and_original_live_reservation_preserved=True,scientific_verdict=unit.UNKNOWN,
        actual_STOP_ACK=False,new_writes=0,new_native_integrations=0))
    invoice=dict(request_id=unit.OLD_REQUEST,child_run_uuid=unit.OLD_RUN,runtime_source=unit.OLD_SOURCE,
        child_manifest_sha256=unit.OLD_MANIFEST,scientific_verdict=unit.UNKNOWN,live_financial_upper=16000,
        financial_charged=dict(physics_ticks=16490))
    settlement=raw(dict(invoice=invoice))
    for key,value in [('OLD_AUDIT',original),('OLD_CLOSURE',closed),('OLD_SETTLEMENT',settlement)]:
        monkeypatch.setattr(unit,key,unit.sha(value))
    continuation=dict(original_commissioning_requested=3,engineering_replacements_requested=1,
        replacement_of_request_id=unit.OLD_REQUEST,replacement_condition_id=ids[0],replacement_index=1,
        original_condition_ids=ids[1:],history_sha256='a'*64,engineering_fix_evidence_sha256=unit.REPAIR_SHA)
    budget=raw(dict(request_id=unit.COLD_REQUEST,requested=dict(episode_starts=4),continuation=continuation))
    manifest=raw(dict(scope=unit.ROLE,run_uuid='SYNTHETIC_SOURCE02',code=dict(vla_commit=unit.COLD_SOURCE),
        data=dict(manifest_sha256=unit.PLAN_SHA256),execution=dict(sealed_inputs={
        'plans/DEC1_SOURCE_CPU_BUDGET.json':unit.sha(budget),unit.HISTORY_PLAN:'a'*64,unit.REPAIR_PLAN:unit.REPAIR_SHA})))
    monkeypatch.setattr(unit,'COLD_MANIFEST',unit.sha(manifest))
    return dict(base=(plan,pop,original,closed,settlement),ids=ids,manifest=manifest,budget=budget,invoice=invoice)


def cold_kwargs(sample,statuses):
    wrapper=unit.document(audited(sample['ids'],statuses,unit.COLD_SOURCE))
    report=json.loads(wrapper['stdout']);rows=report['result']['conditions']
    for row in rows:
        if row['status']!='NOT_RUN':row.update(main_integrations=400,shadow_integrations=490)
    wrapper_raw=receipt(report);result_raw=raw(report['result'])
    attempts=[dict(condition_id=r['condition_id'],status=r['status'],actual_stop_returned=True,unknown_effects=[])
              for r in rows if r['status']!='NOT_RUN']
    n=len(attempts)
    known=dict(schema=unit.KNOWN_SCHEMA,request_id=unit.COLD_REQUEST,
        settlement_id=unit.COLD_REQUEST+'-known-terminal-close',manifest_sha256=unit.COLD_MANIFEST,
        child_run_uuid='SYNTHETIC_SOURCE02',budget_sha256=unit.sha(sample['budget']),plan_sha256=unit.PLAN_SHA256,
        source_sha=unit.COLD_SOURCE,audit_sha256=unit.sha(result_raw),conditions=rows,attempts=attempts,
        actual_live_ticks=400*n,actual_shadow_ticks=490*n,
        financial_charged=dict(physics_ticks=890*n,episode_starts=n),student_neural_ACK_qualification=False,
        target_qualification=False,core64_cells=0,new_training_data=False,native_dispatch_admitted=False)
    known_raw=raw(known);known_sha=unit.sha(known_raw);pairs={}
    continuation=unit.document(sample['budget'])['continuation']
    for side in ('parent','Core'):
        ledger=raw({unit.KNOWN_INDEX:{unit.COLD_REQUEST:known},'source_cpu_reservations':{
            unit.COLD_REQUEST:dict(status='SETTLED',known_terminal_settlement_sha256=known_sha,continuation=continuation)},
            'source_cpu_financial_closeouts':{unit.OLD_REQUEST:sample['invoice']}})
        intent=raw(dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1',request_id=known['settlement_id'],
            target='/synthetic/'+side+'/LEDGER.json',before_sha256='b'*64,proposed_sha256=unit.sha(ledger),native_dispatch_admitted=False))
        ack=raw(dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1',request_id=known['settlement_id'],
            target='/synthetic/'+side+'/LEDGER.json',before_sha256='b'*64,after_sha256=unit.sha(ledger),intent_sha256=unit.sha(intent),
            status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',native_dispatch_admitted=False))
        pairs[side]=dict(ledger_bytes=ledger,intent_bytes=intent,receipt_bytes=ack)
    return dict(cold_audit_bytes=wrapper_raw,cold_audit_sha256=unit.sha(wrapper_raw),
        cold_manifest_bytes=sample['manifest'],cold_manifest_sha256=unit.COLD_MANIFEST,
        cold_budget_bytes=sample['budget'],cold_audit_result_bytes=result_raw,
        known_settlement_bytes=known_raw,known_settlement_sha256=known_sha,once_pairs=pairs)


def test_baseline_preserves_one_unknown_three_notrun(sample):
    r=unit.aggregate(*sample['base'])
    assert r['attempts_observed']==1 and r['engineering_incomplete_unknown']==1
    assert r['current_condition_counts']=={unit.UNKNOWN:1,'NOT_RUN':3}


def test_partial_s1_rejection_preserved_as_known_component_not_task_success(sample):
    r=unit.aggregate(*sample['base'],**cold_kwargs(sample,['KNOWN_REFERENCE_OR_S1_REJECT','NOT_RUN','NOT_RUN','NOT_RUN']))
    assert r['attempts_observed']==2 and r['engineering_incomplete_unknown']==1
    assert r['known_reference_or_S1_rejections']==1 and r['native_safety_failures']==0
    assert r['current_condition_counts']=={'KNOWN_REFERENCE_OR_S1_REJECT':1,'NOT_RUN':3}
    assert r['observed_engineering_replacements']==1 and r['attempt_records'][0]['result']['status']==unit.UNKNOWN


def test_all_four_known_retains_five_actual_history_rows_and_no_neural_qualification(sample):
    statuses=['KNOWN_REFERENCE_OR_S1_REJECT','COMPONENT_COMPLETE','NATIVE_SAFETY_FAIL','COMPONENT_COMPLETE']
    r=unit.aggregate(*sample['base'],**cold_kwargs(sample,statuses))
    assert r['attempts_observed']==5 and len(r['attempt_records'])==5 and r['engineering_incomplete_unknown']==1
    assert sum(x['role']=='ENGINEERING_REPLACEMENT' for x in r['attempt_records'])==1
    assert sum(x['role']=='FIRST_ATTEMPT_OF_PREVIOUSLY_NOT_RUN' for x in r['attempt_records'])==3
    assert not r['student_neural_ACK_qualification'] and not r['target_WBC_qualification']
    assert r['core64_cells']==0 and not r['native_dispatch_admitted']


@pytest.mark.parametrize('fault',['no_receipt','proposal_only','missing_Core','changed_ledger','wrong_intent',
    'same_target','lost_old_unknown','audit_changed','duplicate_attempt','receipt_student','NOT_RUN_charged',
    'wrong_replacement','duplicate_valid_failure','unsettled','source01_as_cold'])
def test_unbound_or_repeated_replacement_cannot_be_merged(sample,fault):
    k=cold_kwargs(sample,['KNOWN_REFERENCE_OR_S1_REJECT','NOT_RUN','NOT_RUN','NOT_RUN'])
    if fault=='no_receipt':k['known_settlement_bytes']=None
    elif fault=='proposal_only':k['once_pairs']['parent']['receipt_bytes']=raw(dict(status='PROPOSAL_ONLY'))
    elif fault=='missing_Core':k['once_pairs'].pop('Core')
    elif fault=='changed_ledger':k['once_pairs']['Core']['ledger_bytes']+=b' '
    elif fault=='wrong_intent':k['once_pairs']['Core']['intent_bytes']+=b' '
    elif fault=='same_target':k['once_pairs']['Core']=deepcopy(k['once_pairs']['parent'])
    elif fault=='audit_changed':k['cold_audit_result_bytes']+=b' '
    elif fault=='source01_as_cold':k['cold_manifest_bytes']=raw(dict(scope=unit.ROLE,run_uuid=unit.OLD_RUN))
    elif fault in ('wrong_replacement','duplicate_valid_failure'):
        b=unit.document(k['cold_budget_bytes']);b['continuation']['replacement_condition_id']=sample['ids'][1]
        if fault=='duplicate_valid_failure':b['continuation']['engineering_replacements_requested']=2
        k['cold_budget_bytes']=raw(b)
    elif fault in ('lost_old_unknown','unsettled'):
        pair=k['once_pairs']['parent'];ledger=unit.document(pair['ledger_bytes'])
        if fault=='lost_old_unknown':ledger['source_cpu_financial_closeouts'][unit.OLD_REQUEST]['scientific_verdict']='KNOWN'
        else:ledger['source_cpu_reservations'][unit.COLD_REQUEST]['status']='RUNNING'
        pair['ledger_bytes']=raw(ledger)  # actual receipt intentionally must not be resealed
    else:
        known=unit.document(k['known_settlement_bytes'])
        if fault=='duplicate_attempt':known['attempts'].append(deepcopy(known['attempts'][0]))
        elif fault=='receipt_student':known['student_neural_ACK_qualification']=True
        elif fault=='NOT_RUN_charged':known['financial_charged']['episode_starts']=4
        k['known_settlement_bytes']=raw(known);k['known_settlement_sha256']=unit.sha(k['known_settlement_bytes'])
    with pytest.raises((ValueError,KeyError)):unit.aggregate(*sample['base'],**k)
