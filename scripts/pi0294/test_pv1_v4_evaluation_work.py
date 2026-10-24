"""CPU evidence/scenario fixtures; no native admission or scientific results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.pi0294.pv1 import baseline as b
from scripts.pi0294.pv1 import evaluation_work as w
from scripts.pi0294.pv1.evaluation_platform import ENDPOINTS
from scripts.pi0294.test_pv1_baseline import fixtures,raw
from scripts.pi0294.test_pv1_evaluation_plan import groups

POP=raw(dict(groups=groups()))


def evidence(candidate='B0_CORR',count=14):
    result=[]
    for original in fixtures()[:count]:
        item=deepcopy(original)
        row=json.loads(item['adjudication']);saved=json.loads(item['result'])
        audit=json.loads(item['audit']);ready=json.loads(item['model_ready']);chain=json.loads(item['causal_chain'])
        row['model']=candidate;row['attempt_id']=row['attempt_id'].replace('B0_CORR',candidate)
        saved['id']=row['attempt_id'];item['result']=raw(saved);row['result_sha256']=hashlib.sha256(item['result']).hexdigest()
        item['adjudication']=raw(row);audit.update(attempt_id=row['attempt_id'],adjudication=row);item['audit']=raw(audit)
        ready['label']=ready['label'].replace('B0_CORR',candidate)
        ready['identity'].update(label=candidate,global_step=2048 if candidate=='B0_CORR' else 4096,
            checkpoint_manifest_sha256=ENDPOINTS[candidate]);item['model_ready']=raw(ready)
        chain.update(attempt_id=row['attempt_id'],adjudication_sha256=hashlib.sha256(item['adjudication']).hexdigest(),identity=ready['identity'])
        item['causal_chain']=raw(chain);result.append(item)
    return result


@pytest.fixture(autouse=True)
def synthetic_population_only():
    with patch.object(b,'POPULATION',hashlib.sha256(POP).hexdigest()):yield


@pytest.mark.parametrize('candidate',list(ENDPOINTS))
def test_partial_valid_negatives_produce_remaining_work_without_admitting_training(candidate):
    items=evidence(candidate);before=deepcopy(items)
    history=b.partial_history(POP,candidate=candidate,evidence=items)
    assert items==before and history['valid_conditions']==14 and history['remaining_eligible_conditions']==34
    assert history['successes']==dict(T1=0,T2=0) and history['complete_DEV48'] is False
    assert history['training_or_selection_admission'] is False and history['native_or_resource_admission'] is False
    assert len(history['all_actual_attempts'])==14
    with pytest.raises(PermissionError,match='INCOMPLETE_DEV48'):
        b.admission(POP,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,evidence=items,
            candidate=candidate,checkpoint_manifest_sha256=ENDPOINTS[candidate])


def test_original_48_valid_negative_training_gate_still_has_exact_completeness_contract():
    items=evidence(count=48)
    admitted=b.admission(POP,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,evidence=items)
    assert admitted['status']=='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE'
    assert admitted['successes']==dict(T1=0,T2=0)
    history=b.partial_history(POP,candidate='B0_CORR',evidence=items)
    assert history['complete_DEV48'] is True and history['training_or_selection_admission'] is False


@pytest.mark.parametrize('change',['duplicate','wrong_stats','missing_chain','failed_audit','bad_raw_hash','unsafe','wrong_candidate'])
def test_partial_parser_keeps_all_original_per_attempt_byte_and_safety_gates(change):
    items=evidence()
    if change=='duplicate':items.append(deepcopy(items[0]))
    if change=='wrong_stats':
        ready=json.loads(items[0]['model_ready']);ready['identity']['statistics_sha256']='f'*64;items[0]['model_ready']=raw(ready)
    if change=='missing_chain':items[0].pop('causal_chain')
    if change=='failed_audit':
        audit=json.loads(items[0]['audit']);audit['checks']['shadow_did_not_mutate_live']=False;items[0]['audit']=raw(audit)
    if change=='bad_raw_hash':items[0]['result']=raw(dict(id='other'))
    if change=='unsafe':
        row=json.loads(items[0]['adjudication']);row['live_native_failures']=1;items[0]['adjudication']=raw(row)
    if change=='wrong_candidate':
        row=json.loads(items[0]['adjudication']);row['model']='A_CONTROL';items[0]['adjudication']=raw(row)
    with pytest.raises((PermissionError,KeyError)):b.partial_history(POP,candidate='B0_CORR',evidence=items)


def test_all_three_histories_are_explicit_and_known_valid_cases_cannot_repeat():
    for histories in ({'A_CONTROL':[]},{'B0_CORR':[],'A_CONTROL':[],'REPLICA':[]}):
        with pytest.raises(PermissionError):w.remaining(POP,evidence_by_candidate=histories)
    result=w.remaining(POP,evidence_by_candidate=dict(B0_CORR=evidence(),A_CONTROL=[],B_COVERAGE=[]))
    assert [j['conditions'] for j in result['jobs']]==[10,24,24,24,24,24]
    assert result['all_valid_outcomes']==14 and result['resources']['episode_starts']==130
    assert result['resources']['neural_queries']==10502 and result['resources']['optimizer_updates']==0
    assert result['resources']['H200_s']==173160 and result['resources']['PBS_submissions']==6
    assert result['resources']['Core_CPU_s']==511800 and result['resources']['Hopper_CPU_s']==523800
    assert result['resources']['local_CPU_s']==43800 and result['resources']['physics_ticks']==4160000
    assert all(j['native_platform']=='VANDA_A40' for j in result['jobs'])
    done={json.loads(x['adjudication'])['condition_id'] for x in evidence()}
    assert not done.intersection(result['jobs'][0]['condition_ids'])
    assert all(j['repeat_original_fixed_probes_authorized'] is False for j in result['jobs'])
    assert result['native_or_resource_admission'] is False


def test_full_regular_envelope_reproduces_existing_complete_scope_without_authorizing_it():
    work=w.remaining(POP,evidence_by_candidate=dict(B0_CORR=evidence(),A_CONTROL=[],B_COVERAGE=[]))
    report=Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/S2_DEV130_CLOSED_COST_AND_FULL_UPPER_SCOPE_DRAFT_20261024.json'
    actual=json.loads(report.read_text());dims=actual['projection']
    value=w.diagnostic(work,used={k:r['used'] for k,r in dims.items()},reserved={},caps={k:r['cap'] for k,r in dims.items()})
    assert value['shortfalls']==actual['regular_work_shortfalls']
    proposed={k:r['cap'] for k,r in dims.items()}
    proposed.update(H200_s=121*3600,PBS_submissions=34,Hopper_CPU_s=213*3600,Hopper_storage_bytes=810*1024**3)
    scenario=w.diagnostic(work,used={k:r['used'] for k,r in dims.items()},reserved={},caps=proposed)
    assert scenario['mathematical_fit'] is True and scenario['native_or_resource_admission'] is False
    assert scenario['caps_are_only_scenario_not_authority'] is True
    work['original_final_protection']['H200_s']=0
    with pytest.raises(PermissionError):w.diagnostic(work,used={},reserved={},caps=proposed)


def test_completed_histories_have_no_new_regular_jobs_and_do_not_authorize_more():
    work=w.remaining(POP,evidence_by_candidate={candidate:evidence(candidate,48) for candidate in ENDPOINTS})
    assert work['all_valid_outcomes']==144 and work['jobs']==[]
    assert all(x==0 for x in work['resources'].values())
    assert work['native_or_resource_admission'] is False and work['new_engineering_attempt_pool_authorized'] is False


def test_two_diagnosed_engineering_attempts_stay_unmeasured_instead_of_becoming_success():
    first=evidence(count=1)[0];items=[]
    for suffix in ('','-REPLACEMENT-01'):
        item=deepcopy(first);row=json.loads(item['adjudication']);result=json.loads(item['result'])
        row.update(attempt_id=row['attempt_id']+suffix,status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,
            failure_stage='TRANSPORT');result['id']=row['attempt_id'];item['result']=raw(result)
        row['result_sha256']=hashlib.sha256(item['result']).hexdigest();item['adjudication']=raw(row)
        audit=json.loads(item['audit']);audit.update(attempt_id=row['attempt_id'],adjudication=row);item['audit']=raw(audit)
        item.pop('causal_chain');item['engineering_diagnosis']=raw(dict(status='CONFIRMED_ENGINEERING_FAILURE_NO_POLICY_REPLAY',
            original_adjudication_sha256=hashlib.sha256(item['adjudication']).hexdigest(),
            original_result_sha256=row['result_sha256'],native_process_terminal_verified=True,
            unknown_effects=[],policy_outcome_valid=False,evidence_sha256=dict(native_terminal='a'*64)))
        items.append(item)
    result=b.partial_history(POP,candidate='B0_CORR',evidence=items)
    assert len(result['all_actual_attempts'])==2 and result['valid_conditions']==0
    assert result['remaining_eligible_conditions']==47 and result['exhausted_engineering_conditions']==1
    assert result['complete_DEV48'] is False and result['training_or_selection_admission'] is False
