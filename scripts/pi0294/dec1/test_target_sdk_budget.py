"""Original forecaster plus explicit CPU-only paired component fixtures."""
from copy import deepcopy
import hashlib
import pytest
from . import target_sdk_budget as unit
from . import future_budget as original
from .test_future_budget import fixtures, amendment, raw, NOW, GIB


@pytest.fixture
def bundle(monkeypatch):
    s,progress,plan=fixtures();s['campaign_run_uuid']=progress['campaign_run_uuid']=unit.CAMPAIGN
    a=raw(amendment());monkeypatch.setattr(original,'VAP1_ADOPTION',hashlib.sha256(a).hexdigest())
    # Tight LOCAL ceiling: original whole plan4080s leaves exactly1s.
    p=dict(run_uuid=unit.CAMPAIGN,T0_epoch=unit.T0,hard_closeout_epoch=unit.HARD_CLOSE,
        unknown_effects=[],Hopper_unknown_effects=[],pending_updates=[],
        CPU_s=dict(CoreLab=140000.,local_PC=57600.-4080.-1,Hopper=7000.),
        CPU_reserved_s=dict(CoreLab=0.,local_PC=0.,Hopper=0.),engine_starts=100,
        network_transfer_accounted_bytes=1000)
    c=dict(authority_release=unit.common.AUTHORITY,VAP1_authority_release=unit.common.VAP1_AUTHORITY,
        campaign_run_uuid=unit.CAMPAIGN,original_T0_epoch=unit.T0,hard_close_epoch=unit.HARD_CLOSE,
        unknown_effects=[],CPU_s=100.,constructors=10,
        source_cpu_reserved=dict.fromkeys(unit.common.DIMS,0),target_zero_counter_reserved=dict.fromkeys(unit.common.DIMS,0))
    for label in ('used','dec1_used'):s[label]['local_storage_bytes']=8*GIB
    for label in ('storage_peaks','dec1_storage_peaks'):s[label]['local_storage_bytes']=8*GIB
    s['used'].update(engine_starts=100,transfer_bytes=1000,local_CPU_s=p['CPU_s']['local_PC'])
    s['dec1_used'].update(Core_storage_bytes=20*GIB,engine_starts=10,transfer_bytes=0)
    plan['storage_lifecycle']['new_output_peak_upper_bytes']['local_storage_bytes']=8*GIB
    footprint=dict(schema='DEC1_SDK_ACTUAL_INCLUSIVE_LOCAL_NAMESPACES_V1',
        complete_existing_namespace_set=list(unit.NAMES),SDK_namespace='target-sdk-CPU-fixture',
        allocated_bytes_by_namespace={n:1024 for n in (*unit.NAMES,'target-sdk-CPU-fixture')},
        protected_future_additional_upper_bytes=GIB+3*64*1024**2,observed_epoch=NOW)
    return s,progress,plan,p,c,footprint,a


def reserved(bundle):
    s,progress,plan,p,c,footprint,a=deepcopy(bundle)
    row=dict(schema=unit.ROW_SCHEMA,scope=unit.ROLE,request_id='SDK_CPU_FIXTURE',requested=deepcopy(unit.REQUESTED),
        remaining=deepcopy(unit.REQUESTED),funded_work_id=unit.WORK,component_id=unit.COMPONENT,
        unknown_effects=[],status='RESERVED',manifest_sha256='a'*64)
    for ledger in (p,c):
        ledger[unit.ROWS]={'SDK_CPU_FIXTURE':deepcopy(row)}
        ledger['target_sdk_client_reserved']=deepcopy(unit.REQUESTED)
    p['CPU_reserved_s'].update(CoreLab=3600.,local_PC=120.)
    return s,progress,plan,p,c,footprint,a


def call(value):
    s,progress,plan,p,c,footprint,a=value
    return unit.forecast(snapshot=s,plan=plan,parent_bytes=unit.common.encode(p),Core_bytes=unit.common.encode(c),
        namespace_proof=footprint,now_epoch=NOW,progress_bytes=raw(progress),amendment_bytes=a)


def test_paired_cpu_hold_funds_same_work_without_shrinking_any_matrix_row(bundle):
    v=reserved(bundle);before=deepcopy(v);r=call(v);s,progress,plan,p,c,_,_=v
    assert v==before and r['admissible_forecast'] is True
    old=r['SDK_funding']['full_original_forecast_before_component_credit']
    assert old['DEC1_shortfalls']['local_CPU_s']==119
    assert r['SDK_funding']['CPU_credit']==dict(Core_CPU_s=3600,local_CPU_s=120)
    assert r['mandatory_jobs']==old['mandatory_jobs']
    matrix=next(j for j in r['mandatory_jobs'] if j['kind']=='MATRIX_ENVELOPE')
    assert matrix['resources']==dict(plan['remaining_matrix_envelope']['resources'],optimizer_updates=0)
    assert matrix['resources']['episode_starts']==88 and matrix['resources']['PBS_submissions']==3
    view=r['SDK_funding']['actual_spent_and_held']
    assert view['actual_reserved']==unit.REQUESTED
    assert view['original_actual_snapshot']==s and view['pending_shown_as_committed_uppers_not_measured_use'] is True
    assert r['future_required_additional']['local_CPU_s']+p['CPU_s']['local_PC']+120==57599
    for k in ('H200_s','PBS_submissions','physics_ticks','engine_starts','neural_queries','optimizer_updates','transfer_bytes'):
        assert r['future_required_additional'][k]==old['future_required_additional'][k]
    assert r['original_final_protection']==old['original_final_protection']
    assert r['target_cost_feasibility']=='UNKNOWN_UNTIL_NATIVE_COMMISSIONING'
    assert r['SDK_funding']['native_dispatch_admitted'] is False


def test_settled_credit_is_only_actual_financial_charge_already_in_totals(bundle):
    v=reserved(bundle);s,progress,plan,p,c,fp,a=v;rid='SDK_CPU_FIXTURE'
    amount=deepcopy(unit.REQUESTED);amount.update(CoreLab_CPU_s=1000.,local_CPU_s=40.)
    record=dict(schema='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1',scope=unit.ROLE,request_id=rid,
        manifest_sha256='a'*64,financial_charged=amount,target_task_native_qualified=False)
    for ledger in (p,c):
        ledger[unit.ROWS][rid].update(status='SETTLED',remaining=dict.fromkeys(unit.common.DIMS,0),
            settlement_sha256=unit.common.sha(unit.common.encode(record)))
        ledger[unit.SETTLEMENTS]={rid:deepcopy(record)}
        ledger['target_sdk_client_reserved']=dict.fromkeys(unit.common.DIMS,0)
    p['CPU_reserved_s'].update(CoreLab=0.,local_PC=0.)
    p['CPU_s']['CoreLab']+=1000;p['CPU_s']['local_PC']+=40
    p['engine_starts']+=2;p['network_transfer_accounted_bytes']+=amount['transfer_bytes']
    c['CPU_s']+=1000;c['constructors']+=2
    r=call(v)
    assert r['SDK_funding']['CPU_credit']==dict(Core_CPU_s=1000.,local_CPU_s=40.)
    assert r['admissible_forecast'] is True
    assert r['future_required_additional']['local_CPU_s']+p['CPU_s']['local_PC']==57599


@pytest.mark.parametrize('fault',['one_side','unknown','dropped_hold','CPU_cap','wrong_work','wrong_role',
    'old_clock','stale_namespace','omitted_namespace','missing_future_output','local_overflow',
    'other_source_hold','hidden_parent_hold','second_overlay','reduced_matrix','not_settled_in_totals'])
def test_no_partial_unknown_other_work_or_unprotected_space_credit(bundle,fault):
    v=reserved(bundle);s,progress,plan,p,c,fp,a=v
    if fault=='one_side':c[unit.ROWS]={}
    elif fault=='unknown':p[unit.ROWS]['SDK_CPU_FIXTURE']['unknown_effects']=['UNKNOWN']
    elif fault=='dropped_hold':
        for x in (p,c):x[unit.ROWS]['SDK_CPU_FIXTURE']['remaining']['CoreLab_CPU_s']=0
    elif fault=='CPU_cap':
        for x in (p,c):
            for field in ('requested','remaining'):x[unit.ROWS]['SDK_CPU_FIXTURE'][field]['local_CPU_s']=121
            x['target_sdk_client_reserved']['local_CPU_s']=121
    elif fault=='wrong_work':
        for x in (p,c):x[unit.ROWS]['SDK_CPU_FIXTURE']['funded_work_id']='TRAIN:B_COVERAGE'
    elif fault=='wrong_role':
        for x in (p,c):x[unit.ROWS]['SDK_CPU_FIXTURE']['scope']='OLD_R4_OR_TARGET_TASK'
    elif fault=='old_clock':c['original_T0_epoch']+=1
    elif fault=='stale_namespace':fp['observed_epoch']-=121
    elif fault=='omitted_namespace':fp['allocated_bytes_by_namespace'].pop(unit.NAMES[0])
    elif fault=='missing_future_output':fp['protected_future_additional_upper_bytes']=0
    elif fault=='local_overflow':fp['allocated_bytes_by_namespace'][unit.NAMES[0]]=8*GIB
    elif fault=='other_source_hold':c['source_cpu_reserved']['CoreLab_CPU_s']=1
    elif fault=='hidden_parent_hold':p['CPU_reserved_s']['local_PC']+=1
    elif fault=='second_overlay':s['target_sdk_component_arithmetic_view']={'CPU_FIXTURE':True}
    elif fault=='reduced_matrix':plan['remaining_matrix_envelope']['resources']['local_CPU_s']=3480
    elif fault=='not_settled_in_totals':c['CPU_s']=-1
    with pytest.raises((PermissionError,KeyError)):call(v)


def test_GPU_shortfall_or_original_clock_failure_never_disappears(bundle):
    v=reserved(bundle);v[0]['used']['H200_s']=200000;v[0]['dec1_used']['H200_s']=48*3600
    r=call(v);assert r['admissible_forecast'] is False and r['DEC1_shortfalls']['H200_s']>0
    assert r['SDK_funding']['full_original_forecast_before_component_credit']['DEC1_shortfalls']['H200_s']==r['DEC1_shortfalls']['H200_s']


def test_closed_failed_attempt_keeps_full_cost_before_one_bounded_same_work_retry(bundle):
    v=reserved(bundle);s,progress,plan,p,c,fp,a=v;first='SDK_CPU_FIXTURE';second='SDK_CPU_FIXTURE_SECOND'
    old=deepcopy(p[unit.ROWS][first]);charge=deepcopy(unit.REQUESTED)
    receipt=dict(schema='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1',scope=unit.ROLE,request_id=first,
        manifest_sha256=old['manifest_sha256'],financial_charged=charge,target_task_native_qualified=False)
    for ledger in (p,c):
        ledger[unit.ROWS][first].update(status='SETTLED',remaining=dict.fromkeys(unit.common.DIMS,0),
            settlement_sha256=unit.common.sha(unit.common.encode(receipt)))
        ledger[unit.SETTLEMENTS]={first:deepcopy(receipt)}
        ledger[unit.ROWS][second]=dict(old,request_id=second)
    p['CPU_s']['CoreLab']+=3600;p['CPU_s']['local_PC']+=120
    p['engine_starts']+=2;p['network_transfer_accounted_bytes']+=charge['transfer_bytes']
    c['CPU_s']+=3600;c['constructors']+=2
    result=call(v)
    assert result['admissible_forecast'] is True
    assert result['SDK_funding']['CPU_credit']==dict(Core_CPU_s=7200,local_CPU_s=240)
    assert result['SDK_funding']['actual_spent_and_held']['actual_spent']==charge
    matrix=next(j for j in result['mandatory_jobs'] if j['kind']=='MATRIX_ENVELOPE')
    assert matrix['resources']['Core_CPU_s']==200000 and matrix['resources']['local_CPU_s']==3600
    assert matrix['resources']['episode_starts']==88 and matrix['resources']['PBS_submissions']==3
    for ledger in (p,c):
        for rid in ('THIRD','FOURTH','FIFTH'):ledger[unit.ROWS][rid]=dict(old,request_id=rid)
    with pytest.raises(PermissionError):call(v)


def test_three_closed_engineering_costs_and_one_new_hold_share_original_matrix_cpu(bundle):
    v=reserved(bundle);s,progress,plan,p,c,fp,a=v;template=deepcopy(p[unit.ROWS]['SDK_CPU_FIXTURE'])
    for ledger in (p,c):ledger[unit.ROWS]={};ledger[unit.SETTLEMENTS]={}
    total=0
    for rid,cpu in (('ENTRY_FAILED',3600),('LINK_FAILED',607.956897),('LOADER_FAILED',607.859879)):
        amount=dict(unit.REQUESTED,CoreLab_CPU_s=cpu);total+=cpu
        receipt=dict(schema='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1',scope=unit.ROLE,request_id=rid,
            manifest_sha256=template['manifest_sha256'],financial_charged=amount,target_task_native_qualified=False)
        for ledger in (p,c):
            ledger[unit.ROWS][rid]=dict(template,request_id=rid,status='SETTLED',remaining=dict.fromkeys(unit.common.DIMS,0),
                settlement_sha256=unit.common.sha(unit.common.encode(receipt)))
            ledger[unit.SETTLEMENTS][rid]=deepcopy(receipt)
    for ledger in (p,c):ledger[unit.ROWS]['NEW_REQUEST']=dict(template,request_id='NEW_REQUEST')
    p['CPU_s']['CoreLab']+=total;p['CPU_s']['local_PC']+=360;p['engine_starts']+=6
    p['network_transfer_accounted_bytes']+=3*unit.REQUESTED['transfer_bytes']
    c['CPU_s']+=total;c['constructors']+=6
    r=call(v)
    assert r['admissible_forecast'] is True
    assert r['SDK_funding']['CPU_credit']==dict(Core_CPU_s=total+3600,local_CPU_s=480)
    assert next(x for x in r['mandatory_jobs'] if x['kind']=='MATRIX_ENVELOPE')['resources']==dict(plan['remaining_matrix_envelope']['resources'],optimizer_updates=0)
