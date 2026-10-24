"""Full original forecast and paired closed SDK receipts, CPU fixtures only."""
from copy import deepcopy
import pytest

from . import target_native_io_budget as unit
from .test_target_sdk_budget import bundle,NOW,GIB
from .test_target_sdk_resources import packet
from .test_target_sdk_closed_costs import closed
from .test_future_budget import raw


@pytest.fixture
def context(bundle,closed):
    s,progress,plan,_,_,space,amend=deepcopy(bundle)
    p,c,d,closures=deepcopy(closed)
    p['CPU_s']['local_PC']=50000.
    p['network_transfer_accounted_bytes']+=100000
    s['used']['local_CPU_s']=s['dec1_used']['local_CPU_s']=50000.
    for label,base in (('used',100),('dec1_used',80)):
        s[label]['Core_storage_bytes']=base*GIB+unit.SOURCE_COUNTER_OUTPUT
    s['storage_peaks']['Core_storage_bytes']=s['used']['Core_storage_bytes']
    s['dec1_storage_peaks']['Core_storage_bytes']=s['dec1_used']['Core_storage_bytes']
    plan['storage_lifecycle']['new_output_peak_upper_bytes']['Core_storage_bytes']=90*GIB
    s['actual_DEC1_ledger_sha256']=unit.common.sha(unit.common.encode(d))
    s['dec1_used']['engine_starts']=d['constructors']
    s['dec1_used']['transfer_bytes']=d['network_transfer_accounted_bytes']
    inputs=dict(snapshot=s,plan=plan,progress_bytes=raw(progress),amendment_bytes=amend,now_epoch=NOW)
    return dict(p=p,c=c,d=d,closures=closures,space=space,inputs=inputs,VAP1=amend)


def held(v):
    v=deepcopy(v);p,c=v['p'],v['c'];rid='CPU_NATIVE_IO_FIXTURE'
    row=dict(schema=unit.ROW_SCHEMA,scope=unit.ROLE,request_id=rid,child_run_uuid='CPU-current-IO',
        manifest_sha256='f'*64,source_sha='a'*40,funded_work_id=unit.WORK,component_id=unit.COMPONENT,
        requested=deepcopy(unit.REQUESTED),remaining=deepcopy(unit.REQUESTED),status='RESERVED',unknown_effects=[])
    for ledger in (p,c):ledger[unit.ROWS]={rid:deepcopy(row)};ledger[unit.AGGREGATE]=deepcopy(unit.REQUESTED)
    p['CPU_reserved_s'].update(CoreLab=3600,local_PC=60)
    return v


def call(v,*,now=NOW):
    args=deepcopy(v['inputs']);args.pop('now_epoch')
    return unit.forecast(parent_bytes=unit.common.encode(v['p']),Core_bytes=unit.common.encode(v['c']),
        DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'],namespace_proof=v['space'],now_epoch=now,**args)


def project_SDK(v):
    x=unit.closed_sdk.propose(*map(unit.common.encode,(v['p'],v['c'],v['d'])),closures=v['closures'])
    v['d']=x['proposed_DEC1'];s=v['inputs']['snapshot']
    s['actual_DEC1_ledger_sha256']=unit.common.sha(unit.common.encode(v['d']))
    s['dec1_used']['engine_starts']=v['d']['constructors']
    s['dec1_used']['transfer_bytes']=v['d']['network_transfer_accounted_bytes']


def test_full_eight_holds_add_no_IO_credit_and_keep_every_matrix_and_controller_upper(context):
    v=held(context);before=deepcopy(v);r=call(v);baseline=call(context)
    assert v==before and r['admissible_forecast'] is True
    view=r['native_IO_accounting'];assert view['actual_reserved']==unit.REQUESTED
    assert not any(view['IO_credit'].values()) and view['original_actual_snapshot']==v['inputs']['snapshot']
    assert r['mandatory_jobs']==baseline['mandatory_jobs']
    assert r['future_required_additional']==baseline['future_required_additional']
    assert r['original_final_protection']==baseline['original_final_protection']
    assert r['SDK_funding']['CPU_credit']==baseline['SDK_funding']['CPU_credit']
    arithmetic=r['SDK_funding']['actual_spent_and_held']['original_actual_snapshot']
    assert arithmetic['target_native_IO_component_arithmetic_view']['IO_held']==unit.REQUESTED
    assert r['target_cost_feasibility']=='UNKNOWN_UNTIL_NATIVE_COMMISSIONING'
    assert view['native_dispatch_admitted'] is False and view['unknown_journal_not_cleared_or_reissued']


def test_missing_and_actually_adopted_SDK_markers_have_identical_full_costs(context):
    before=held(context);missing=call(before)
    adopted=deepcopy(before);project_SDK(adopted);r=call(adopted)
    a=missing['SDK_funding']['actual_spent_and_held'];b=r['SDK_funding']['actual_spent_and_held']
    for label in ('used','dec1_used'):
        assert a['original_actual_snapshot'][label]==b['original_actual_snapshot'][label]
    assert r['future_required_additional']==missing['future_required_additional']
    assert missing['native_IO_accounting']['SDK_projection_missing']['constructors']==2
    assert r['native_IO_accounting']['SDK_projection_missing']['constructors']==0
    assert r['native_IO_accounting']['SDK_existing_native_projection_deducted_once']['constructors']==2
    assert adopted['d']!=before['d'] # Only the explicit fixture projection changed this context.


def test_tight_LOCAL_cap_refuses_IO_instead_of_deleting_future_controller_cost(context):
    v=held(context);r=call(v);future=r['future_required_additional']['local_CPU_s']
    v['p']['CPU_s']['local_PC']=57600-future-59
    for label in ('used','dec1_used'):v['inputs']['snapshot'][label]['local_CPU_s']=v['p']['CPU_s']['local_PC']
    r=call(v)
    assert r['DEC1_shortfalls']['local_CPU_s']==1 and not r['admissible_forecast']
    assert r['future_required_additional']['local_CPU_s']==future


def test_all_original_grants_DEC1_bytes_and_actual_SDK_closures_roundtrip_without_rewriting(context):
    v=context;before=deepcopy(v);raw=unit.forecast_pack(v['inputs'],DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'])
    args,db,cl=unit.unpack_forecast(raw)
    assert args==v['inputs'] and db==unit.common.encode(v['d']) and cl==v['closures'] and v==before


@pytest.mark.parametrize('fault',['one_side','unknown','dropped_hold','extra_attempt','nonzero_physics',
    'hidden_parent','wrong_aggregate','missing_closure','changed_closure','wrong_snapshot_sha',
    'stale_namespace','missing_namespace','LOCAL_output_overflow','SDK_storage_already_applied',
    'other_source_hold','constructor_base','transfer_base','wrong_storage_base','double_arithmetic'])
def test_no_partial_hidden_stale_unclosed_or_double_accounting_context(context,fault):
    v=held(context);p,c,s,d=v['p'],v['c'],v['inputs']['snapshot'],v['d'];rid='CPU_NATIVE_IO_FIXTURE'
    if fault=='one_side':c[unit.ROWS]={}
    elif fault=='unknown':p[unit.ROWS][rid]['unknown_effects']=['UNKNOWN']
    elif fault=='dropped_hold':
        for ledger in (p,c):ledger[unit.ROWS][rid]['remaining']['CoreLab_CPU_s']=0
    elif fault=='extra_attempt':
        for ledger in (p,c):ledger[unit.ROWS]['SECOND']=deepcopy(ledger[unit.ROWS][rid])
    elif fault=='nonzero_physics':
        for ledger in (p,c):ledger[unit.ROWS][rid]['requested']['physics_ticks']=1
    elif fault=='hidden_parent':p['CPU_reserved_s']['CoreLab']+=1
    elif fault=='wrong_aggregate':c[unit.AGGREGATE]['transfer_bytes']=0
    elif fault=='missing_closure':v['closures']={}
    elif fault=='changed_closure':next(iter(v['closures'].values()))['lease_bytes']+=b' '
    elif fault=='wrong_snapshot_sha':s['actual_DEC1_ledger_sha256']='0'*64
    elif fault=='stale_namespace':v['space']['observed_epoch']-=121
    elif fault=='missing_namespace':v['space']['allocated_bytes_by_namespace'].pop('science-next')
    elif fault=='LOCAL_output_overflow':v['space']['allocated_bytes_by_namespace']['science-next']=8*GIB
    elif fault=='SDK_storage_already_applied':s['closed_SDK_storage_projection']={}
    elif fault=='other_source_hold':c['source_cpu_reserved']['constructors']=1
    elif fault=='constructor_base':s['dec1_used']['engine_starts']+=1
    elif fault=='transfer_base':s['dec1_used']['transfer_bytes']+=1
    elif fault=='wrong_storage_base':s['used']['Core_storage_bytes']+=1
    elif fault=='double_arithmetic':s['target_native_IO_component_arithmetic_view']={'old':True}
    with pytest.raises((PermissionError,KeyError)):call(v)


def test_non_CPU_shortfall_and_original_clock_failure_remain_STOP(context):
    v=held(context);v['inputs']['snapshot']['dec1_used']['H200_s']=200000
    v['inputs']['snapshot']['used']['H200_s']=200000
    r=call(v);assert not r['admissible_forecast'] and r['DEC1_shortfalls']['H200_s']>0
    v=held(context);now=unit.STOP_NEW-1000
    v['space']['observed_epoch']=now
    v['inputs']['snapshot']['observed_epoch']=now
    progress=unit.common.obj(v['inputs']['progress_bytes']);progress['observed_epoch']=now
    v['inputs']['progress_bytes']=unit.common.encode(progress)
    r=call(v,now=now)
    assert not r['admissible_forecast'] and 'NATIVE_IO_NEW_COMPONENT_CUTOFF' in r['clock_failures']
