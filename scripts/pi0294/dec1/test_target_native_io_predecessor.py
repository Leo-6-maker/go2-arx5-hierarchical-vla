"""Synthetic isolated CPU history; never a native closure/admission proof."""
import base64
from copy import deepcopy
import pytest

from . import target_native_io_predecessor as unit
from . import target_native_io_resources as resources
from .test_target_native_io_resources import proposal,manifest,reserve,terminal,settle
from .test_target_native_io_budget import context
from .test_target_sdk_budget import bundle,NOW
from .test_target_sdk_resources import packet
from .test_target_sdk_closed_costs import closed


@pytest.fixture
def continuation(proposal,monkeypatch):
    v=deepcopy(proposal);v['q'].update(request_id=unit.REQUEST,child_run_uuid=unit.RUN)
    first=reserve(v);args=terminal(first)
    result=unit.common.obj(args['result_bytes'])
    result.update(status='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE',completed_stages=[],
        error="ModuleNotFoundError: No module named 'rclpy'",constructor_labels=[],
        actual_constructor_attempts=0,core64_cells=0)
    rb=unit.common.encode(result);actor=unit.common.obj(args['actor_proof_bytes'])
    actor['result_sha256']=unit.common.sha(rb)
    args.update(result_bytes=rb,actor_proof_bytes=unit.common.encode(actor))
    final=settle(first,args);old=first['reservation'];record=final['settlement']
    monkeypatch.setattr(unit,'SOURCE',old['source_sha']);monkeypatch.setattr(unit,'MANIFEST',old['manifest_sha256'])
    lease=dict(status='CLOSED',scope=unit.ROLE,run_uuid=unit.RUN,source_sha=old['source_sha'],
        manifest_sha256=old['manifest_sha256'],settlement_sha256=unit.common.sha(unit.common.encode(record)),
        all_registered_actors_terminal=True,target_task_native_qualified=False)
    lb=unit.common.encode(lease);lh=unit.common.sha(lb)
    cb=unit.common.encode(final['proposed_Core']);pb=unit.common.encode(final['proposed_parent'])
    dual=dict(status='ACTUAL_CURRENT_NATIVE_IO_DUAL_ONCE_SETTLED_AND_LEASE_CLOSED',settlement=record,
        Core=dict(closed_lease_sha256=lh,closed_lease_base64=base64.b64encode(lb).decode(),Core_after_sha256=unit.common.sha(cb)),
        LOCAL_readback=dict(status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT',receipt_binding_valid=True,actual_sha256=unit.common.sha(pb)),
        LOCAL_receipt=dict(request_id=record['settlement_id'],after_sha256=unit.common.sha(pb),
            status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'),LOCAL_after_sha256=unit.common.sha(pb))
    independent=dict(status='INDEPENDENT_CURRENT_NATIVE_IO_CLOSED_READBACK',lease_sha256=lh,
        Core_after_sha256=unit.common.sha(cb),LOCAL_after_sha256=unit.common.sha(pb),all_8_holds_zero=True,
        LOCAL_all_8_holds_zero=True,registered_actors_active=[],new_native_writes=0,old_SDK_UNKNOWN_unchanged=True)
    fields=dict(dual_bytes=unit.common.encode(dual),lease_bytes=lb,result_bytes=rb,
        actor_proof_bytes=args['actor_proof_bytes'],independent_bytes=unit.common.encode(independent))
    # Isolated synthetic fixture anchors never leave these tests. Actual root
    # captured-byte checks use the unmodified exact public historical hashes.
    monkeypatch.setattr(unit,'ANCHORS',{k:unit.common.sha(raw) for k,raw in fields.items()})
    v['p'],v['c']=final['proposed_parent'],final['proposed_Core']
    v['predecessor']=unit.pack(fields)
    q,x=v['q'],v['x'];q.update(request_id='bdfe76bc-272c-4b17-96d4-2167a920e725',
        child_run_uuid='CPU_ONLY_one_original_ROS_cold_repair',closed_predecessor_packet_sha256=unit.common.sha(v['predecessor']))
    for key,raw in (('parent_before_sha256',pb),('Core_before_sha256',cb)):
        q[key]=x[key]=unit.common.sha(raw)
    return v,fields


def next_proposal(v):
    m=manifest(v);m['execution']['sealed_inputs'][unit.PACK]=unit.common.sha(v['predecessor'])
    return resources.reserve(*map(unit.common.encode,(v['p'],v['c'],v['q'],m,v['x'])),v['VAP1'],
        forecast_inputs=v['inputs'],DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'],
        namespace_proof=v['space'],now_epoch=NOW,predecessor_bytes=v['predecessor'])


def test_one_exact_closed_zero_constructor_continuation_retains_costs_and_all_future_work(continuation):
    v,_=continuation;before=deepcopy(v);r=next_proposal(v)
    assert v==before and len(r['proposed_parent'][resources.budget.ROWS])==2
    old=r['proposed_parent'][resources.budget.ROWS][unit.REQUEST]
    assert old==v['p'][resources.budget.ROWS][unit.REQUEST]
    assert r['proposed_parent']['CPU_s']==v['p']['CPU_s']
    assert r['proposed_parent']['CPU_reserved_s']['local_PC']==60
    assert r['forecast']['admissible_forecast'] and not r['native_dispatch_admitted']
    assert not any(r['forecast']['native_IO_accounting']['IO_credit'].values())
    final=settle(r,terminal(r))
    resources.budget.rows(final['proposed_parent'],final['proposed_Core'])
    assert final['proposed_parent'][resources.budget.ROWS][unit.REQUEST]==old
    assert not any(final['proposed_parent']['CPU_reserved_s'].values())
    v['p'],v['c']=final['proposed_parent'],final['proposed_Core']
    with pytest.raises(PermissionError):next_proposal(v) # No third attempt/pool.


@pytest.mark.parametrize('fault',('raw_changed','missing_lease','old_request','old_child','unsettled',
    'dropped_paid_cost','unsealed_packet','swapped_packet','extra_row'))
def test_no_replay_partial_unknown_or_unfunded_cold_continuation(continuation,fault):
    v,fields=continuation
    if fault=='raw_changed':v['predecessor']+=b' '
    elif fault=='missing_lease':
        fields.pop('lease_bytes')
        with pytest.raises(PermissionError):unit.pack(fields)
        return
    elif fault=='old_request':v['q']['request_id']=unit.REQUEST
    elif fault=='old_child':v['q']['child_run_uuid']=unit.RUN
    elif fault=='unsettled':
        for ledger in (v['p'],v['c']):ledger[resources.budget.ROWS][unit.REQUEST]['status']='RESERVED'
    elif fault=='dropped_paid_cost':v['p'][resources.budget.SETTLEMENTS][unit.REQUEST]['financial_charged']['local_CPU_s']=0
    elif fault=='unsealed_packet':v['q']['closed_predecessor_packet_sha256']='0'*64
    elif fault=='swapped_packet':
        fields['result_bytes']+=b' ';v['predecessor']=unit.pack(fields)
    elif fault=='extra_row':
        for ledger in (v['p'],v['c']):ledger[resources.budget.ROWS]['third']=deepcopy(ledger[resources.budget.ROWS][unit.REQUEST])
    with pytest.raises((PermissionError,KeyError)):next_proposal(v)


@pytest.mark.parametrize('fault',('model_started','physics','unknown_error','active_lease','active_actor','missing_LOCAL','changed_row'))
def test_even_a_synthetically_reanchored_unsafe_or_incomplete_history_rejects(continuation,monkeypatch,fault):
    v,fields=continuation
    key='result_bytes';value=unit.common.obj(fields[key])
    if fault=='model_started':value['actual_constructor_attempts']=1
    elif fault=='physics':value['physics_ticks']=1
    elif fault=='unknown_error':value['error']='UNKNOWN_NATIVE_WRITE'
    elif fault=='active_lease':key='lease_bytes';value=unit.common.obj(fields[key]);value['status']='ACTIVE'
    elif fault=='active_actor':key='actor_proof_bytes';value=unit.common.obj(fields[key]);value['current_matching_actors']=1
    elif fault=='missing_LOCAL':key='independent_bytes';value=unit.common.obj(fields[key]);value['LOCAL_all_8_holds_zero']=False
    elif fault=='changed_row':v['p'][resources.budget.ROWS][unit.REQUEST]['manifest_sha256']='0'*64
    fields[key]=unit.common.encode(value)
    monkeypatch.setattr(unit,'ANCHORS',{k:unit.common.sha(raw) for k,raw in fields.items()})
    v['predecessor']=unit.pack(fields);v['q']['closed_predecessor_packet_sha256']=unit.common.sha(v['predecessor'])
    with pytest.raises(PermissionError):next_proposal(v)
