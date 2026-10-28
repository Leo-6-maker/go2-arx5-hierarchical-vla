"""Pure closed-counter accounting fixtures; no native mirror or C0 PASS."""
from copy import deepcopy
import pytest
from . import target_zero_counter_contract as c
from . import target_zero_counter_projection as unit
from .test_target_zero_counter import fixture,terminal


def closed(fixture):
    r,rid,raw,actor=terminal(fixture);s=c.settle(c.encode(r['proposed_parent']),c.encode(r['proposed_Core']),rid,
        result_bytes=raw,actor_proof_bytes=c.encode(actor));record=s['settlement'];row=s['proposed_parent']['target_zero_counter_reservations'][rid]
    lease=c.encode(dict(status='CLOSED',scope=c.ROLE,run_uuid=row['child_run_uuid'],settlement_sha256=row['settlement_sha256']))
    proof=c.encode(dict(schema='DEC1_TARGET_ZERO_COUNTER_ACTUAL_CORE_CLOSURE_V1',settlement=record,actor_proof_sha256=record['actor_proof_sha256']))
    io=dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1',request_id=record['settlement_id'],
        status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',native_dispatch_admitted=False,after_sha256='a'*64)
    dual=c.encode(dict(schema='DEC1_TARGET_ZERO_COUNTER_DUAL_SETTLEMENT_AND_CLOSURE_V1',settlement=record,
        all_registered_actors_terminal=True,Core=dict(settlement_sha256=row['settlement_sha256'],closed_lease_sha256=c.sha(lease),
            proof_sha256=c.sha(proof),receipt=io,lease_receipt=io),LOCAL=dict(receipt=io,
            readback=dict(status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT',receipt_binding_valid=True,actual_sha256='a'*64))))
    h=dict(authority_release=c.AUTHORITY,campaign_run_uuid=s['proposed_Core']['campaign_run_uuid'],original_T0_epoch=1792348196,
        hard_close_epoch=c.HARD_CLOSE,unknown_effects=[],constructors=50,network_transfer_accounted_bytes=1073741824,
        optimizer_updates=7939,B_COVERAGE=dict(last_native_step=2048),source02_marker='PRESERVED')
    return s['proposed_parent'],s['proposed_Core'],h,{rid:dict(dual_bytes=dual,lease_bytes=lease,Core_proof_bytes=proof)}


def test_only_missing_closed_counter_costs_projected_no_new_global_spend_or_training_change(fixture):
    p,k,h,b=closed(fixture);before=deepcopy((p,k,h,b));r=unit.propose(*map(c.encode,(p,k,h)),closures=b)
    assert r['delta_fields']==dict(constructors=2,network_transfer_accounted_bytes=256*1024**2)
    assert not r['new_global_charge'] and (p,k,h,b)==before
    after=r['proposed_Hopper_DEC1'];assert after['optimizer_updates']==7939 and after['source02_marker']=='PRESERVED'
    again=unit.propose(*map(c.encode,(p,k,after)),closures=b);assert not any(again['delta_fields'].values())
    assert again['total_closed_Core_output_upper_bytes']==64*1024**2


@pytest.mark.parametrize('fault',['lease','dual','live','missing','record','marker'])
def test_projection_rejects_missing_or_not_actually_closed_target_counter(fixture,fault):
    p,k,h,b=closed(fixture);rid=next(iter(b))
    if fault=='lease':v=c.obj(b[rid]['lease_bytes']);v['status']='ACTIVE';b[rid]['lease_bytes']=c.encode(v)
    elif fault=='dual':v=c.obj(b[rid]['dual_bytes']);v['all_registered_actors_terminal']=False;b[rid]['dual_bytes']=c.encode(v)
    elif fault=='live':p['target_zero_counter_reservations'][rid]['status']='RESERVED';p['target_zero_counter_reservations'][rid]['remaining']=deepcopy(c.REQUESTED)
    elif fault=='missing':b={}
    elif fault=='record':p['target_zero_counter_settlements'][rid]['target_task_native_qualified']=True
    elif fault=='marker':h[unit.MARKER]={'unknown':{}}
    with pytest.raises(PermissionError):unit.propose(*map(c.encode,(p,k,h)),closures=b)


def test_counter_storage_adds_after_source_once_and_never_adds_local_peak_twice(fixture):
    p,k,h,b=closed(fixture);h=unit.propose(*map(c.encode,(p,k,h)),closures=b)['proposed_Hopper_DEC1']
    s=dict(closed_source_storage_projection={'old':'unchanged'},used=dict(Core_storage_bytes=100,local_storage_bytes=999),
        dec1_used=dict(Core_storage_bytes=80,local_storage_bytes=999),storage_peaks=dict(Core_storage_bytes=200),
        dec1_storage_peaks=dict(Core_storage_bytes=150))
    result=unit.augment(s,*map(c.encode,(p,k,h)),closures=b)
    assert result['used']['Core_storage_bytes']==100+64*1024**2 and result['used']['local_storage_bytes']==999
    with pytest.raises(PermissionError):unit.augment(result,*map(c.encode,(p,k,h)),closures=b)


def test_no_new_counter_does_not_add_a_scientific_or_resource_gate_to_B(fixture):
    p,k,h,b=closed(fixture)
    for ledger in (p,k):
        ledger.pop('target_zero_counter_reservations');ledger.pop('target_zero_counter_settlements')
        ledger.pop('target_zero_counter_reserved')
    r=unit.propose(*map(c.encode,(p,k,h)),closures={})
    assert r['delta_fields']=={'constructors':0,'network_transfer_accounted_bytes':0}
    assert r['proposed_Hopper_DEC1']==h and r['total_closed_Core_output_upper_bytes']==0
