"""Typed closed SDK fixtures, no IO writer, process, SDK or native admission."""
from copy import deepcopy
import pytest
from . import target_sdk_closed_costs as unit, target_sdk_resources as resources
from .test_target_sdk_resources import packet,reserve
from .test_target_sdk_budget import bundle

@pytest.fixture
def closed(packet):
    q=packet['q'];rid=q['request_id'];r=reserve(packet)
    proof=dict(schema='DEC1_TARGET_SDK_ACTUAL_TERMINAL_ACTOR_PROOF_V1',request_id=rid,manifest_sha256=r['reservation']['manifest_sha256'],
        all_registered_actors_terminal=True,current_matching_actors=0,pid=123,proc_start='987',result_present=False,result_sha256=None)
    result=resources.settle(unit.common.encode(r['proposed_parent']),unit.common.encode(r['proposed_Core']),rid,
        actor_proof_bytes=unit.common.encode(proof),result_bytes=None,LOCAL_meter_bytes=None)
    p,c=result['proposed_parent'],result['proposed_Core'];s=result['settlement'];row=p[unit.budget.ROWS][rid];sid=s['settlement_id']
    lease=dict(status='CLOSED',run_uuid=row['child_run_uuid'],scope=unit.budget.ROLE,uid=2009,owner='EXISTING_LOCAL_PC',
        authority_release=unit.common.PARENT_AUTHORITY,manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],
        all_registered_actors_terminal=True,target_task_native_qualified=False,settlement_sha256=unit.common.sha(unit.common.encode(s)))
    lb=unit.common.encode(lease)
    def receipt(after):return dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1',request_id=sid,
        status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',native_dispatch_admitted=False,after_sha256=after)
    cr=receipt(unit.common.sha(unit.common.encode(c)));lr=receipt(unit.common.sha(lb));pr=receipt(unit.common.sha(unit.common.encode(p)))
    coreproof=unit.common.encode(dict(settlement=s,Core_receipt=cr,lease_receipt=lr))
    dual=dict(status='ACTUAL_CURRENT_SDK_DUAL_ONCE_SETTLED_AND_LEASE_CLOSED',settlement=s,
        Core=dict(closed_lease_sha256=unit.common.sha(lb),closure_proof_sha256=unit.common.sha(coreproof),Core_receipt=cr,lease_receipt=lr,Core_after_sha256=cr['after_sha256']),
        LOCAL_receipt=pr,LOCAL_after_sha256=pr['after_sha256'],LOCAL_readback=dict(receipt_binding_valid=True,actual_sha256=pr['after_sha256']))
    d=dict(authority_release=unit.common.AUTHORITY,campaign_run_uuid=unit.budget.CAMPAIGN,original_T0_epoch=unit.budget.T0,
        hard_close_epoch=unit.budget.HARD_CLOSE,unknown_effects=[],constructors=54,network_transfer_accounted_bytes=10000,
        GPU_reserved_s=0,CPU_reserved_s=0,optimizer_updates_reserved=0,neural_queries_reserved=0,network_transfer_reserved_bytes=0,GPU_site_output_reserved_bytes=0,
        CPU_s=149898.836,optimizer_updates=11714,physics_ticks=30621,episode_starts=5,
        closed_Core_source_financial_projections={'OLD_UNKNOWN':'UNCHANGED'},closed_Core_target_zero_counter_projections={'C0':'UNCHANGED'},
        training_runs={'B_FAILED':1829,'B_LOST':2970,'B_SAVED':4096})
    return p,c,d,{rid:dict(dual_bytes=unit.common.encode(dual),lease_bytes=lb)}

def call(v):
    p,c,d,closures=v
    return unit.propose(*map(unit.common.encode,(p,c,d)),closures=closures)

def test_only_missing_constructor_and_transfer_projection_no_global_or_GPU_charge(closed):
    before=deepcopy(closed);r=call(closed);p,c,d,_=closed
    assert closed==before and r['delta_fields']==dict(constructors=2,network_transfer_accounted_bytes=64*1024**2)
    assert r['proposed_DEC1']['CPU_s']==d['CPU_s'] and r['proposed_DEC1']['optimizer_updates']==11714
    assert r['proposed_DEC1']['episode_starts']==5 and r['proposed_DEC1']['physics_ticks']==30621
    assert r['proposed_DEC1']['training_runs']==d['training_runs']
    assert r['new_global_charge'] is r['global_ledger_mutation'] is r['native_dispatch_admitted'] is False


def test_existing_marker_only_allows_exact_readonly_zero_delta_and_one_storage_augmentation(closed):
    p,c,d,cl=closed;r=call(closed);again=unit.propose(*map(unit.common.encode,(p,c,r['proposed_DEC1'])),closures=cl)
    assert not any(again['delta_fields'].values()) and again['new_markers']=={}
    snap=dict(used=dict(Core_storage_bytes=100*1024**3+7147094016,local_storage_bytes=8*1024**3),
        dec1_used=dict(Core_storage_bytes=80*1024**3+7147094016,local_storage_bytes=8*1024**3),
        storage_peaks=dict(Core_storage_bytes=100*1024**3+7147094016),dec1_storage_peaks=dict(Core_storage_bytes=80*1024**3+7147094016))
    after=unit.storage(snap,projection=again)
    assert after['used']['Core_storage_bytes']==snap['used']['Core_storage_bytes']+512*1024**2
    assert after['used']['local_storage_bytes']==8*1024**3 and 'closed_SDK_storage_projection' not in snap
    with pytest.raises(PermissionError):unit.storage(after,projection=again)
    with pytest.raises(PermissionError):unit.storage(snap,projection=r)
    bad=deepcopy(snap);bad['used']['Core_storage_bytes']=100*1024**3+3506438144
    with pytest.raises(PermissionError):unit.storage(bad,projection=again)

@pytest.mark.parametrize('fault',['one_side','active','missing_closure','bad_native_proof','wrong_uid','wrong_source',
    'unclosed','unknown','other_hold','neural_hold','bad_receipt','bad_LOCAL','changed_existing_marker'])
def test_partial_unknown_active_unclosed_or_modified_costs_never_project(closed,fault):
    p,c,d,cl=deepcopy(closed);rid=next(iter(cl));row=p[unit.budget.ROWS][rid];dual=unit.common.obj(cl[rid]['dual_bytes']);lease=unit.common.obj(cl[rid]['lease_bytes'])
    if fault=='one_side':c[unit.budget.ROWS]={}
    elif fault=='active':row['status']='RESERVED';c[unit.budget.ROWS][rid]=deepcopy(row)
    elif fault=='missing_closure':cl={}
    elif fault=='bad_native_proof':dual['Core']['closure_proof_sha256']='0'*64
    elif fault=='wrong_uid':lease['uid']=123
    elif fault=='wrong_source':lease['source_sha']='0'*40
    elif fault=='unclosed':lease['status']='ACTIVE'
    elif fault=='unknown':d['unknown_effects']=['UNKNOWN']
    elif fault=='other_hold':c['source_cpu_reserved']['physics_ticks']=1
    elif fault=='neural_hold':d['GPU_reserved_s']=1
    elif fault=='bad_receipt':dual['LOCAL_receipt']['request_id']='OTHER'
    elif fault=='bad_LOCAL':dual['LOCAL_after_sha256']='0'*64
    else:
        r=unit.propose(*map(unit.common.encode,(p,c,d)),closures=cl);d=r['proposed_DEC1'];d[unit.MARKER][rid]['financial_charged']['constructors']=1
    if rid in cl:cl[rid]=dict(dual_bytes=unit.common.encode(dual),lease_bytes=unit.common.encode(lease))
    with pytest.raises((PermissionError,KeyError)):unit.propose(*map(unit.common.encode,(p,c,d)),closures=cl)
