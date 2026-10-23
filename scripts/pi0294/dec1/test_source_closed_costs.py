"""Pure source settlement projection contracts, no owner/native operations."""
import sys
_BLOCK=[]
_GUARD_ACTIVE=False
def deny(event,args):
    if _GUARD_ACTIVE and (event.startswith(('socket.','subprocess.')) or event in ('os.system','os.fork','os.forkpty','os.posix_spawn')):
        _BLOCK.append(event);raise PermissionError('NO_EXTERNAL_EFFECT')
sys.addaudithook(deny)
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_closed_costs as u
assert not _BLOCK


def dims(starts=4,ticks=128000):return dict(zip(u.DIMS,(100.,20.,starts,ticks,12,500,200,100)))


def io(sid):return dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1',request_id=sid,
    status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',native_dispatch_admitted=False,after_sha256='e'*64)

@pytest.fixture
def facts(monkeypatch):
    # Fully synthetic branch receipts. Pinned old constants are changed only here.
    old_amount=dims(1,16490);old_amount['constructors']=6
    invoice=dict(settlement_id=u.OLD+'-terminal-bound-close',financial_charged=old_amount,
        scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',actual_returned_STOP_ACK=False,live_financial_upper=16000)
    ih=u.sha(u.encode(invoice));monkeypatch.setattr(u,'INVOICE',ih)
    row=lambda rid:dict(schema='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1',
        role='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_V1',authority_release=u.AUTHORITY,
        request_id=rid,child_run_uuid='source01' if rid==u.OLD else 'source02',
        status='SETTLED',unknown_effects=[],requested=dims(),remaining=dict.fromkeys(u.DIMS,0))
    old=row(u.OLD);old.update(financial_closeout_sha256=ih,scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',full_unresolved_live_bound_charged=True)
    known_amount=dims(1,1200)
    known=dict(schema='DEC1_SOURCE_CPU_KNOWN_TERMINAL_SETTLEMENT_V1',request_id=u.NEW,
        settlement_id=u.NEW+'-known-terminal-close',manifest_sha256='14ce2f841c8db3de7e2eb9c191e95fc9b26720f628493774e4fd954012556b0f',
        source_sha='47dc6001227572f031ebc3cc33d56a487027d8cb',financial_charged=known_amount,
        child_ledger_sha256='c'*64,conditions=[dict(status=s) for s in ('NATIVE_SAFETY_FAIL','NOT_RUN','NOT_RUN','NOT_RUN')],
        actual_live_ticks=800,actual_shadow_ticks=400,student_neural_ACK_qualification=False,
        target_qualification=False,core64_cells=0,native_dispatch_admitted=False)
    kh=u.sha(u.encode(known));new=row(u.NEW);new.update(known_terminal_settlement_sha256=kh,
        scientific_verdict='SOURCE_COMPONENT_RESULTS_KNOWN_NOT_STUDENT_QUALIFICATION')
    ledger=dict(authority_release=u.AUTHORITY,campaign_run_uuid='campaign',original_T0_epoch=1,hard_close_epoch=1792952996,
        unknown_effects=[],source_cpu_reservations={u.OLD:old,u.NEW:new},source_cpu_reserved=dict.fromkeys(u.DIMS,0),
        source_cpu_financial_closeouts={u.OLD:invoice},source_cpu_known_terminal_settlements={u.NEW:known},
        CPU_s=10000.,episode_starts=2,physics_ticks=17690,constructors=18)
    c=deepcopy(ledger);p=deepcopy(ledger);p.update(run_uuid='campaign',T0_epoch=1,hard_closeout_epoch=1792952996,Hopper_unknown_effects=[])
    d=dict(authority_release=u.AUTHORITY,campaign_run_uuid='campaign',original_T0_epoch=1,hard_close_epoch=1792952996,
        unknown_effects=[],CPU_s=9876.5,GPU_allocated_s=77777,GPU_reserved_s=0,optimizer_updates=7939,neural_queries=444,
        new_submissions=['all original history'],training_runs={'A':{'last_native_step':4096},'B_COVERAGE':{'last_native_step':2048}},
        episode_starts=1,physics_ticks=16490,constructors=6,network_transfer_accounted_bytes=100,
        old_1024_checkpoint={'unchanged':True},old_lost_updates=771)
    def dual(record,kind,lease,proof=None):
        sid=record['settlement_id'];core=dict(Core_receipt=io(sid),lease_receipt=io(sid),lease_sha256=u.sha(u.encode(lease)),
            proof_sha256='f'*64 if proof is None else u.sha(u.encode(proof)))
        value=dict(status='SOURCE_DUAL_FINANCIAL_TERMINAL_BOUND_SETTLED_CORE_LEASE_CLOSED_COMPONENT_UNKNOWN' if kind=='old'
            else 'SOURCE_DUAL_KNOWN_TERMINAL_SETTLED_LEASE_CLOSED_NOT_STUDENT_QUALIFICATION',
            Core=core,LOCAL=io(sid),LOCAL_readback=dict(status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT',actual_sha256='e'*64,receipt_binding_valid=True))
        if kind=='old':value['invoice']=record;core['invoice_sha256']=u.sha(u.encode(record))
        else:value['settlement']=record;core['settlement_sha256']=u.sha(u.encode(record))
        return value
    lease1=dict(run_uuid='source01',status='CLOSED',actual_STOP_ACK=False,original_child_ledger_unknowns_preserved=True,financial_closeout_sha256='f'*64)
    proof2=dict(schema='DEC1_SOURCE_CORE_KNOWN_TERMINAL_CLOSURE_V1',settlement=known,settlement_sha256=kh,
        Core_receipt=io(known['settlement_id']),original_child_ledger_unchanged_sha256='c'*64)
    lease2=dict(run_uuid='source02',status='CLOSED',all_registered_workers_terminal=True,known_terminal_settlement_sha256=kh,
        known_terminal_proof_sha256=u.sha(u.encode(proof2)))
    dual1=dual(invoice,'old',lease1);dual2=dual(known,'new',lease2,proof2)
    monkeypatch.setattr(u,'OLD_DUAL',u.sha(u.encode(dual1)));monkeypatch.setattr(u,'OLD_LEASE',u.sha(u.encode(lease1)))
    oldmarker=dict(schema='DEC1_CLOSED_CORE_SOURCE_NONADDITIVE_COST_PROJECTION_V1',request_id=u.OLD,source_invoice_sha256=ih,
        dual_financial_receipt_sha256=u.sha(u.encode(dual1)),financial_charged=deepcopy(old_amount),
        already_inside_global_parent_spend=True,adds_no_global_charge=True,scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',
        historical_actual_parent_hash='do not replace')
    d[u.OLD_MARKER]={u.OLD:oldmarker}
    closures={u.OLD:dict(dual_receipt_bytes=u.encode(dual1),closed_lease_bytes=u.encode(lease1)),
        u.NEW:dict(dual_receipt_bytes=u.encode(dual2),closed_lease_bytes=u.encode(lease2),Core_proof_bytes=u.encode(proof2))}
    # CPython audit hooks cannot be removed. Keep this hook inactive during
    # collection and every unrelated test in the same pytest process. Both this
    # module and storage consumers activate it only while using this fixture.
    global _GUARD_ACTIVE
    _GUARD_ACTIVE=True
    try:
        yield dict(p=p,c=c,d=d,closures=closures,oldmarker=oldmarker)
    finally:
        _GUARD_ACTIVE=False
        assert not _BLOCK


def call(f):return u.propose(u.encode(f['p']),u.encode(f['c']),u.encode(f['d']),closures=f['closures'])


def test_B2048_counts_only_unprojected_known_safety_failure_and_retains_every_old_field(facts):
    before=deepcopy(facts);out=call(facts);after=out['proposed_Hopper_DEC1']
    assert out['per_request'][u.OLD]['already_projected'] is True
    assert out['per_request'][u.OLD]['delta_fields']==dict.fromkeys(u.FIELDS.values(),0)
    assert out['delta_fields']==dict(episode_starts=1,physics_ticks=1200,constructors=12,network_transfer_accounted_bytes=100)
    for key in ('CPU_s','GPU_allocated_s','GPU_reserved_s','optimizer_updates','neural_queries','new_submissions','training_runs','old_1024_checkpoint','old_lost_updates'):
        assert after[key]==before['d'][key]
    assert after[u.OLD_MARKER]==before['d'][u.OLD_MARKER] and facts==before
    assert out['native_dispatch_admitted'] is False and out['no_new_global_charge'] is True
    assert out['unprojected_Core_output_upper_bytes']==500 and not _BLOCK


def test_second_projection_has_zero_delta_without_mutating_old_marker(facts):
    first=call(facts);facts['d']=first['proposed_Hopper_DEC1'];again=call(facts)
    assert not any(again['delta_fields'].values()) and again['new_markers']=={}
    assert again['proposed_Hopper_DEC1']==facts['d']


def test_active_Source02_preserves_full8_pending_without_scientific_success_gate(facts):
    for ledger in (facts['p'],facts['c']):
        row=ledger['source_cpu_reservations'][u.NEW];row.update(status='RUNNING',remaining=dims())
        ledger['source_cpu_reserved']=dims();del ledger[u.KNOWN_INDEX][u.NEW]
        row.pop('known_terminal_settlement_sha256');row.pop('scientific_verdict')
    del facts['closures'][u.NEW]
    out=call(facts);assert out['active_source_remaining']==dims() and out['pending_adapter_required'] is True
    assert not any(out['delta_fields'].values()) and out['new_markers']=={}


@pytest.mark.parametrize('change',[
    lambda f:f['c'].update(unknown_effects=['UNKNOWN']),
    lambda f:f['p']['source_cpu_reservations'][u.NEW].update(status='UNKNOWN'),
    lambda f:f['c']['source_cpu_reservations'][u.NEW]['remaining'].update(physics_ticks=1),
    lambda f:f['d'][u.OLD_MARKER][u.OLD]['financial_charged'].update(physics_ticks=0),
    lambda f:f['d'][u.OLD_MARKER][u.OLD].update(source_invoice_sha256='0'*64),
    lambda f:f['closures'][u.NEW].update(closed_lease_bytes=b'{}'),
    lambda f:f['closures'][u.NEW].update(Core_proof_bytes=b'{}'),
    lambda f:f['closures'].pop(u.NEW),
    lambda f:f['p']['source_cpu_reserved'].update(physics_ticks=1),
    lambda f:f['c'].update(physics_ticks=0),
])
def test_inconsistent_or_unknown_costs_and_missing_closed_proofs_reject(facts,change):
    change(facts)
    with pytest.raises((PermissionError,KeyError)):call(facts)
