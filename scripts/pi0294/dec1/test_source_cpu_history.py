"""Synthetic history contracts; no native qualification or external effects."""
import sys
_BLOCK=[]
_ACTIVE=True
def _deny(event,args):
    if _ACTIVE and (event.startswith(('socket.','subprocess.')) or event in ('os.system','os.fork','os.forkpty','os.posix_spawn')):
        _BLOCK.append(event);raise PermissionError('NO_EXTERNAL_EFFECT')
sys.addaudithook(_deny)
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
_s=importlib.util.spec_from_file_location('source_cpu_history',Path(__file__).with_name('source_cpu_history.py'))
u=importlib.util.module_from_spec(_s);_s.loader.exec_module(u)
assert not _BLOCK
_ACTIVE=False


@pytest.fixture(autouse=True)
def no_external_effects():
    global _ACTIVE
    before=len(_BLOCK);_ACTIVE=True
    try:yield
    finally:
        _ACTIVE=False
        assert len(_BLOCK)==before


def base(row=None):return dict(zip(u.DIMS,(100.,20.,4,128000,326,500,200,100))) if row is None else deepcopy(row)

@pytest.fixture
def facts(monkeypatch):
    # Every SHA patched below is synthetic. Actual deployed constants stay fixed.
    child=dict(run_uuid=u.OLD_CHILD,unknown_effects=[{'identifier':u.OLD_CONDITION,'error':'SourceComponentUnknown'}],
        reservations={u.OLD_CONDITION:dict(maximum=16000,kind='live_and_recovery_integrations')})
    invoice=dict(request_id=u.OLD_REQUEST,child_run_uuid=u.OLD_CHILD,scientific_verdict=u.VERDICT,
        actual_returned_STOP_ACK=False,source_or_target_qualified=False,native_dispatch_admitted=False,
        original_T0_epoch=1,settled_epoch=10.,hard_close_epoch=100.,live_financial_upper=16000,
        original_live_reservation_and_unknowns=dict(unknown_effects=deepcopy(child['unknown_effects']),reservations=deepcopy(child['reservations'])),
        financial_charged=dict(zip(u.DIMS,(10.,5.,1,16490,6,500,200,100))))
    invoice_hash=u.sha(u.encode(invoice));monkeypatch.setattr(u,'INVOICE_SHA',invoice_hash)
    row=dict(schema=u.ROW_SCHEMA,role=u.ROLE,request_id=u.OLD_REQUEST,child_run_uuid=u.OLD_CHILD,
        child_root='/registered/campaign/children/'+u.OLD_CHILD,status='SETTLED',unknown_effects=[],
        requested=base(),remaining=dict.fromkeys(u.DIMS,0),financial_closeout_sha256=invoice_hash,
        scientific_verdict=u.VERDICT,full_unresolved_live_bound_charged=True)
    c=dict(unknown_effects=[],original_T0_epoch=1,hard_close_epoch=100.,CPU_s=100.,episode_starts=1,physics_ticks=16490,constructors=6,
        source_cpu_reservations={u.OLD_REQUEST:deepcopy(row)},source_cpu_reserved=dict.fromkeys(u.DIMS,0),
        source_cpu_financial_closeouts={u.OLD_REQUEST:deepcopy(invoice)})
    p=dict(unknown_effects=[],Hopper_unknown_effects=[],T0_epoch=1,hard_closeout_epoch=100.,
        source_cpu_reservations={u.OLD_REQUEST:deepcopy(row)},source_cpu_reserved=dict.fromkeys(u.DIMS,0),
        source_cpu_financial_closeouts={u.OLD_REQUEST:deepcopy(invoice)},
        physics_ticks=dict(live=40,shadow=10,reconstruction=0,closed_source_component_financial_upper=16490),
        old_history={'failed_updates':771,'old_D_episodes':48},CPU_s={'CoreLab':31840.5})
    audit=dict(schema='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1',
        conditions=[dict(condition_id=cid,status=u.VERDICT if cid==u.OLD_CONDITION else 'NOT_RUN') for cid in (u.OLD_CONDITION,*u.NOT_RUN)])
    readback=dict(status='INDEPENDENT_DUAL_FINANCIAL_CLOSURE_RETAINED_UNKNOWN_AND_FILES_CONFIRMED',
        invoice_sha256=invoice_hash,active_workers=0,source_lease_status='CLOSED',original_child_ledger_unchanged=True,
        unknown_and_original_live_reservation_preserved=True,original_failure_files_readback=13,Core_ledger=deepcopy(c))
    docs=dict(dual_receipt=dict(invoice=invoice,Core=dict(invoice_sha256=invoice_hash,proof_sha256='f'*64)),
        original_child_ledger=child,closed_lease=dict(status='CLOSED',run_uuid=u.OLD_CHILD,actual_STOP_ACK=False,
            scientific_verdict=u.VERDICT,original_child_ledger_unknowns_preserved=True,financial_closeout_sha256='f'*64),
        original_audit=dict(returncode=0,read_only=True,stdout=json.dumps(dict(result=audit))),
        closure_readback=dict(returncode=0,read_only=True,stdout=json.dumps(readback)))
    pins={k:u.sha(u.encode(v)) for k,v in docs.items()};monkeypatch.setattr(u,'PINS',pins)
    pack=dict(schema=u.SCHEMA,documents={k:dict(base64=base64.b64encode(u.encode(v)).decode()) for k,v in docs.items()})
    raw=u.encode(pack)
    return dict(parent=p,Core=c,raw=raw,row=row,invoice=invoice,docs=docs)


def history(f):return u.validate_history(f['raw'],f['parent'],f['Core'],now_epoch=20.)


def continuation(h):
    return dict(history_sha256=h['history_sha256'],original_commissioning_requested=3,
        engineering_replacements_requested=1,replacement_condition_id=u.OLD_CONDITION,
        replacement_of_request_id=u.OLD_REQUEST,replacement_index=1,original_condition_ids=list(u.NOT_RUN))


def newrow(h):
    return dict(schema=u.ROW_SCHEMA,role=u.ROLE,request_id='new-request',child_run_uuid='new-child',
        child_root='/registered/campaign/children/new-child',status='RESERVED',unknown_effects=[],
        requested=base(),remaining=base(),continuation=continuation(h))


def test_legacy_no_history_keeps_four_and_three_bucket_behavior():
    assert u.validate_history(None,{}, {},now_epoch=None) is None
    assert u.row_slots(dict(requested={'episode_starts':4}),{})==dict(commissioning_count=4,engineering_count=0)
    assert u.physics_total({'physics_ticks':dict(live=1,shadow=2,reconstruction=3)})==6
    with pytest.raises(PermissionError):u.validate_continuation(dict(requested={'episode_starts':4},continuation={}))


def test_exact_closed_old_counts_one_and_no_refund_or_mutation(facts):
    before=deepcopy(facts);h=history(facts)
    assert u.row_slots(facts['row'],facts['Core'],h)==dict(commissioning_count=1,engineering_count=0)
    assert facts==before and facts['row']['requested']['episode_starts']==4
    assert h['engineering_permission_verified'] is False and 'native_dispatch_admitted' not in h
    assert u.physics_total(facts['parent'],h)==16540 and not _BLOCK


def test_continuation_three_original_one_replacement_and_native_row(facts):
    h=history(facts);r=newrow(h);assert u.row_slots(r,facts['Core'],h)==dict(commissioning_count=3,engineering_count=1)
    assert u.validate_continuation(r,h)==dict(commissioning_count=3,engineering_count=1)
    for ledger in (facts['parent'],facts['Core']):
        ledger['source_cpu_reservations'][r['request_id']]=deepcopy(r);ledger['source_cpu_reserved']=base()
    h2=history(facts);assert u.row_slots(r,facts['Core'],h2)['engineering_count']==1
    other=deepcopy(r);other.update(request_id='third',child_run_uuid='third-child')
    with pytest.raises(PermissionError):u.row_slots(other,facts['Core'],h2)


@pytest.mark.parametrize('change',[
    lambda f:f['parent'].update(unknown_effects=['new unknown']),
    lambda f:f['Core'].update(unknown_effects=['new unknown']),
    lambda f:f['row'].update(status='RUNNING'),
    lambda f:f['Core']['source_cpu_reservations'][u.OLD_REQUEST]['requested'].update(episode_starts=1),
    lambda f:f['Core']['source_cpu_reservations'][u.OLD_REQUEST]['remaining'].update(physics_ticks=1),
    lambda f:f['Core']['source_cpu_reservations'][u.OLD_REQUEST].update(unknown_effects=['unknown']),
    lambda f:f['parent']['physics_ticks'].update(closed_source_component_financial_upper=0),
    lambda f:f['parent']['physics_ticks'].update(other_financial_upper=1),
    lambda f:f['parent']['source_cpu_financial_closeouts'].update(unpinned=deepcopy(f['invoice'])),
])
def test_bad_current_history_or_uncovered_financial_bucket_reject(facts,change):
    change(facts)
    # f['row'] is a separate preserved fixture, so propagate only its intentional status mutation.
    if facts['row']['status']!='SETTLED':facts['Core']['source_cpu_reservations'][u.OLD_REQUEST]['status']=facts['row']['status']
    with pytest.raises(PermissionError):history(facts)


@pytest.mark.parametrize('field,value',[
    ('history_sha256','bad'),('original_commissioning_requested',4),('original_commissioning_requested',True),
    ('engineering_replacements_requested',2),('replacement_condition_id','new-selected-condition'),
    ('replacement_of_request_id','another-request'),('replacement_index',2),
    ('original_condition_ids',[u.NOT_RUN[0]]*3),('original_condition_ids',[u.OLD_CONDITION,*u.NOT_RUN[:2]])])
def test_continuation_cannot_duplicate_condition_or_expand_allocation(facts,field,value):
    h=history(facts);row=newrow(h);row['continuation'][field]=value
    with pytest.raises(PermissionError):u.validate_continuation(row,h)


@pytest.mark.parametrize('now',[True,float('inf'),9.,100.])
def test_invalid_or_late_epoch_reject(facts,now):
    with pytest.raises(PermissionError):u.validate_history(facts['raw'],facts['parent'],facts['Core'],now_epoch=now)


def test_any_pinned_raw_bytes_change_reject(facts):
    pack=u.obj(facts['raw']);pack['documents']['original_child_ledger']['base64']=base64.b64encode(b'{}').decode()
    with pytest.raises(PermissionError):u.validate_history(u.encode(pack),facts['parent'],facts['Core'],now_epoch=20.)



def repair_packet(facts):
    h=history(facts);b=newrow(h)
    b['continuation']['engineering_fix_evidence_sha256']=u.REPAIR_SHA
    m=dict(execution=dict(sealed_inputs={u.REPAIR_PLAN:u.REPAIR_SHA}))
    # Exact actual published evidence is a read-only local text fixture, not execution.
    root=Path(__file__).resolve().parents[3]
    raw=(root/'reports/PI0294/design/WDR1_SOURCE_COMPONENT_NATIVE_FIRST_ATTEMPT_AND_STOP_READBACK_CPU_ACTUAL_20261023.json').read_bytes()
    return h,b,m,raw


def test_exact_published_cold_repair_bound_without_native_permission(facts):
    h,b,m,raw=repair_packet(facts)
    out=u.validate_repair(m,b,h,raw,source_members=deepcopy(u.REPAIR_MEMBERS))
    assert out['engineering_evidence_sha256']==u.REPAIR_SHA
    assert out['actual_native_reexecution_proven'] is False and 'native_dispatch_admitted' not in out
    assert u.validate_repair({}, {},None,None) is None


@pytest.mark.parametrize('bad',['bytes','manifest','budget','members'])
def test_bad_repair_binding_or_changed_cold_member_reject(facts,bad):
    h,b,m,raw=repair_packet(facts);members=deepcopy(u.REPAIR_MEMBERS)
    if bad=='bytes':raw+=b' '
    if bad=='manifest':m['execution']['sealed_inputs'][u.REPAIR_PLAN]='0'*64
    if bad=='budget':b['continuation']['engineering_fix_evidence_sha256']='0'*64
    if bad=='members':members['scripts/pi0294/dec1/source_sync_stop.py']='0'*64
    with pytest.raises(PermissionError):u.validate_repair(m,b,h,raw,source_members=members)
