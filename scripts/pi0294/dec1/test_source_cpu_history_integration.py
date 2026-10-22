"""Real intake/proposal/runtime seams with synthetic metadata, no native GO."""
import sys
_BLOCK=[]
_ACTIVE=True
def _deny(event,args):
    if _ACTIVE and (event.startswith(('socket.','subprocess.')) or event in ('os.system','os.fork','os.forkpty','os.posix_spawn')):
        _BLOCK.append(event);raise PermissionError('NO_EXTERNAL_EFFECT')
sys.addaudithook(_deny)
import base64
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_intake as intake
from scripts.pi0294.dec1 import source_cpu_reservation as reserve
from scripts.pi0294.dec1 import source_cpu_runtime as runtime
from scripts.pi0294.dec1 import source_cpu_history as history
from scripts.pi0294.dec1.test_source_cpu_intake import packet, encode, sha
from scripts.pi0294.dec1.test_source_cpu_reservation import facts
from scripts.pi0294.dec1.test_source_commissioning_plan import population
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

# Exact already published report bytes, no fabricated STOP ACK or permission.
ENGINEERING_FIXTURE_EXTENSION = {'engineering_fix_evidence_sha256':history.REPAIR_SHA}


def _zero():
    return dict.fromkeys(intake.DIMS, 0)


def _remainder(f):
    p=f['p']; parent=p['parent']; b=p['budget']
    pending={k:sum(c.get(k+'_reserved',0) for c in parent['children'])
             for k in ('episode_starts','physics_ticks','engine_starts')}
    b['parent_remaining_after_spent_reserved_and_final'].update(
        CoreLab_CPU_s=320*3600-parent['CPU_s']['CoreLab']-parent['CPU_reserved_s']['CoreLab'],
        local_CPU_s=48*3600-parent['CPU_s']['local_PC']-parent['CPU_reserved_s']['local_PC'],
        episode_starts=900-parent['episode_starts']-pending['episode_starts'],
        physics_ticks=40000000-3200000-sum(parent['physics_ticks'].values())-pending['physics_ticks'],
        constructors=100000-parent['engine_starts']-pending['engine_starts'])


@pytest.fixture
def cold(facts,monkeypatch):
    f=facts; p=f['p']; parent=p['parent']; core=f['current']; b=p['budget']
    # Start from the existing real reservation constructor, before adding history.
    old=deepcopy(reserve.propose(**f['bind']())['reservation'])
    old.update(request_id=history.OLD_REQUEST,child_run_uuid=history.OLD_CHILD,
        child_root='/registered/'+core['campaign_run_uuid']+'/children/'+history.OLD_CHILD,
        child_manifest_sha256='d'*64,status='SETTLED',remaining=_zero(),unknown_effects=[],
        scientific_verdict=history.VERDICT,full_unresolved_live_bound_charged=True)
    assert old['requested']['episode_starts']==4
    charged=_zero()|dict(CoreLab_CPU_s=1039.537288565,local_CPU_s=720,
        episode_starts=1,physics_ticks=16490,constructors=6,
        CoreLab_output_bytes=3506438144,local_output_bytes=67108864,transfer_bytes=1073741824)
    original_unknown=dict(reservations={history.OLD_CONDITION:dict(
        kind='live_and_recovery_integrations',maximum=16000,started_epoch=p['now_epoch']-10)},
        unknown_effects=[dict(identifier=history.OLD_CONDITION,error='SourceComponentUnknown',reservation_retained=True)])
    invoice=dict(request_id=history.OLD_REQUEST,child_run_uuid=history.OLD_CHILD,
        scientific_verdict=history.VERDICT,actual_returned_STOP_ACK=False,
        source_or_target_qualified=False,native_dispatch_admitted=False,
        settled_epoch=p['now_epoch']-1,original_T0_epoch=parent['T0_epoch'],
        hard_close_epoch=intake.HARD_CLOSE,live_financial_upper=16000,
        original_live_reservation_and_unknowns=deepcopy(original_unknown),financial_charged=charged)
    invoice_sha=sha(history.encode(invoice)); monkeypatch.setattr(history,'INVOICE_SHA',invoice_sha)
    old['financial_closeout_sha256']=invoice_sha
    for ledger in (parent,core):
        ledger.update(source_cpu_reservations={history.OLD_REQUEST:deepcopy(old)},
            source_cpu_reserved=_zero(),source_cpu_financial_closeouts={history.OLD_REQUEST:deepcopy(invoice)})
    core['CPU_s']+=charged['CoreLab_CPU_s'];core['episode_starts']=1
    core['physics_ticks']=16490;core['constructors']=6
    used=deepcopy(b['DEC1_used'])
    for k in intake.DIMS:used[k]+=charged[k]
    b['DEC1_used']=used;core['source_cpu_current_classification']=deepcopy(used)
    parent['CPU_s']['CoreLab']+=charged['CoreLab_CPU_s'];parent['CPU_s']['local_PC']+=720
    parent['episode_starts']+=1;parent['engine_starts']+=6
    parent['physics_ticks']['closed_source_component_financial_upper']=16490
    parent['Hopper_unknown_effects']=[]
    b.update(source_commissioning_used=1,source_commissioning_reserved=0)
    _remainder(f)
    child=dict(run_uuid=history.OLD_CHILD,**deepcopy(original_unknown))
    proof_sha='c'*64
    lease=dict(status='CLOSED',run_uuid=history.OLD_CHILD,actual_STOP_ACK=False,
        scientific_verdict=history.VERDICT,original_child_ledger_unknowns_preserved=True,
        financial_closeout_sha256=proof_sha)
    conditions=[dict(condition_id=c,status=history.VERDICT if c==history.OLD_CONDITION else 'NOT_RUN')
                for c in (history.OLD_CONDITION,*history.NOT_RUN)]
    def wrapper(v):return dict(returncode=0,read_only=True,stdout=encode(v).decode(),stderr='')
    audit=wrapper(dict(result=dict(schema='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1',conditions=conditions)))
    readback=wrapper(dict(status='INDEPENDENT_DUAL_FINANCIAL_CLOSURE_RETAINED_UNKNOWN_AND_FILES_CONFIRMED',
        invoice_sha256=invoice_sha,active_workers=0,source_lease_status='CLOSED',
        original_child_ledger_unchanged=True,unknown_and_original_live_reservation_preserved=True,
        original_failure_files_readback=13,Core_ledger=deepcopy(core)))
    # Production invoice bytes preserve insertion order; do not sort the
    # nested invoice after computing its full-byte identity.
    docs=dict(dual_receipt=history.encode(dict(invoice=invoice,Core=dict(invoice_sha256=invoice_sha,proof_sha256=proof_sha))),
        original_child_ledger=encode(child),closed_lease=encode(lease),original_audit=encode(audit),closure_readback=encode(readback))
    monkeypatch.setattr(history,'PINS',{k:sha(v) for k,v in docs.items()})
    history_bytes=encode(dict(schema=history.SCHEMA,documents={
        k:dict(base64=base64.b64encode(v).decode()) for k,v in docs.items()}))
    continuation=dict(history_sha256=sha(history_bytes),original_commissioning_requested=3,
        engineering_replacements_requested=1,replacement_condition_id=history.OLD_CONDITION,
        replacement_of_request_id=history.OLD_REQUEST,original_condition_ids=list(history.NOT_RUN),replacement_index=1)
    continuation.update(deepcopy(ENGINEERING_FIXTURE_EXTENSION))
    b['continuation']=continuation
    report=Path(__file__).resolve().parents[3]/'reports/PI0294/design/WDR1_SOURCE_COMPONENT_NATIVE_FIRST_ATTEMPT_AND_STOP_READBACK_CPU_ACTUAL_20261023.json'
    f['engineering_bytes']=report.read_bytes()
    assert sha(f['engineering_bytes'])==history.REPAIR_SHA
    f['seal']['files'].update(history.REPAIR_MEMBERS)
    f.update(history_bytes=history_bytes,old_row=deepcopy(old),old_child=child,old_invoice=invoice)
    return f


def _bind(f,root):
    # Existing facts.bind seals all original fields. Extend only actual new seals.
    args=f['bind'](child_root=str(root));p=f['p'];b=p['budget'];m=f['m']
    if 'continuation' in b:f['projection']['continuation']=deepcopy(b['continuation'])
    raw=encode(f['projection']); b['native_projection_receipt_sha256']=sha(raw)
    m['execution']['sealed_inputs'][reserve.PROJECTION]=sha(raw)
    m['execution']['sealed_inputs'][intake.PATHS['budget']]=sha(encode(b))
    if 'history_bytes' in f:
        m['execution']['sealed_inputs'][history.HISTORY_PLAN]=sha(f['history_bytes'])
        m['execution']['sealed_inputs'][history.REPAIR_PLAN]=sha(f['engineering_bytes'])
        args['history_bytes']=f['history_bytes']
        args['engineering_bytes']=f['engineering_bytes']
        args['history_parent_bytes']=encode(p['parent'])
    args.update(projection_bytes=raw,budget_bytes=encode(b),manifest_bytes=encode(m))
    return args


def _root(f,tmp_path):
    root=tmp_path/f['current']['campaign_run_uuid']/'children'/f['m']['run_uuid']
    root.mkdir(parents=True,exist_ok=True)
    return root


def _validate(f):
    extra={'history_bytes':f['history_bytes']} if 'history_bytes' in f else {}
    return intake.validate_budget(f['p']['budget'],f['p']['parent'],f['current'],f['p']['adoption'],
                                  now_epoch=f['p']['now_epoch'],**extra)


def _propose(f,root):
    return reserve.propose_parent(parent_bytes=encode(f['p']['parent']),
        adoption_bytes=encode(f['p']['adoption']),**_bind(f,root))


def _runtime(f,root,proposal,*,closeout=False,now=None):
    raw=encode(f['m']);path=root/'RUN_MANIFEST.json'
    if path.exists():path.chmod(0o600)
    path.write_bytes(raw);path.chmod(0o400)
    extra={'history_bytes':f['history_bytes'],'engineering_bytes':f['engineering_bytes'],
           'parent_bytes':encode(f['p']['parent']),'source_members':f['seal']['files']} if 'history_bytes' in f else {}
    runtime._reservation(root,f['m'],f['p']['stage'],f['p']['budget'],
        proposal['Core_proposal']['proposed_ledger'],f['current'],closeout=closeout,
        now=f['p']['now_epoch'] if now is None else now,**extra)


def test_default_legacy_three_layers_unchanged(facts,tmp_path):
    f=facts;root=_root(f,tmp_path)
    requested=_validate(f);proposal=_propose(f,root);_runtime(f,root,proposal)
    assert requested==f['p']['budget']['requested']
    assert 'continuation' not in proposal['reservation']
    assert proposal['native_dispatch_admitted'] is False


def test_cold_three_layers_preserve_unknown_history_and_charge_upper_once(cold,tmp_path):
    f=cold;root=_root(f,tmp_path);before=deepcopy(f['p']['parent']);pre=deepcopy(f['current'])
    assert _validate(f)==f['p']['budget']['requested']
    proposal=_propose(f,root);_runtime(f,root,proposal)
    # Exercise the complete pure intake as well, including all original15 pins.
    p=f['p']
    accepted=intake.validate(f['m'],stage_bytes=encode(p['stage']),plan_bytes=encode(p['plan']),
        population_bytes=p['population_bytes'],adoption_bytes=encode(p['adoption']),
        parent_bytes=encode(p['parent']),dec1_bytes=encode(f['current']),budget_bytes=encode(p['budget']),
        source_files=p['source_files'],now_epoch=p['now_epoch'],history_bytes=f['history_bytes'],
        engineering_bytes=f['engineering_bytes'])
    assert accepted['native_dispatch_admitted'] is False
    parent=proposal['proposed_ledger'];core=proposal['Core_proposal']['proposed_ledger'];row=proposal['reservation']
    assert parent['source_cpu_reservations'][row['request_id']]==core['source_cpu_reservations'][row['request_id']]==row
    assert row['continuation']==f['p']['budget']['continuation']==f['projection']['continuation']
    assert row['continuation']['history_sha256']==f['m']['execution']['sealed_inputs'][history.HISTORY_PLAN]
    for ledger in (parent,core):assert ledger['source_cpu_reservations'][history.OLD_REQUEST]==f['old_row']
    assert parent['CPU_s']==before['CPU_s'] and parent['physics_ticks']==before['physics_ticks']
    assert parent['physics_ticks']['closed_source_component_financial_upper']==16490
    assert core['CPU_s']==pre['CPU_s'] and core['physics_ticks']==pre['physics_ticks']==16490
    assert f['old_child']['unknown_effects'] and f['old_child']['reservations'][history.OLD_CONDITION]['maximum']==16000
    assert row['requested']['episode_starts']==4 and f['old_row']['requested']['episode_starts']==4
    verified=history.validate_history(f['history_bytes'],before,pre,now_epoch=f['p']['now_epoch'])
    assert history.row_slots(f['old_row'],pre,verified)==dict(commissioning_count=1,engineering_count=0)
    assert history.row_slots(row,core,verified)==dict(commissioning_count=3,engineering_count=1)
    assert not proposal['native_dispatch_admitted']


@pytest.mark.parametrize('fault',['replacement_twice','old_reused','changed_condition','history_digest','new_unknown','upper_twice','lost_upper','DEC1_cap','site_cap'])
def test_cold_bad_case_rejected_by_intake_and_parent_proposal(cold,tmp_path,fault):
    f=cold;b=f['p']['budget'];c=b['continuation'];root=_root(f,tmp_path)
    if fault=='replacement_twice':c['engineering_replacements_requested']=2
    elif fault=='old_reused':b['request_id']=history.OLD_REQUEST
    elif fault=='changed_condition':c['original_condition_ids'][0]=history.OLD_CONDITION
    elif fault=='history_digest':c['history_sha256']='0'*64
    elif fault=='new_unknown':f['current']['unknown_effects']=['UNKNOWN_NEW_EFFECT']
    elif fault=='upper_twice':f['p']['parent']['physics_ticks']['closed_source_component_financial_upper']*=2;_remainder(f)
    elif fault=='lost_upper':del f['p']['parent']['physics_ticks']['closed_source_component_financial_upper'];_remainder(f)
    elif fault=='DEC1_cap':b['DEC1_reserved']['physics_ticks']=intake.CAPS['physics_ticks']
    elif fault=='site_cap':b['site_remaining']['physics_ticks']=b['requested']['physics_ticks']-1
    before=deepcopy((f['p']['parent'],f['current']))
    with pytest.raises(PermissionError):_validate(f)
    with pytest.raises(PermissionError):_propose(f,root)
    assert (f['p']['parent'],f['current'])==before


@pytest.mark.parametrize('fault',['row_history','missing_seal','bytes_changed','old_unknown','third_request','aggregate','native_cap'])
def test_runtime_independently_rejects_post_proposal_tamper(cold,tmp_path,fault):
    f=cold;root=_root(f,tmp_path);_validate(f);proposal=_propose(f,root)
    core=proposal['Core_proposal']['proposed_ledger'];row=core['source_cpu_reservations'][f['p']['budget']['request_id']]
    if fault=='row_history':row['continuation']['history_sha256']='0'*64
    elif fault=='missing_seal':f['m']['execution']['sealed_inputs'].pop(history.HISTORY_PLAN)
    elif fault=='bytes_changed':f['history_bytes']+=b' '
    elif fault=='old_unknown':core['source_cpu_reservations'][history.OLD_REQUEST]['unknown_effects']=['NEW_UNKNOWN_WRITE']
    elif fault=='third_request':
        third=deepcopy(row);third.update(request_id='illegal-third',child_run_uuid='third',child_root='/registered/third')
        core['source_cpu_reservations']['illegal-third']=third
    elif fault=='aggregate':core['source_cpu_reserved']['physics_ticks']-=1
    elif fault=='native_cap':core['physics_ticks']=intake.CAPS['physics_ticks']
    # Bind deliberately changed manifest bytes only; never repair a bad row/hash.
    if fault=='missing_seal':row['child_manifest_sha256']=sha(encode(f['m']))
    with pytest.raises(PermissionError):_runtime(f,root,proposal)


def test_expired_history_cannot_start_but_metadata_closeout_preserves_costs(cold,tmp_path):
    f=cold;root=_root(f,tmp_path);proposal=_propose(f,root)
    with pytest.raises(PermissionError):_runtime(f,root,proposal,now=intake.HARD_CLOSE+1)
    _runtime(f,root,proposal,closeout=True,now=intake.HARD_CLOSE+1)


def test_current_unknown_is_preserved_for_readonly_closeout_never_new_work(cold,tmp_path):
    f=cold;root=_root(f,tmp_path);proposal=_propose(f,root)
    core=proposal['Core_proposal']['proposed_ledger'];row=core['source_cpu_reservations'][f['p']['budget']['request_id']]
    core['unknown_effects']=['ACTUAL_EFFECT_UNRESOLVED']
    row.update(status='UNKNOWN',unknown_effects=['ACTUAL_EFFECT_UNRESOLVED'])
    before=deepcopy(core)
    with pytest.raises(PermissionError):_runtime(f,root,proposal)
    _runtime(f,root,proposal,closeout=True,now=intake.HARD_CLOSE+1)
    assert core==before


@pytest.mark.parametrize('fault',['missing_report','report_bytes','wrong_cold_repair'])
def test_financial_history_never_substitutes_for_exact_cold_repair(cold,tmp_path,fault):
    f=cold;root=_root(f,tmp_path)
    if fault=='report_bytes':f['engineering_bytes']+=b' '
    else:f['seal']['files']['scripts/pi0294/dec1/source_sync_stop.py']='0'*64
    if fault=='missing_report':
        # Leave the correct sealed report identity and remove its supplied bytes.
        args=_bind(f,root)
        args['engineering_bytes']=None
        with pytest.raises(PermissionError):reserve.propose_parent(parent_bytes=encode(f['p']['parent']),
            adoption_bytes=encode(f['p']['adoption']),**args)
    else:
        with pytest.raises(PermissionError):_propose(f,root)
