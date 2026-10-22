"""Pure exact source financial history/count helpers; no IO, caps or native GO.

Use optional history only for the one pinned financially SETTLED UNKNOWN. Its
engineering cause, cold repair/CI, actual closed effects and fresh resources
remain original independent owner gates. Counting never proves those gates.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math

HISTORY_PLAN='plans/SOURCE_CLOSED_FINANCIAL_HISTORY.private.json'
SCHEMA='DEC1_SOURCE_CPU_CLOSED_FINANCIAL_HISTORY_V1'
VALIDATED='DEC1_SOURCE_CPU_VALIDATED_FINANCIAL_HISTORY_V1'
OLD_REQUEST='23da1121-b1cd-4fef-83b4-a28e45e108b1'
OLD_CHILD='dec1-source-component-01'
OLD_CONDITION='DEC1-SOURCE-COMMISSION-F0-000-T1-I_TIME'
NOT_RUN=('DEC1-SOURCE-COMMISSION-F0-000-T2-I_PROGRESS',
         'DEC1-SOURCE-COMMISSION-F1-000-T1-I_PROGRESS',
         'DEC1-SOURCE-COMMISSION-F1-000-T2-I_TIME')
INVOICE_SHA='391dc276e498418a8b8abfb17342f527e3573eeb3edbe569081f5fa78b3fd809'
PINS=dict(dual_receipt='f27cb94173dbb751bb14f4c5172a1141e72c43f54aeaec64926abd9111014283',
 original_child_ledger='256d94a2347f6546c83aab5963cb2808730c9913907592acc742807a4288021d',
 closed_lease='20bf44b798a746910cf2b406a760d096d91322ec46a288abd2ca0de278f12777',
 original_audit='d12895ad02bafbeccbc48ae03064292fbca20245df6bdd6abf93424b87598803',
 closure_readback='46bfdc1ed854483b618904f5ce787178d0aabcb6b8d06df200c5b4d581edd214')
DIMS=('CoreLab_CPU_s','local_CPU_s','episode_starts','physics_ticks','constructors',
      'CoreLab_output_bytes','local_output_bytes','transfer_bytes')
VERDICT='ENGINEERING_INCOMPLETE_UNKNOWN'
ROLE='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_V1'
ROW_SCHEMA='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1'


def need(ok,reason):
    if not ok:raise PermissionError('SOURCE_HISTORY_'+reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encode(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()


def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def number(v,integer=False):
    need(type(v) in (int,float) and math.isfinite(v) and v>=0
         and (not integer or type(v) is int),'TYPED_FINITE_NONNEGATIVE')
    return v


def dims(v):
    need(type(v) is dict and set(v)==set(DIMS),'COMPLETE_EIGHT_DIMENSIONS')
    for k,n in v.items():number(n,k not in ('CoreLab_CPU_s','local_CPU_s'))
    return v


def obj(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    need(type(raw) is bytes,'IMMUTABLE_HISTORY_BYTES')
    v=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(PermissionError('SOURCE_HISTORY_NONFINITE_JSON')))
    need(type(v) is dict,'DOCUMENT_OBJECT');return v


def _wrapper(v):
    need(v.get('returncode')==0 and type(v.get('returncode')) is int
         and v.get('read_only') is True and type(v.get('stdout')) is str,'SUCCESSFUL_READ_ONLY_WRAPPER')
    return obj(v['stdout'].encode())


def _marker(history):
    need(type(history) is dict and history.get('schema')==VALIDATED
         and history.get('document_pins')==PINS and history.get('invoice_sha256')==INVOICE_SHA
         and type(history.get('history_sha256')) is str and len(history['history_sha256'])==64
         and all(c in '0123456789abcdef' for c in history['history_sha256']),
         'VALIDATED_HISTORY_REQUIRED')


def _old_row(ledger,invoice,original):
    need(type(ledger) is dict and ledger.get('unknown_effects')==[],'NO_GLOBAL_OR_WRITE_UNKNOWN')
    rows=ledger.get('source_cpu_reservations');need(type(rows) is dict and OLD_REQUEST in rows,'OLD_ROW_PRESENT')
    row=rows[OLD_REQUEST]
    need(row.get('schema')==ROW_SCHEMA and row.get('role')==ROLE and row.get('request_id')==OLD_REQUEST
         and row.get('status')=='SETTLED' and row.get('unknown_effects')==[]
         and row.get('remaining')==dict.fromkeys(DIMS,0)
         and same(row.get('requested'),original['requested'])
         and same(row,original)
         and row.get('financial_closeout_sha256')==INVOICE_SHA
         and row.get('scientific_verdict')==VERDICT and row.get('full_unresolved_live_bound_charged') is True
         and ledger.get('source_cpu_financial_closeouts',{}).get(OLD_REQUEST)==invoice,
         'EXACT_OLD_SETTLED_REQUESTED4_REMAINS_IMMUTABLE')
    need(len(rows)<=2,'NO_THIRD_SOURCE_REQUEST')
    total=dict.fromkeys(DIMS,0);children=set();roots=set()
    for key,r in rows.items():
        need(type(r) is dict and r.get('request_id')==key and r.get('unknown_effects')==[]
             and r.get('status') in ('RESERVED','RUNNING','SETTLED')
             and r.get('schema')==ROW_SCHEMA and r.get('role')==ROLE,'NO_OTHER_SOURCE_UNKNOWN')
        need(type(r.get('child_run_uuid')) is str and r['child_run_uuid']
             and type(r.get('child_root')) is str and r['child_root'].startswith('/')
             and r['child_run_uuid'] not in children and r['child_root'] not in roots,'UNIQUE_SOURCE_CHILD')
        children.add(r['child_run_uuid']);roots.add(r['child_root'])
        requested,left=dims(r['requested']),dims(r['remaining'])
        need(all(left[k]<=requested[k] for k in DIMS),'NO_EXPANDED_REMAINING')
        if r['status']=='SETTLED':need(not any(left.values()),'SETTLED_ZERO_REMAINING')
        for k in DIMS:total[k]+=left[k]
    need(same(dims(ledger['source_cpu_reserved']),total),'EXACT_CURRENT_SOURCE_AGGREGATE')
    return rows


def validate_history(history_bytes,parent,Core,*,now_epoch):
    """Return immutable-evidence classification; None preserves old call paths.

    Current parent/Core are caller-bound snapshots. Pinned readback is historical
    evidence only and must never substitute for current native admission.
    """
    if history_bytes is None:return None
    pack=obj(history_bytes);need(pack.get('schema')==SCHEMA and set(pack)=={'schema','documents'},'EXACT_HISTORY_SCHEMA')
    docs=pack['documents'];need(type(docs) is dict and set(docs)==set(PINS),'ALL_FIVE_PINNED_DOCUMENTS')
    values={}
    for key,pin in PINS.items():
        need(type(docs[key]) is dict and set(docs[key])=={'base64'},'EXACT_DOCUMENT_WRAPPER')
        raw=base64.b64decode(docs[key]['base64'],validate=True)
        need(sha(raw)==pin,'EXACT_DOCUMENT_BYTES:'+key);values[key]=obj(raw)
    close=values['dual_receipt'];invoice=close['invoice']
    need(sha(encode(invoice))==INVOICE_SHA==close['Core']['invoice_sha256']
         and invoice.get('request_id')==OLD_REQUEST and invoice.get('child_run_uuid')==OLD_CHILD
         and invoice.get('scientific_verdict')==VERDICT and invoice.get('actual_returned_STOP_ACK') is False
         and invoice.get('source_or_target_qualified') is False and invoice.get('native_dispatch_admitted') is False,
         'FINANCIAL_CLOSURE_NEVER_SCIENCE_OR_ENGINEERING_PROOF')
    now=number(now_epoch)
    need(number(invoice['settled_epoch'])<=now<number(invoice['hard_close_epoch']),'ORIGINAL_CLOCK_NOT_LATE')
    original_unknown=invoice['original_live_reservation_and_unknowns'];child=values['original_child_ledger'];lease=values['closed_lease']
    need(child['run_uuid']==OLD_CHILD and child['unknown_effects']==original_unknown['unknown_effects']
         and child['reservations']==original_unknown['reservations'] and bool(child['unknown_effects'])
         and child['reservations'][OLD_CONDITION]['maximum']==invoice['live_financial_upper']==16000,
         'OLD_CHILD_UNKNOWN_AND_LIVE_UPPER_PRESERVED')
    need(lease['status']=='CLOSED' and lease['run_uuid']==OLD_CHILD and lease['actual_STOP_ACK'] is False
         and lease['scientific_verdict']==VERDICT and lease['original_child_ledger_unknowns_preserved'] is True
         and lease['financial_closeout_sha256']==close['Core']['proof_sha256'],'EXACT_CLOSED_LEASE_STILL_UNKNOWN')
    audit_outer=_wrapper(values['original_audit']);audit=audit_outer['result'];rr=audit['conditions']
    need(audit.get('schema')=='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1'
         and len(rr)==4 and {r['condition_id'] for r in rr}=={OLD_CONDITION,*NOT_RUN}
         and all(r['status']==(VERDICT if r['condition_id']==OLD_CONDITION else 'NOT_RUN') for r in rr),
         'ONE_UNKNOWN_THREE_NOT_RUN_NOT_VALID_POLICY_FAILURE')
    readback=_wrapper(values['closure_readback'])
    need(readback.get('status')=='INDEPENDENT_DUAL_FINANCIAL_CLOSURE_RETAINED_UNKNOWN_AND_FILES_CONFIRMED'
         and readback.get('invoice_sha256')==INVOICE_SHA and readback.get('active_workers')==0
         and type(readback.get('active_workers')) is int and readback.get('source_lease_status')=='CLOSED'
         and readback.get('original_child_ledger_unchanged') is True
         and readback.get('unknown_and_original_live_reservation_preserved') is True
         and readback.get('original_failure_files_readback')==13,'EXACT_INDEPENDENT_ACTUAL_CLOSURE_READBACK')
    for field in ('CPU_s','episode_starts','physics_ticks','constructors'):
        need(number(Core[field],field!='CPU_s')>=number(readback['Core_ledger'][field],field!='CPU_s'),
             'PRESERVED_CUMULATIVE_CORE_COST:'+field)
    original=readback['Core_ledger']['source_cpu_reservations'][OLD_REQUEST]
    need(dims(original['requested'])['episode_starts']==4,'ORIGINAL_REQUESTED4_NOT_REWRITTEN')
    history=dict(schema=VALIDATED,history_sha256=sha(history_bytes),document_pins=deepcopy(PINS),
        invoice_sha256=INVOICE_SHA,invoice=deepcopy(invoice),original_row=deepcopy(original),
        commissioning_count=1,engineering_count=0,engineering_permission_verified=False)
    for ledger in (parent,Core):
        rows=_old_row(ledger,invoice,original)
        for key,row in rows.items():
            if key!=OLD_REQUEST:row_slots(row,ledger,history)
    need(parent.get('Hopper_unknown_effects')==[]
         and parent['T0_epoch']==Core['original_T0_epoch']==invoice['original_T0_epoch']
         and parent['hard_closeout_epoch']==Core['hard_close_epoch']==invoice['hard_close_epoch'],
         'ORIGINAL_GLOBAL_CLOCK_AND_KNOWN_WRITES')
    # Validate every invoice covering the special financial bucket, never guess0.
    physics_total(parent,history)
    return history


def validate_continuation(budget,history=None):
    """Classify requested starts only, never certify engineering permission."""
    if history is None:
        need('continuation' not in budget,'CONTINUATION_REQUIRES_EXACT_HISTORY')
        return dict(commissioning_count=number(budget['requested']['episode_starts'],True),engineering_count=0)
    _marker(history);binding=budget.get('continuation')
    need(type(binding) is dict and binding.get('history_sha256')==history['history_sha256']
         and type(binding.get('original_commissioning_requested')) is int
         and binding['original_commissioning_requested']==3
         and type(binding.get('engineering_replacements_requested')) is int
         and binding['engineering_replacements_requested']==1
         and binding.get('replacement_condition_id')==OLD_CONDITION
         and binding.get('replacement_of_request_id')==OLD_REQUEST
         and type(binding.get('replacement_index')) is int and binding['replacement_index']==1
         and binding.get('original_condition_ids')==list(NOT_RUN)
         and type(budget['requested']['episode_starts']) is int
         and budget['requested']['episode_starts']==4,'EXACT_ORIGINAL3_ENGINEERING1_UNIQUE_CONDITIONS')
    return dict(commissioning_count=3,engineering_count=1)


def row_slots(row,ledger,history=None):
    """Old behavior without history; exact unique continuation when supplied."""
    if history is None:return validate_continuation(row,None)
    _marker(history);_old_row(ledger,history['invoice'],history['original_row'])
    key=row.get('request_id');need(type(key) is str and key,'REQUEST_ID')
    if key==OLD_REQUEST:
        need(same(row,ledger['source_cpu_reservations'][OLD_REQUEST]),'EXACT_OLD_ROW_NOT_CANDIDATE_REUSE')
        return dict(commissioning_count=1,engineering_count=0)
    others={k for k in ledger['source_cpu_reservations'] if k!=OLD_REQUEST}
    need(not others or others=={key},'NO_THIRD_OR_DUPLICATE_REPLACEMENT_REQUEST')
    need(row.get('schema')==ROW_SCHEMA and row.get('role')==ROLE and row.get('unknown_effects')==[]
         and type(row.get('child_run_uuid')) is str and bool(row['child_run_uuid'])
         and row['child_run_uuid']!=OLD_CHILD,'NO_REUSE_OLD_UNKNOWN_CHILD_OR_OTHER_UNKNOWN')
    return validate_continuation(row,history)


def physics_total(parent,history=None):
    """Sum original physics plus exactly invoice-covered financial upper once."""
    physics=parent['physics_ticks'];base={'live','shadow','reconstruction'}
    need(type(physics) is dict,'PHYSICS_BUCKET_OBJECT')
    if history is None:
        need(set(physics)==base,'LEGACY_THREE_PHYSICS_BUCKETS')
    else:
        _marker(history);invoices=parent.get('source_cpu_financial_closeouts')
        need(type(invoices) is dict and set(invoices)=={OLD_REQUEST}
             and same(invoices[OLD_REQUEST],history['invoice']),'EACH_FINANCIAL_BUCKET_INVOICE_PINNED')
        total=sum(number(i['financial_charged']['physics_ticks'],True) for i in invoices.values())
        need(set(physics)==base|{'closed_source_component_financial_upper'}
             and number(physics['closed_source_component_financial_upper'],True)==total,
             'FINANCIAL_UPPER_EXACT_INVOICE_SUM_NOT_NATIVE_LIVE')
    return sum(number(v,True) for v in physics.values())


REPAIR_PLAN='plans/SOURCE_ENGINEERING_FIX_EVIDENCE.json'
REPAIR_SHA='28221c1935f960b51945516924088b76d414cbaa26eafb9c2e208fe3f766d8ba'
REPAIR_MEMBERS={
 'scripts/pi0294/dec1/source_sync_stop.py':'952d6a349066bb8c4feb2b14ce0f3f0a1447ffa5cfecc280b5cecb91c603f8f4',
 'scripts/pi0294/dec1/test_source_sync_stop.py':'01b296617470231a67ece2f71ccce78ecc35b597d758270f3967f14851f76b6e'}


def validate_repair(manifest,budget,history,engineering_bytes,*,source_members=None):
    """Authenticate published cause/cold repair facts, never grant dispatch.

    Historical report pending/no-replay statements remain historical evidence.
    Actual replacement permission still requires original authority, closed
    effects and fresh standard/native admission; no boolean substitutes for it.
    """
    if history is None:return None
    _marker(history);validate_continuation(budget,history)
    need(type(engineering_bytes) is bytes and sha(engineering_bytes)==REPAIR_SHA,'EXACT_PUBLISHED_REPAIR_BYTES')
    report=obj(engineering_bytes);first=report['first_component'];repair=report['repair']
    need(report.get('schema')=='WDR1_SOURCE_COMPONENT_NATIVE_FIRST_ATTEMPT_AND_STOP_READBACK_CPU_ACTUAL_V1'
         and report.get('executed_source')=='55e73c7b6e2c6584f90b14d64ef9f5b842fd2e2c'
         and report['actual_registration']['request_id']==OLD_REQUEST
         and first['condition_id']==OLD_CONDITION and first['actual_returned_STOP_ack'] is False
         and first['error']=='PrefixFailure: SOURCE_SHADOW_CHANGED_LIVE_OR_RNG'
         and first['original_native_safety_rejection_on_third_shadow'] is True
         and first['original_source_unknowns_and_16000_live_reservation_retained'] is True
         and repair['files_sha256']==REPAIR_MEMBERS
         and type(repair['tests_passed']) is int and repair['tests_passed']==107
         and type(repair['regression_cases_added']) is int and repair['regression_cases_added']==3
         and repair['numerical_engine_or_original_S1_changed'] is False
         and repair['native_deployment_or_reexecution'] is False
         and repair['original_failed_capsule_and_receipts_immutable'] is True,
         'EXACT_OLD_CAUSE_S1_REJECTION_AND_COLD_CPU_REPAIR_FACTS')
    need(manifest['execution']['sealed_inputs'].get(REPAIR_PLAN)==REPAIR_SHA
         and budget['continuation'].get('engineering_fix_evidence_sha256')==REPAIR_SHA,
         'SEALED_MANIFEST_AND_CONTINUATION_REPAIR_BINDING')
    if source_members is not None:
        need(type(source_members) is dict and all(source_members.get(k)==v for k,v in REPAIR_MEMBERS.items()),
             'ACTUAL_COLD_SOURCE_MEMBERS_EXACT_REPAIR_HASHES')
    return dict(engineering_evidence_sha256=REPAIR_SHA,repair_member_hashes=deepcopy(REPAIR_MEMBERS),
                old_scientific_verdict=VERDICT,actual_native_reexecution_proven=False)
