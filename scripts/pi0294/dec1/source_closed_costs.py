"""Pure future B cold-boundary source-cost projection; no IO or native GO.

LOCAL already charged global costs. Only missing cumulative DEC1 source costs
are proposed. Native GPU/CPU/updates/checkpoints and old markers are preserved.
Caller owns actual B2048 audit/once settlement/dual-close/inactive CAS gates.
"""
from copy import deepcopy
import hashlib
import json
import math

OLD='23da1121-b1cd-4fef-83b4-a28e45e108b1'
NEW='081c5299-5594-4ff6-8676-d48b9652cf52'
INVOICE='391dc276e498418a8b8abfb17342f527e3573eeb3edbe569081f5fa78b3fd809'
OLD_DUAL='f27cb94173dbb751bb14f4c5172a1141e72c43f54aeaec64926abd9111014283'
OLD_LEASE='20bf44b798a746910cf2b406a760d096d91322ec46a288abd2ca0de278f12777'
OLD_MARKER='closed_Core_source_financial_projections'
NEW_MARKER='closed_Core_source_known_terminal_projections'
KNOWN_INDEX='source_cpu_known_terminal_settlements'
DIMS=('CoreLab_CPU_s','local_CPU_s','episode_starts','physics_ticks','constructors',
      'CoreLab_output_bytes','local_output_bytes','transfer_bytes')
FIELDS={'episode_starts':'episode_starts','physics_ticks':'physics_ticks',
        'constructors':'constructors','transfer_bytes':'network_transfer_accounted_bytes'}
KNOWN={'COMPONENT_COMPLETE','KNOWN_REFERENCE_OR_S1_REJECT','NATIVE_SAFETY_FAIL'}
AUTHORITY='a7a7ecd9a6be62a7221af25a5e32a0009b076a84'


def need(ok,why):
    if not ok:raise PermissionError('B_SOURCE_COSTS_'+why)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encode(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()


def obj(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    need(type(raw) is bytes,'IMMUTABLE_BYTES')
    v=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(PermissionError('B_SOURCE_COSTS_NONFINITE_JSON')))
    need(type(v) is dict,'OBJECT_REQUIRED');return v


def num(v,integer=False):
    need(type(v) in (int,float) and math.isfinite(v) and v>=0
         and (not integer or type(v) is int),'FINITE_NONNEGATIVE_TYPED_COST')
    return v


def dims(v):
    need(type(v) is dict and set(v)==set(DIMS),'COMPLETE_EIGHT_DIMENSIONS')
    for k,n in v.items():num(n,k not in ('CoreLab_CPU_s','local_CPU_s'))
    return v


def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def _io(receipt,sid):
    need(receipt.get('schema')=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
         and receipt.get('request_id')==sid
         and receipt.get('status')=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
         and receipt.get('native_dispatch_admitted') is False,'EXACT_ONCE_METADATA_RECEIPT')


def _closed(rid,row,p,c,evidence):
    dual_raw=evidence['dual_receipt_bytes'];lease_raw=evidence['closed_lease_bytes']
    dual=obj(dual_raw);lease=obj(lease_raw)
    need(row['status']=='SETTLED' and row['remaining']==dict.fromkeys(DIMS,0)
         and lease['run_uuid']==row['child_run_uuid'] and lease['status']=='CLOSED',
         'SETTLED_ZERO_AND_EXACT_CLOSED_CHILD_LEASE')
    if rid==OLD:
        record=p['source_cpu_financial_closeouts'][rid]
        need(same(record,c['source_cpu_financial_closeouts'][rid])
             and sha(encode(record))==INVOICE==row['financial_closeout_sha256']
             and sha(dual_raw)==OLD_DUAL and sha(lease_raw)==OLD_LEASE
             and dual['invoice']==record and dual['Core']['invoice_sha256']==INVOICE
             and dual['status']=='SOURCE_DUAL_FINANCIAL_TERMINAL_BOUND_SETTLED_CORE_LEASE_CLOSED_COMPONENT_UNKNOWN'
             and record['scientific_verdict']=='ENGINEERING_INCOMPLETE_UNKNOWN'
             and record['actual_returned_STOP_ACK'] is False and record['live_financial_upper']==16000
             and row['scientific_verdict']=='ENGINEERING_INCOMPLETE_UNKNOWN'
             and row['full_unresolved_live_bound_charged'] is True
             and lease['actual_STOP_ACK'] is False and lease['original_child_ledger_unknowns_preserved'] is True
             and lease['financial_closeout_sha256']==dual['Core']['proof_sha256'],
             'EXACT_OLD_FINANCIAL_UNKNOWN_NEVER_SCIENCE_PASS')
        rh=INVOICE;sid=record['settlement_id'];kind='FINANCIAL_UNKNOWN'
    else:
        record=p[KNOWN_INDEX][rid];rh=sha(encode(record));sid=record['settlement_id'];kind='KNOWN_TERMINAL'
        need(rid==NEW and same(record,c[KNOWN_INDEX][rid])
             and record['schema']=='DEC1_SOURCE_CPU_KNOWN_TERMINAL_SETTLEMENT_V1'
             and record['request_id']==NEW
             and record['manifest_sha256']=='14ce2f841c8db3de7e2eb9c191e95fc9b26720f628493774e4fd954012556b0f'
             and record['source_sha']=='47dc6001227572f031ebc3cc33d56a487027d8cb'
             and row['known_terminal_settlement_sha256']==rh
             and row['scientific_verdict']=='SOURCE_COMPONENT_RESULTS_KNOWN_NOT_STUDENT_QUALIFICATION'
             and dual['status']=='SOURCE_DUAL_KNOWN_TERMINAL_SETTLED_LEASE_CLOSED_NOT_STUDENT_QUALIFICATION'
             and dual['settlement']==record and dual['Core']['settlement_sha256']==rh
             and lease['all_registered_workers_terminal'] is True
             and lease['known_terminal_settlement_sha256']==rh
             and lease['known_terminal_proof_sha256']==dual['Core']['proof_sha256'],
             'EXACT_NEW_KNOWN_TERMINAL_INDEX_DUAL_AND_LEASE')
        proof_raw=evidence['Core_proof_bytes'];proof=obj(proof_raw)
        need(sha(proof_raw)==dual['Core']['proof_sha256']
             and proof['schema']=='DEC1_SOURCE_CORE_KNOWN_TERMINAL_CLOSURE_V1'
             and proof['settlement']==record and proof['settlement_sha256']==rh
             and proof['Core_receipt']==dual['Core']['Core_receipt']
             and proof['original_child_ledger_unchanged_sha256']==record['child_ledger_sha256'],
             'EXACT_NATIVE_CLOSURE_PROOF_BYTES')
        statuses=[r['status'] for r in record['conditions']]
        need(len(statuses)==4 and all(s in KNOWN|{'NOT_RUN'} for s in statuses)
             and record['financial_charged']['episode_starts']==sum(s!='NOT_RUN' for s in statuses)
             and record['financial_charged']['physics_ticks']==num(record['actual_live_ticks'],True)+num(record['actual_shadow_ticks'],True)
             and record['student_neural_ACK_qualification'] is False
             and record['target_qualification'] is False and record['core64_cells']==0
             and record['native_dispatch_admitted'] is False,'KNOWN_FAILURE_ALLOWED_UNKNOWN_NOT_ALLOWED')
    need(sha(lease_raw)==dual['Core']['lease_sha256']
         and dual['LOCAL_readback']['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
         and dual['LOCAL_readback']['actual_sha256']==dual['LOCAL']['after_sha256']
         and dual['LOCAL_readback']['receipt_binding_valid'] is True,
         'ACTUAL_DUAL_ONCE_WRITE_AND_LEASE_BYTE_BINDING')
    for receipt in (dual['Core']['Core_receipt'],dual['Core']['lease_receipt'],dual['LOCAL']):_io(receipt,sid)
    amount=deepcopy(dims(record['financial_charged']));requested=dims(row['requested'])
    need(all(amount[k]<=requested[k] for k in DIMS),'CHARGED_WITHIN_ORIGINAL_FULL_UPPER')
    return dict(kind=kind,receipt_sha256=rh,dual_sha256=sha(dual_raw),amount=amount,record=record)


def propose(parent_bytes,Core_bytes,Hopper_bytes,*,closures):
    """Return per-request delta/proposed copy; owner must still enforce B cold IO.

    closures[rid] supplies actual dual_receipt_bytes and closed_lease_bytes;
    Source02 additionally needs exact Core_proof_bytes. All readback/actor truth
    is independently authenticated by the owner, never inferred by this code.
    Active Source02 remains pending and must use the separate pending adapter.
    """
    p,c,d=map(obj,(parent_bytes,Core_bytes,Hopper_bytes))
    need(c.get('authority_release')==d.get('authority_release')==AUTHORITY
         and c['campaign_run_uuid']==d['campaign_run_uuid']==p['run_uuid']
         and c['original_T0_epoch']==d['original_T0_epoch']==p['T0_epoch']
         and c['hard_close_epoch']==d['hard_close_epoch']==p['hard_closeout_epoch']==1792952996,
         'ORIGINAL_SCOPE_AND_CLOCK')
    need(p.get('unknown_effects')==p.get('Hopper_unknown_effects')==c.get('unknown_effects')==d.get('unknown_effects')==[],
         'NO_UNRECONCILED_PARENT_CORE_OR_HOPPER_WRITE')
    rows=p['source_cpu_reservations'];need(same(rows,c['source_cpu_reservations'])
         and set(rows)<={OLD,NEW} and OLD in rows,'EXACT_DUAL_ALLOWED_SOURCE_ROWS')
    need(type(closures) is dict and set(closures)<=set(rows),'NO_UNREGISTERED_CLOSURE')
    pending=dict.fromkeys(DIMS,0);records={};all_charged=dict.fromkeys(DIMS,0)
    for rid,row in rows.items():
        need(row.get('schema')=='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1'
             and row.get('role')=='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_V1'
             and row.get('authority_release')==AUTHORITY
             and row.get('request_id')==rid and row.get('unknown_effects')==[]
             and row.get('status') in ('RESERVED','RUNNING','SETTLED'),'NO_UNKNOWN_SOURCE_ROW')
        requested,left=dims(row['requested']),dims(row['remaining'])
        need(all(left[k]<=requested[k] for k in DIMS),'NO_EXPANDED_PENDING')
        if row['status']=='SETTLED':
            need(rid in closures,'SETTLED_REQUIRES_ACTUAL_CLOSURE_BYTES')
            records[rid]=_closed(rid,row,p,c,closures[rid])
            for k,v in records[rid]['amount'].items():all_charged[k]+=v
        else:
            need(rid==NEW and rid not in closures and same(left,requested),
                 'ACTIVE_NEW_SOURCE_FULL_BOUND_NOT_A_CLOSED_RECEIPT')
            need(rid not in p.get(KNOWN_INDEX,{}) and rid not in c.get(KNOWN_INDEX,{}),
                 'ACTIVE_SOURCE_CANNOT_ALREADY_HAVE_KNOWN_TERMINAL_INDEX')
            for k,v in left.items():pending[k]+=v
    need(same(dims(p['source_cpu_reserved']),pending) and same(dims(c['source_cpu_reserved']),pending),
         'EXACT_FULL_EIGHT_PENDING_AGGREGATE')
    if 'source_cpu_current_classification' in c:
        need(all(dims(c['source_cpu_current_classification'])[k]>=all_charged[k] for k in DIMS),
             'COMPLETE_CORE_IO_CPU_CLASSIFICATION_NOT_RESET')
    for k,field in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),('physics_ticks','physics_ticks'),('constructors','constructors')):
        need(num(c[field],k!='CoreLab_CPU_s')>=all_charged[k],'CUMULATIVE_CORE_COST_NOT_RESET:'+field)
    after=deepcopy(d);delta=dict.fromkeys(FIELDS.values(),0);by_request={};new_markers={}
    for name,allowed in ((OLD_MARKER,{OLD}),(NEW_MARKER,{NEW})):
        need(type(d.get(name,{})) is dict and set(d.get(name,{}))<=allowed,'NO_UNRECOGNIZED_EXISTING_MARKER')
    need(set(d.get(OLD_MARKER,{}))<=set(records) and set(d.get(NEW_MARKER,{}))<=set(records),
         'NO_CLOSED_HOPPER_MARKER_FOR_ACTIVE_OR_MISSING_SOURCE')
    for rid,r in records.items():
        name=OLD_MARKER if rid==OLD else NEW_MARKER;existing=d.get(name,{}).get(rid)
        amount=r['amount'];addition={field:amount[k] for k,field in FIELDS.items()}
        if existing is not None:
            expected_schema='DEC1_CLOSED_CORE_SOURCE_NONADDITIVE_COST_PROJECTION_V1' if rid==OLD else 'DEC1_CLOSED_CORE_SOURCE_KNOWN_NONADDITIVE_COST_PROJECTION_V1'
            hashkey='source_invoice_sha256' if rid==OLD else 'source_settlement_sha256'
            need(existing.get('schema')==expected_schema and existing.get('request_id')==rid
                 and existing.get(hashkey)==r['receipt_sha256']
                 and existing.get('dual_financial_receipt_sha256' if rid==OLD else 'dual_known_receipt_sha256')==r['dual_sha256']
                 and same(dims(existing['financial_charged']),amount)
                 and existing.get('already_inside_global_parent_spend') is True
                 and existing.get('adds_no_global_charge') is True,'EXACT_COMPLETE_EXISTING_MARKER_NO_GUESSED_PARTIAL')
            addition=dict.fromkeys(FIELDS.values(),0)
        else:
            marker=dict(schema='DEC1_CLOSED_CORE_SOURCE_NONADDITIVE_COST_PROJECTION_V1' if rid==OLD else 'DEC1_CLOSED_CORE_SOURCE_KNOWN_NONADDITIVE_COST_PROJECTION_V1',
                request_id=rid,financial_charged=deepcopy(amount),already_inside_global_parent_spend=True,adds_no_global_charge=True,
                actual_parent_ledger_sha256=sha(parent_bytes),actual_Core_ledger_sha256=sha(Core_bytes),source_or_target_qualification=False)
            marker['scientific_verdict']=r['record'].get('scientific_verdict','SOURCE_COMPONENT_RESULTS_KNOWN_NOT_STUDENT_QUALIFICATION')
            marker['source_invoice_sha256' if rid==OLD else 'source_settlement_sha256']=r['receipt_sha256']
            marker['dual_financial_receipt_sha256' if rid==OLD else 'dual_known_receipt_sha256']=r['dual_sha256']
            after.setdefault(name,{})[rid]=marker;new_markers[rid]=deepcopy(marker)
        for field,n in addition.items():delta[field]+=n
        by_request[rid]=dict(kind=r['kind'],delta_fields=addition,already_projected=existing is not None)
    for field,n in delta.items():after[field]=num(d.get(field,0),True)+n
    # Do not hard-bind A5891/B0 or any training step; every unrelated byte-level
    # field in the parsed object stays the same. Native application is external.
    allowed={OLD_MARKER,NEW_MARKER,*FIELDS.values()}
    need(all(same(after[k],v) for k,v in d.items() if k not in allowed),'PRESERVE_ALL_GPU_UPDATE_CPU_AND_TRAINING_HISTORY')
    return dict(status='PURE_B_SOURCE_COST_DELTA_NOT_NATIVE_ADMISSION',before_Hopper_DEC1_sha256=sha(Hopper_bytes),
        proposed_Hopper_DEC1=after,delta_fields=delta,per_request=by_request,new_markers=new_markers,
        total_closed_Core_output_upper_bytes=all_charged['CoreLab_output_bytes'],
        unprojected_Core_output_upper_bytes=sum(records[k]['amount']['CoreLab_output_bytes'] for k in new_markers),
        active_source_remaining=deepcopy(pending),pending_adapter_required=any(pending.values()),
        no_new_global_charge=True,native_dispatch_admitted=False,
        actual_B2048_audit_once_settlement_dual_closure_and_inactive_CAS_still_required=True)
