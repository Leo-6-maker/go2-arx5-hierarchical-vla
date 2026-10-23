"""Cold B ledger/storage projection of already closed target-counter costs.

LOCAL global charges remain unchanged. Only missing DEC1 constructor/transfer
classification is proposed; actual B2048 closure/inactive native CAS remains
with the original owner. A counter PASS never becomes task qualification.
"""
from copy import deepcopy
from . import target_zero_counter_contract as c

MARKER='closed_Core_target_zero_counter_projections'


def propose(parent_bytes,Core_bytes,Hopper_bytes,*,closures):
    p,k,h=map(c.obj,(parent_bytes,Core_bytes,Hopper_bytes));need=c.need
    need(p['run_uuid']==k['campaign_run_uuid']==h['campaign_run_uuid']=='20261018T182956Z-localpc-pv1'
         and p['T0_epoch']==k['original_T0_epoch']==h['original_T0_epoch']==1792348196
         and p['hard_closeout_epoch']==k['hard_close_epoch']==h['hard_close_epoch']==c.HARD_CLOSE
         and k['authority_release']==h['authority_release']==c.AUTHORITY,'PROJECTION_ORIGINAL_SCOPE_CLOCK')
    need(p['unknown_effects']==p.get('Hopper_unknown_effects',[])==k['unknown_effects']==h['unknown_effects']==[],
         'NO_UNRECONCILED_WRITE')
    rows=p.get('target_zero_counter_reservations',{})
    need(c.same(rows,k.get('target_zero_counter_reservations',{})) and set(closures)==set(rows),
         'ALL_TARGET_COUNTER_ROWS_AND_ACTUAL_CLOSURES')
    need(not any(c.totals(p).values()) and not any(c.totals(k).values()),'ALL_TARGET_COUNTER_HOLDS_REALLY_CLOSED')
    after=deepcopy(h);delta=dict(constructors=0,network_transfer_accounted_bytes=0);output=0;markers={}
    need(set(h.get(MARKER,{}))<=set(rows),'NO_UNREGISTERED_TARGET_COUNTER_MARKER')
    for rid,row in rows.items():
        record=p['target_zero_counter_settlements'][rid]
        need(c.same(record,k['target_zero_counter_settlements'][rid])
             and record['schema']=='DEC1_TARGET_ZERO_COUNTER_TERMINAL_SETTLEMENT_V1'
             and record['scope']==c.ROLE and record['request_id']==rid
             and record['manifest_sha256']==row['manifest_sha256']
             and row['settlement_sha256']==c.sha(c.encode(record))
             and record['target_task_native_qualified'] is False and record['core64_cells']==0,
             'EXACT_KNOWN_OR_FINANCIALLY_CLOSED_COUNTER_RECORD_NOT_TASK_PASS')
        b=closures[rid];dual=c.obj(b['dual_bytes']);lease=c.obj(b['lease_bytes']);proof=c.obj(b['Core_proof_bytes'])
        need(dual['schema']=='DEC1_TARGET_ZERO_COUNTER_DUAL_SETTLEMENT_AND_CLOSURE_V1'
             and dual['settlement']==record and dual['all_registered_actors_terminal'] is True
             and dual['Core']['settlement_sha256']==row['settlement_sha256']
             and dual['Core']['closed_lease_sha256']==c.sha(b['lease_bytes'])
             and dual['Core']['proof_sha256']==c.sha(b['Core_proof_bytes'])
             and lease['status']=='CLOSED' and lease['run_uuid']==row['child_run_uuid']
             and lease['scope']==c.ROLE and lease['settlement_sha256']==row['settlement_sha256']
             and proof['schema']=='DEC1_TARGET_ZERO_COUNTER_ACTUAL_CORE_CLOSURE_V1'
             and proof['settlement']==record and proof['actor_proof_sha256']==record['actor_proof_sha256'],
             'ACTUAL_DUAL_RECEIPT_LEASE_AND_CLOSURE_PROOF_BYTES')
        for io in (dual['Core']['receipt'],dual['Core']['lease_receipt'],dual['LOCAL']['receipt']):
            need(io['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
                 and io['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
                 and io['native_dispatch_admitted'] is False
                 and io['request_id']==record['settlement_id'],'EXACT_ONCE_METADATA_RECEIPTS')
        need(dual['LOCAL']['readback']['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
             and dual['LOCAL']['readback']['receipt_binding_valid'] is True
             and dual['LOCAL']['readback']['actual_sha256']==dual['LOCAL']['receipt']['after_sha256'],
             'ACTUAL_LOCAL_AFTER_BYTES_CONFIRMED')
        amount=c.vector(record['financial_charged']);need(amount['episode_starts']==amount['physics_ticks']==0
             and all(amount[d]<=row['requested'][d] for d in c.DIMS),'NO_HIDDEN_TARGET_POLICY_OR_PHYSICS_COST')
        output+=amount['CoreLab_output_bytes']
        marker=dict(schema='DEC1_CLOSED_TARGET_ZERO_COUNTER_NONADDITIVE_PROJECTION_V1',
            request_id=rid,settlement_sha256=row['settlement_sha256'],dual_sha256=c.sha(b['dual_bytes']),
            financial_charged=deepcopy(amount),already_charged_globally=True,new_global_charge=False,
            original_scientific_verdict=record['scientific_verdict'],target_task_native_qualified=False)
        old=h.get(MARKER,{}).get(rid)
        if old is not None:need(c.same(old,marker),'NO_PARTIAL_OR_CHANGED_CLOSED_COUNTER_PROJECTION')
        else:
            after.setdefault(MARKER,{})[rid]=marker;markers[rid]=marker
            delta['constructors']+=amount['constructors'];delta['network_transfer_accounted_bytes']+=amount['transfer_bytes']
    for key,x in delta.items():after[key]=h.get(key,0)+x
    return dict(status='PURE_CLOSED_TARGET_COUNTER_PROJECTION_NOT_NATIVE_ADMISSION',
        proposed_Hopper_DEC1=after,delta_fields=delta,new_markers=markers,
        total_closed_Core_output_upper_bytes=output,new_global_charge=False,native_dispatch_admitted=False)


def augment(snapshot,parent_bytes,Core_bytes,Hopper_bytes,*,closures):
    """After original source storage callback, add closed counter output ONCE."""
    p=propose(parent_bytes,Core_bytes,Hopper_bytes,closures=closures);need=c.need
    need(not any(p['delta_fields'].values()),'ACTUAL_INACTIVE_COUNTER_COST_MIRROR_REQUIRED')
    need('closed_source_storage_projection' in snapshot and 'closed_target_zero_counter_storage_projection' not in snapshot,
         'AFTER_ORIGINAL_SOURCE_CALLBACK_ONE_APPLICATION')
    s=deepcopy(snapshot);total=p['total_closed_Core_output_upper_bytes']
    for used,peaks in (('used','storage_peaks'),('dec1_used','dec1_storage_peaks')):
        s[used]['Core_storage_bytes']+=total
        s[peaks]['Core_storage_bytes']=max(s[peaks]['Core_storage_bytes'],s[used]['Core_storage_bytes'])
    s['closed_target_zero_counter_storage_projection']=dict(Core_output_bound_added_once=total,
        actual_Hopper_DEC1_sha256=c.sha(Hopper_bytes),current_parent_sha256=c.sha(parent_bytes),
        exact_closed_Core_sha256=c.sha(Core_bytes),no_second_global_charge=True,
        fresh_inclusive_LOCAL_namespace_must_also_include_counter=True,native_dispatch_admitted=False)
    return s
