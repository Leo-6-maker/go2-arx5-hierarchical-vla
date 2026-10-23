"""Pure closed SDK cost projection into DEC1; no writer or native admission.

Global costs are already charged in the original LOCAL/Core ledgers. Preserve
all GPU/global bytes and historical source/counter markers. Financial SDK
constructor uppers are never reported as actual MuJoCo models.
"""
from copy import deepcopy
from . import target_sdk_cost_rows as budget, target_zero_counter_contract as common

MARKER='closed_Core_target_SDK_projections'
SCHEMA='DEC1_CLOSED_TARGET_SDK_NONADDITIVE_COST_PROJECTION_V1'
need=budget.need

def _receipt(value,rid):
    need(value['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
         and value['request_id']==rid
         and value['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
         and value['native_dispatch_admitted'] is False,'SDK_CLOSED_EXACT_ONCE_RECEIPT')


def propose(parent_bytes,Core_bytes,DEC1_bytes,*,closures):
    p,c,d=map(common.obj,(parent_bytes,Core_bytes,DEC1_bytes));rows,spent,held=budget.rows(p,c)
    need(rows and not any(held.values()) and set(closures)==set(rows),'SDK_ALL_COMPLETE_CLOSED_ROWS_NO_PENDING')
    need(p['run_uuid']==c['campaign_run_uuid']==d['campaign_run_uuid']==budget.CAMPAIGN
         and p['T0_epoch']==c['original_T0_epoch']==d['original_T0_epoch']==budget.T0
         and p['hard_closeout_epoch']==c['hard_close_epoch']==d['hard_close_epoch']==budget.HARD_CLOSE
         and c['authority_release']==d['authority_release']==common.AUTHORITY
         and p['unknown_effects']==p.get('Hopper_unknown_effects',[])==p['pending_updates']==c['unknown_effects']==d['unknown_effects']==[],
         'SDK_CLOSED_ORIGINAL_CLOCK_AUTHORITY_AND_NO_UNKNOWN_WRITE')
    for ledger in (p,c):
        for field in ('source_cpu_reserved','target_zero_counter_reserved','target_sdk_client_reserved'):
            need(not any(common.vector(ledger.get(field,dict.fromkeys(common.DIMS,0))).values()),'SDK_CLOSED_ALL_CORE_COMPONENT_HOLDS_ZERO')
    for field in ('GPU_reserved_s','CPU_reserved_s','optimizer_updates_reserved','neural_queries_reserved','network_transfer_reserved_bytes','GPU_site_output_reserved_bytes'):
        need(d[field]==0,'SDK_CLOSED_NO_CURRENT_NEURAL_OR_CONTROL_RESERVATION')
    old=d.get(MARKER,{})
    need(type(old) is dict and set(old)<=set(rows),'SDK_CLOSED_NO_UNRECOGNIZED_PROJECTION')
    after=deepcopy(d);delta=dict(constructors=0,network_transfer_accounted_bytes=0);new={}
    total_output=0
    for rid,row in rows.items():
        need(row['status']=='SETTLED','SDK_CLOSED_ONLY_SETTLED')
        s=p[budget.SETTLEMENTS][rid];rh=common.sha(common.encode(s));amount=common.vector(s['financial_charged'])
        raw=closures[rid]['dual_bytes'];dual=common.obj(raw);lease_raw=closures[rid]['lease_bytes'];lease=common.obj(lease_raw)
        need(dual['status']=='ACTUAL_CURRENT_SDK_DUAL_ONCE_SETTLED_AND_LEASE_CLOSED'
             and common.same(dual['settlement'],s) and s['request_id']==rid
             and lease['run_uuid']==row['child_run_uuid'] and lease['status']=='CLOSED'
             and lease['scope']==budget.ROLE and lease['owner']=='EXISTING_LOCAL_PC' and lease['uid']==2009
             and lease['authority_release']==common.PARENT_AUTHORITY
             and lease['manifest_sha256']==row['manifest_sha256'] and lease['source_sha']==row['source_sha']
             and lease['all_registered_actors_terminal'] is True and lease['target_task_native_qualified'] is False
             and lease['settlement_sha256']==rh and common.sha(lease_raw)==dual['Core']['closed_lease_sha256'],
             'SDK_CLOSED_ACTUAL_DUAL_SETTLEMENT_AND_LEASE_BYTES')
        proof=common.encode(dict(settlement=s,Core_receipt=dual['Core']['Core_receipt'],lease_receipt=dual['Core']['lease_receipt']))
        need(common.sha(proof)==dual['Core']['closure_proof_sha256'],'SDK_CLOSED_EXACT_NATIVE_CLOSURE_PROOF')
        sid=s['settlement_id']
        for receipt in (dual['Core']['Core_receipt'],dual['Core']['lease_receipt'],dual['LOCAL_receipt']):_receipt(receipt,sid)
        need(dual['Core']['Core_after_sha256']==dual['Core']['Core_receipt']['after_sha256'],
             'SDK_CLOSED_CORE_ONCE_AFTER_BINDING')
        need(dual['LOCAL_readback']['receipt_binding_valid'] is True
             and dual['LOCAL_readback']['actual_sha256']==dual['LOCAL_after_sha256']==dual['LOCAL_receipt']['after_sha256']
             and s['target_task_native_qualified'] is False and s['core64_cells']==0
             and amount['episode_starts']==amount['physics_ticks']==0,'SDK_CLOSED_NO_TASK_OR_PHYSICS_CLAIM')
        marker=dict(schema=SCHEMA,request_id=rid,settlement_sha256=rh,dual_sha256=common.sha(raw),
            closed_lease_sha256=common.sha(lease_raw),financial_charged=deepcopy(amount),
            original_scientific_verdict=s['scientific_verdict'],already_charged_globally=True,new_global_charge=False,
            financial_constructor_upper_not_actual_MuJoCo_models=True,target_task_native_qualified=False)
        if rid in old:need(common.same(old[rid],marker),'SDK_CLOSED_EXACT_EXISTING_MARKER_NEVER_GUESSED_REPAIR')
        else:
            after.setdefault(MARKER,{})[rid]=marker;new[rid]=deepcopy(marker)
            delta['constructors']+=amount['constructors'];delta['network_transfer_accounted_bytes']+=amount['transfer_bytes']
        total_output+=amount['CoreLab_output_bytes']
    for field,value in delta.items():after[field]=d[field]+value
    need(after['constructors']<=12000 and after['network_transfer_accounted_bytes']<=128*1024**3,'SDK_CLOSED_ORIGINAL_SITE_CAPS')
    allowed={MARKER,*delta}
    need(all(common.same(after[k],v) for k,v in d.items() if k not in allowed),'SDK_CLOSED_PRESERVE_NEURAL_CPU_CLOCK_SOURCE_AND_COUNTER_HISTORY')
    return dict(status='PURE_CLOSED_SDK_PROJECTION_NOT_NATIVE_ADMISSION',before_DEC1_sha256=common.sha(DEC1_bytes),
        parent_sha256=common.sha(parent_bytes),Core_sha256=common.sha(Core_bytes),proposed_DEC1=after,delta_fields=delta,
        new_markers=new,total_closed_Core_output_upper_bytes=total_output,complete_financial_charged=spent,
        global_ledger_mutation=False,new_global_charge=False,native_dispatch_admitted=False)


def storage(snapshot,*,projection):
    """Once after actual DEC1 marker adoption; preserve the old source base."""
    need(not any(projection['delta_fields'].values()),'SDK_CLOSED_REAL_NATIVE_PROJECTION_REQUIRED')
    need('closed_SDK_storage_projection' not in snapshot,'SDK_CLOSED_ONE_STORAGE_APPLICATION')
    s=deepcopy(snapshot);old=7147094016
    need(s['used']['Core_storage_bytes']==100*1024**3+old
         and s['dec1_used']['Core_storage_bytes']==80*1024**3+old,
         'SDK_CLOSED_ORIGINAL_SOURCE_AND_COUNTER_BASE_ONCE')
    total=projection['total_closed_Core_output_upper_bytes']
    for label,peaks in (('used','storage_peaks'),('dec1_used','dec1_storage_peaks')):
        s[label]['Core_storage_bytes']+=total
        s[peaks]['Core_storage_bytes']=max(s[peaks]['Core_storage_bytes'],s[label]['Core_storage_bytes'])
    s['closed_SDK_storage_projection']=dict(total_Core_output_upper_bytes_added_once=total,
        parent_sha256=projection['parent_sha256'],Core_sha256=projection['Core_sha256'],
        global_costs_already_paid=True,target_task_native_qualified=False)
    return s
