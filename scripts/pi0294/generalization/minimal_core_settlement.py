"""Independent raw512 terminal proposal on the original canonical counters."""
from copy import deepcopy
from decimal import Decimal
import json
from . import minimal_core_accounting as c
from scripts.pi0294.pv1 import mix_core_delegation as protocol
from scripts.pi0294.pv1.evaluation_core_accounting import terminal_values
from scripts.pi0294.pv1.mix_core_settlement import physical

need=c.need


def merge_terminal(grant_bytes,state_bytes,global_bytes,local_bytes,*,native_proof,effects,
                   physical_bytes,delivery_receipt_bytes,now_epoch):
    b=protocol._body(grant_bytes);state=protocol.validate_state(grant_bytes,state_bytes)
    g,l=map(json.loads,(global_bytes,local_bytes));c.paired(g,l)
    rid=b['request']['request_id'];marker=protocol._marker(grant_bytes)
    need(g.get(protocol.FIELD,{}).get(rid)==l.get(protocol.FIELD,{}).get(rid)==marker,
         'ONE_STILL_HELD_CANONICAL_CUSTODY')
    need(g[c.FIELD][rid]==l[c.FIELD][rid]==b['issued_row'] and now_epoch>=state['local_observed_epoch']
        and state['local_observed_epoch']<=native_proof['finished_epoch'],
        'NO_OTHER_WRITER_OR_EFFECT_AFTER_TERMINAL')
    need(json.loads(delivery_receipt_bytes)==c.endpoint_reference(),'ORIGINAL_DELIVERED_FIXED512_ENDPOINT')
    row=deepcopy(state['row']);p,phys=physical(physical_bytes,row)
    need(p['reservations']=={},'ALL_PHYSICAL_EFFECTS_KNOWN_NOT_A_TIMEOUT')
    amounts=terminal_values(row,request_id=rid,native_proof=native_proof,effects=effects,now_epoch=now_epoch)
    # Durable store receipts must be independently read by the caller before
    # this pure fold. A completed scientific query is not inferred from debit.
    v,w=deepcopy(g),deepcopy(l)
    for event in state['events']:
        n=event['count']
        if event['kind']=='generation_queries':
            v['generation_queries']+=n;v['used']['neural_queries']+=n;v['reserved']['neural_queries']-=n
        elif event['kind']=='model_loads':v['model_loads']+=n
        else:v['CPU_s']+=n;v['CPU_reserved_s']-=n
    q,loads,cpu,gpu,output=(amounts[k]for k in ('queries','loads','cpu','gpu','output'))
    v['generation_queries']+=q-amounts['charged'];v['used']['neural_queries']+=q-amounts['charged']
    v['model_loads']+=loads-amounts['paid_loads'];v['CPU_s']+=cpu-amounts['paid_cpu']
    v['used']['H200_s']+=gpu;v['reserved']['H200_s']-=c.HOLD['H200_s']
    v['reserved']['neural_queries']-=row['reserved_neural_queries']-len(amounts['unknown'])
    v['CPU_reserved_s']=float(Decimal(str(v['CPU_reserved_s']))-Decimal(80000)+Decimal(str(amounts['paid_cpu'])))
    v['storage_reserved_bytes']-=c.HOLD['Hopper_storage_bytes']
    v['new_storage_current_bytes']+=output;v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    row.update(status=c.STATES[3],native_exit_status=native_proof['native_exit_status'],conservative_GPU_s=gpu,
        conservative_CPU_s=cpu,query_attempts=q,model_load_attempts=loads,
        unknown_query_intents=deepcopy(amounts['unknown']),reserved_neural_queries=len(amounts['unknown']),
        terminal_native_sha256=c.digest(native_proof),terminal_effect_summary_sha256=c.digest(effects),settled_epoch=now_epoch)
    for x in (v,w):x[c.FIELD][rid]=deepcopy(row)
    w['GPU_reserved_s']-=c.HOLD['H200_s'];w['GPU_allocated_s']+=gpu
    w['CPU_reserved_s']['Hopper']-=c.HOLD['Hopper_CPU_s'];w['CPU_s']['Hopper']+=cpu
    w['CPU_reserved_s']['CoreLab']-=c.HOLD['Core_CPU_s'];w['CPU_s']['CoreLab']+=cpu+phys['physical_CPU_s']
    w['neural_queries']+=q;w['episode_starts']+=phys['episodes'];w['engine_starts']+=phys['constructors']
    w['physics_ticks']['live']+=phys['live'];w['physics_ticks']['shadow']+=phys['shadow']
    for name,n in dict(episode_starts=26,constructors=2108,physics_ticks=832000,
        CoreLab_output_bytes=c.HOLD['Core_storage_bytes'],transfer_bytes=c.HOLD['transfer_bytes']).items():
        need(w['Core_native_GPU_reserved'][name]>=n,'OWN_FULL_FINITE_HOLD_'+name)
        w['Core_native_GPU_reserved'][name]-=n
    w['network_transfer_accounted_bytes']+=c.HOLD['transfer_bytes']
    w['Core_native_GPU_current_storage_bytes']=w.get('Core_native_GPU_current_storage_bytes',0)+c.HOLD['Core_storage_bytes']
    closed=dict(marker,status=protocol.MERGED,terminal_native_sha256=c.digest(native_proof),
        terminal_effect_view_sha256=c.sha(state_bytes),merged_epoch=now_epoch,effects=len(state['events']),
        actual_global_before_sha256=c.sha(global_bytes),actual_LOCAL_before_sha256=c.sha(local_bytes))
    for x in (v,w):x[protocol.FIELD][rid]=deepcopy(closed)
    v['revision']+=1;c.paired(v,w)
    need(v[c.old.CORE_ROWS]==g[c.old.CORE_ROWS] and v['PBS_submissions']==g['PBS_submissions']
        and v['ARX5_minimal_first512_execution_package']==g['ARX5_minimal_first512_execution_package'],
        'NO_REWRITING_E2_D25_OR_OLD_PACKAGE')
    receipt=dict(schema='ARX5_MINIMAL_CORE_FIXED512_KNOWN_NATIVE_FINANCIAL_TERMINAL_V1',request_id=rid,
        terminal_native_sha256=c.digest(native_proof),effect_view_sha256=c.sha(state_bytes),
        GPU_s=gpu,neural_CPU_s=cpu,physical=phys,query_attempts=q,model_load_attempts=loads,
        scoped_unknown_query_intents=amounts['unknown'],unknown_queries_remain_charged_and_held=True,
        Core_output_charged_upper_bytes=c.HOLD['Core_storage_bytes'],transfer_charged_upper_bytes=c.HOLD['transfer_bytes'],
        observed_epoch=now_epoch,scientific_stage_completed=False,leases_closed=False)
    return dict(proposed_global=v,proposed_LOCAL=w,receipt=receipt,original_onceIO_required=True,
        native_IO_performed=False,leases_closed=False,scientific_stage_advanced=False)
