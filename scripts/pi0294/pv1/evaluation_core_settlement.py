"""Known native Core GPU effects settle once into the original two ledgers.

Pure proposals only. Native unit exit, actual journal, transfer and allocated
storage readbacks remain required. Queue-only/unknown starts cannot be refunded
through this path. Physical-worker and control leases settle independently.
"""
from copy import deepcopy
import hashlib
import json

from . import evaluation_core_accounting as c
from .evaluation_admission import resource_floor


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_SETTLEMENT_'+why)


def propose(global_bytes,local_bytes,*,request_id,native_proof,effects,transfer_proof,now_epoch):
    need(type(global_bytes) is bytes and type(local_bytes) is bytes,'FULL_ACTUAL_ORIGINAL_LEDGER_BYTES')
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.scope.validate_pair(g,l)
    before=g[c.FIELD][request_id];held=l.get('Core_native_GPU_reservations',{}).get(request_id)
    need(type(held) is dict and held.get('status')=='RESERVED_NOT_SPENT'
         and held.get('request_id')==request_id and held.get('request_sha256')==before['request_sha256']
         and held.get('source_sha')==before['source_sha'] and held.get('run_uuid')==before['child_run_uuid']
         and held.get('Core_CPU_upper_s')==c.CPU_UPPER
         and held.get('Core_output_upper_bytes')==before['additional_output_upper_bytes']
         and held.get('transfer_upper_bytes')==2*c.MODEL_BYTES
         and request_id not in l.get('Core_native_GPU_terminal_settlements',{}),
         'ONCE_EXACT_UNSETTLED_PAIRED_LOCAL_HOLD')
    t=transfer_proof
    need(type(t) is dict and t.get('schema')=='PV1_CORE_MODEL_TRANSFER_TERMINAL_READBACK_V1'
         and t.get('request_id')==request_id and t.get('request_sha256')==before['request_sha256']
         and t.get('source_sha')==before['source_sha']
         and t.get('all_effects_accounted') is True and t.get('unknown_effects')==[]
         and type(t.get('accounted_bytes')) is int and 0<=t['accounted_bytes']<=2*c.MODEL_BYTES,
         'KNOWN_ACTUAL_TRANSFER_OR_CONSERVATIVE_CHARGED_BOUND_NO_UNKNOWN_REFUND')
    after=c.settle(g,request_id=request_id,native_proof=native_proof,effects=effects,now_epoch=now_epoch)
    row=after[c.FIELD][request_id];local=deepcopy(l)
    cpu=row['conservative_CPU_s'];gpu=row['conservative_GPU_s'];output=native_proof['new_resident_allocated_upper_bytes']
    need(local['GPU_reserved_s']>=c.GPU_UPPER
         and local['CPU_reserved_s']['CoreLab']>=c.CPU_UPPER
         and local['CPU_reserved_s']['Hopper']>=c.CPU_UPPER
         and local['Core_native_GPU_reserved']['CoreLab_output_bytes']>=before['additional_output_upper_bytes']
         and local['Core_native_GPU_reserved']['transfer_bytes']>=2*c.MODEL_BYTES,'ALL_NATIVE_ESCROWS_STILL_PRESENT')
    local['GPU_reserved_s']-=c.GPU_UPPER;local['GPU_allocated_s']+=gpu
    for site in ('CoreLab','Hopper'):
        local['CPU_reserved_s'][site]-=c.CPU_UPPER;local['CPU_s'][site]+=cpu
    local['neural_queries']+=row['query_attempts']
    local['network_transfer_accounted_bytes']+=t['accounted_bytes']
    local['Core_native_GPU_reserved']['CoreLab_output_bytes']-=before['additional_output_upper_bytes']
    local['Core_native_GPU_reserved']['transfer_bytes']-=2*c.MODEL_BYTES
    local['Core_native_GPU_current_storage_bytes']=local.get('Core_native_GPU_current_storage_bytes',0)+output
    receipt=dict(schema='PV1_CORE_GPU_KNOWN_DUAL_SETTLEMENT_PROPOSAL_V1',request_id=request_id,
        request_sha256=before['request_sha256'],native_handle=before['native_handle'],source_sha=before['source_sha'],
        execution_platform='CORELAB_RTX4080',legacy_Hopper_CPU_is_site_projection=True,
        GPU_s=gpu,Core_CPU_s=cpu,GPU_site_CPU_s=cpu,query_attempts=row['query_attempts'],
        model_load_attempts=row['model_load_attempts'],transfer_accounted_bytes=t['accounted_bytes'],
        new_resident_allocated_upper_bytes=output,unknown_query_intents=row['unknown_query_intents'],
        native_terminal_sha256=c.digest(native_proof),actual_effects_sha256=c.digest(effects),transfer_sha256=c.digest(t),
        original_GLOBAL_before_sha256=hashlib.sha256(global_bytes).hexdigest(),
        original_LOCAL_before_sha256=hashlib.sha256(local_bytes).hexdigest(),settled_epoch=now_epoch,
        original_PBS_and_training_preserved=True,new_PBS=0,new_optimizer_updates=0,IO_performed=False)
    local.setdefault('Core_native_GPU_terminal_settlements',{})[request_id]=receipt
    local['Core_native_GPU_reservations'][request_id].update(status='SETTLED_KNOWN_NATIVE_TERMINAL',
        terminal_settlement_sha256=c.digest(receipt))
    if row['unknown_query_intents']:
        local['unknown_effects'].append(dict(kind='CORE_GPU_UNKNOWN_QUERIES_PRESERVED',request_id=request_id,
            native_handle=before['native_handle'],indices=deepcopy(row['unknown_query_intents'])))
    c.scope.validate_pair(after,local)
    # Unknown effects deliberately prevent future admission; do not weaken
    # resource_floor just to manufacture a clean terminal receipt.
    if not row['unknown_query_intents']:resource_floor(after,local)
    return dict(proposed_global=after,proposed_LOCAL=local,receipt=receipt,IO_performed=False,
        scientific_or_native_admission=False,lease_close_performed=False)
