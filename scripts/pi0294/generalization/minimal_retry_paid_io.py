"""Pure exact paid-output candidate for one zero-effect first512 replacement.

This does not install a marker, change accounting, credit a ledger, or grant
native dispatch. Actual readers must additionally enforce combined physical
output and control-directory ceilings before an adoption can be considered.
"""
import math
from . import minimal_execution as e,minimal_retry_custody as custody,minimal_scope as scope

def candidate(global_bytes,local_bytes,*,native_readback,closed_readback,old_cold_close_readback,root_inventory,new_request_id,new_nonce,new_root,now_epoch):
    import json
    prior=custody.inspect(global_bytes,local_bytes,native_readback=native_readback,closed_readback=closed_readback)
    g=json.loads(global_bytes);r=e.rows(g)[0];base=g[e.FIELD]
    e.need(e._uuid(new_request_id) and e._uuid(new_nonce)
        and new_request_id!=r['request_id'] and new_nonce!=r['submission_nonce']
        and type(new_root)is str and new_root.endswith('/arx5-minimal-raw-parent-first512-retry-01')
        and all(v.get('request_id')!=new_request_id and v.get('submission_nonce')!=new_nonce
                for v in [*g['PBS_submissions'],*g.get(e.original.CORE_ROWS,{}).values()]),
        'RETRY_NEW_SINGLE_BOUND_IDENTITY')
    e.need(type(now_epoch)in(int,float) and math.isfinite(now_epoch)
        and now_epoch+59520<r['stop_new_training_epoch'], 'RETRY_COMPLETE_ORIGINAL_TRAINING_WINDOW')
    inv=root_inventory
    e.need(old_cold_close_readback.get('status')=='INDEPENDENT_OLD_COLD_ONLY_LEASE_CLOSED_ONCE_READBACK'
        and old_cold_close_readback.get('new_writes')==0 and len(old_cold_close_readback.get('lease_sha256',''))==64,
        'RETRY_FIRST_COLD_ROLE_ACTUALLY_CLOSED')
    e.need(inv['status']=='ACTUAL_TWO_CLOSED_ROOTS_UNIQUE_INODE_READONLY' and inv['new_writes']==0
        and set(inv['roots'])=={'arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02'}
        and type(inv['unique_allocated_bytes'])is int and 0<inv['unique_allocated_bytes']<=e.CONTROL['Hopper_storage_bytes']
        and now_epoch>=inv['observed_epoch'] and now_epoch-inv['observed_epoch']<=120,
        'RETRY_ACTUAL_CLOSED_METADATA_WITHIN_ORIGINAL_PAID_CONTROL')
    for v in inv['roots'].values():
        e.need(not any(row['path'].endswith(('.safetensors','.pt','.pth','.ckpt','.bin')) for row in v['rows']),
            'RETRY_NO_PRIOR_MODEL_PAYLOAD_AND_NO_FALSE_ZERO_FROM_OWNER_LABEL')
    allowance=e.native_resources()['Hopper_storage_bytes']
    e.need(prior['settled_original_output_gross_bytes']==allowance
        and g['new_storage_current_bytes']>=base['before']['new_storage_current_bytes']+e.CONTROL['Hopper_storage_bytes']+allowance
        and g['new_storage_peak_bytes']>=g['new_storage_current_bytes'],
        'RETRY_EXACT_ALREADY_CHARGED_ORIGINAL_GROSS_AND_CURRENT_PEAK')
    # Keep the paid original output allowance and the separate engineering
    # metadata allowance in the forecast. No historical number is decreased.
    dims=prior['gross_duplicate_charge_forecast'];projection={};shortfalls={}
    for k,row in dims.items():
        amount=row['projected']
        if k=='Hopper_storage_bytes':amount=amount-allowance+scope.components()['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT'][k]
        cap=row['cap'];projection[k]=dict(projected=amount,cap=cap,shortfall=max(0,amount-cap))
        if projection[k]['shortfall']:shortfalls[k]=projection[k]['shortfall']
    return dict(status='PAID_OUTPUT_REUSE_CANDIDATE_ONLY_NOT_ADOPTED',failed_request_id=r['request_id'],
        failed_job_id=r['job_id'],new_request_id=new_request_id,new_nonce=new_nonce,new_root=new_root,
        old_paid_output_allowance=allowance,old_actual_control_allocated=inv['unique_allocated_bytes'],
        combined_control_allocated_ceiling=e.CONTROL['Hopper_storage_bytes']+scope.components()['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT']['Hopper_storage_bytes'],
        replacement_output_allocated_ceiling=allowance,old_retained_artifacts_must_remain_unchanged=True,
        full_dimensions=projection,shortfalls=shortfalls,old_costs_current_and_peak_unchanged=True,
        periodic_actual_physical_enforcement_implemented=False,original_row_history_binding_implemented=False,
        native_registration_or_permission=False,actual_credit_granted=False)
