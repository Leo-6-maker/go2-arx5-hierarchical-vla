"""Pure once charge for actual Vanda migration metadata, within existing caps.

The original64MiB charge is retained. This supplements it by8MiB, including
2MiB of remaining finite audit/receipt output. It grants no training budget.
Native callers must CAS both actual Vanda ledgers and independently read back.
"""
from copy import deepcopy
import math
from . import vanda_control_handoff as handoff

FIELD='Vanda_controller_administrative_output_once'
DELTA=8*1024**2
OLD_UPPER=67108864
PRIOR_PROJECTION=64995328
ACTUAL_MIGRATION=5885952
FUTURE_AUDIT=2*1024**2

def need(ok,reason):
    if not ok:raise PermissionError('VANDA_ADMINISTRATIVE_STORAGE_'+reason)

def prepare(h,d,*,readback,now_epoch):
    need(type(now_epoch) in (int,float) and math.isfinite(now_epoch)
         and 1792348196<=now_epoch<1792952996,'ORIGINAL_CLOCK')
    need(FIELD not in h and FIELD not in d,'NO_SECOND_CHARGE')
    marker=h.get(handoff.FIELD)
    need(isinstance(marker,dict) and marker==d.get(handoff.FIELD)
         and marker.get('grant_sha256')==handoff.GRANT_SHA
         and marker.get('old_parent_mutations_permitted') is False
         and marker.get('origin_is_fresh_native_read') is False,'REAL_MIGRATED_SINGLE_AUTHORITY')
    need(h['optimizer_updates']==d['optimizer_updates']==8744 and h['generation_queries']==1197
         and h['reserved']==dict(H200_s=22560,optimizer_updates=2048,neural_queries=29)
         and h['CPU_reserved_s']==54000 and h['storage_reserved_bytes']==70000000000
         and h['unknown_effects']==h['pending_updates']==d['unknown_effects']==[],
         'FAILED_JOB_RESERVATIONS_AND_ALL_PAID_EFFECTS_RETAINED')
    need(readback.get('status')=='INDEPENDENT_MIGRATED_AUTHORITY_LEDGER_AND_STORAGE_BYTES'
         and type(readback.get('metadata_allocated_upper_bytes')) is int
         and readback['metadata_allocated_upper_bytes']==ACTUAL_MIGRATION
         and readback.get('combined_upper_bytes')==PRIOR_PROJECTION+ACTUAL_MIGRATION
         and readback.get('fits_prepaid_output') is False,'ACTUAL_FORECAST_MISS_RETAINED')
    need(PRIOR_PROJECTION+ACTUAL_MIGRATION+FUTURE_AUDIT<=OLD_UPPER+DELTA,'COMPLETE_ADMINISTRATIVE_OUTPUT_ENVELOPE')
    for v in (h['new_storage_current_bytes'],h['new_storage_peak_bytes'],d['new_GPU_site_output_allocated_upper_bytes'],d['GPU_site_output_reserved_bytes']):
        need(type(v) is int and v>=0,'FINITE_NONNEGATIVE_ACTUAL_STORAGE_COUNTERS')
    need(h['new_storage_current_bytes']<=h['new_storage_peak_bytes'],'RESIDENT_WITHIN_PEAK')
    need(h['new_storage_current_bytes']+h['storage_reserved_bytes']+DELTA+40*1024**3<=615*1024**3
         and d['new_GPU_site_output_allocated_upper_bytes']+d['GPU_site_output_reserved_bytes']+DELTA<=549*1024**3,
         'EXISTING549_615_CAPS_AND_ORIGINAL_FINAL40GIB')
    nh,nd=deepcopy(h),deepcopy(d)
    nh['new_storage_current_bytes']+=DELTA
    nh['new_storage_peak_bytes']=max(nh['new_storage_peak_bytes'],nh['new_storage_current_bytes'])
    nd['new_GPU_site_output_allocated_upper_bytes']+=DELTA
    receipt=dict(schema='VANDA_ADMINISTRATIVE_OUTPUT_WITHIN_EXISTING_CAPS_V1',authority_release=handoff.RELEASE,
        grant_sha256=handoff.GRANT_SHA,additional_charged_output_upper_bytes=DELTA,
        original64MiB_charge_preserved=True,actual_migration_allocated_upper_bytes=ACTUAL_MIGRATION,
        original_four_root_projection_upper_bytes=PRIOR_PROJECTION,remaining_audit_output_upper_bytes=FUTURE_AUDIT,
        cumulative_control_output_upper_bytes=OLD_UPPER+DELTA,new_GPU_PBS_updates_queries_models_physics=0,
        cap_or_clock_increase=False,old_origin_reconciliation_still_required=True,charged_epoch=now_epoch)
    nh[FIELD]=deepcopy(receipt);nd[FIELD]=deepcopy(receipt)
    return nh,nd,receipt
