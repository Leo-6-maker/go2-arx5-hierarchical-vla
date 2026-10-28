"""Pure closed-source storage seam for a future cold B snapshot, never GO.

ACTIVE source work uses a distinct full pending native adapter. This function
cannot silently drop that work, replay costs, or alter any training state.
"""
from copy import deepcopy
from . import source_closed_costs as costs


def augment(snapshot,*,parent_bytes,Core_bytes,Hopper_bytes,closures,
            allocated_local_namespace_bytes,observed_epoch,now_epoch):
    need=costs.need
    projection=costs.propose(parent_bytes,Core_bytes,Hopper_bytes,closures=closures)
    need(not projection['pending_adapter_required'],
         'ACTIVE_SOURCE_NEEDS_COMPLETE_SEPARATE_NATIVE_PENDING_ADAPTER')
    need(not any(projection['delta_fields'].values()),'REAL_INACTIVE_HOPPER_COST_CATCHUP_REQUIRED')
    need(0<=costs.num(now_epoch)-costs.num(observed_epoch)<1800,'FRESH_ACTUAL_NAMESPACE_OBSERVATION')
    s=deepcopy(snapshot);d=costs.obj(Hopper_bytes);p=costs.obj(parent_bytes)
    need('closed_source_storage_projection' not in s,'ONE_APPLICATION_TO_ORIGINAL_FRESH_SNAPSHOT')
    need(s['authority_release']==d['authority_release']==costs.AUTHORITY
         and s['campaign_run_uuid']==d['campaign_run_uuid']==p['run_uuid']
         and s['T0_epoch']==d['original_T0_epoch']==p['T0_epoch']==1792348196
         and s['actual_DEC1_ledger_sha256']==costs.sha(Hopper_bytes),
         'ORIGINAL_CLOCK_AND_EXACT_ACTUAL_HOPPER_BYTES')
    need(s['used']['Core_storage_bytes']==100*1024**3
         and s['dec1_used']['Core_storage_bytes']==80*1024**3,
         'ORIGINAL_MATURE_CORE_ENVELOPES_BEFORE_ANY_SOURCE_INCREMENT')
    inclusive=s['dec1_used']['local_storage_bytes'];allocated=costs.num(allocated_local_namespace_bytes,True)
    need(allocated+512*1024**2<inclusive==8*1024**3,'ACTUAL_ALL_SOURCE_NAMESPACES_INSIDE_ORIGINAL_LOCAL_PEAK')
    total=projection['total_closed_Core_output_upper_bytes']
    for label,peaks in (('used','storage_peaks'),('dec1_used','dec1_storage_peaks')):
        value=s[label]['Core_storage_bytes']+total
        s[label]['Core_storage_bytes']=value;s[peaks]['Core_storage_bytes']=max(s[peaks]['Core_storage_bytes'],value)
    s['closed_source_storage_projection']=dict(schema='B_DUAL_KIND_CLOSED_CORE_SOURCE_STORAGE_PROJECTION_V1',
        total_closed_Core_output_bound_added_once=total,source_parent_sha256=costs.sha(parent_bytes),
        source_Core_sha256=costs.sha(Core_bytes),actual_Hopper_DEC1_sha256=costs.sha(Hopper_bytes),
        actual_namespace_bytes=allocated,local_inclusive_peak_bytes=inclusive,
        closed_requests=deepcopy(projection['per_request']),no_second_global_charge=True,
        old_source_UNKNOWN_financial_upper_preserved=True,native_dispatch_admitted=False)
    return s
