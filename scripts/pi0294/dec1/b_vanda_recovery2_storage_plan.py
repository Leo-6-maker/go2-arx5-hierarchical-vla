"""Cold storage plan from the exact approved draft and actual closed costs.

Pure preparation, never native admission or a grant. Original numerical code,
all future envelopes, clocks and final protection remain unchanged.
"""
from copy import deepcopy
import hashlib
import json
from . import b_vanda_recovery2 as r


def cold_plan(plan,global_ledger,DEC1,*,grant_bytes,draft_bytes):
    grant=r.grant(grant_bytes)
    if hashlib.sha256(draft_bytes).hexdigest()!=grant['approved_draft_sha256']:
        raise PermissionError('RECOVERY2_EXACT_APPROVED_STORAGE_DRAFT_BYTES')
    r.validate_adopted(global_ledger,DEC1)
    if (not r.enabled(plan) or global_ledger['unknown_effects'] or global_ledger['pending_updates']
            or DEC1['unknown_effects'] or any(global_ledger['reserved'].values())
            or global_ledger['CPU_reserved_s'] or global_ledger['storage_reserved_bytes']
            or any(DEC1.get(k,0) for k in ('GPU_reserved_s','CPU_reserved_s','GPU_site_output_reserved_bytes'))
            or any(x['status']!='NATIVE_TERMINAL' for x in global_ledger['PBS_submissions'])):
        raise PermissionError('RECOVERY2_INACTIVE_CLOSED_COST_BOUNDARY_ONLY')
    draft=json.loads(draft_bytes)
    # The original draft explicitly funds every future output, including the
    # same full training bound and original three matrix jobs. Add only actual
    # closed costs accrued since it; never silently shrink that envelope.
    dimensions=draft['scope_review']['all_dimensions']
    row=dimensions['Hopper_storage_bytes']
    future=row['future']
    if (type(future) is not int or future<=0 or row['DEC1_total']!=row['DEC1_used']+future
            or row['global_total']!=row['global_used']+future+40*1024**3):
        raise PermissionError('RECOVERY2_ORIGINAL_COMPLETE_OUTPUT_AND_FINAL40G')
    used=DEC1['new_GPU_site_output_allocated_upper_bytes']
    global_used=global_ledger['new_storage_current_bytes']
    if used<row['DEC1_used'] or global_used<row['global_used']:
        raise PermissionError('RECOVERY2_NO_CLOSED_OUTPUT_REFUND_OR_COST_ROLLBACK')
    result=deepcopy(plan);old=plan['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']
    peak=max(old,used+future)
    if peak>614*1024**3 or global_used+future+40*1024**3>680*1024**3:
        raise PermissionError('RECOVERY2_ACTUAL_COST_AND_ALL_FUTURE_STORAGE_EXCEED_EXISTING_CAPS')
    result['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']=peak
    return result
