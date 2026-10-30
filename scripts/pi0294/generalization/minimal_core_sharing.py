"""Direct user GPU0 sharing for the same unstarted E1 allocation only.

Retains both earlier cold rows, all costs, the original device and qualification
gates. Foreign compute is observed, never controlled. No IO occurs on import.
"""
from copy import deepcopy
import json
from . import minimal_core_accounting as c

FIELD='ARX5_fixed512_direct_GPU0_sharing'
PLAN='plans/E1_FIXED512_GPU0_DIRECT_HUMAN_SHARING.json'
GPU='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
RID='f6eb2990-707d-439a-b99a-f8c89c05e0ff'
NONCE='5f0da9f9-d737-4346-bec1-60939cf8574c'
RUN='arx5-e1-fixed512-n294101-03'
need=c.need


def grant():
    return dict(schema='ARX5_E1_FIXED512_GPU0_DIRECT_USER_SHARING_V1',
        direct_human_answer='GPU0允许共享任务挂载',request_id=RID,submission_nonce=NONCE,
        GPU_uuid=GPU,candidate='E1',checkpoint_manifest_sha256=c.CHECKPOINT,
        checkpoint_step=512,remaining_conditions=24,optimizer_updates=0,
        same_scientific_allocation_and_clock=True,additional_GPU_queries_physics=0,
        original_30GiB_total_device_and_native_qualification_unchanged=True,
        foreign_process_control=False,infrastructure_changes=False,automatic_retry=False)


GRANT_SHA=c.digest(grant())


def allowed(r):
    if 'shared_GPU_grant_sha256' not in r:return False
    if r.get('execution_policy_sha256')==c.V7_POLICY:
        need(c.typed(r)and r['shared_GPU_grant_sha256']==c.V7_POLICY
            and r.get('GPU_uuid')==GPU and r.get('GPU_sharing_direct_human_answer')=='GPU0允许共享任务挂载'
            and r.get('evaluation_candidate')=='E1'and r.get('checkpoint_step')==512
            and r.get('checkpoint_manifest_sha256')==c.CHECKPOINT
            and r.get('scope_grant_sha256')==c.GRANT_SHA and r.get('optimizer_updates')==0,
            'V7_SAME_DIRECT_GPU0_PERMISSION_AND_FIXED_E1_SCIENCE')
        return True
    need(c.typed(r) and r.get('shared_GPU_grant_sha256')==GRANT_SHA
        and r.get('request_id')==RID and r.get('submission_nonce')==NONCE
        and r.get('run_uuid',r.get('child_run_uuid'))==RUN and r.get('GPU_uuid')==GPU
        and r.get('evaluation_candidate')=='E1' and r.get('checkpoint_step')==512
        and r.get('checkpoint_manifest_sha256')==c.CHECKPOINT
        and r.get('scope_grant_sha256')==c.GRANT_SHA and r.get('optimizer_updates')==0,
        'DIRECT_SHARED_GPU0_ONLY_SAME_UNSTARTED_E1_NOT_OLD_B24')
    return True


def manifest(m,r):
    if allowed(r):
        plan,pin=(c.V7_PLAN,c.V7_POLICY)if r.get('execution_policy_sha256')==c.V7_POLICY else(PLAN,GRANT_SHA)
        need(m['execution']['sealed_inputs'].get(plan)==pin,
             'DIRECT_GPU0_SHARING_PRESEALED_NOT_CALLER_FLAG')


def predecessor(g):
    """The earlier path-repair marker continues to validate its exact old row."""
    if FIELD not in g:return None
    return g[FIELD]['preserved_unshared_row']


def validate(g):
    rows=g.get(c.FIELD,{})
    if FIELD not in g:
        need(not any('shared_GPU_grant_sha256'in r for r in rows.values()),'NO_UNADOPTED_SHARED_E1')
        return None
    m=g[FIELD];old=m['preserved_unshared_row'];c.validate_row(old);row=rows[RID];allowed(row)
    need(m['schema']=='ARX5_E1_SAME_ZERO_ALLOCATION_DIRECT_GPU0_COLD_BOUNDARY_V1'
        and m['grant_sha256']==GRANT_SHA and m['preserved_unshared_row_sha256']==c.digest(old)
        and old['run_uuid']=='arx5-e1-fixed512-n294101-02'
        and old['status']==c.STATES[0] and old['native_handle']is None and c.progress(old)==(0,0,0)
        and row['request_sha256']==m['new_request_sha256']
        and row['run_manifest_sha256']==m['new_manifest_sha256']
        and row['physical_manifest_sha256']==m['new_physical_manifest_sha256']
        and m['retained_old_metadata_Core_CPU_upper_s']==300
        and m['additional_GPU_queries_physics']==0 and m['automatic_retry']is False,
        'EXACT_ONE_DIRECT_USER_SHARED_COLD_BOUNDARY')
    allowed_changes={'source_sha','run_uuid','native_unit','run_manifest_sha256','cold_input_audit_sha256'}
    need(set(row['request'])==set(old['request'])|{'shared_GPU_grant_sha256'}
        and all(row['request'][k]==v for k,v in old['request'].items()if k not in allowed_changes),
        'SAME_ID_NONCE_CLOCK_MODEL_PANEL_SAFETY_AND_HOLD')
    return m


def propose(global_bytes,local_bytes,*,request_bytes,manifest_bytes,physical_manifest_bytes,
            closed_readback_bytes,now_epoch):
    from .minimal_core_cold import closed
    from .minimal_core_identity import request_identity
    g,l,r,m,p,proof=map(json.loads,(global_bytes,local_bytes,request_bytes,manifest_bytes,
        physical_manifest_bytes,closed_readback_bytes))
    c.paired(g,l);need(FIELD not in g and FIELD not in l,'ONE_DIRECT_GPU0_BOUNDARY')
    old=g[c.FIELD][RID];closed(proof,old,now_epoch=now_epoch)
    need(proof['LOCAL_sha256']==c.sha(local_bytes),'CURRENT_CANONICAL_CLOSED_BOUNDARY')
    request_identity(m,r,c.sha(manifest_bytes));manifest(p,r)
    need(p['run_uuid']==r['run_uuid'] and p['code']['vla_commit']==r['source_sha']
        and p['authority']['scope_grant_sha256']==c.GRANT_SHA
        and all(p['limits'][k]==v for k,v in dict(CPU_s=93900,physics_ticks=832000,
            policy_episodes=26,engine_starts=2108).items()),'ORIGINAL_FULL_PHYSICAL_ENVELOPE')
    row=deepcopy(old);row.update(r);row.update(request=deepcopy(r),request_sha256=c.sha(request_bytes),
        child_run_uuid=r['run_uuid'],physical_manifest_sha256=c.sha(physical_manifest_bytes))
    marker=dict(schema='ARX5_E1_SAME_ZERO_ALLOCATION_DIRECT_GPU0_COLD_BOUNDARY_V1',
        grant_sha256=GRANT_SHA,preserved_unshared_row=old,preserved_unshared_row_sha256=c.digest(old),
        closed_readback_sha256=c.sha(closed_readback_bytes),new_request_sha256=c.sha(request_bytes),
        new_manifest_sha256=c.sha(manifest_bytes),new_physical_manifest_sha256=c.sha(physical_manifest_bytes),
        retained_old_metadata_Core_CPU_upper_s=300,additional_GPU_queries_physics=0,
        automatic_retry=False,adopted_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for x in(v,w):x[c.FIELD][RID]=deepcopy(row);x[FIELD]=deepcopy(marker)
    w['CPU_s']['CoreLab']+=300;v['revision']+=1;c.paired(v,w)
    need(v['used']==g['used'] and v['reserved']==g['reserved']
        and v['PBS_submissions']==g['PBS_submissions'] and v[c.old.CORE_ROWS]==g[c.old.CORE_ROWS]
        and v['ARX5_minimal_first512_execution_package']==g['ARX5_minimal_first512_execution_package'],
        'OLD_COSTS_ESCROWS_E2_D25_UNCHANGED_NO_ADDITIONAL_NATIVE_HOLD')
    return dict(proposed_global=v,proposed_LOCAL=w,native_permission=False,
        original_onceIO_required=True)
