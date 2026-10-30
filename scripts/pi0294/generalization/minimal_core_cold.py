"""One exact zero-effect cold source repair, same still-unstarted allocation.

No new GPU/query/physics hold, no native dispatch and no reopening closed roles.
The original onceIO owner must independently read both canonical adoptions.
"""
from copy import deepcopy
import base64,json
from . import minimal_core_accounting as c

FIELD='ARX5_fixed512_exact_zero_cold_source_rebind'
SCHEMA='ARX5_FIXED512_ZERO_MODEL_CANONICAL_LOCAL_PATH_REPAIR_V1'
need=c.need


def validate(g):
    if FIELD not in g:return None
    m=g[FIELD];old=m['preserved_old_row'];c.validate_row(old)
    from .minimal_core_sharing import predecessor
    row=predecessor(g) or g[c.FIELD][m['request_id']]
    need(m['schema']==SCHEMA and m['old_row_sha256']==c.digest(old)
        and old['status']==c.STATES[0]and old['native_handle']is None and c.progress(old)==(0,0,0)
        and old['run_uuid']=='arx5-e1-fixed512-n294101-01'
        and row['run_uuid']=='arx5-e1-fixed512-n294101-02'
        and m['new_request_sha256']==row['request_sha256']
        and m['new_manifest_sha256']==row['run_manifest_sha256']
        and m['new_physical_manifest_sha256']==row['physical_manifest_sha256']
        and m['retained_old_metadata_Core_CPU_upper_s']==300
        and m['additional_GPU_queries_physics']==0
        and m['old_costs_and_escrows_retained']is m['original_clock_unchanged']is True,
        'EXACT_ONE_RETAINED_ZERO_MODEL_SOURCE_REBIND')
    allowed={'source_sha','run_uuid','native_unit','run_manifest_sha256','cold_input_audit_sha256'}
    need(set(old['request'])==set(row['request'])
        and all(row['request'][k]==v for k,v in old['request'].items()if k not in allowed),
        'NO_SCOPE_CLOCK_OR_SCIENTIFIC_CHANGE_IN_COLD_REBIND')
    return m


def closed(value,row,*,now_epoch):
    from scripts.pi0294.pv1.mix_core_delegation import _receipt
    from scripts.pi0294.pv1.successor_core_context import LEDGER_SCHEMA
    need(value.get('schema')=='ARX5_FIXED512_THREE_COLD_ROLES_ACTUAL_ZERO_CLOSED_V1'
        and value.get('request_id')==row['request_id']and value.get('source_sha')==row['source_sha']
        and value.get('LOCAL_sha256') and 0<=now_epoch-value['observed_epoch']<=120,
        'FRESH_EXACT_COLD_CLOSE_NOT_A_NATIVE_EXIT_INFERENCE')
    roles=value['roles'];need(set(roles)=={'owner','core','hopper'},'ALL_THREE_UNUSED_ROLES_CLOSED')
    for role,item in roles.items():
        before=base64.b64decode(item['lease_before_base64'],validate=True)
        after=base64.b64decode(item['lease_after_base64'],validate=True)
        old,new=map(json.loads,(before,after))
        _receipt(base64.b64decode(item['intent_base64']),base64.b64decode(item['receipt_base64']),
            before=c.sha(before),after=c.sha(after))
        need(old['status']=='ACTIVE'and new['status']=='CLOSED'
            and new['run_uuid']==old['run_uuid']==item['run_uuid']
            and new['authority_release']==old['authority_release']==c.TASK_SOURCE
            and new['deadline_epoch']==old['deadline_epoch']==row['deadline_epoch']
            and all(new[k]==v for k,v in old.items()if k!='status')
            and item['worker_records']==[]and item['native_start_files']==[],
            'ORIGINAL_LEASE_BYTES_NO_WORKER_OR_START_EFFECT:'+role)
    need(roles['core']['manifest_sha256']==row['physical_manifest_sha256']
        and roles['hopper']['manifest_sha256']==row['run_manifest_sha256'],
        'EXACT_OLD_BOUND_PHYSICAL_AND_NEURAL_MANIFESTS')
    for props in value['native_units'].values():
        need(props.get('MainPID')=='0'and props.get('LoadState')=='not-found'
            and props.get('ActiveState')=='inactive','NEVER_CREATED_OWNER_AND_GPU_UNITS')
    need(set(value['native_units'])=={'owner','neural'},'BOTH_OWN_NATIVE_UNITS')
    p=json.loads(base64.b64decode(value['physical_ledger_base64'],validate=True))
    need(p['schema']==LEDGER_SCHEMA and p['run_uuid']==row['run_uuid']
        and p['authority_release']==c.TASK_SOURCE and p['unknown_effects']==[]and p['reservations']=={}
        and all(p[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
        and p['Core_CPU_s']==300,'EXACT_OLD_ZERO_PHYSICS_WITH_RETAINED_METADATA_UPPER')
    return p


def propose(global_bytes,local_bytes,*,request_bytes,manifest_bytes,physical_manifest_bytes,
            closed_readback_bytes,now_epoch):
    g,l,r,m,p,proof=map(json.loads,(global_bytes,local_bytes,request_bytes,manifest_bytes,
        physical_manifest_bytes,closed_readback_bytes))
    c.paired(g,l);need(FIELD not in g and FIELD not in l,'ONE_PATH_REPAIR_NOT_AN_UNLIMITED_COLD_RETRY')
    need(len(g.get(c.FIELD,{}))==1,'ONE_ALREADY_RESERVED_E1_NOT_A_NEW_ALLOCATION')
    old=next(iter(g[c.FIELD].values()));c.validate_row(old)
    need(old['status']==c.STATES[0]and old['native_handle']is None and c.progress(old)==(0,0,0),
         'OLD_REQUEST_NEVER_BOUND_OR_CONSUMED')
    closed(proof,old,now_epoch=now_epoch)
    need(proof['LOCAL_sha256']==c.sha(local_bytes),'ACTUAL_SAME_CANONICAL_LOCAL_CLOSED_BOUNDARY')
    allowed={'source_sha','run_uuid','native_unit','run_manifest_sha256','cold_input_audit_sha256'}
    need(set(r)==set(old['request'])and all(r[k]==v for k,v in old['request'].items()if k not in allowed)
        and r['run_uuid']=='arx5-e1-fixed512-n294101-02'
        and old['run_uuid']=='arx5-e1-fixed512-n294101-01'
        and r['source_sha']!=old['source_sha'],'SAME_ID_NONCE_CLOCK_PANEL_AND_OWN_FINITE_HOLD')
    from .minimal_core_identity import request_identity
    request_identity(m,r,c.sha(manifest_bytes))
    need(p.get('platform_evaluation_version')==c.VERSION and p['candidate']=='E1'
        and p['run_uuid']==r['run_uuid']and p['code']['vla_commit']==r['source_sha']
        and p['authority']['scope_grant_sha256']==c.GRANT_SHA
        and all(p['limits'][k]==v for k,v in dict(CPU_s=93900,physics_ticks=832000,policy_episodes=26,engine_starts=2108).items()),
         'NEW_COLD_PHYSICAL_PEER_SAME_COMPLETE_ENVELOPE')
    row=deepcopy(old);row.update(r);row.update(request=deepcopy(r),request_sha256=c.sha(request_bytes),
        child_run_uuid=r['run_uuid'],physical_manifest_sha256=c.sha(physical_manifest_bytes))
    marker=dict(schema=SCHEMA,request_id=r['request_id'],preserved_old_row=deepcopy(old),
        old_row_sha256=c.digest(old),closed_readback_sha256=c.sha(closed_readback_bytes),
        new_request_sha256=c.sha(request_bytes),new_manifest_sha256=c.sha(manifest_bytes),
        new_physical_manifest_sha256=c.sha(physical_manifest_bytes),
        retained_old_metadata_Core_CPU_upper_s=300,additional_GPU_queries_physics=0,
        old_costs_and_escrows_retained=True,original_clock_unchanged=True,adopted_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for x in(v,w):x[c.FIELD][r['request_id']]=deepcopy(row);x[FIELD]=deepcopy(marker)
    w['CPU_s']['CoreLab']+=300;v['revision']+=1;c.paired(v,w)
    need(v['used']==g['used']and v['reserved']==g['reserved']
        and v['PBS_submissions']==g['PBS_submissions']and v[c.old.CORE_ROWS]==g[c.old.CORE_ROWS]
        and v['ARX5_minimal_first512_execution_package']==g['ARX5_minimal_first512_execution_package'],
         'NO_EXTRA_EFFECT_HOLD_OR_OLD_E2_D25_PACKAGE_CHANGE')
    return dict(proposed_global=v,proposed_LOCAL=w,original_onceIO_required=True,native_permission=False,
        additional_training_GPU_queries_physics=0,old_closed_roles_reopened=False)
