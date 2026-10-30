"""Cold source routing only for the actual never-started06 request.

No new request, native attempt, resources, costs or clock. Three real closed
roles and fresh complete source bytes precede the original onceIO rebind.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a
FIELD='V5_CD_D25_unstarted06_cold07'
RUN='v5-cd-d25-n294101-07'
SOURCE06='94c3989505a327663fe3ada41ca206c4c7c3dfbc'
RID='9c771bec-ba9c-4a44-aefa-f424c58f5909'
GLOBAL='be0415daba3f30a8a9a9d0b51fc7ceb7c3bfa4952fa3212a7b1d574d36e19e0c'
LOCAL='5e93f79117ebe8ad9a6a6d3c59d5bcb70312859bee48a650c28954e448d99b1c'
def need(ok,why):
    if not ok:raise PermissionError('D25_UNSTARTED06_COLD07_'+why)
def digest(v):return a.sha(a.raw(v))

def validate(g,repair_marker):
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_d25_admission_repair as r
    old=m['preserved_prestart_row'];c.validate_row(old)
    need(m['schema']=='D25_ACTUAL_NEVER_STARTED06_COLD07_V1'
         and m['authority_sha256']==r.AUTHORITY and m['repair06_sha256']==digest(repair_marker)
         and m['before_GLOBAL_sha256']==GLOBAL and m['before_LOCAL_sha256']==LOCAL
         and old['request_id']==repair_marker['replacement_request_id']==m['request_id']==RID
         and old['submission_nonce']==m['submission_nonce']==repair_marker['replacement_submission_nonce']
         and old['status']==c.STATES[0]and old['native_handle']is None and c.progress(old)==(0,0,0)
         and old['source_sha']==SOURCE06 and old['child_run_uuid']==r.RUN
         and m['source_sha']!=SOURCE06 and re.fullmatch('[0-9a-f]{40}',m['source_sha'])
         and m['run_uuid']==RUN and re.fullmatch('[0-9a-f]{64}',m['source_seal_sha256'])
         and m['new_native_requests']==m['new_control_payment']==m['new_tensor_transfers']==0,
         'EXACT_UNSTARTED_IDENTITY_AND_ORIGINAL_FINANCIAL_REPAIR')
    proof=m['closed06_readback']
    need(proof['status']=='ACTUAL_THREE_COLD06_LEASES_CLOSED_NO_NATIVE_OR_MODEL_START'
         and proof['GLOBAL_sha256']==GLOBAL and proof['LOCAL_sha256']==LOCAL
         and proof['registry_sha256']=='ec1065f7c05bd642949e79b8339eeec9ea06f21385d828cb5bc448f3d24eb351'
         and proof['new07_roots_absent']is proof['all_units_never_started']is True
         and set(proof['roles'])=={'owner','core','hopper'}
         and digest(proof)==m['closed06_readback_sha256']
         and m['T_CD']==old['T_CD']and m['deadline_epoch']==old['deadline_epoch']
         and repair_marker['adopted_epoch']<=m['adopted_epoch']<min(old['deadline_epoch']-28860,1793404800),
         'REAL_CLOSED_BOUNDARY_ORIGINAL_CLOCK')
    for role,v in proof['roles'].items():
        need(v['root'].endswith('/'+('v5-cd-d25-eval-owner-06'if role=='owner'else r.RUN))
             and v['lease']==v['root']+'/control/OWN_LEASE.private.json'
             and re.fullmatch('[0-9a-f]{64}',v['closed_lease_sha256']),'EXACT_CLOSED_ROLE:'+role)
    return m


def prepare(global_bytes,local_bytes,*,authority_bytes,closed_readback,source_seal_bytes,frozen,source_readback,now_epoch):
    from . import mix_core_accounting as c,mix_core_d25_admission_repair as r
    from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    g,l=map(json.loads,(global_bytes,local_bytes));c.checked(g);c.paired(g,l);prior=r.validate(g)
    need(a.sha(global_bytes)==GLOBAL and a.sha(local_bytes)==LOCAL and a.sha(authority_bytes)==r.AUTHORITY
         and FIELD not in g and FIELD not in l and prior is not None
         and g['unknown_effects']==g['pending_updates']==l['unknown_effects']==l['pending_updates']==l['pending_queries']==[]
         and all(x['status']==c.STATES[3]for k,x in g[c.FIELD].items()if k!=RID), 'ONE_EXACT_UNUSED_CANONICAL_PAIR')
    seal=json.loads(source_seal_bytes)
    need(0<=now_epoch-closed_readback['observed_epoch']<=120
         and len(frozen)==15 and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(k)==v for k,v in frozen.items())
         and source_readback['status']=='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK'
         and source_readback['source_sha']==seal['source_sha']
         and source_readback['source_seal_sha256']==a.sha(source_seal_bytes)
         and source_readback['file_sha256']==seal['files']and source_readback['uid']==2009
         and source_readback['hostname']=='corelab-4080s'
         and source_readback['model_calls']==source_readback['physics_ticks']==0
         and 0<=now_epoch-source_readback['observed_epoch']<=120,'FRESH_CLOSED_BOUNDARY_AND_ALL_NEW_SOURCE_BYTES')
    m=dict(schema='D25_ACTUAL_NEVER_STARTED06_COLD07_V1',authority_sha256=r.AUTHORITY,repair06_sha256=digest(prior),
        before_GLOBAL_sha256=GLOBAL,before_LOCAL_sha256=LOCAL,preserved_prestart_row=deepcopy(g[c.FIELD][RID]),
        request_id=RID,submission_nonce=prior['replacement_submission_nonce'],source_sha=seal['source_sha'],
        source_seal_sha256=a.sha(source_seal_bytes),run_uuid=RUN,new_native_requests=0,new_control_payment=0,new_tensor_transfers=0,
        closed06_readback=deepcopy(closed_readback),closed06_readback_sha256=digest(closed_readback),
        full_source_readback_sha256=digest(source_readback),T_CD=prior['T_CD'],deadline_epoch=prior['deadline_epoch'],adopted_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for x in(v,w):x[FIELD]=deepcopy(m)
    validate(v,prior);c.checked(v);c.paired(v,w)
    return dict(before_global=g,before_LOCAL=l,assembly_global=v,assembly_LOCAL=w)


def finalize(plan,assembly):
    from . import mix_core_accounting as c,mix_core_d25_admission_repair as r,mix_core_d25_cold_source as cold
    v,w=deepcopy(plan['assembly_global']),deepcopy(plan['assembly_LOCAL']);m=validate(v,v[r.FIELD]);old=m['preserved_prestart_row'];row=assembly['record']
    mutable={'source_sha','child_run_uuid','created_epoch','run_manifest_sha256','request_sha256','cold_input_audit_sha256',cold.DESCRIPTOR}
    need(set(row)==set(old)and all(row[k]==val for k,val in old.items()if k not in mutable)
         and row['source_sha']==m['source_sha']and row['child_run_uuid']==RUN
         and row[cold.DESCRIPTOR]==digest(m),'ONLY_SOURCE_AND_IMMUTABLE_COLD_IDENTITY_FIELDS')
    for x in(v,w):x[c.FIELD][RID]=deepcopy(row)
    stage=w['V5_CD_Core_CPU_substages'][c.key('D25')]
    need(stage['status']=='RESERVED_NOT_SPENT'and stage['request_id']==RID,'SAME_UNSPENT_ORIGINAL_PHYSICAL_STAGE')
    stage.update(run_uuid=RUN,source_sha=m['source_sha'],run_manifest_sha256=assembly['rows']['core']['manifest_sha256']);v['revision']+=1
    need(all(v[k]==x for k,x in plan['before_global'].items()if k not in(c.FIELD,'revision'))
         and all(w[k]==x for k,x in plan['before_LOCAL'].items()if k not in(c.FIELD,'V5_CD_Core_CPU_substages')),'NO_NEW_HOLD_FEE_CLOCK_OR_HISTORY')
    c.checked(v);c.paired(v,w)
    return dict(proposed_global=v,proposed_LOCAL=w,new_native_requests=0,native_admission=False)
