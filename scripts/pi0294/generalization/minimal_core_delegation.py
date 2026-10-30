"""New512 grant on the unchanged finite Core once-debit wire protocol."""
from copy import deepcopy
import json
from . import minimal_core_accounting as c, minimal_core_native as native

GRANT='ARX5_MINIMAL_CORE_FIXED512_FINITE_GRANT_V1'
need=c.need


def body(grant_bytes):
    from scripts.pi0294.pv1 import mix_core_delegation as protocol,mix_core_identity as identity
    grant=json.loads(grant_bytes)
    need(set(grant)=={'schema','body','body_sha256'} and grant['schema']==GRANT
        and grant['body_sha256']==c.digest(grant['body']),'FIXED512_GRANT_BODY_HASH')
    b=grant['body'];r=b['request'];m=b['manifest'];row=b['issued_row']
    need(set(b)=={'mode','request','manifest','issued_row','admission','source_inputs','issued_epoch',
        'execution_not_after_epoch','accounting_not_after_epoch','limits','original_GLOBAL_revision'},
        'EXACT_SHARED_FINITE_GRANT_FIELDS')
    identity.request_identity(m,r,c.digest(m));c.validate_row(row)
    need(b['mode']==r.get('global_accounting_mode')==m['execution'].get('global_accounting_mode')==protocol.MODE
        and row['status']==c.STATES[1] and c.progress(row)==(0,0,0)
        and row['request_sha256']==c.digest(r) and row['run_manifest_sha256']==c.digest(m),
        'BOUND_ZERO_EFFECT_NEW512_ON_SHARED_ONCE_PROTOCOL')
    p=b['admission'];allocation=p['allocation']
    need(p['schema']==native.PROOF and p['request_id']==r['request_id']
        and p['request_sha256']==c.digest(r) and p['GLOBAL_sha256']==b['source_inputs']['global_bytes']
        and p['LOCAL_sha256']==b['source_inputs']['local_bytes'] and p['snapshot_sha256']==b['source_inputs']['snapshot_bytes']
        and p['shortfalls']=={} and p['all_remaining_costs_fit'] is True
        and p['original_matrix_and_final_protected'] is True and p['model_permission'] is p['physics_authorized'] is False,
        'EXACT_COMPLETE_RESOURCE_JOIN_NOT_A_MODEL_GO')
    identity.sealed_identity(m,r,allocation,run_uuid=r['run_uuid'],manifest_sha256=c.digest(m),
        request_sha256=c.digest(r),same_bound_process=row)
    need(b['source_inputs']['manifest_bytes']==c.digest(m) and b['source_inputs']['request_bytes']==c.digest(r)
        and 0<=b['issued_epoch']-allocation['native_process']['observed_epoch']<=120
        and b['execution_not_after_epoch']==allocation['allocation_end_epoch']-2700
        and b['accounting_not_after_epoch']==allocation['allocation_end_epoch']
        and b['issued_epoch']<b['execution_not_after_epoch']<b['accounting_not_after_epoch']<=r['deadline_epoch'],
        'FRESH_BOUND_GRANT_WITH_ORIGINAL_CLOSEOUT_RESERVE')
    need(b['limits']==dict(generation_queries=2097,model_loads=2,CPU_s=80000,GPU_s=28860,
        output_bytes=c.HOLD['Hopper_storage_bytes'],optimizer_updates=0),'EXACT_ONCE_RESERVED_JOB_LIMITS')
    return b


def issue(**inputs):
    from scripts.pi0294.pv1 import mix_core_delegation as protocol
    need(set(inputs)==set(protocol._INPUTS)|{'now_epoch'},'COMPLETE_NATIVE_INPUT_BYTES')
    proof=native.recompute(**inputs)
    g,l,m,r=(json.loads(inputs[k])for k in ('global_bytes','local_bytes','manifest_bytes','request_bytes'))
    for k,v in [('global_bytes',g),('local_bytes',l),('manifest_bytes',m),('request_bytes',r)]:
        need(inputs[k]==c.raw(v),'CANONICAL_ACTUAL_'+k)
    row=deepcopy(g[c.FIELD][r['request_id']])  # Full paired validation in recompute.
    need(g.get(protocol.FIELD,{})==l.get(protocol.FIELD,{})
        and all(v['status']==protocol.MERGED for v in g.get(protocol.FIELD,{}).values())
        and r['request_id'] not in g.get(protocol.FIELD,{}),'ONE_ORIGINAL_FINITE_CUSTODY_OWNER')
    b=dict(mode=protocol.MODE,request=r,manifest=m,issued_row=row,admission=proof,
        source_inputs={k:c.sha(inputs[k]) for k in protocol._INPUTS},issued_epoch=inputs['now_epoch'],
        execution_not_after_epoch=proof['allocation']['allocation_end_epoch']-2700,
        accounting_not_after_epoch=proof['allocation']['allocation_end_epoch'],
        limits=dict(generation_queries=2097,model_loads=2,CPU_s=80000,GPU_s=28860,
                    output_bytes=c.HOLD['Hopper_storage_bytes'],optimizer_updates=0),
        original_GLOBAL_revision=g['revision'])
    grant=dict(schema=GRANT,body=b,body_sha256=c.digest(b));marker=protocol._marker(c.raw(grant))
    v,w=deepcopy(g),deepcopy(l)
    for x in (v,w):x.setdefault(protocol.FIELD,{})[r['request_id']]=deepcopy(marker)
    v['revision']+=1
    # Marker-only append: all allocations/counters just validated by recompute
    # are unchanged, and the same exact marker is written to both targets.
    need(v[c.FIELD]==g[c.FIELD]and w[c.FIELD]==l[c.FIELD]
        and v[protocol.FIELD]==w[protocol.FIELD],'ONLY_MATCHED_FINITE_MARKER_APPEND')
    return dict(grant=grant,proposed_global=v,proposed_LOCAL=w,original_onceIO_required=True,
                native_IO_performed=False,model_permission=False,physics_authorized=False)


def activate(grant_bytes,*,global_bytes,local_bytes,global_intent_bytes,global_receipt_bytes,
             local_intent_bytes,local_receipt_bytes,now_epoch):
    from scripts.pi0294.pv1 import mix_core_delegation as protocol
    b=body(grant_bytes);marker=protocol._marker(grant_bytes);g,l=map(json.loads,(global_bytes,local_bytes));c.paired(g,l)
    need(0<=now_epoch-b['issued_epoch']<=120,'FRESH_INDEPENDENT_DUAL_READBACK')
    for x,k in [(g,'global_bytes'),(l,'local_bytes')]:
        rid=b['request']['request_id'];need(x.get(protocol.FIELD,{}).get(rid)==marker,'ACTUAL_HELD_MARKER')
        before=deepcopy(x);del before[protocol.FIELD][rid]
        if not before[protocol.FIELD]:del before[protocol.FIELD]
        if k=='global_bytes':
            need(before['revision']==b['original_GLOBAL_revision']+1,'ONE_ADOPTION_REVISION');before['revision']-=1
        need(c.digest(before)==b['source_inputs'][k],'EXACT_ADOPTION_ONLY_NOT_OTHER_CHANGES')
    protocol._receipt(global_intent_bytes,global_receipt_bytes,before=b['source_inputs']['global_bytes'],after=c.sha(global_bytes))
    protocol._receipt(local_intent_bytes,local_receipt_bytes,before=b['source_inputs']['local_bytes'],after=c.sha(local_bytes))
    need(json.loads(global_intent_bytes)['target']!=json.loads(local_intent_bytes)['target'],'DISTINCT_CANONICAL_TARGETS')
    activation=dict(epoch=now_epoch,GLOBAL_sha256=c.sha(global_bytes),LOCAL_sha256=c.sha(local_bytes),
        GLOBAL_intent_sha256=c.sha(global_intent_bytes),GLOBAL_receipt_sha256=c.sha(global_receipt_bytes),
        LOCAL_intent_sha256=c.sha(local_intent_bytes),LOCAL_receipt_sha256=c.sha(local_receipt_bytes))
    return protocol._initial(grant_bytes,activation)
