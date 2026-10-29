"""Human removed the PBS count ceiling; real costs and admission remain.

Legacy finite-vector readers receive a rolling arithmetic bound, never a new
submission budget: it includes the current count plus all originally declared
PBS work and the next one request. It grows with actual history, so count alone
can never stop sequential same-scope work. Native resource/identity/once gates
still decide whether that one request is legal.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math

FIELD='V6_human_GPU_PBS_count_unlimited'
SCHEMA='PI0294_GPU_PBS_COUNT_UNLIMITED_ONCE_ADOPTION_V1'
GRANT_SHA='72a72ac3e5fd482e67c40ec9fed803bcd7a4581a467c66f4089f97519ed5abc2'
TEXT='GPU PBS提交次数不做限制，以跑通科学主线作为第一目标'
CLOCK=(1792932749.8133287,1793451149.8133287,1793537549.8133287)
def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,why):
    if not ok:raise PermissionError('PBS_COUNT_AUTHORITY_'+why)
def grant(body):
    need(type(body)is bytes and sha(body)==GRANT_SHA,'EXACT_DIRECT_HUMAN_GRANT_BYTES')
    v=json.loads(body)
    need(v['schema']=='PI0294_GPU_PBS_COUNT_UNLIMITED_DIRECT_HUMAN_AUTHORITY_V1'
         and v['human_instruction']==TEXT and v['PBS_submission_count_limit']is None
         and v['scope']=='EXISTING_V6_SCIENTIFIC_WORK_AND_SAME_SCOPE_ENGINEERING_REPLACEMENTS'
         and all(v[k]is False for k in('other_caps_changed','scientific_recipe_changed','concurrent_GPU_limit_changed','clocks_reset','native_adopted'))
         and tuple(v[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))==CLOCK
         and all(v[k]is True for k in('preserve_original_costs_and_failed_unknown_records','preserve_unique_request_onceIO_and_unknown_no_replay','preserve_remaining_matrix_and_final_physical_resources')),
         'COUNT_ONLY_NO_OTHER_RESOURCE_SCIENCE_OR_RUNTIME_PERMISSION')
    return v
def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    grant(base64.b64decode(m['grant_base64'],validate=True))
    need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
         and m['PBS_submission_count_limit']is None and m['new_GPU_permission']is False
         and m['new_resource_costs']=={} and m['other_caps_changed']is False
         and type(m['adopted_epoch'])in(int,float)and math.isfinite(m['adopted_epoch'])
         and CLOCK[0]<=m['adopted_epoch']<CLOCK[1]
         and type(m['before_PBS_count'])is int and m['before_PBS_count']>=41
         and g['used']['PBS_submissions']==len(g['PBS_submissions'])>=m['before_PBS_count']
         and type(m['before_GLOBAL_sha256'])is str and len(m['before_GLOBAL_sha256'])==64,
         'EXACT_ADOPTED_COUNT_ONLY_AND_RETAINED_REAL_HISTORY')
    return m
def history_upper(g,old):
    """Retain exact old row checks, retire only their obsolete outer count cap."""
    return max(old,g['used']['PBS_submissions'])if validate(g)is not None else old
def legacy_vector_bound(g,old):
    if validate(g)is None:return old
    # All counts are real/declared integers. This is a compatibility bound for
    # arithmetic-only old vectors, not a hard cap or extra funded work.
    declared=0
    for value in g.values():
        if not isinstance(value,dict):continue
        for key in('original_resources','remaining_resources'):
            row=value.get(key,{})
            if isinstance(row,dict):
                n=row.get('PBS_submissions',0)
                need(type(n)is int and n>=0,'DECLARED_NONNEGATIVE_ORIGINAL_WORK')
                declared+=n
    return max(old,g['used']['PBS_submissions']+declared+3+1)
def propose(global_bytes,grant_bytes,*,now_epoch):
    grant(grant_bytes);g=json.loads(global_bytes)
    need(FIELD not in g and type(now_epoch)in(int,float)and math.isfinite(now_epoch)
         and CLOCK[0]<=now_epoch<CLOCK[1]and g['pending_updates']==g['unknown_effects']==[],
         'ONCE_ONLY_AFTER_EXACT_CURRENT_UNFENCED_READ')
    need(all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions']),
         'CLOSED_NATIVE_BOUNDARY_NO_HOT_SWITCH')
    v=deepcopy(g);v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,
        grant_base64=base64.b64encode(grant_bytes).decode(),PBS_submission_count_limit=None,
        adopted_epoch=now_epoch,before_PBS_count=g['used']['PBS_submissions'],
        before_GLOBAL_sha256=sha(global_bytes),new_GPU_permission=False,
        new_resource_costs={},other_caps_changed=False)
    v['revision']+=1;validate(v);return v
