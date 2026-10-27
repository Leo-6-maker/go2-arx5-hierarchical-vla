"""Exact human-approved one-hour overlay; original seven-cap marker is immutable."""
import base64
import json
import math
from copy import deepcopy
from . import minimal_adoption as adoption

FIELD='V5_CD_C50_GPU147_TO148_ONLY_authority'
GRANT_SHA='a161314047ad8aea6ce0c468353f9da4d252521f5c6f4c7eba84f6ee652e66a9'
DRAFT_SHA='27d8c0f14a9888ca71f12d046b4c6f4103f49e13c72db2e48bc32b3024daf2fb'
FAILED='75c63c6e-c3f6-41cc-adae-a62971b26e74'
TERMINAL='ed5d75e19020ff8fcb846e30f1381a8c0336c0556a066385be38b73770068a43'
BEFORE=529200
AFTER=532800

def need(ok,why):
    if not ok:raise PermissionError('V5_GPU148_'+why)

def grant(b):
    need(type(b)is bytes and adoption.sha(b)==GRANT_SHA,'EXACT_DIRECT_HUMAN_BYTES')
    h=json.loads(b)
    need(h['schema']=='V5_CORE_C50_GPU147_TO148_DIRECT_HUMAN_GRANT_V1'
        and h['human_answer']=='批准仅GPU总上限147→148小时'
        and h['draft_sha256']==DRAFT_SHA and h['failed_request_id']==FAILED
        and h['caps']=={'H200_s':{'before':BEFORE,'after':AFTER}}
        and all(h[k]is True for k in('other_caps_unchanged','no_new_PBS_or_training_or_conditions',
        'all_older_costs_peaks_markers_unknowns_and_checkpoints_preserved',
        'original_matrix3_and_final20GPUh40GiB_protected','clocks_not_reset','scientific_recipe_R0_S1_scoring_unchanged'))
        and h['new_native_admission_by_this_document']is False,'ONLY_GPU_HOUR')
    return h

def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    need(adoption.validate(g)is not None,'ORIGINAL_SEVEN_CAPS_REQUIRED')
    grant(base64.b64decode(m['grant_base64'],validate=True))
    from scripts.pi0294.pv1 import successor_cd as s
    clock=s.validated(g);r=g['V5_CD_Core_allocations'][FAILED]
    need(m['schema']=='V5_CD_C50_GPU147_TO148_ONLY_AUTHORITY_V1'
        and m['grant_sha256']==GRANT_SHA and m['before_H200_s']==BEFORE and m['after_H200_s']==AFTER
        and m['terminal_native_sha256']==TERMINAL==r['terminal_native_sha256']
        and r['status']=='NATIVE_TERMINAL'and r['native_exit_status']==1
        and r['query_attempts']==r['model_load_attempts']==0
        and r['unknown_query_intents']==[]
        and all(m[k]==clock[k]for k in('T_CD','deadline_epoch','stop_new_training_epoch'))
        and type(m['adopted_epoch'])in(int,float)and math.isfinite(m['adopted_epoch'])
        and clock['T_CD']<=m['adopted_epoch']<clock['deadline_epoch']
        and m['new_native_admission']is False,'EXACT_TERMINAL_CLOCK_AND_ONLY_CAP')
    for k in('before_GLOBAL_sha256','before_LOCAL_sha256','closed_readback_sha256'):
        need(type(m[k])is str and len(m[k])==64 and all(c in'0123456789abcdef'for c in m[k]),'FULL_SOURCE_IDENTITY')
    return m

def propose(global_bytes,local_bytes,grant_bytes,*,closed_readback,now_epoch):
    from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a,successor_cd as s
    grant(grant_bytes);g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    need(FIELD not in g and FIELD not in l,'ONCE_NO_READOPTION')
    p=closed_readback
    need(p['status']=='ACTUAL_75C63C6E_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
        and p['request_id']==FAILED and p['host']=='corelab-4080s'and p['uid']==2009
        and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
        and 0<=now_epoch-p['observed_epoch']<=120,'FRESH_REAL_DUAL_AND_THREE_CLOSES')
    need(g['unknown_effects']==g['pending_updates']==[]
        and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
        and all(r['status']=='NATIVE_TERMINAL'for r in g[c.FIELD].values()),'COLD_NO_ACTIVE_EFFECTS')
    need(set(p['leases'])=={'GPU','physical','control'}and all(v['status']=='CLOSED'
        and v['terminal_native_sha256']==TERMINAL and v['all_actual_workers_terminal']is True
        for v in p['leases'].values()),'ACTUAL_ALL_THREE_CLOSED')
    clock=s.validated(g)
    m=dict(schema='V5_CD_C50_GPU147_TO148_ONLY_AUTHORITY_V1',grant_sha256=GRANT_SHA,
        grant_base64=base64.b64encode(grant_bytes).decode(),before_H200_s=BEFORE,after_H200_s=AFTER,
        terminal_native_sha256=TERMINAL,before_GLOBAL_sha256=a.sha(global_bytes),before_LOCAL_sha256=a.sha(local_bytes),
        closed_readback_sha256=a.sha(a.raw(p)),adopted_epoch=now_epoch,new_native_admission=False,
        **{k:clock[k]for k in('T_CD','deadline_epoch','stop_new_training_epoch')})
    v,w=deepcopy(g),deepcopy(l);v[FIELD]=deepcopy(m);w[FIELD]=deepcopy(m);v['revision']+=1
    validate(v);validate(w);c.checked(v);c.paired(v,w)
    need(all(v[k]==x for k,x in g.items()if k!='revision')and all(w[k]==x for k,x in l.items()),'ONLY_APPEND_MARKER_REVISION')
    # A closed zero-model panel has no live reservation. Restoring its full
    # original hold is a distinct once continuation and supplies the forecast.
    return dict(proposed_global=v,proposed_LOCAL=w,new_native_admission=False)
