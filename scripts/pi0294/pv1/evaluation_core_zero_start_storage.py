"""Exact human-approved two storage cap amendment; pure once adoption only."""
from copy import deepcopy
import hashlib,json,math

FIELD='Core_B_zero_start_two_storage_caps'
GRANT_SHA='8f5998df61c2935c8cd3248e6cb4338150dfea271659931fcf17efad161d6101'
SCHEMA='V4A_B_ZERO_START_TWO_STORAGE_CAP_GRANT_V1'
OLD={'Hopper_storage_bytes':810*1024**3,'Core_storage_bytes':300*1024**3}
CAPS={'Hopper_storage_bytes':811*1024**3,'Core_storage_bytes':329*1024**3}
T0=1792348196;STOP_NEW=1792866596

def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_ZERO_STORAGE_'+why)

def grant(raw):
    need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==GRANT_SHA,'EXACT_PUBLISHED_HUMAN_GRANT_BYTES')
    value=json.loads(raw);need(value['schema']==SCHEMA and value['approved']=={'GPU_site_storage_GiB':{'before':810,'after':811},'Core_storage_GiB':{'before':300,'after':329}},'ONLY_THE_TWO_APPROVED_CAPS')
    return value

def validate(g):
    from . import evaluation_scope as scope
    scope.validate_global(g)
    m=g.get(FIELD)
    need(type(m) is dict and set(m)=={'schema','grant_sha256','before_sha256','adopted_epoch','caps'}
         and m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA and m['caps']==CAPS
         and type(m['before_sha256']) is dict and set(m['before_sha256'])=={'GLOBAL','LOCAL'}
         and all(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in m['before_sha256'].values())
         and type(m['adopted_epoch']) in (int,float) and math.isfinite(m['adopted_epoch'])
         and T0<=m['adopted_epoch']<STOP_NEW,'EXACT_ONCE_ADOPTED_TWO_STORAGE_CAP_MARKER')
    return m

def adopt(global_bytes,LOCAL_bytes,grant_bytes,*,now_epoch):
    from . import evaluation_core_accounting as c
    from .evaluation_core_zero_start_recovery import FAILED,PREVIOUS_A
    grant(grant_bytes);g,l=json.loads(global_bytes),json.loads(LOCAL_bytes)
    c.checked(g);c.scope.validate_pair(g,l)
    need(FIELD not in g and FIELD not in l and not g['unknown_effects'] and not g['pending_updates']
         and not l['unknown_effects'] and not l['pending_updates'] and not l['pending_queries']
         and set(g[c.FIELD])=={FAILED,PREVIOUS_A}
         and all(r['status']==c.STATES[3] for r in g[c.FIELD].values())
         and l['Core_native_GPU_reservations'][FAILED]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL'
         and all(r['status']=='CLOSED' for r in g['V4_DEV_control_reservations'].values())
         and T0<=now_epoch<STOP_NEW,'INACTIVE_KNOWN_CLOSED_PAIR_ONE_ADOPTION_NO_REPLAY')
    g,l=deepcopy(g),deepcopy(l)
    marker=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,before_sha256=dict(GLOBAL=hashlib.sha256(global_bytes).hexdigest(),LOCAL=hashlib.sha256(LOCAL_bytes).hexdigest()),adopted_epoch=now_epoch,caps=dict(CAPS))
    for v in (g,l):v[FIELD]=deepcopy(marker)
    c.checked(g);c.scope.validate_pair(g,l)
    return dict(proposed_global=g,proposed_LOCAL=l,marker=marker,new_compute=False,IO_performed=False)
