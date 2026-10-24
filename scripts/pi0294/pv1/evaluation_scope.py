"""Exact human-approved V4 DEV resource package; pure metadata proposals.

No dispatch, model release, training permission, retry or clock extension.
Both existing ledgers must adopt the same marker before a new DEV request.
"""
from copy import deepcopy
import hashlib
import json
import math

RELEASE='108ab0a09790fa148d72f6792815de9453054ecb'
GRANT_SHA='1e41c5d745e2e600d608c52ce72ef8c5dfef3b6e1acc89476e0526f3a2db0f4f'
APPROVED=1792828822.3443167
FIELD='V4_S2_global_resource_and_saved_B4096_reuse'
CAMPAIGN='20261018T182956Z-localpc-pv1'
T0=1792348196
STOP_NEW=1792866596
HARD_CLOSE=1792952996
HISTORY_SHA='1ef8841d614631add59b35d247b9c728fd50efa219ee0f12b050dfdcc02b7f96'
RUNS_SHA='62f8c82e535b9536ff79d8bb58565002d028da76eb529e4918b56f09836b0202'
B_CHECKPOINT='be7ab71dee0fb68e17efae06a04ff8dde9cd776ff429c1d6c408ae01323a9c19'
B_CLOSURE='e875123dd3f6e713ce24d3d0f5f7a294dde3ee64cd86723135b1aacda3462519'
OLD=dict(H200_s=96*3600,PBS_submissions=28,Hopper_CPU_s=160*3600,Hopper_storage_bytes=680*1024**3)
CAPS=dict(H200_s=121*3600,PBS_submissions=34,Hopper_CPU_s=213*3600,Hopper_storage_bytes=810*1024**3)


def need(ok,why):
    if not ok:raise PermissionError('PV1_V4_EXACT_SCOPE_'+why)


def raw(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(raw(value)).hexdigest()


def grant(data):
    need(type(data) is bytes and hashlib.sha256(data).hexdigest()==GRANT_SHA,'PUBLISHED_HUMAN_GRANT_BYTES')
    value=json.loads(data)
    need(value['human_answer']=='批准整个工作包' and value['approval_recorded_epoch']==APPROVED
         and value['status']=='HUMAN_AUTHORIZED_EXACT_PACKAGE_NOT_NATIVE_GO'
         and value['saved_B4096_reuse_exception']['human_authorized'] is True,
         'DIRECT_ANSWER_NOT_PROPOSAL_OR_NATIVE_GO')
    return value


def history(value):
    """The original 25 rows cannot be edited, dropped or made active again."""
    rows=value.get('PBS_submissions')
    need(type(rows) is list and len(rows)>=25 and digest(rows[:25])==HISTORY_SHA
         and all(r['status']=='NATIVE_TERMINAL' for r in rows[:25]),'PRESERVE_EXACT_ALL25_CLOSED_ROWS')


def marker(now,before):
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,adopted_epoch=now,
        original_T0_epoch=T0,hard_close_epoch=HARD_CLOSE,
        original_history25_sha256=HISTORY_SHA,original_training_runs_sha256=RUNS_SHA,
        before_sha256=deepcopy(before),old_global_caps=deepcopy(OLD),authorized_global_caps=deepcopy(CAPS),
        saved_B4096_checkpoint_sha256=B_CHECKPOINT,saved_B4096_closed_proof_sha256=B_CLOSURE,
        outer_B_owner_exit=1,read_only_original_DEV_reuse_only=True,
        DEC1_and_other_caps_unchanged=True,new_training_authorized=False,
        native_or_scientific_admission=False)


def validate_global(value):
    # Do not call global cap/check functions here: they call this validator.
    from scripts.pi0294.dec1 import b_precuda_retry as prior
    prior.validate_global(value);history(value)
    need(value.get('campaign_run_uuid')==CAMPAIGN and digest(value.get('training_runs'))==RUNS_SHA,
         'ORIGINAL_CAMPAIGN_AND_FROZEN_TRAINING_PROGRESS')
    m=value.get(FIELD);now=m.get('adopted_epoch') if type(m) is dict else None
    before=m.get('before_sha256') if type(m) is dict else None
    need(type(now) in (int,float) and math.isfinite(now) and APPROVED<=now<STOP_NEW
         and type(before) is dict and set(before)=={'GLOBAL','LOCAL'}
         and all(type(s) is str and len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in before.values())
         and m==marker(now,before),'EXACT_ADOPTED_MARKER')
    return m


def validate_pair(global_ledger,local):
    m=validate_global(global_ledger);history(local)
    need(local.get('run_uuid')==CAMPAIGN and local.get('T0_epoch')==T0
         and local.get('hard_closeout_epoch')==HARD_CLOSE and local.get(FIELD)==m,
         'SAME_NATIVE_AND_LOCAL_MARKERS_OR_NO_NEW_REQUEST')
    return m


def caps(global_ledger):
    if FIELD in global_ledger:validate_global(global_ledger);return dict(CAPS)
    return dict(OLD)


def _zero_reserved(value):
    def zero(v):
        if type(v) is dict:return all(zero(x) for x in v.values())
        if type(v) is list:return not v
        return type(v) in (int,float) and v==0
    return all(zero(v) for k,v in value.items() if 'reserved' in k.lower())


def adopt(global_bytes,local_bytes,grant_bytes,closure_bytes,*,now_epoch):
    """Return two existing-ledger proposals. Caller persists each with once IO."""
    grant(grant_bytes)
    need(type(global_bytes) is bytes and type(local_bytes) is bytes and type(closure_bytes) is bytes,
         'RAW_CURRENT_BYTES_NOT_SUMMARY')
    from .future_budget import _read
    g,l=_read(global_bytes)[0],_read(local_bytes)[0]
    from .evaluation_accounting import checked
    checked(g);history(g);history(l)
    need(FIELD not in g and FIELD not in l and len(g['PBS_submissions'])==len(l['PBS_submissions'])==25
         and digest(g['training_runs'])==RUNS_SHA and g['PBS_submissions']==l['PBS_submissions']
         and g['used']['H200_s']==l['GPU_allocated_s']==103148
         and g['optimizer_updates']==l['optimizer_updates']==11714
         and g['generation_queries']==l['neural_queries']==1231
         and g['CPU_s']==l['CPU_s']['Hopper']==156569.658597758
         and g['new_storage_current_bytes']==g['new_storage_peak_bytes']==677711804228
         and _zero_reserved(g) and _zero_reserved(l)
         and g['pending_updates']==g['unknown_effects']==l['pending_updates']==l['pending_queries']==l['unknown_effects']==l['Hopper_unknown_effects']==[]
         and l['status']=='ACTIVE_CONTROL_PLANE' and l['run_uuid']==CAMPAIGN
         and l['T0_epoch']==T0 and l['hard_closeout_epoch']==HARD_CLOSE,
         'ONCE_COMPLETE_INACTIVE_25ROW_BOUNDARY_KEEP_ALL_COSTS')
    need(hashlib.sha256(closure_bytes).hexdigest()==B_CLOSURE,'EXACT_INDEPENDENT_B4096_DUAL_CLOSURE_BYTES')
    c=json.loads(closure_bytes)
    need(c['status']=='ACTUAL1481574_F0_FAILED_READER_FINANCIALLY_SETTLED_BOTH_VANDA_LEASES_CLOSED'
         and c['original_owner_success'] is False and c['updates']==11714 and c['queries']==1231,
         'PRESERVE_OUTER_OWNER_FAILURE_NOT_RELABEL_SUCCESS')
    before=dict(GLOBAL=hashlib.sha256(global_bytes).hexdigest(),LOCAL=hashlib.sha256(local_bytes).hexdigest())
    g,l=deepcopy(g),deepcopy(l);m=marker(now_epoch,before)
    g[FIELD]=deepcopy(m);l[FIELD]=deepcopy(m)
    validate_pair(g,l);checked(g)
    return dict(proposed_global=g,proposed_LOCAL=l,marker=m,native_dispatch_admitted=False,
        revision_or_IO_performed=False)


def require_B_reuse(global_ledger,local,*,checkpoint_sha256,closure_bytes):
    validate_pair(global_ledger,local)
    need(checkpoint_sha256==B_CHECKPOINT and type(closure_bytes) is bytes
         and hashlib.sha256(closure_bytes).hexdigest()==B_CLOSURE,'ONLY_EXACT_CLOSED_B4096_DEV_REUSE')
    return dict(checkpoint_manifest_sha256=B_CHECKPOINT,original_owner_exit=1,
        native_model_release=False,physics_authorized=False,read_only_DEV_reuse=True)
