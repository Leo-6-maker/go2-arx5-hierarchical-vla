"""Exact zero-update continuation; only adopt adds two identical markers.

No runner, transport, dispatch, lineage creation, cost refund or file write.
HR validators used here must remain baseline validators (no callback here).
"""
from copy import deepcopy
import math
from pathlib import Path
import stat

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.future_budget import _read
from scripts.pi0294.dec1 import heartbeat_recovery as HR
from scripts.pi0294.dec1 import lost_save_recovery as recovery

RELEASE='871094ee31b18c47183c7403459ebd5fdad88b1f'
GRANT_SHA='f32922b15138816517de3e9bd63d4c413698f74b49c20908bb1d4dfea9a279a8'
PLAN='plans/ZERO_UPDATE_RESUME_AUTHORIZATION.json'
FIELD='source_zero_update_resume'
FAILED_JOB='1477127.stdct-mgmt-02'
FAILED_REQUEST='bbb04faa-fd21-4dbd-af73-1ee8e4c0661b'
FAILED_SOURCE='df6427f791fdbe8f175906f240eb3413bf096cbf'
CLOSURE_SHA='fb0a148dc06b7ba1fecd02158d7ec2f76e5b4617e94ae8ae08acd3ad1293cc93'
CONTROL_CLOSURE_SHA='a5f854b15f73a2d9c98db6e1728bf68e1653b3c52a432e677a3fc4ebd81f6349'
REQUEST_SHA='4b40b38de8142245880e6f55e9325ce8c86b27aebeb3a8de572ab8f7bf5b7b26'
NATIVE_SHA='376d093505f3329f1836ddb76b314426b9fc0e2c4244ec99808c91fe23117214'
EFFECT_SHA='cf3a444185a6d111d425fe04fc65b1556567cf5bf9122518938218b6a3526e8b'
APPROVED_EPOCH=1792695381.3922973
T0=1792348196
CUTOFF=1792866596
GIB=1024**3
SEMANTICS=dict(original_A2048_seed=True,cursor=65536,new_request_new_registration_new_lease_required=True,
    reuse_logical_A_CONTROL_HEARTBEAT_RECOVERY2_lineage_with_zero_new_effects=True,
    no_closed_request_or_owner_restart=True,A_remaining_updates=2048,
    B_independent_original_CORR_remaining_updates=4096,all_cumulative_updates_at_completion=9987,
    old1024_2819_and_zero_update_failure_costs_preserved=True,all_checkpoints_retained=True,no_repeat_A12probes=True)
MARKER_KEYS={'authority_release','grant_sha256','failed_job_id','failed_request_id','failed_source_sha',
    'closed_native_receipt_sha256','closed_control_receipt_sha256','logical_lineage_key',
    'original_seed_step','original_seed_checkpoint_sha256','original_T0_epoch','adopted_epoch'}

def fail(reason):raise PermissionError('DEC1_ZERO_UPDATE_RESUME_'+reason)

def grant(raw):
    value,digest=_read(raw)
    if (RELEASE=='TO_BE_PINNED' or digest!=GRANT_SHA
            or value.get('schema')!='VAP1_ZERO_UPDATE_RESUME_AUTHORIZATION_V1'
            or value.get('status')!='HUMAN_AUTHORIZED_EXACT_PRESERVED_COST_RESUME'
            or value.get('approved_epoch')!=APPROVED_EPOCH
            or value.get('narrow_changes')!={'DEC1_new_PBS':[12,13],
                'DEC1_GPU_site_output_GiB':[418,483],'global_GPU_site_output_GiB':[485,550]}
            or value.get('prior_heartbeat_release')!=HR.RELEASE or value.get('prior_heartbeat_grant_sha256')!=HR.GRANT_SHA
            or value.get('failed_job')!=FAILED_JOB or value.get('failed_request')!=FAILED_REQUEST
            or value.get('failed_source')!=FAILED_SOURCE
            or value.get('closed_gpu_receipt_sha256')!=CLOSURE_SHA
            or value.get('closed_control_receipt_sha256')!=CONTROL_CLOSURE_SHA
            or value.get('seed_checkpoint2048_sha256')!=HR.SEED_CHECKPOINT_SHA
            or value.get('original_T0_epoch')!=T0
            or value.get('original_clock')!=dict(stop_new_jobs_epoch=CUTOFF,stop_training_epoch=1792909796,
                stop_physics_epoch=1792924196,hard_closeout_epoch=1792952996)
            or value.get('recovery_semantics')!=SEMANTICS or value.get('updates_cap_unchanged')!=9987
            or value.get('global_PBS_cap_unchanged')!=24 or value.get('all_checkpoints_preserved') is not True
            or value.get('retirement_credit_bytes')!=0):fail('EXACT_FIXED_GRANT_THREE_CHANGES_AND_PRESERVED_LINEAGE')
    return value

def enabled(row):
    keys=('zero_update_resume_release','zero_update_resume_grant_sha256')
    if not any(k in row for k in keys):return False
    if RELEASE=='TO_BE_PINNED' or row.get(keys[0])!=RELEASE or row.get(keys[1])!=GRANT_SHA:
        fail('NO_PARTIAL_OR_UNBOUND_NEW_SCOPE')
    return True

def caps(row):
    # HR.caps will delegate the new scope here. Never call it back.
    if not enabled(row):
        if not HR.enabled(row):return HR.storage.caps(row)
        if not HR.seg.enabled(row) or not HR.storage.enabled(row):fail('ORIGINAL_HEARTBEAT_CAP_SCOPE')
        return 485*GIB,418*GIB
    if not HR.enabled(row):fail('ORIGINAL_HEARTBEAT_SCOPE_REQUIRED')
    HR.lineage(row)
    return 550*GIB,483*GIB

def pbs_limit(row):
    if enabled(row):caps(row);return 13
    if HR.enabled(row):return 12
    from scripts.pi0294.dec1.segmented_training import enabled as segmented
    return 11 if segmented(row) else 8

def _history(h):
    rows=[r for r in h.get('PBS_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if len(rows)!=1:fail('ONE_PRESERVED_FAILED_ROW')
    row=rows[0]
    if (row.get('status')!='NATIVE_TERMINAL' or row.get('job_id')!=FAILED_JOB
            or row.get('source_sha')!=FAILED_SOURCE or row.get('request_sha256')!=REQUEST_SHA
            or row.get('training_lineage_key')!=HR.LINEAGE or row.get('arm')!='A_CONTROL'
            or row.get('start_native_step')!=2048 or row.get('committed_optimizer_updates')!=0
            or row.get('query_attempts')!=0 or row.get('reserved_optimizer_updates')!=0
            or row.get('reserved_neural_queries')!=0 or row.get('terminal_native_sha256')!=NATIVE_SHA
            or row.get('terminal_effect_summary_sha256')!=EFFECT_SHA
            or row.get('conservative_H200_s')!=754 or row.get('conservative_CPU_s')!=259
            or h.get('used',{}).get('H200_s',0)<49117 or h.get('CPU_s',0)<259
            or h.get('generation_queries',0)<1134):fail('EXACT_ZERO_EFFECT_ROW_AND_PAID_COSTS_RETAINED')
    return row

def _marker(h):
    m=h.get(FIELD)
    expected=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=FAILED_JOB,
        failed_request_id=FAILED_REQUEST,failed_source_sha=FAILED_SOURCE,closed_native_receipt_sha256=CLOSURE_SHA,
        closed_control_receipt_sha256=CONTROL_CLOSURE_SHA,logical_lineage_key=HR.LINEAGE,
        original_seed_step=2048,original_seed_checkpoint_sha256=HR.SEED_CHECKPOINT_SHA,original_T0_epoch=T0)
    if (RELEASE=='TO_BE_PINNED' or type(m) is not dict or set(m)!=MARKER_KEYS
            or any(m.get(k)!=v for k,v in expected.items()) or type(m.get('adopted_epoch')) not in (int,float)
            or not math.isfinite(m['adopted_epoch']) or not APPROVED_EPOCH<=m['adopted_epoch']<CUTOFF):
        fail('EXACT_NEW_MARKER')
    return m

def validate_global(h):
    HR.validate_global(h)  # Baseline only: HR must not call back into here.
    m=_marker(h);_history(h)
    if h[HR.FIELD]['adopted_epoch']>m['adopted_epoch']:fail('NEW_AFTER_OLD_ADOPTION')
    return m

def validate_adopted(h,d):
    HR.validate_adopted(h,d)  # Baseline only: do not redirect HR validation here.
    m=validate_global(h);row=_history(h)
    rows=[r for r in d.get('new_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if (m!=d.get(FIELD) or len(rows)!=1 or rows[0].get('status')!='NATIVE_TERMINAL'
            or rows[0].get('job_id')!=row['job_id']):fail('BOTH_MARKERS_AND_FAILED_NATIVE_ROWS')
    return m

def _zero(value):
    if isinstance(value,dict):return all(_zero(v) for v in value.values())
    if isinstance(value,(list,tuple)):return not value
    return type(value) in (int,float) and value==0

def _closed_known(h,d):
    if any(v.get('unknown_effects')!=[] or v.get('pending_updates',[])!=[]
            or any(not _zero(x) for k,x in v.items() if 'reserved' in k)
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')
                for r in v.get('PBS_submissions' if v is h else 'new_submissions',[])) for v in (h,d)):
        fail('ALL_CLOSED_UNRESERVED_KNOWN_EFFECTS')

def adopt(h,d,*,grant_bytes,closure_bytes,control_closure_bytes,now_epoch):
    grant(grant_bytes);HR.validate_adopted(h,d)
    closure,cs=_read(closure_bytes);control,ccs=_read(control_closure_bytes)
    row=_history(h);e=closure.get('effects',{});s=closure.get('settlement',{});gc=control.get('GPU_close',{})
    _closed_known(h,d)
    if (FIELD in h or FIELD in d or cs!=CLOSURE_SHA or ccs!=CONTROL_CLOSURE_SHA
            or h.get('optimizer_updates')!=3843 or d.get('optimizer_updates')!=3843
            or h.get('training_runs',{}).get(HR.LINEAGE)!=dict(last_native_step=2048,recipe_sha256=recovery.RECIPE)
            or h.get('training_runs',{}).get('B_COVERAGE') not in
                (None,dict(last_native_step=0,recipe_sha256=recovery.MATCHED_RECIPES['B_COVERAGE']))
            or closure.get('status')!='CLOSED_RECOVERY2_PREMODEL_HEARTBEAT_FAILURE_PRESERVED'
            or closure.get('job_id')!=FAILED_JOB or closure.get('latest_complete_step')!=2048
            or closure.get('checkpoints')!={} or closure.get('recoverable_full_state') is not False
            or closure.get('prior_complete_state_preserved') is not True
            or closure.get('committed_updates_without_new_complete_state')!=0 or closure.get('retired_or_deleted_files')!=0
            or e.get('schema')!='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1'
            or e.get('job_id')!=FAILED_JOB or e.get('request_id')!=FAILED_REQUEST
            or e.get('request_sha256')!=REQUEST_SHA or e.get('source_sha')!=FAILED_SOURCE
            or e.get('arm')!='A_CONTROL' or e.get('start_native_step')!=2048 or e.get('last_native_step')!=2048
            or e.get('optimizer_updates_committed')!=0 or e.get('query_attempts')!=0 or e.get('model_load_attempts')!=0
            or e.get('pending_updates')!=[] or e.get('unknown_query_intents')!=[]
            or e.get('effect_journal_complete') is not True or e.get('recipe_sha256')!=recovery.RECIPE
            or e.get('new_physics')!=0 or e.get('new_episodes')!=0
            or s.get('status')!='EXACT_RECOVERY2_PREMODEL_HEARTBEAT_FAILURE_SETTLED'
            or s.get('job_id')!=FAILED_JOB or s.get('request_id')!=FAILED_REQUEST
            or s.get('updates')!=3843 or s.get('unknown_effects')!=[]
            or control.get('status')!='CLOSED_CPU_RECOVERY2_PREMODEL_FAILURE_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT'
            or control.get('job_id')!=FAILED_JOB or control.get('old_A1024_costs_preserved') is not True
            or control.get('new_A_recovery_step')!=2048 or control.get('unknown_effects')!=[]
            or control.get('new_GPU_PBS_models_queries_updates_physics')!=0
            or control.get('settled_submission')!=row or gc.get('lease_status')!='CLOSED'
            or gc.get('terminal_receipt_sha256')!=cs or gc.get('terminal_receipt')!=closure
            or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED_EPOCH<=now_epoch<CUTOFF
            or any(type(v.get('closed_epoch')) not in (int,float) or not math.isfinite(v['closed_epoch'])
                or not T0<=v['closed_epoch']<=now_epoch for v in (closure,control))):
        fail('EXACT_CLOSED_ZERO_EFFECTS_NO_SEED_OR_COST_REWRITE')
    result,dec=deepcopy(h),deepcopy(d)
    m=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=FAILED_JOB,failed_request_id=FAILED_REQUEST,
        failed_source_sha=FAILED_SOURCE,closed_native_receipt_sha256=cs,closed_control_receipt_sha256=ccs,
        logical_lineage_key=HR.LINEAGE,original_seed_step=2048,original_seed_checkpoint_sha256=HR.SEED_CHECKPOINT_SHA,
        original_T0_epoch=T0,adopted_epoch=now_epoch)
    result[FIELD]=deepcopy(m);dec[FIELD]=deepcopy(m)
    validate_adopted(result,dec)
    return result,dec

def _readonly(path):
    path=Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode&0o222:
        fail('REGULAR_READONLY_GRANT')
    return path

def admitted(root,manifest,request):
    if not enabled(request):return False
    caps(request)  # Do not call HR.admitted: its integration may call here.
    path=_readonly(Path(root)/PLAN);raw=path.read_bytes()
    if (sha(path)!=GRANT_SHA or manifest.get('authority',{}).get('zero_update_resume_release')!=RELEASE
            or manifest.get('authority',{}).get('zero_update_resume_grant_sha256')!=GRANT_SHA
            or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or request.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA):fail('SEALED_MANIFEST_REQUEST_GRANT')
    grant(raw)
    native=_readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name)
    if native.read_bytes()!=raw:fail('EXACT_NATIVE_AUTHORITY_BYTES')
    return True
