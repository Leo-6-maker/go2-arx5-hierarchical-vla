"""Exact A2 heartbeat-loss authorization; no dispatch, rollback or cost reset.

The global-cap caller first validates its original storage/global baseline.
This module intentionally does not change lost_save_recovery or call back
through storage.global_cap_from_ledger, so the new cap path cannot recurse.
"""
from copy import deepcopy
import math
from pathlib import Path

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.future_budget import _read
from . import lost_save_recovery as recovery
from . import segmented_training as seg
from . import storage_expansion as storage
from .authority import AUTHORITY, PARENT_AUTHORITY, HARD_CLOSE

RELEASE='d8908d7aeb4e01bb49016d1a1aabc801b91c9c86'
GRANT_SHA='b27902561f14d963bc1b307b6cbe7a73db6556b08ab52128f4eca5b928903f18'
PLAN='plans/A2_HEARTBEAT_RECOVERY_AUTHORIZATION.json'
FIELD='source_heartbeat_recovery'
LINEAGE='A_CONTROL_HEARTBEAT_RECOVERY2'
FAILED_JOB='1475719.stdct-mgmt-02'
FAILED_REQUEST='9d1582d5-1c87-41bf-ad53-4e049cc21fe9'
CLOSURE_SHA='22e1f8d50c54b9f9bfa83cde3ca0075d0472aafed375db4da1c4a50ec7ee3a77'
CONTROL_CLOSURE_SHA='abadf79ee98699adb57a2a52802c87485a534208565c75df6e13195aac97cb4b'
SEED_CHECKPOINT_SHA='1336326d908bbc91c36f871f0846b316c2bea08bada2bcee0b1bcc9bcb0e5486'
T0=1792348196
NEW_JOBS_CUTOFF=1792866596
APPROVED_EPOCH=1792687959.244639
GIB=1024**3
_MARKER_KEYS={'authority_release','grant_sha256','failed_job_id','failed_request_id',
    'failed_lineage_key','failed_lineage_last_native_step','known_irrecoverable_optimizer_updates',
    'replacement_lineage_key','seed_native_step','seed_checkpoint_sha256',
    'closed_native_receipt_sha256','closed_control_receipt_sha256','original_T0_epoch','adopted_epoch'}


def fail(reason):
    raise PermissionError('DEC1_HEARTBEAT_RECOVERY_'+reason)


def grant(raw):
    value,digest=_read(raw)
    if (digest!=GRANT_SHA or value.get('schema')!='VAP1_A2_HEARTBEAT_RECOVERY_AUTHORIZATION_V1'
            or value.get('status')!='HUMAN_AUTHORIZED_EXACT_PRESERVED_COST_RECOVERY'
            or value.get('narrow_changes')!={'DEC1_cumulative_updates':[9216,9987],
                'DEC1_new_PBS':[11,12],'DEC1_GPU_site_output_GiB':[352,418],
                'global_GPU_site_output_GiB':[419,485]}
            or value.get('original_T0_epoch')!=T0
            or value.get('original_clock')!={'stop_new_jobs_epoch':1792866596,
                'stop_training_epoch':1792909796,'stop_physics_epoch':1792924196,
                'hard_closeout_epoch':HARD_CLOSE}
            or value.get('failed_job')!=FAILED_JOB or value.get('failed_request')!=FAILED_REQUEST
            or value.get('closed_gpu_receipt_sha256')!=CLOSURE_SHA
            or value.get('closed_control_receipt_sha256')!=CONTROL_CLOSURE_SHA
            or value.get('seed_checkpoint2048_sha256')!=SEED_CHECKPOINT_SHA
            or value.get('prior_storage_release')!=storage.RELEASE
            or value.get('retirement_credit_bytes')!=0
            or value.get('all_checkpoints_preserved') is not True):
        fail('EXACT_FIXED_GRANT_AND_FOUR_NARROW_CHANGES')
    expected=dict(latest_complete_step=2048,saved_cursor=65536,failed_lineage_actual_step=2819,
        old_lost_updates=1024,new_lost_updates=771,all_charged_updates_before_recovery=3843,
        A_remaining_updates=2048,B_remaining_updates=4096,total_required_updates=9987,
        new_named_lineage_required=True,failed_lineage_or_cost_rollback=False,
        new_request_and_registration_required=True,B_independent_original_CORR=True,
        original12_A_probes_preserved_no_repeat=True,proposed_lineage_key=LINEAGE,
        seed_full_state_step=2048,failed_lineage_preserved=recovery.LINEAGE,
        seed_is_imported_state_not_new_charged_updates=True)
    if value.get('recovery_semantics')!=expected:
        fail('EXACT_PRESERVED_LOSS_AND_IMPORTED_STATE_SEMANTICS')
    return value


def enabled(row):
    keys=('heartbeat_recovery_release','heartbeat_recovery_grant_sha256')
    if not any(k in row for k in keys):return False
    if row.get(keys[0])!=RELEASE or row.get(keys[1])!=GRANT_SHA:
        fail('NO_PARTIAL_OR_UNBOUND_HEARTBEAT_AUTHORIZATION')
    return True


def caps(row):
    from . import zero_update_recovery as zero
    if zero.enabled(row):return zero.caps(row)
    if not enabled(row):return storage.caps(row)
    if not seg.enabled(row) or not storage.enabled(row):
        fail('NEW_CAP_REQUIRES_ORIGINAL_STORAGE_AND_SEGMENT_SCOPE')
    return 485*GIB,418*GIB


def lineage(row):
    if not enabled(row):return recovery.lineage(row)
    arm=row.get('arm',row.get('training_block',{}).get('arm'))
    expected=LINEAGE if arm=='A_CONTROL' else 'B_COVERAGE'
    if (arm not in ('A_CONTROL','B_COVERAGE')
            or row.get('recovery_authorization_release')!=recovery.RELEASE
            or row.get('recovery_adoption_sha256')!=recovery.GRANT_SHA
            or not seg.enabled(row) or not storage.enabled(row)
            or row.get('training_lineage_key')!=expected):
        fail('EXACT_OLD_RECOVERY_STORAGE_SEGMENT_AND_NEW_A_OR_ORIGINAL_B_LINEAGE')
    return expected


def _marker(h):
    m=h.get(FIELD)
    if (type(m) is not dict or set(m)!=_MARKER_KEYS
            or m.get('authority_release')!=RELEASE or m.get('grant_sha256')!=GRANT_SHA
            or m.get('failed_job_id')!=FAILED_JOB or m.get('failed_request_id')!=FAILED_REQUEST
            or m.get('failed_lineage_key')!=recovery.LINEAGE
            or m.get('failed_lineage_last_native_step')!=2819
            or m.get('known_irrecoverable_optimizer_updates')!=771
            or m.get('replacement_lineage_key')!=LINEAGE or m.get('seed_native_step')!=2048
            or m.get('seed_checkpoint_sha256')!=SEED_CHECKPOINT_SHA
            or m.get('closed_native_receipt_sha256')!=CLOSURE_SHA
            or m.get('closed_control_receipt_sha256')!=CONTROL_CLOSURE_SHA
            or m.get('original_T0_epoch')!=T0
            or type(m.get('adopted_epoch')) not in (int,float)
            or not math.isfinite(m['adopted_epoch'])
            or not APPROVED_EPOCH<=m['adopted_epoch']<NEW_JOBS_CUTOFF):
        fail('EXACT_COMPLETE_ADOPTED_MARKER_AND_ORIGINAL_CLOCK')
    return m


def _history(h, *, seeded):
    runs=h.get('training_runs',{})
    rows=[r for r in h.get('PBS_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    count=h.get('optimizer_updates')
    if (h.get('authority_release')!=PARENT_AUTHORITY
            or h.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or h.get('DEC1_source_pair_platform')!='VANDA_A40'
            or type(count) is not int or count<3843 or h.get('used',{}).get('optimizer_updates')!=count
            or runs.get('A_CONTROL')!=dict(last_native_step=1024,recipe_sha256=recovery.RECIPE)
            or runs.get(recovery.LINEAGE)!=dict(last_native_step=2819,recipe_sha256=recovery.RECIPE)
            or len(rows)!=1 or rows[0].get('job_id')!=FAILED_JOB
            or rows[0].get('status')!='NATIVE_TERMINAL'
            or rows[0].get('committed_optimizer_updates')!=771):
        fail('ORIGINAL_LOST1024_FAILED2819_AND_ALL3843_COSTS_PRESERVED')
    if seeded:
        run=runs.get(LINEAGE,{})
        if (set(run)!={'last_native_step','recipe_sha256'}
                or run.get('recipe_sha256')!=recovery.RECIPE
                or type(run.get('last_native_step')) is not int
                or not 2048<=run['last_native_step']<=4096):
            fail('EXPLICIT_IMPORTED2048_NEW_LINEAGE_NO_PROGRESS_ROLLBACK')
    return rows[0]


def validate_global(h):
    """Validate the new global extension after the caller's old 419GiB checks.

    Do not call storage.global_cap_from_ledger here: its authorized extension
    calls this function. The exact old storage marker remains mandatory.
    """
    previous=storage._marker(h)
    m=_marker(h);_history(h,seeded=True)
    if previous['adopted_epoch']>m['adopted_epoch']:
        fail('NEW_ADOPTION_MUST_FOLLOW_OLD_STORAGE_ADOPTION')
    return m


def validate_adopted(h,d):
    storage.validate_adopted(h,d)
    m=validate_global(h)
    rows=[r for r in d.get('new_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if (m!=d.get(FIELD) or h.get('optimizer_updates')!=d.get('optimizer_updates')
            or type(d.get('optimizer_updates')) is not int or d['optimizer_updates']<3843
            or len(rows)!=1 or rows[0].get('status')!='NATIVE_TERMINAL'
            or rows[0].get('job_id')!=FAILED_JOB):
        fail('BOTH_NATIVE_MARKERS_AND_FAILED_SUBMISSION_COSTS_AGREE')
    return m


def _zero(v):
    if isinstance(v,dict):return all(_zero(x) for x in v.values())
    if isinstance(v,(tuple,list)):return not v
    return type(v) in (int,float) and v==0


def _unreserved(v):
    return all(_zero(x) for k,x in v.items() if 'reserved' in k)


def adopt(h,d,*,grant_bytes,closure_bytes,control_closure_bytes,now_epoch):
    grant(grant_bytes);storage.validate_adopted(h,d)
    closure,closure_sha=_read(closure_bytes);control,control_sha=_read(control_closure_bytes)
    row=_history(h,seeded=False);effects=closure.get('effects',{});settlement=closure.get('settlement',{})
    gpu_close=control.get('GPU_close',{});b=h.get('training_runs',{}).get('B_COVERAGE')
    dr=[r for r in d.get('new_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if (FIELD in h or FIELD in d or LINEAGE in h.get('training_runs',{})
            or closure_sha!=CLOSURE_SHA or control_sha!=CONTROL_CLOSURE_SHA
            or closure.get('status')!='CLOSED_A2_HEARTBEAT_FAILURE_2819_PRESERVED'
            or closure.get('job_id')!=FAILED_JOB
            or closure.get('latest_complete_step')!=2048
            or closure.get('checkpoints',{}).get('2048',{}).get('complete_sha256')!=SEED_CHECKPOINT_SHA
            or closure.get('recoverable_full_state') is not False
            or closure.get('prior_complete_state_preserved') is not True
            or closure.get('retired_or_deleted_files')!=0
            or effects.get('job_id')!=FAILED_JOB or effects.get('request_id')!=FAILED_REQUEST
            or effects.get('source_sha')!='3a1420e76c492196fa83905f4bde92924cc12c28'
            or effects.get('arm')!='A_CONTROL' or effects.get('start_native_step')!=2048
            or effects.get('last_native_step')!=2819 or effects.get('optimizer_updates_committed')!=771
            or effects.get('query_attempts')!=17 or effects.get('model_load_attempts')!=2
            or effects.get('pending_updates')!=[] or effects.get('unknown_query_intents')!=[]
            or effects.get('recipe_sha256')!=recovery.RECIPE or effects.get('effect_journal_complete') is not True
            or settlement.get('updates')!=3843 or settlement.get('unknown_effects')!=[]
            or control.get('status')!='CLOSED_CPU_A2_HEARTBEAT_FAILURE_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT'
            or control.get('job_id')!=FAILED_JOB or control.get('new_A_recovery_step')!=2819
            or control.get('old_A1024_costs_preserved') is not True or control.get('unknown_effects')!=[]
            or control.get('new_GPU_PBS_models_queries_updates_physics')!=0
            or gpu_close.get('lease_status')!='CLOSED' or gpu_close.get('terminal_receipt_sha256')!=closure_sha
            or gpu_close.get('terminal_receipt')!=closure or control.get('settled_submission')!=row
            or len(dr)!=1 or dr[0].get('status')!='NATIVE_TERMINAL' or dr[0].get('job_id')!=FAILED_JOB
            or h.get('optimizer_updates')!=3843 or d.get('optimizer_updates')!=3843
            or b is not None and b!=dict(last_native_step=0,recipe_sha256=recovery.MATCHED_RECIPES['B_COVERAGE'])
            or h.get('unknown_effects')!=[] or d.get('unknown_effects')!=[]
            or h.get('pending_updates')!=[] or d.get('pending_updates',[])!=[]
            or not _unreserved(h) or not _unreserved(d)
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for r in h.get('PBS_submissions',[]))
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for r in d.get('new_submissions',[]))
            or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED_EPOCH<=now_epoch<NEW_JOBS_CUTOFF
            or type(closure.get('closed_epoch')) not in (int,float)
            or type(control.get('closed_epoch')) not in (int,float)
            or not math.isfinite(closure['closed_epoch']) or not math.isfinite(control['closed_epoch'])
            or not T0<=max(closure['closed_epoch'],control['closed_epoch'])<=now_epoch):
        fail('ONCE_EXACT_CLOSED_HEARTBEAT_LOSS_NO_RESERVATION_UNKNOWN_OR_COST_RESET')
    result,dec=deepcopy(h),deepcopy(d)
    marker=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=FAILED_JOB,
        failed_request_id=FAILED_REQUEST,failed_lineage_key=recovery.LINEAGE,
        failed_lineage_last_native_step=2819,known_irrecoverable_optimizer_updates=771,
        replacement_lineage_key=LINEAGE,seed_native_step=2048,seed_checkpoint_sha256=SEED_CHECKPOINT_SHA,
        closed_native_receipt_sha256=closure_sha,closed_control_receipt_sha256=control_sha,
        original_T0_epoch=T0,adopted_epoch=now_epoch)
    result[FIELD]=deepcopy(marker);dec[FIELD]=deepcopy(marker)
    result['training_runs'][LINEAGE]=dict(last_native_step=2048,recipe_sha256=recovery.RECIPE)
    validate_adopted(result,dec)
    return result,dec


def admitted(root,manifest,request):
    if not enabled(request):return False
    lineage(request)
    if storage.admitted(root,manifest,request) is not True:
        fail('ORIGINAL_STORAGE_AND_SEGMENT_SEALED_ADMISSION_REQUIRED')
    path=storage._readonly(Path(root)/PLAN)
    if (sha(path)!=GRANT_SHA
            or manifest.get('authority',{}).get('heartbeat_recovery_release')!=RELEASE
            or manifest.get('authority',{}).get('heartbeat_recovery_grant_sha256')!=GRANT_SHA
            or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or request.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA):
        fail('EXACT_MANIFEST_REQUEST_AND_READONLY_HEARTBEAT_GRANT')
    raw=path.read_bytes();grant(raw)
    native=storage._readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name)
    if native.read_bytes()!=raw:fail('EXACT_SAME_NATIVE_HEARTBEAT_GRANT_BYTES')
    return True
