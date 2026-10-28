"""Exact human-approved B1024 import after the settled1829 failure.

Pure authorization/accounting only. Old A/B histories and all costs are
immutable; imported optimizer state is not a new charged update.
"""
from copy import deepcopy
import math
from pathlib import Path
from scripts.pi0294.pv1.future_budget import _read
from scripts.pi0294.native_dataset import sha
from . import zero_update_recovery as zero

RELEASE='09ae7896b90e4a9447ee401a2c1b5763a0fa745d'
GRANT_SHA='c122a853baf84b49a00af89caa24ffea72690fc6791e18a8112b2345a2ecc8a2'
PLAN='plans/B1_HEARTBEAT_RECOVERY_AUTHORIZATION.json'
FIELD='source_B_heartbeat_recovery'
SCOPE_KEYS=('B_heartbeat_recovery_release','B_heartbeat_recovery_grant_sha256')
LINEAGE='B_COVERAGE_HEARTBEAT_RECOVERY1'
FAILED_JOB='1478518.stdct-mgmt-02'
FAILED_REQUEST='22cdd5d4-c735-4dfc-901c-3b9dda2ca246'
FAILED_SOURCE='0cf86cbc67a3a9639488d1b5080284f45ddad40e'
REQUEST_SHA='ff5aa468d934c04d383208257adc278f65def4aaa047e386c3a77862cd319d49'
CLOSURE_SHA='fa35994a6245d963bc8687d1f42ad7c43f14352fb32efb76a595990268677505'
CONTROL_CLOSURE_SHA='08c24af740062c92570ca0a256ac7e561e24367a67ed9e44f0615dc722801c92'
SEED_SHA='7da8aad42b835edb1181c38daaa1859ffb94516bce693609ebde4e2c10a38e2b'
PROBE_SHA='074ffbcce72e9917caecd70a2855ca67eb9b60afbb20dd6daf9ab98343130154'
RECIPE='eebe7c57a25d5a1672659457b9d9905d059123d6b5961eeaad843d6e05ff4ac9'
APPROVED_EPOCH=1792725773
T0=1792348196
CUTOFF=1792866596
GIB=1024**3
CHANGES={'DEC1_cumulative_updates':[9987,10792],'DEC1_new_PBS':[13,14],
    'global_PBS':[24,25],'DEC1_GPU_site_output_GiB':[483,549],
    'global_GPU_site_output_GiB':[550,615]}
SEMANTICS=dict(new_named_lineage=LINEAGE,new_request_new_registration_new_leases=True,
    seed_step=1024,seed_is_imported_not_new_updates=True,original_CORR0_B_model_family_unchanged=True,
    A_weights_or_optimizer_inherited=False,first_segment=[1024,2048],second_segment=[2048,4096],
    remaining_updates=3072,required_cumulative_updates=10792,
    original_12_B1024_probes_preserved_no_repeat=True,
    original_B_arm_recipe_R0_WBC_S1_gripper_scoring_unchanged=True)


def fail(reason):raise PermissionError('DEC1_B_HEARTBEAT_RECOVERY_'+reason)


def grant(raw):
    v,digest=_read(raw)
    if (digest!=GRANT_SHA or v.get('schema')!='VAP1_B1_HEARTBEAT_RECOVERY_AUTHORIZATION_V1'
            or v.get('status')!='HUMAN_AUTHORIZED_EXACT_PRESERVED_COST_RECOVERY'
            or v.get('approval_recorded_epoch')!=APPROVED_EPOCH or v.get('narrow_changes')!=CHANGES
            or v.get('approved_draft_release')!='44a2397bdafb64c0d6ea6da8d05607ff9747e178'
            or v.get('approved_draft_sha256')!='f4ba8f060156db85e0383e3aeb2aecb5456ee378f828739e020e4b0c6aae504b'
            or v.get('prior_zero_update_release')!=zero.RELEASE
            or v.get('prior_zero_update_grant_sha256')!=zero.GRANT_SHA
            or v.get('failed_job')!=FAILED_JOB or v.get('failed_request')!=FAILED_REQUEST
            or v.get('failed_source')!=FAILED_SOURCE or v.get('closed_gpu_receipt_sha256')!=CLOSURE_SHA
            or v.get('closed_control_receipt_sha256')!=CONTROL_CLOSURE_SHA
            or v.get('latest_full_state')!=dict(step=1024,cursor=32768,members=15,
                checkpoint_manifest_sha256=SEED_SHA,origin_job_id=FAILED_JOB,original_12_probe_result_sha256=PROBE_SHA)
            or v.get('old_B_lineage_preserved')!=dict(name='B_COVERAGE',actual_failed_step=1829,
                paid_updates=1829,unrecoverable_since_full_state=805)
            or v.get('recovery_semantics')!=SEMANTICS or v.get('all_charged_updates_before_recovery')!=7720
            or v.get('original_T0_epoch')!=T0 or v.get('original_clock')!=dict(stop_new_jobs_epoch=CUTOFF,
                stop_training_epoch=1792909796,stop_physics_epoch=1792924196,hard_closeout_epoch=1792952996)
            or v.get('original_final')!=dict(GPU_s=72000,Hopper_storage_bytes=40*GIB)
            or v.get('other_caps_recipe_R0_WBC_S1_scoring_unchanged') is not True
            or v.get('all_checkpoints_preserved') is not True or v.get('retirement_credit_bytes')!=0):
        fail('EXACT_FIVE_CHANGE_GRANT_AND_PRESERVED805_LOSS')
    return v


def enabled(row):
    if not any(k in row for k in SCOPE_KEYS):return False
    if tuple(row.get(k) for k in SCOPE_KEYS)!=(RELEASE,GRANT_SHA):fail('NO_PARTIAL_OR_UNBOUND_SCOPE')
    return True


def lineage(row):
    from . import b_vanda_recovery2 as vanda2
    if vanda2.enabled(row):return vanda2.lineage(row)
    if not enabled(row):fail('EXPLICIT_NEW_B_SCOPE')
    arm=row.get('arm',row.get('training_block',{}).get('arm'))
    if (arm!='B_COVERAGE' or row.get('training_lineage_key')!=LINEAGE
            or not zero.enabled(row) or not zero.HR.enabled(row)
            or row.get('recovery_authorization_release')!=zero.recovery.RELEASE
            or row.get('recovery_adoption_sha256')!=zero.recovery.GRANT_SHA
            or not zero.HR.seg.enabled(row) or not zero.HR.storage.enabled(row)):
        fail('ONLY_SAME_B_ARM_RECIPE_NEW_LINEAGE_WITH_ALL_PRIOR_SCOPES')
    return LINEAGE


def caps(row):
    lineage(row)
    from . import b_vanda_recovery2 as vanda2
    if vanda2.enabled(row):return 680*GIB,614*GIB
    return 615*GIB,549*GIB


def _marker_value(now):
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=FAILED_JOB,
        failed_request_id=FAILED_REQUEST,failed_lineage_key='B_COVERAGE',failed_lineage_last_native_step=1829,
        known_irrecoverable_optimizer_updates=805,replacement_lineage_key=LINEAGE,seed_native_step=1024,
        seed_checkpoint_sha256=SEED_SHA,original_probe_result_sha256=PROBE_SHA,
        closed_native_receipt_sha256=CLOSURE_SHA,closed_control_receipt_sha256=CONTROL_CLOSURE_SHA,
        original_T0_epoch=T0,adopted_epoch=now)


def _marker(h):
    m=h.get(FIELD);now=m.get('adopted_epoch') if isinstance(m,dict) else None
    if (type(now) not in (int,float) or not math.isfinite(now) or not APPROVED_EPOCH<=now<CUTOFF
            or m!=_marker_value(now)):fail('EXACT_COMPLETE_ADOPTED_MARKER')
    return m


def _history(h,*,seeded):
    rows=[r for r in h.get('PBS_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if len(rows)!=1:fail('ONE_PRESERVED_FAILED_B_ROW')
    row=rows[0]
    expected=dict(status='NATIVE_TERMINAL',job_id=FAILED_JOB,source_sha=FAILED_SOURCE,request_sha256=REQUEST_SHA,
        arm='B_COVERAGE',training_lineage_key='B_COVERAGE',start_native_step=0,committed_optimizer_updates=1829,
        query_attempts=29,conservative_H200_s=14196,conservative_CPU_s=15060,reserved_optimizer_updates=0,
        reserved_neural_queries=0,native_exit_status=1,
        terminal_native_sha256='c4ce568d3d5f56cb36b5a2c15d7d581f2dfaec7e88165ce9f1ace404deba9ffc',
        terminal_effect_summary_sha256='89515d0d921a84626843894a432ac26a97415a2c27668bcf2d98ee8fa580713f')
    runs=h.get('training_runs',{})
    if (any(row.get(k)!=v for k,v in expected.items())
            or runs.get('B_COVERAGE')!=dict(last_native_step=1829,recipe_sha256=RECIPE)
            or runs.get(zero.HR.LINEAGE)!=dict(last_native_step=4096,recipe_sha256=zero.recovery.RECIPE)
            or type(h.get('optimizer_updates')) is not int or h['optimizer_updates']<7720
            or h.get('used',{}).get('optimizer_updates')!=h['optimizer_updates']
            or h.get('used',{}).get('H200_s',0)<75658 or h.get('CPU_s',0)<103291.82213385901
            or h.get('generation_queries',0)<1180):fail('ALL_PAID_HISTORY_AND_CLOSED_A4096_B1829_PRESERVED')
    if seeded:
        run=runs.get(LINEAGE,{})
        if (set(run)!={'last_native_step','recipe_sha256'} or run.get('recipe_sha256')!=RECIPE
                or type(run.get('last_native_step')) is not int or not 1024<=run['last_native_step']<=4096):
            fail('EXPLICIT_IMPORTED1024_NEW_PROGRESS_NO_OLD_B_ROLLBACK')
    return row


def validate_global(h):
    zero.validate_global(h)
    m=_marker(h);_history(h,seeded=True)
    if h[zero.FIELD]['adopted_epoch']>m['adopted_epoch']:fail('NEW_AFTER_PRIOR_ADOPTION')
    return m


def validate_adopted(h,d):
    zero.validate_adopted(h,d)
    m=validate_global(h)
    rows=[r for r in d.get('new_submissions',[]) if r.get('request_id')==FAILED_REQUEST]
    if (m!=d.get(FIELD) or len(rows)!=1 or rows[0].get('job_id')!=FAILED_JOB
            or rows[0].get('status')!='NATIVE_TERMINAL'
            or d.get('optimizer_updates')!=h['optimizer_updates']):fail('DUAL_MARKERS_AND_PRESERVED_FAILED_ROW')
    return m


def adopt(h,d,*,grant_bytes,closure_bytes,control_closure_bytes,now_epoch):
    grant(grant_bytes);zero.validate_adopted(h,d);zero._closed_known(h,d)
    c,cs=_read(closure_bytes);cc,ccs=_read(control_closure_bytes);row=_history(h,seeded=False)
    e=c.get('effects',{});gc=cc.get('GPU_close',{})
    if (FIELD in h or FIELD in d or LINEAGE in h.get('training_runs',{})
            or cs!=CLOSURE_SHA or ccs!=CONTROL_CLOSURE_SHA
            or c.get('status')!='CLOSED_B1_HEARTBEAT_FAILURE_1829_B1024_PRESERVED'
            or c.get('job_id')!=FAILED_JOB or c.get('latest_complete_step')!=1024
            or c.get('checkpoints')!={'1024':dict(complete_sha256=SEED_SHA,members=15,actual_consumed_cursor=32768)}
            or c.get('committed_updates_without_new_complete_state')!=805 or c.get('recoverable_full_state') is not False
            or c.get('prior_complete_state_preserved') is not True or c.get('retired_or_deleted_files')!=0
            or any(e.get(k)!=v for k,v in dict(job_id=FAILED_JOB,request_id=FAILED_REQUEST,
                request_sha256=REQUEST_SHA,source_sha=FAILED_SOURCE,arm='B_COVERAGE',start_native_step=0,
                optimizer_updates_committed=1829,last_native_step=1829,query_attempts=29,model_load_attempts=4,
                pending_updates=[],unknown_query_intents=[],recipe_sha256=RECIPE,effect_journal_complete=True,
                new_physics=0,new_episodes=0).items())
            or c.get('native_terminal',{}).get('job_state')!='F' or c['native_terminal'].get('Exit_status')!=1
            or c.get('settlement',{}).get('updates')!=7720 or c['settlement'].get('unknown_effects')!=[]
            or cc.get('status')!='CLOSED_CPU_B1_FAILED1829_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT'
            or cc.get('job_id')!=FAILED_JOB or cc.get('new_B_native_step')!=1829
            or cc.get('old_A1024_costs_preserved') is not True or cc.get('unknown_effects')!=[]
            or cc.get('new_GPU_PBS_models_queries_updates_physics')!=0 or cc.get('settled_submission')!=row
            or gc.get('lease_status')!='CLOSED' or gc.get('terminal_receipt_sha256')!=cs or gc.get('terminal_receipt')!=c
            or h.get('optimizer_updates')!=7720 or d.get('optimizer_updates')!=7720
            or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED_EPOCH<=now_epoch<CUTOFF
            or any(type(v.get('closed_epoch')) not in (int,float) or not math.isfinite(v['closed_epoch'])
                or not T0<=v['closed_epoch']<=now_epoch for v in (c,cc))):
        fail('ONCE_EXACT_CLOSED_KNOWN_B1024_IMPORT_NO_REFUND_OR_RESTART')
    hh,dd=deepcopy(h),deepcopy(d)
    m=_marker_value(now_epoch);hh[FIELD]=deepcopy(m);dd[FIELD]=deepcopy(m)
    hh['training_runs'][LINEAGE]=dict(last_native_step=1024,recipe_sha256=RECIPE)
    validate_adopted(hh,dd)
    return hh,dd


def admitted(root,manifest,request):
    if not enabled(request):return False
    lineage(request)
    path=zero._readonly(Path(root)/PLAN);raw=path.read_bytes()
    if (sha(path)!=GRANT_SHA or any(v.get(SCOPE_KEYS[0])!=RELEASE or v.get(SCOPE_KEYS[1])!=GRANT_SHA
            for v in (manifest.get('authority',{}),request))
            or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or request.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA):fail('EXACT_READONLY_SEALED_NATIVE_GRANT')
    grant(raw)
    native=zero._readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name)
    if native.read_bytes()!=raw:fail('SAME_NATIVE_AUTHORITY_BYTES')
    return True
