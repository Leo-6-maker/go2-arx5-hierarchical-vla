"""DEC1 source-first runway arithmetic; legacy PV1 release gates are separate.

An unqualified target gets an explicit budget ceiling, not an invented timing
measurement. This permits the source-first schedule authorized by DEC1 without
claiming that the target matrix or the old PV1 release can already be finished.
"""
import math

from scripts.pi0294.pv1 import future_budget as old
from .authority import AUTHORITY, BANK, HARD_CLOSE, PARENT_AUTHORITY

GIB=1024**3
CAPS=dict(old.CAPS,H200_s=48*3600,PBS_submissions=8,optimizer_updates=8192,
    episode_starts=136,neural_queries=14000,physics_ticks=5000000,engine_starts=12000,
    Core_CPU_s=96*3600,Hopper_CPU_s=96*3600,local_CPU_s=8*3600,
    transfer_bytes=64*GIB,Hopper_storage_bytes=64*GIB,Core_storage_bytes=64*GIB,local_storage_bytes=8*GIB)


def fail(reason):
    raise PermissionError('DEC1_FUTURE_'+reason)


def forecast(*,snapshot,progress_bytes,plan,now_epoch):
    progress,progress_sha=old._read(progress_bytes)
    for v in (snapshot,progress):
        old._fresh(v['observed_epoch'],now_epoch,120)
        if (v.get('authority_release')!=AUTHORITY or v.get('unknown_effects')!=[]
                or v.get('native_safety_faults')!=0):
            fail('FRESH_AUTHORITY_CLEAR_NATIVE_EFFECTS')
    t0=old._number(snapshot['T0_epoch'],integer=True)
    clock=dict(stop_new_jobs_epoch=t0+144*3600,stop_training_epoch=t0+156*3600,
               stop_physics_epoch=t0+160*3600,hard_closeout_epoch=t0+168*3600)
    if (snapshot.get('schema')!='DEC1_SINGLE_OWNER_FRESH_RESOURCE_SNAPSHOT_V1'
            or snapshot.get('original_authority_release')!=PARENT_AUTHORITY
            or snapshot['clock']!=clock or clock['hard_closeout_epoch']!=HARD_CLOSE
            or not t0<=now_epoch<clock['stop_new_jobs_epoch']
            or progress.get('schema')!='DEC1_SOURCE_TRAIN_OWNER_PROGRESS_V1'
            or progress.get('campaign_run_uuid')!=snapshot['campaign_run_uuid']
            or progress.get('T0_epoch')!=t0
            or progress.get('new_bank_manifest_sha256')!=BANK
            or snapshot.get('new_bank_manifest_sha256')!=BANK
            or set(progress.get('candidates',{}))!={'A_CONTROL','B_COVERAGE'}):
        fail('ORIGINAL_WINDOW_FIXED_BANK_AND_TWO_ARMS_ONLY')
    schedule=snapshot['schedule']
    if any(type(schedule.get(k)) is not bool for k in ('target_ready','interfaces_ready','core_matrix_complete')):
        fail('EXPLICIT_OBSERVED_SCHEDULE_STATE')
    old._sha(schedule['evidence_sha256'])
    if schedule['target_ready'] and schedule['interfaces_ready'] and not schedule['core_matrix_complete']:
        fail('QUALIFIED_CORE_MATRIX_HAS_PRIORITY_OVER_SOURCE_TRAINING')
    if (plan.get('schema')!='DEC1_PRESEALED_SOURCE_AND_MATRIX_RESOURCE_PLAN_V1'
            or plan.get('authority_release')!=AUTHORITY
            or plan.get('old_PV1_release_or_LOCKBOX_admitted') is not False
            or plan.get('target_cost_is_measured') is not False
            or plan.get('matrix_is_budget_ceiling_not_feasibility_proof') is not True):
        fail('EXPLICIT_NONADDITIVE_SOURCE_FIRST_PLAN_NOT_OLD_RELEASE')
    used=old._vector(snapshot['used']);reserved=old._vector(snapshot['reserved'])
    dec_used=old._vector(snapshot['dec1_used']);dec_reserved=old._vector(snapshot['dec1_reserved'])
    for key in CAPS:
        if dec_used[key]>used[key] or dec_reserved[key]>reserved[key]:
            fail('DEC1_COSTS_INSIDE_NOT_ADDED_TO_ORIGINAL')
    for label,vector in (('global',used),('dec1',dec_used)):
        peaks=snapshot['storage_peaks' if label=='global' else 'dec1_storage_peaks']
        if set(peaks)!={'Hopper_storage_bytes','Core_storage_bytes','local_storage_bytes'}:
            fail('ALL_SITE_CURRENT_AND_PEAK_STORAGE')
        for key,peak in peaks.items():
            old._number(peak,integer=True)
            if not vector[key]<=peak<=(old.CAPS if label=='global' else CAPS)[key]:
                fail('CURRENT_STORAGE_CANNOT_ERASE_HISTORICAL_PEAK')
    pad=old._number(plan['accounting_pad_s'],integer=True)
    wall_max=min(8*3600,old._number(plan['site_wall_limit_s'],integer=True,positive=True))
    overhead=old._number(plan['qualification_load_save_close_upper_s'],positive=True)
    per_update=old._number(plan['seconds_per_update_upper'],positive=True)
    per_probe=old._number(plan['seconds_per_archived_probe_upper'],positive=True)
    if pad<60 or overhead<3300:
        fail('NATIVE_PAD_QUALIFICATION_AND_45_MIN_CLOSEOUT')
    old._sha(plan['performance_evidence_sha256'])
    payload=old._number(plan['full_checkpoint_payload_upper_bytes'],integer=True,positive=True)
    old._sha(plan['checkpoint_size_evidence_sha256'])
    jobs=[];seen=set();committed=0
    for arm,row in progress['candidates'].items():
        step=old._number(row['optimizer_updates_committed'],integer=True)
        if step>4096 or type(row.get('probe_1024_complete')) is not bool:
            fail('EXACT_COMMITTED_LOCAL_STEP_AND_PROBE_STATE')
        if step>1024 and not row['probe_1024_complete']:
            fail('FIXED_1024_PROBES_BEFORE_LATER_UPDATES')
        if step:
            old._sha(row['checkpoint_complete_sha256'])
        if row['probe_1024_complete']:
            if step<1024:fail('NO_PROBE_BEFORE_ITS_CHECKPOINT')
            old._sha(row['probe_receipt_sha256'])
        committed+=step;cursor=step;probe_done=row['probe_1024_complete']
        blocks=[j for j in plan['training_jobs'] if j['arm']==arm]
        for block in blocks:
            start=old._number(block['start_step'],integer=True);stop=old._number(block['stop_step'],integer=True)
            probe_only=block.get('probe_only',False)
            if type(probe_only) is not bool or start!=cursor:
                fail('EXACT_REMAINING_BLOCK_CURSOR_NOT_RESTART_OR_ROLLBACK')
            if probe_only:
                if start!=stop or start!=1024 or probe_done:fail('ONE_MISSING_FIXED_PROBE_ONLY_BLOCK')
                count=0;probes=12;probe_done=True
            else:
                if not start<stop<=4096 or stop!=(1024 if start<1024 else 4096):
                    fail('WHOLE_SOURCE_PHASES_KEEP_ORIGINAL_1024_GATE')
                if start>=1024 and not probe_done:fail('FIXED_PROBE_BEFORE_SOURCE_REMAINDER')
                count=stop-start;probes=12 if stop==1024 else 0
                cursor=stop;probe_done=probe_done or stop==1024
            resources=old._vector(block['resources']);wall=old._number(block['walltime_upper_s'],integer=True,positive=True)
            work_id=block['work_id']
            if (not isinstance(work_id,str) or work_id in seen or not work_id.startswith('TRAIN:'+arm+':')
                    or not overhead+count*per_update+probes*per_probe<=wall<=wall_max
                    or resources['H200_s']!=wall+pad or resources['PBS_submissions']!=1
                    or resources['optimizer_updates']!=count or resources['neural_queries']<probes
                    or resources['Hopper_CPU_s']<=0 or resources['local_CPU_s']<=0
                    or resources['Hopper_storage_bytes']<payload
                    or resources['engine_starts']<2
                    or any(resources[k] for k in ('episode_starts','physics_ticks'))):
                fail('FIXED_TRAINING_JOB_ALL_DIMENSION_UPPER')
            seen.add(work_id);jobs.append(dict(block,kind='TRAIN',resources=resources))
        if cursor!=4096 or not probe_done:fail('BOTH_FULL_4096_ENDPOINTS_AND_FIXED_PROBES_FUNDED')
    if len(jobs)!=len(plan['training_jobs']) or used['optimizer_updates']<committed or dec_used['optimizer_updates']!=committed:
        fail('NO_EXTRA_ARM_OR_UNCHARGED_UPDATES')
    matrix=plan['remaining_matrix_envelope'];mr=old._vector(matrix['resources'])
    if schedule['core_matrix_complete']:
        if any(mr.values()):fail('COMPLETED_CORE_NOT_SILENTLY_REPEATED')
    else:
        if (matrix.get('core_cells')!=64 or matrix.get('commissioning_upper')!=8
                or matrix.get('engineering_replacements_upper')!=16
                or matrix.get('valid_failure_replays')!=0
                or mr['episode_starts']<88 or mr['neural_queries']<88*80
                or mr['physics_ticks']<88*32000 or mr['engine_starts']<88*81
                or mr['H200_s']<=0 or mr['Core_CPU_s']<=0
                or mr['PBS_submissions']<math.ceil(mr['H200_s']/(wall_max+pad))
                or mr['optimizer_updates']!=0):
            fail('FULL_MATRIX_64_AND_BOUNDED_COMMISSIONING_REPLACEMENT_CEILING')
        jobs.append(dict(work_id='DEC1_CORE_MATRIX_AND_COMMISSIONING',kind='MATRIX_ENVELOPE',
            walltime_upper_s=mr['H200_s'],resources=mr))
    funded={};all_reserved={k:0 for k in CAPS};charged=0
    for row in snapshot['work_reservations']:
        key=row['work_id'];rv=old._vector(row['resources'])
        cold=row.get('reservation_scope')=='COLD_NON_GPU_ONLY'
        if (key in funded or rv['PBS_submissions'] or row.get('kind')!='TRAIN'
                or type(row.get('submission_already_charged')) is not bool
                or cold and (row.get('job_id') is not None or row['submission_already_charged']
                    or any(rv[k] for k in ('H200_s','optimizer_updates','neural_queries')))
                or not cold and (not row.get('job_id') or not row['submission_already_charged'])):
            fail('ONE_EXACT_NATIVE_ALLOCATION_RESERVATION_CREDIT')
        funded[key]=row;charged+=int(row['submission_already_charged'])
        for k in CAPS:all_reserved[k]+=rv[k]
    if (all_reserved!=reserved or len(funded)>1 or charged>used['PBS_submissions']
            or funded and snapshot.get('current_native_request_work_id') not in funded):
        fail('NO_HIDDEN_OTHER_JOB_OR_DOUBLE_CREDIT')
    extra={k:0 for k in CAPS};credit={k:0 for k in CAPS}
    queue=old._number(plan['queue_delay_upper_s_per_job']);cursor=now_epoch
    training_finish=now_epoch;submission_times=[]
    for job in jobs:
        rv=job['resources'];row=funded.pop(job['work_id'],None);cr=old._vector(row['resources']) if row else old._vector({})
        if row and row.get('reservation_scope')!='COLD_NON_GPU_ONLY':
            cr['PBS_submissions']=int(row['submission_already_charged'])
            if cr['H200_s']!=rv['H200_s']:fail('CURRENT_JOB_FULL_ENVELOPE_MATCH')
        else:
            submission_times.append(cursor);cursor+=queue*rv['PBS_submissions']
        cursor+=job['walltime_upper_s']
        if job['kind']=='TRAIN':training_finish=cursor
        for k in CAPS:
            amount=min(rv[k],cr[k]);credit[k]+=amount;extra[k]+=rv[k]-amount
    if funded:fail('RESERVATION_NOT_BOUND_TO_PLANNED_REMAINING_WORK')
    # Storage is resident peak, unlike cumulative GPU/CPU/transfer counters.
    # A later checkpoint stage still needs its actual pre-write space guard;
    # this sealed lifecycle does not authorize deletion or assert HF readback.
    storage=plan['storage_lifecycle']
    if (storage.get('schema')!='DEC1_PRESEALED_CHECKPOINT_STORAGE_LIFECYCLE_V1'
            or storage.get('full_resume_members_preserved') is not True
            or storage.get('retire_only_after_actual_independent_readback') is not True
            or storage.get('actual_transition_guard_before_later_checkpoint') is not True):
        fail('NO_UNVERIFIED_CHECKPOINT_DELETION_OR_UNBOUNDED_STORAGE_PLAN')
    old._sha(storage['implementation_or_execution_plan_sha256'])
    peaks=storage['new_output_peak_upper_bytes']
    if set(peaks)!={'Hopper_storage_bytes','Core_storage_bytes','local_storage_bytes'}:
        fail('ALL_SITE_PRESEALED_OUTPUT_PEAKS')
    for key,peak in peaks.items():
        old._number(peak,integer=True,positive=True)
        if peak>CAPS[key] or peak<dec_used[key]+dec_reserved[key] or key=='Hopper_storage_bytes' and peak<2*payload:
            fail('FULL_CHECKPOINT_TRANSIENT_AND_RESIDENT_PEAK_WITHIN_CAP')
        extra[key]=max(0,peak-dec_used[key]-dec_reserved[key])
    finish=cursor+old._number(plan['package_closeout_wall_s'],positive=True)
    global_over={k:used[k]+reserved[k]+extra[k]+old.FINAL.get(k,0)-old.CAPS[k]
                 for k in CAPS if used[k]+reserved[k]+extra[k]+old.FINAL.get(k,0)>old.CAPS[k]}
    dec_over={k:dec_used[k]+dec_reserved[k]+extra[k]-CAPS[k]
              for k in CAPS if dec_used[k]+dec_reserved[k]+extra[k]>CAPS[k]}
    time_errors=[]
    if submission_times and max(submission_times)>=clock['stop_new_jobs_epoch']:time_errors.append('NEW_JOB_CUTOFF')
    if training_finish>=clock['stop_training_epoch']:time_errors.append('TRAINING_CUTOFF')
    if cursor>=clock['stop_physics_epoch']:time_errors.append('MATRIX_CEILING_AFTER_PHYSICS_CUTOFF')
    if finish>=clock['hard_closeout_epoch']:time_errors.append('ORIGINAL_HARD_CLOSE')
    return dict(schema='DEC1_SOURCE_FIRST_NONADDITIVE_FORECAST_V1',authority_release=AUTHORITY,
        admissible_forecast=not global_over and not dec_over and not time_errors,
        status='PASS_SOURCE_FIRST_RESOURCE_ENVELOPE_NOT_SITE_OR_TASK_QUALIFICATION' if not global_over and not dec_over and not time_errors else 'STOP_RESOURCE_OR_CLOCK',
        progress_sha256=progress_sha,mandatory_jobs=jobs,matching_reserved_credits=credit,
        future_required_additional=extra,original_final_protection=old.FINAL,
        global_shortfalls=global_over,DEC1_shortfalls=dec_over,clock_failures=time_errors,
        projected_finish_epoch=finish,clock=clock,target_cost_feasibility='UNKNOWN_UNTIL_NATIVE_COMMISSIONING',
        matrix_ceiling_not_proven_runtime=True,old_PV1_release_and_LOCKBOX_admitted=False,
        optional_stress_funded=False,site_permissions_and_scientific_guards_still_required=True)
