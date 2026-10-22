"""DEC1 source-first runway arithmetic; legacy PV1 release gates are separate.

An unqualified target gets an explicit budget ceiling, not an invented timing
measurement. This permits the source-first schedule authorized by DEC1 without
claiming that the target matrix or the old PV1 release can already be finished.
"""
import math
import hashlib

from scripts.pi0294.pv1 import future_budget as old
from .authority import AUTHORITY, BANK, HARD_CLOSE, PARENT_AUTHORITY

GIB=1024**3
CAPS=dict(old.CAPS,H200_s=48*3600,PBS_submissions=8,optimizer_updates=8192,
    episode_starts=136,neural_queries=14000,physics_ticks=5000000,engine_starts=12000,
    Core_CPU_s=96*3600,Hopper_CPU_s=96*3600,local_CPU_s=8*3600,
    transfer_bytes=64*GIB,Hopper_storage_bytes=64*GIB,Core_storage_bytes=64*GIB,local_storage_bytes=8*GIB)
VAP1='43f7fe6f86c683f2e38e5c8802eb8cde208e575c'
VAP1_ADOPTION='a26aeb5076e075791e1dfea4e5da0ccaae05ccbb7fa578a1583dea1c4764ed89'


def fail(reason):
    raise PermissionError('DEC1_FUTURE_'+reason)


def effective_caps(amendment_bytes=None,recovery_bytes=None,segment_bytes=None,storage_bytes=None,heartbeat_bytes=None,zero_update_bytes=None):
    if zero_update_bytes is not None:
        from .zero_update_recovery import grant
        grant(zero_update_bytes)
        if heartbeat_bytes is None:fail('ZERO_UPDATE_REQUIRES_ORIGINAL_HEARTBEAT_AUTHORIZATION')
        return dict(effective_caps(amendment_bytes,recovery_bytes,segment_bytes,storage_bytes,heartbeat_bytes),PBS_submissions=13,Hopper_storage_bytes=483*GIB)
    if heartbeat_bytes is not None:
        from .heartbeat_recovery import grant
        grant(heartbeat_bytes)
        if storage_bytes is None:fail('HEARTBEAT_REQUIRES_ORIGINAL_STORAGE_AUTHORIZATION')
        return dict(effective_caps(amendment_bytes,recovery_bytes,segment_bytes,storage_bytes),
            optimizer_updates=9987,PBS_submissions=12,Hopper_storage_bytes=418*GIB)
    if storage_bytes is not None:
        from .storage_expansion import grant
        grant(storage_bytes)
        if segment_bytes is None:fail('STORAGE_REQUIRES_ORIGINAL_SEGMENT_AUTHORIZATION')
        return dict(effective_caps(amendment_bytes,recovery_bytes,segment_bytes),Hopper_storage_bytes=352*GIB)
    if segment_bytes is not None:
        from .segmented_training import grant
        grant(segment_bytes)
        if recovery_bytes is None:fail('SEGMENT_REQUIRES_ORIGINAL_LOST_SAVE_AUTHORIZATION')
        return dict(effective_caps(amendment_bytes,recovery_bytes),H200_s=52*3600,
            PBS_submissions=11,Hopper_storage_bytes=176*GIB)
    if recovery_bytes is not None:
        from .lost_save_recovery import grant
        grant(recovery_bytes)
        if amendment_bytes is None:fail('RECOVERY_REQUIRES_ORIGINAL_VAP1_ADOPTION')
        return dict(effective_caps(amendment_bytes),optimizer_updates=9216)
    if amendment_bytes is None:return dict(CAPS)
    value,digest=old._read(amendment_bytes)
    expected=dict(local_CPU_s=[28800,57600],CoreLab_CPU_s=[345600,576000],
        GPU_site_output_bytes=[64*GIB,160*GIB],CoreLab_output_bytes=[64*GIB,128*GIB],
        transfer_bytes=[64*GIB,128*GIB])
    if (digest!=VAP1_ADOPTION or value.get('schema')!='VAP1_ADOPTION_AND_DELTA_BUDGET_V1'
            or value.get('authority_release')!=VAP1 or value.get('original_T0_epoch')!=1792348196
            or value.get('original_hard_close_epoch')!=HARD_CLOSE
            or value.get('cumulative_subcap_delta')!=expected
            or value.get('old_clocks_and_counters_reset') is not False
            or value.get('GPU_global_hours')!=96 or value.get('GPU_DEC1_hours')!=48
            or value.get('global_PBS_limit')!=24 or value.get('DEC1_new_PBS_limit')!=8
            or value.get('concurrent_GPUs')!=1 or value.get('final_GPU_protected_s')!=72000):
        fail('EXACT_NATIVE_VAP1_NONADDITIVE_ADOPTION_NO_GPU_OR_CLOCK_CHANGE')
    return dict(CAPS,local_CPU_s=57600,Core_CPU_s=576000,Hopper_storage_bytes=160*GIB,
        Core_storage_bytes=128*GIB,transfer_bytes=128*GIB)


def forecast(*,snapshot,progress_bytes,plan,now_epoch,amendment_bytes=None,recovery_bytes=None,segment_bytes=None,storage_bytes=None,heartbeat_bytes=None,zero_update_bytes=None):
    caps=effective_caps(amendment_bytes,recovery_bytes,segment_bytes,storage_bytes,heartbeat_bytes,zero_update_bytes)
    global_caps=dict(old.CAPS)
    if storage_bytes is not None:
        from .storage_expansion import RELEASE as STORAGE_RELEASE,GRANT_SHA as STORAGE_SHA
        if any(v.get('storage_authorization_release')!=STORAGE_RELEASE or v.get('storage_grant_sha256')!=STORAGE_SHA for v in (snapshot,plan)):
            fail('EXACT_STORAGE_ONLY_SCOPE_ON_SNAPSHOT_AND_PLAN')
        global_caps['Hopper_storage_bytes']=419*GIB
    elif any('storage_authorization_release' in v or 'storage_grant_sha256' in v for v in (snapshot,plan)):
        fail('NO_UNBOUND_STORAGE_SCOPE')
    from . import heartbeat_recovery as heartbeat
    for value in (snapshot,plan):
        if heartbeat.enabled(value)!=(heartbeat_bytes is not None):fail('EXACT_HEARTBEAT_GRANT_SNAPSHOT_PLAN_SCOPE')
    if heartbeat_bytes is not None:global_caps['Hopper_storage_bytes']=485*GIB
    from . import zero_update_recovery as zero
    for value in (snapshot,plan):
        if zero.enabled(value)!=(zero_update_bytes is not None):fail('EXACT_ZERO_UPDATE_RESUME_PLAN_SNAPSHOT_SCOPE')
    if zero_update_bytes is not None:global_caps['Hopper_storage_bytes']=550*GIB
    segment=segment_bytes is not None
    if segment:
        from .segmented_training import GRANT_SHA as SEGMENT_SHA,RELEASE as SEGMENT_RELEASE,RATE,CPU_PROFILE
        if (snapshot.get('segmented_grant_sha256')!=SEGMENT_SHA or plan.get('segmented_authorization_release')!=SEGMENT_RELEASE
                or plan.get('seconds_per_update_upper',0)<RATE):fail('SEGMENT_FRESH_SCOPE_AND_ACTUAL_COST_BOUND')
    progress,progress_sha=old._read(progress_bytes)
    lost=0
    if recovery_bytes is not None:
        from .lost_save_recovery import GRANT_SHA,RELEASE
        if (snapshot.get('source_recovery_grant_sha256')!=GRANT_SHA
                or progress.get('source_recovery_authorization_release')!=RELEASE
                or progress.get('known_irrecoverable_optimizer_updates')!=1024):
            fail('EXACT_KNOWN_LOST_UPDATES_NOT_FAKE_ACTIVE_CHECKPOINT_PROGRESS')
        lost=1024
    elif progress.get('known_irrecoverable_optimizer_updates',0):fail('NO_UNAUTHORIZED_LOST_UPDATE_CREDIT')
    if zero.enabled(progress)!=(zero_update_bytes is not None):fail('EXACT_ZERO_UPDATE_RESUME_PROGRESS_SCOPE')
    if heartbeat.enabled(progress)!=(heartbeat_bytes is not None):fail('EXACT_HEARTBEAT_PROGRESS_SCOPE')
    if heartbeat_bytes is not None:
        if (progress.get('heartbeat_failed_paid_updates')!=771
                or progress.get('failed_lineage_preserved_step')!=2819
                or progress['candidates']['A_CONTROL'].get('training_lineage_key')!=heartbeat.LINEAGE
                or progress['candidates']['B_COVERAGE'].get('training_lineage_key')!='B_COVERAGE'):
            fail('OLD_PAID771_AND_EXPLICIT_NEW_SEEDED_LINEAGE_NO_COST_RESET')
        lost+=771
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
    for key in caps:
        if dec_used[key]>used[key] or dec_reserved[key]>reserved[key]:
            fail('DEC1_COSTS_INSIDE_NOT_ADDED_TO_ORIGINAL')
    for label,vector in (('global',used),('dec1',dec_used)):
        peaks=snapshot['storage_peaks' if label=='global' else 'dec1_storage_peaks']
        if set(peaks)!={'Hopper_storage_bytes','Core_storage_bytes','local_storage_bytes'}:
            fail('ALL_SITE_CURRENT_AND_PEAK_STORAGE')
        for key,peak in peaks.items():
            old._number(peak,integer=True)
            if not vector[key]<=peak<=(global_caps if label=='global' else caps)[key]:
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
            if heartbeat_bytes is not None:
                if not heartbeat.enabled(block) or block.get('training_lineage_key')!=(heartbeat.LINEAGE if arm=='A_CONTROL' else 'B_COVERAGE'):
                    fail('EXACT_HEARTBEAT_RECOVERY_PLAN_LINEAGE')
            elif heartbeat.enabled(block):fail('NO_UNBOUND_HEARTBEAT_PLAN_BLOCK')
            if zero.enabled(block)!=(zero_update_bytes is not None):fail('EXACT_ZERO_UPDATE_RESUME_TRAIN_BLOCK_SCOPE')
            start=old._number(block['start_step'],integer=True);stop=old._number(block['stop_step'],integer=True)
            probe_only=block.get('probe_only',False)
            long_run=block.get('same_allocation_1024_probes',False)
            if type(long_run) is not bool:fail('EXPLICIT_LONG_ALLOCATION_BOOLEAN')
            if long_run:
                from .long_training import POLICY_SHA
                if (amendment_bytes is None or plan.get('long_allocation_policy_sha256')!=POLICY_SHA
                        or probe_only or (not segment and (stop!=4096 or start>=1024))):
                    fail('LONG_ALLOCATION_EXACT_VAP1_POLICY_WITH_FIXED_IN_JOB_PROBES')
                if segment and ((arm,start,stop) not in (('A_CONTROL',32,2048),('A_CONTROL',2048,4096),
                        ('B_COVERAGE',0,2048),('B_COVERAGE',2048,4096))
                        or block.get('segmented_authorization_release')!=SEGMENT_RELEASE
                        or block.get('segmented_grant_sha256')!=SEGMENT_SHA
                        or block.get('CPU_envelope_profile')!=CPU_PROFILE):fail('EXACT_AUTHORIZED_FOUR_SEGMENT_PLAN')
            if type(probe_only) is not bool or start!=cursor:
                fail('EXACT_REMAINING_BLOCK_CURSOR_NOT_RESTART_OR_ROLLBACK')
            if probe_only:
                if start!=stop or start!=1024 or probe_done:fail('ONE_MISSING_FIXED_PROBE_ONLY_BLOCK')
                count=0;probes=12;probe_done=True
            else:
                if not start<stop<=4096 or stop!=((2048 if start<1024 else 4096) if segment else (4096 if long_run else (1024 if start<1024 else 4096))):
                    fail('WHOLE_SOURCE_PHASES_KEEP_ORIGINAL_1024_GATE')
                if start>=1024 and not probe_done:fail('FIXED_PROBE_BEFORE_SOURCE_REMAINDER')
                count=stop-start;probes=12 if stop==1024 or long_run and start<1024 else 0
                cursor=stop;probe_done=probe_done or stop==1024 or long_run and start<1024
            resources=old._vector(block['resources']);wall=old._number(block['walltime_upper_s'],integer=True,positive=True)
            if plan.get('native_platform')=='VANDA_A40':
                from .a40_allocation import cpu_envelope
                if resources['Hopper_CPU_s']!=cpu_envelope(block,wall):
                    fail('FUTURE_A40_CPU_ENVELOPE_MUST_MATCH_ENFORCED_COLD_PROFILE')
            work_id=block['work_id']
            if (not isinstance(work_id,str) or work_id in seen or not work_id.startswith('TRAIN:'+arm+':')
                    or not overhead+count*per_update+probes*per_probe<=wall<=wall_max
                    or resources['H200_s']!=wall+pad or resources['PBS_submissions']!=1
                    or resources['optimizer_updates']!=count or resources['neural_queries']<probes
                    or resources['Hopper_CPU_s']<=0 or resources['local_CPU_s']<=0
                    or resources['Hopper_storage_bytes']<payload
                    or resources['engine_starts']<(3 if long_run else 2)
                    or any(resources[k] for k in ('episode_starts','physics_ticks'))):
                fail('FIXED_TRAINING_JOB_ALL_DIMENSION_UPPER')
            seen.add(work_id);jobs.append(dict(block,kind='TRAIN',resources=resources))
        if cursor!=4096 or not probe_done:fail('BOTH_FULL_4096_ENDPOINTS_AND_FIXED_PROBES_FUNDED')
    if len(jobs)!=len(plan['training_jobs']) or used['optimizer_updates']<committed+lost or dec_used['optimizer_updates']!=committed+lost:
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
    funded={};all_reserved={k:0 for k in caps};charged=0
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
        for k in caps:all_reserved[k]+=rv[k]
    if (all_reserved!=reserved or len(funded)>1 or charged>used['PBS_submissions']
            or funded and snapshot.get('current_native_request_work_id') not in funded):
        fail('NO_HIDDEN_OTHER_JOB_OR_DOUBLE_CREDIT')
    extra={k:0 for k in caps};credit={k:0 for k in caps}
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
        for k in caps:
            amount=min(rv[k],cr[k]);credit[k]+=amount;extra[k]+=rv[k]-amount
    if funded:fail('RESERVATION_NOT_BOUND_TO_PLANNED_REMAINING_WORK')
    if heartbeat_bytes is not None:
        # Keep the original three future controller ceilings conservatively
        # even after one is prepaid; this grants no additional CPU or IO.
        controls=dict(Hopper_CPU_s=5520,local_CPU_s=3600,transfer_bytes=805306368)
        if plan.get('heartbeat_remaining_control_upper')!=controls:
            fail('EXPLICIT_ALL_REMAINING_CONTROL_AND_INCIDENT_COSTS')
        for key,cost in controls.items():extra[key]+=cost
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
        if peak>caps[key] or peak<dec_used[key]+dec_reserved[key] or key=='Hopper_storage_bytes' and peak<2*payload:
            fail('FULL_CHECKPOINT_TRANSIENT_AND_RESIDENT_PEAK_WITHIN_CAP')
        extra[key]=max(0,peak-dec_used[key]-dec_reserved[key])
    finish=cursor+old._number(plan['package_closeout_wall_s'],positive=True)
    global_over={k:used[k]+reserved[k]+extra[k]+old.FINAL.get(k,0)-global_caps[k]
                 for k in caps if used[k]+reserved[k]+extra[k]+old.FINAL.get(k,0)>global_caps[k]}
    dec_over={k:dec_used[k]+dec_reserved[k]+extra[k]-caps[k]
              for k in caps if dec_used[k]+dec_reserved[k]+extra[k]>caps[k]}
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
        optional_stress_funded=False,site_permissions_and_scientific_guards_still_required=True,
        effective_DEC1_VAP1_caps=caps,effective_global_caps=global_caps,budget_amendment_sha256=hashlib.sha256(amendment_bytes).hexdigest() if amendment_bytes is not None else None)
