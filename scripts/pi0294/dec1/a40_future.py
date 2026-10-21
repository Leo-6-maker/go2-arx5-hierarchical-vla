"""A40 evidence adapter around the existing DEC1 remaining-work arithmetic."""
from scripts.pi0294.pv1 import future_budget as old
from .alternate_accounting import digest
from .authority import AUTHORITY, PARENT_AUTHORITY, HARD_CLOSE
from .future_budget import forecast


def fail(message):
    raise PermissionError('VAP1_A40_FUTURE_'+message)


def recompute(*, request_bytes, ledger_bytes, DEC1_ledger_bytes, snapshot_bytes,
              progress_bytes, plan_bytes, amendment_bytes, now_epoch, job_id,
              purpose='NATIVE_PRETRAIN'):
    from .a40_allocation import request_scope, PLATFORM, T0
    r,request_sha=old._read(request_bytes);request_scope(r,now=now_epoch)
    if purpose not in ('PREQUEUE','NATIVE_PRETRAIN'):
        fail('EXPLICIT_PREQUEUE_OR_FRESH_NATIVE_PRETRAIN_PURPOSE')
    if purpose=='PREQUEUE' and (job_id is not None or not 0<=now_epoch-r['created_epoch']<=120):
        fail('FRESH_UNSUBMITTED_PREQUEUE_REQUEST_ONLY')
    h,ledger_sha=old._read(ledger_bytes);d,dec_sha=old._read(DEC1_ledger_bytes)
    s,snapshot_sha=old._read(snapshot_bytes);p,plan_sha=old._read(plan_bytes)
    if (h.get('authority_release')!=PARENT_AUTHORITY or d.get('authority_release')!=AUTHORITY
            or h.get('campaign_run_uuid')!=r['campaign_run_uuid']
            or d.get('campaign_run_uuid')!=r['campaign_run_uuid']
            or h.get('unknown_effects')!=[] or h.get('pending_updates')!=[] or d.get('unknown_effects')!=[]
            or s.get('actual_Hopper_ledger_sha256')!=ledger_sha
            or s.get('actual_DEC1_ledger_sha256')!=dec_sha
            or p.get('native_platform')!=PLATFORM
            or p.get('performance_basis')!='UNMEASURED_A40_OWNER_CEILING_REVALIDATE_FIRST32'
            or p.get('H200_timing_is_A40_measurement') is not False
            or p.get('first32_are_counted_A_CONTROL_updates') is not True
            or s.get('paired_source_training_platform')!=PLATFORM):
        fail('ACTUAL_NATIVE_LEDGER_BYTES_AND_EXPLICIT_NOT_MEASURED_A40_CEILING')
    if h['used']['PBS_submissions']!=len(h['PBS_submissions']):
        fail('NO_UNCOUNTED_SUBMISSIONS')
    if (any(x.get('stage')=='P2_PRIMARY' and x.get('native_platform')!=PLATFORM
            and x.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')
            for x in h['PBS_submissions'])
            or any(h.get('training_runs',{}).get(name,{}).get('last_native_step',0)>0
                for name in ('A_CONTROL','B_COVERAGE')) and h.get('DEC1_source_pair_platform')!=PLATFORM):
        fail('SAME_PLATFORM_PAIR_NOT_A_PARALLEL_HOPPER_SOURCE_ARM_OR_MIGRATION')
    used=old._vector(s['used']);reserved=old._vector(s['reserved'])
    du=old._vector(s['dec1_used']);dr=old._vector(s['dec1_reserved'])
    expected=dict(h['used'],Hopper_CPU_s=h['CPU_s'],
        Hopper_storage_bytes=h['new_storage_current_bytes'])
    expected_reserved=dict(h['reserved'],Hopper_CPU_s=h.get('CPU_reserved_s',0),
        Hopper_storage_bytes=h['storage_reserved_bytes'])
    for observed,native in ((used,expected),(reserved,expected_reserved)):
        if any(observed[k]!=v for k,v in native.items()):
            fail('GLOBAL_SPENT_AND_ALL_PLATFORM_RESERVATIONS_EXACT_NOT_RESET')
    expected_d=dict(H200_s=d['GPU_allocated_s'],PBS_submissions=len(d['new_submissions']),
        optimizer_updates=d['optimizer_updates'],neural_queries=d['neural_queries'],
        Hopper_CPU_s=d['CPU_s'],episode_starts=d['episode_starts'],physics_ticks=d['physics_ticks'],
        engine_starts=d['constructors'],transfer_bytes=d.get('network_transfer_accounted_bytes',0),
        Hopper_storage_bytes=d.get('new_GPU_site_output_allocated_upper_bytes',0))
    expected_dr=dict(H200_s=d['GPU_reserved_s'],Hopper_CPU_s=d.get('CPU_reserved_s',0),
        neural_queries=d.get('neural_queries_reserved',0),
        optimizer_updates=d.get('optimizer_updates_reserved',0),
        transfer_bytes=d.get('network_transfer_reserved_bytes',0),
        Hopper_storage_bytes=d.get('GPU_site_output_reserved_bytes',0))
    for observed,native in ((du,expected_d),(dr,expected_dr)):
        if any(observed[k]!=v for k,v in native.items()):
            fail('DEC1_NATIVE_SPENT_RESERVED_CPU_AND_IO_EXACT')
    forecasted=forecast(snapshot=s,progress_bytes=progress_bytes,plan=p,
        amendment_bytes=amendment_bytes,now_epoch=now_epoch)
    if not forecasted['admissible_forecast']:
        fail('COMPLETE_SOURCE_PAIR_MATRIX_AND_ORIGINAL_FINAL_RESERVE_DO_NOT_FIT')
    rows=[x for x in h['PBS_submissions'] if x['request_id']==r['request_id']]
    if (purpose=='PREQUEUE' and rows or purpose=='NATIVE_PRETRAIN'
            and (len(rows)!=1 or not job_id or rows[0].get('job_id')!=job_id or rows[0]['request_sha256']!=request_sha)):
        fail('EXACT_CURRENT_RESERVED_NATIVE_REQUEST')
    return dict(schema='VAP1_A40_RECOMPUTED_REMAINING_WORK_ADMISSION_V1',
        status='PASS_ORIGINAL_GLOBAL_AND_DEC1_REMAINING_WORK',purpose=purpose,job_id=job_id,request_sha256=request_sha,
        observed_epoch=now_epoch,original_T0_epoch=T0,hard_close_epoch=HARD_CLOSE,
        native_global_ledger_sha256=ledger_sha,native_DEC1_ledger_sha256=dec_sha,
        global_ledger_canonical_sha256=digest(h),snapshot_sha256=snapshot_sha,plan_sha256=plan_sha,
        DEC1_ledger_canonical_sha256=digest(d),
        same_platform_AB=PLATFORM,all_old_spent_and_reserved_preserved=True,
        uses_H200_timing_as_A40_measurement=False,full_remaining_work_admissible=True,
        final_GPU_protected_s=72000,new_R1_in_AB=False,
        DEC1_GPU_site_output_used_and_reserved_bytes=du['Hopper_storage_bytes']+dr['Hopper_storage_bytes'],
        forecast=forecasted,not_proof_of_A40_throughput_or_final_completion=True)
