"""CPU-only S3 retry preparation; no submission or model side effects."""
from datetime import date
import math
import re

GIB = 1024**3


def nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('INVALID_' + name)
    return value


def prepare(*, authorized_project, project, user, cluster, today, balance_su,
            ledger, own_jobs, quota, filesystem_free_bytes, owned_bytes,
            now, adoption, deferred_after_any=(), dependency_queue_authority=None):
    if project['name'] != authorized_project:
        raise PermissionError('EXPLICIT_PROJECT_BINDING_REQUIRED')
    day = date.fromisoformat(today)
    if (project['active'] is not True or user not in project['users']
            or cluster not in project['cluster']
            or not date.fromisoformat(project['start_date']) <= day <= date.fromisoformat(project['end_date'])):
        raise PermissionError('CURRENT_PROJECT_MEMBERSHIP_CLUSTER_AND_DATES_REQUIRED')
    if nonnegative(balance_su, 'BALANCE_SU') == 0:
        raise PermissionError('POSITIVE_CURRENT_BALANCE_REQUIRED')
    dependencies = sorted(deferred_after_any)
    if own_jobs or dependencies:
        if (dependency_queue_authority != '2026-10-13_DIRECT_USER_ALLOW_PBS_DEPENDENCY_QUEUE_ONLY'
                or not dependencies or len(set(dependencies)) != len(dependencies)
                or set(dependencies) != set(own_jobs)):
            raise PermissionError('NO_OVERLAPPING_OWN_ALLOCATION')
        for job_id, job in own_jobs.items():
            if (not re.fullmatch(r'[0-9]+\.[A-Za-z0-9._-]+', job_id)
                    or job['Job_Owner'].split('@')[0] != user or job['job_state'] != 'R'
                    or int(job['Resource_List']['ngpus']) != 1
                    or job.get('project') == authorized_project):
                raise PermissionError('EXACT_OBSERVED_UNRELATED_OWN_RUNNING_GPU_DEPENDENCY_REQUIRED')
    if (adoption['release'] != '261bc5b37f20aaff1ad7d5742252d3dc1a91fa65'
            or adoption['T0_epoch'] != 1791860304
            or adoption['new_work_cutoff_epoch'] != 1791943104
            or adoption['deadline_epoch'] != 1791946704 or adoption['reset_on_resume']):
        raise PermissionError('ORIGINAL_AUTHORITY_AND_CLOCK_REQUIRED')
    if (any(ledger[k] != 0 for k in ('model_loads', 'model_smoke_updates', 'main_updates'))
            or ledger.get('pending_updates', 0)):
        raise PermissionError('UNEXECUTED_S3_ONLY_NOT_ENGINEERING_MODEL_RETRY')
    submissions = ledger['PBS_submissions']
    if (len(submissions) != 1 or submissions[0]['request_id'] != 'S3-smoke-01'
            or submissions[0]['status'] != 'SUBMIT_FAILED_PRESERVED'
            or submissions[0]['job_id'] is not None
            or submissions[0].get('GPU_reserved_s', 0)
            or submissions[0].get('CPU_reserved_s', 0)):
        raise PermissionError('EXACT_FIRST_SUBMISSION_FAILURE_MUST_BE_RETAINED')
    if quota['user'] != user or quota['source'] != 'NATIVE_HPC_PERSONAL_QUOTA_RAW_BYTES':
        raise PermissionError('PERSONAL_BYTE_QUOTA_REQUIRED')
    age = now - quota['report_epoch']
    if not 0 <= age <= 900:
        raise PermissionError('FRESH_QUOTA_REPORT_REQUIRED')
    used = nonnegative(quota['used_bytes'], 'QUOTA_USED')
    hard = nonnegative(quota['hard_bytes'], 'QUOTA_HARD')
    free = max(0, hard - used)
    forecast = 80 * GIB
    margin = 20 * GIB
    blockers = []
    if free < forecast + margin:
        blockers.append('PERSONAL_QUOTA_FREE_MARGIN')
    if nonnegative(filesystem_free_bytes, 'FILESYSTEM_FREE') < forecast + margin:
        blockers.append('FILESYSTEM_FREE_MARGIN')
    if nonnegative(owned_bytes, 'OWNED_BYTES') + forecast > 180 * GIB:
        blockers.append('OWNED_PEAK_EXCEEDS_180_GIB')
    if now + 1800 >= adoption['new_work_cutoff_epoch']:
        blockers.append('ORIGINAL_NEW_WORK_CUTOFF')
    if nonnegative(ledger['GPU_allocated_s'], 'GPU_LEDGER') + 1800 > 8 * 3600:
        blockers.append('ORIGINAL_GPU_BUDGET')
    cpu = nonnegative(ledger['CPU_s'], 'CPU_USED') + nonnegative(ledger.get('CPU_upper_s', 0), 'CPU_UPPER')
    cpu += sum(nonnegative(x.get('CPU_reserved_s', 0), 'CPU_RESERVED') for x in ledger.get('stages', []))
    if cpu + 7200 > 64 * 3600:
        blockers.append('ORIGINAL_HOPPER_CPU_BUDGET')
    return dict(schema='PI0292_S3_RETRY_PREPARATION_V1',
        status='PREPARED_NOT_SUBMITTED_CAPACITY_HOLD' if blockers else 'PREPARED_REQUIRES_LIVE_SUBMISSION_ADMISSION',
        project=authorized_project, project_end_date=project['end_date'],
        project_membership_verified=True, positive_balance_verified=True,
        request_id='S3-smoke-02', previous_failure_retained=True,
        PBS_calls_used=1, PBS_calls_remaining=2, next_smoke_walltime_s=1800,
        required_remaining_peak_bytes=forecast, required_margin_bytes=margin,
        personal_used_bytes=used, personal_hard_bytes=hard, personal_free_bytes=free,
        additional_headroom_needed_bytes=max(0, forecast + margin - free),
        quota_report_epoch=quota['report_epoch'], observed_epoch=now,
        blockers=blockers, authorizes_execution=False,
        recheck_at_submission_and_allocation=True, main_training=False,
        queue_only_after_any_dependencies=dependencies,
        dependency_queue_authority=dependency_queue_authority if dependencies else None,
        actual_allocation_still_requires_no_other_own_job=True,
        remaining_main_and_eval_must_share_one_allocation=True,
        budget_reset=False, selected='NONE', deployment='NONE')
