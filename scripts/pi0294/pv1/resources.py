"""PV1-specific admission preserves earlier scopes and final-resource reserve."""
import datetime as dt
import math


TOTAL = dict(H200_s=96*3600, PBS_submissions=24, optimizer_updates=16384,
    episode_starts=1000, neural_queries=48000, physics_ticks=40000000, engine_starts=100000)
FINAL = dict(H200_s=20*3600, episode_starts=100, neural_queries=9000,
    live_ticks=1600000, shadow_ticks=1600000, storage_bytes=40*1024**3)


def project_eligible(project, today):
    today = dt.date.fromisoformat(today)
    return bool(project['active'] and project['member']
        and dt.date.fromisoformat(project['start_date']) <= today <= dt.date.fromisoformat(project['end_date'])
        and math.isfinite(project.get('balance_su', 0)) and project.get('balance_su', 0) > 0)


def admit_resources(used, reserved, request, *, final=False, active_GPUs=0, requested_GPUs=0,
                    job_wall_s=0, site_wall_limit_s=8*3600):
    if (isinstance(active_GPUs, bool) or isinstance(requested_GPUs, bool)
            or active_GPUs < 0 or requested_GPUs < 0 or active_GPUs + requested_GPUs > 1
            or requested_GPUs not in (0, 1)):
        raise PermissionError('PV1_SINGLE_H200_NOT_MULTIPLIED_BY_PROJECT_COUNT')
    if requested_GPUs and not 0 < job_wall_s <= min(8*3600, site_wall_limit_s):
        raise PermissionError('PV1_NATIVE_AND_EIGHT_HOUR_JOB_LIMIT')
    if requested_GPUs and (request.get('H200_s', 0) < requested_GPUs * job_wall_s
            or request.get('PBS_submissions', 0) < 1):
        raise PermissionError('PV1_FULL_ALLOCATION_AND_SUBMISSION_MUST_BE_RESERVED')
    for mapping in (used, reserved, request):
        if set(mapping) - set(TOTAL):
            raise PermissionError('PV1_UNKNOWN_RESOURCE_DIMENSION')
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in mapping.values()):
            raise PermissionError('PV1_FINITE_NONNEGATIVE_RESOURCE_ACCOUNTING')
    for key, cap in TOTAL.items():
        reserve = FINAL.get(key, 0)
        if key == 'physics_ticks':
            reserve = FINAL['live_ticks'] + FINAL['shadow_ticks']
        limit = cap if final else cap - reserve
        if used.get(key, 0) + reserved.get(key, 0) + request.get(key, 0) > limit:
            raise PermissionError('PV1_TOTAL_OR_PROTECTED_FINAL_RESERVE:' + key)
    return dict(status='PASS_PV1_BUDGET_ADMISSION_NOT_NATIVE_EXECUTION', final=final,
        concurrent_GPU_limit=1, project_pooling_increases_budget=False)
