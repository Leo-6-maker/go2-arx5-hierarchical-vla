"""Conservative, pure CPU planning from actual completed smoke receipts.

No fixture or estimated rate may authorize a main run. The caller must bind the
returned plan to real receipt hashes and recheck live quota/project/PBS state.
"""
from __future__ import annotations

import math

GIB = 1024**3


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('INVALID_' + name)
    return value


def nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('INVALID_' + name)
    return value


def calculate(*, smoke_gate, smoke_runs, step_records, wrapper, ledger, checkpoint_bytes,
              owned_bytes, filesystem_free_bytes, quota_free_lower_bound_bytes,
              extra_peak_bytes, now, new_work_cutoff):
    if smoke_gate['status'] != 'TRAINER_MODEL_INTERFACE_QUALIFIED':
        raise ValueError('ACTUAL_SMOKE_GATE_REQUIRED')
    if [row['global_step'] for row in smoke_runs] != [4, 2, 4]:
        raise ValueError('ACTUAL_4_VERSUS_2_PLUS_2_REQUIRED')
    if wrapper['status'] != 'SMOKE_COMPLETE_NO_MAIN_STARTED':
        raise ValueError('TERMINAL_SMOKE_WRAPPER_REQUIRED')
    if ledger['main_updates'] != 0 or ledger.get('pending_updates', 0):
        raise ValueError('NO_NEW_PRISTINE_MAIN_AFTER_UPDATE_OR_UNRESOLVED_RESERVATION')
    if not 8 <= ledger['model_smoke_updates'] <= 16:
        raise ValueError('ACTUAL_SMOKE_UPDATE_LEDGER_REQUIRED')
    submissions = ledger['PBS_submissions']
    if not 1 <= len(submissions) < 3:
        raise ValueError('NO_SUBMISSION_REMAINING')
    terminal = {'RECONCILED_TERMINAL', 'SUBMIT_FAILED_PRESERVED'}
    if any(row['status'] not in terminal or row.get('GPU_reserved_s', 0) or row.get('CPU_reserved_s', 0)
           for row in submissions):
        raise ValueError('PREVIOUS_ALLOCATION_NOT_RECONCILED')
    if [row['step'] for row in step_records] != [1, 2, 3, 4]:
        raise ValueError('CONTINUOUS_SMOKE_TIMING_REQUIRED')
    if [row['cursor'] for row in step_records] != [32, 64, 96, 128]:
        raise ValueError('ACTUAL_EFFECTIVE_BATCH_32_REQUIRED')
    for run in smoke_runs:
        if (run['recipe'] != smoke_runs[0]['recipe'] or not run['frozen_backbone_unchanged']
                or not run['changed_trainable_parameters']):
            raise ValueError('SMOKE_RECIPE_OR_WEIGHT_GATE_CHANGED')
    for key, expected in [('effective_batch', 32), ('max_steps', 4), ('workers', 2),
                          ('learning_rate', 1e-4), ('seed', 29200), ('eval_strategy', 'no')]:
        if smoke_runs[0]['recipe'][key] != expected:
            raise ValueError('FROZEN_SMOKE_RECIPE_REQUIRED')
    times = [positive(row['epoch'], 'STEP_TIME') for row in step_records]
    intervals = [positive(b - a, 'STEP_INTERVAL') for a, b in zip(times, times[1:])]
    processes = [row for row in wrapper['steps'] if row['mode'].startswith('smoke-')]
    if [row['mode'] for row in processes] != ['smoke-continuous', 'smoke-first', 'smoke-resume']:
        raise ValueError('ALL_SMOKE_PROCESS_TIMES_REQUIRED')
    if any(row['exit'] != 0 for row in processes):
        raise ValueError('FAILED_SMOKE_PROCESS')
    process_bound = max(positive(row['wall_s'], 'SMOKE_PROCESS_TIME') for row in processes)
    # A full child duration bounds one load/save event; no optimistic subtraction.
    seconds_per_step = max(intervals) * 1.25
    estimated_main_s = math.ceil(4096 * seconds_per_step + 3 * process_bound)
    evaluation_reserve_s = 3 * 3600
    requested_wall_s = math.ceil((estimated_main_s + evaluation_reserve_s + 120) / 60) * 60
    actual_gpu = nonnegative(ledger['GPU_allocated_s'], 'GPU_LEDGER')
    positive(checkpoint_bytes, 'COMPLETE_CHECKPOINT_BYTES')
    nonnegative(owned_bytes, 'OWNED_BYTES')
    nonnegative(filesystem_free_bytes, 'FILESYSTEM_FREE_BYTES')
    positive(extra_peak_bytes, 'EXTRA_PEAK_BYTES')
    positive(now, 'CURRENT_EPOCH')
    positive(new_work_cutoff, 'NEW_WORK_CUTOFF')
    if quota_free_lower_bound_bytes is not None:
        nonnegative(quota_free_lower_bound_bytes, 'QUOTA_FREE_BYTES')
    if extra_peak_bytes < 8 * GIB:
        raise ValueError('EXPLICIT_TRANSFER_AND_TEMPORARY_SPACE_RESERVE_REQUIRED')
    remaining_peak = 2 * checkpoint_bytes + extra_peak_bytes
    blockers = []
    if estimated_main_s > int(4.5 * 3600):
        blockers.append('FULL_4096_EXCEEDS_MAIN_4_5_HOUR_SHARE')
    if actual_gpu + requested_wall_s > 8 * 3600:
        blockers.append('COMBINED_MAIN_EVAL_EXCEEDS_REMAINING_GPU_TIME')
    if now + requested_wall_s >= new_work_cutoff:
        blockers.append('COMBINED_ALLOCATION_EXCEEDS_ORIGINAL_NEW_WORK_CUTOFF')
    cpu_used_upper = nonnegative(ledger['CPU_s'], 'CPU_USED') + nonnegative(ledger.get('CPU_upper_s', 0), 'CPU_UPPER')
    cpu_used_upper += sum(nonnegative(row.get('CPU_reserved_s', 0), 'CPU_RESERVED') for row in ledger.get('stages', []))
    if cpu_used_upper + 4 * requested_wall_s > 64 * 3600:
        blockers.append('HOPPER_CPU_BUDGET')
    if owned_bytes + remaining_peak > 180 * GIB:
        blockers.append('OWNED_PEAK_EXCEEDS_180_GIB')
    headroom = remaining_peak + 20 * GIB
    if filesystem_free_bytes < headroom:
        blockers.append('FILESYSTEM_FREE_MARGIN')
    if quota_free_lower_bound_bytes is None:
        blockers.append('PERSONAL_QUOTA_UNKNOWN_REQUIRES_SEPARATE_BOUNDED_WRITE_AND_OWNER_ADMISSION')
    elif quota_free_lower_bound_bytes < headroom:
        blockers.append('PERSONAL_QUOTA_FREE_MARGIN')
    return dict(schema='PI0292_MAIN_CAPACITY_PLAN_V1',
        status='CAPACITY_ESTIMATE_PASSES_REQUIRES_BOUND_LIVE_ADMISSION' if not blockers else 'CAPACITY_HOLD',
        blockers=blockers, measured_interstep_seconds=intervals, timing_safety_factor=1.25,
        conservative_seconds_per_step=seconds_per_step, estimated_full_4096_main_seconds=estimated_main_s,
        overhead_basis='Three whole smoke child durations for pristine load and two full saves',
        evaluation_reserve_seconds=evaluation_reserve_s, combined_walltime_seconds=requested_wall_s,
        remaining_peak_bytes=remaining_peak, required_free_headroom_bytes=headroom,
        checkpoint_copies_in_forecast=2, extra_transfer_temporary_bytes=extra_peak_bytes,
        allocated_GPU_seconds_before=actual_gpu, submission_calls_used=len(submissions),
        main_optimizer_updates=4096, effective_batch=32, sample_exposures=131072,
        main_and_eval_share_next_allocation=True, quality_thresholds_changed=False,
        authorizes_execution=False)
