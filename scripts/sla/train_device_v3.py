"""Finite W1/R1 operations inside the original executor; no job submission."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np

from go2arx5_vla.sla_adapter.batch_v2 import RNG_VERSION, accumulation_steps
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

ACTIVE_ENGINE = None
STOP_REQUESTED = False
OPERATIONS = {'PROFILE_BATCH_V2': 10, 'QUALIFY_CONTINUOUS_BATCH_V2': 4,
              'QUALIFY_SPLIT_BATCH_V2': 2, 'QUALIFY_RESUME_BATCH_V2': 4,
              'COMPARE_BATCH_V2': 1, 'R1_CANDIDATE500_V2': 500}


def operation_contract(spec):
    from go2arx5_vla.sla_adapter.runtime_v3 import training_matmul_precision
    operation = spec['operation']
    if operation not in OPERATIONS:
        raise ValueError('FINITE_W1_OR_R1_OPERATION_REQUIRED')
    accumulation_steps(spec['microbatch'])
    points = spec['cohort_microbatch_points']
    if (not 1 <= len(points) <= 3 or len(set(points)) != len(points) or 1 not in points
            or spec['microbatch'] not in points):
        raise ValueError('PREREGISTERED_AT_MOST_THREE_MICROBATCH_POINTS')
    for micro in points:
        accumulation_steps(micro)
    if spec['closeout_reserve_s'] < 60 or spec['closeout_reserve_s'] >= spec['max_walltime_s'] / 2:
        raise ValueError('FINITE_CHECKPOINT_CLOSEOUT_RESERVE_REQUIRED')
    if operation == 'QUALIFY_RESUME_BATCH_V2' and not spec.get('resume'):
        raise ValueError('REAL_STEP2_CHECKPOINT_REQUIRED')
    if operation != 'QUALIFY_RESUME_BATCH_V2' and operation != 'R1_CANDIDATE500_V2' and spec.get('resume'):
        raise ValueError('QUALIFICATION_OR_PROFILE_MUST_START_CLEAN')
    training_matmul_precision(spec)
    return OPERATIONS[operation]


def candidate_admission(spec):
    q = read_pin(spec['training_admission'])
    if (q.get('schema') != 'SLA_BATCH_CUDA_RESTORE_RGB_ADMISSION_V2' or not q.get('passes')
            or q['random_stream'] != RNG_VERSION or q['microbatch'] != spec['microbatch']
            or q['parent_manifest_sha256'] != spec['parent_manifest']['sha256']
            or q['target_audit_sha256'] != spec['target_audit']['sha256']
            or q['action_representation_audit_sha256'] != spec['action_representation_audit']['sha256']
            or q['restore_updates'] != 8 or q['batch_comparison_updates'] != 2
            or not 10 <= q['noncandidate_updates'] <= 40 or q['ordinary_RGB_generations'] != 32
            or not q['full_state4_vs2plus2_bit_exact'] or not q['real_model_merged_gradient_comparison_passes']
            or not q['all_training_state_GPU_resident'] or not q['frozen_parameters_unchanged']):
        raise ValueError('REAL_V2_CUDA_GRADIENT_RESTORE_RGB_PROOF_REQUIRED')
    bias = read_pin(spec['native_epsilon_validation'])
    if (bias.get('schema') != 'SLA_LOGIT_EPSILON_NATIVE_EXPERT_VALIDATION_V2' or not bias.get('passes')
            or bias['epsilon'] != EPSILON or bias['native_integrations'] <= 0
            or bias['fully_consumed_H40'] <= 0 or bias['requested_conditions'] < 1
            or bias['task_successes'] != bias['requested_conditions']
            or bias['technical_failures'] != 0 or bias['hard_safety_events'] != 0
            or not bias['original_safety_unchanged'] or not bias['old_failures_retained']):
        raise ValueError('ACTUAL_NATIVE_EXPERT_LOGIT_ENDPOINT_BIAS_VALIDATION_REQUIRED')
    # The producer is the original owner. Retain its sealed terminal evidence;
    # never promote CPU input tests or a standalone boolean as CUDA/native proof.
    for proof in (q, bias):
        if not proof.get('closed_run_evidence'):
            raise ValueError('ACTUAL_CLOSED_RUN_EVIDENCE_REQUIRED')
        for row in proof['closed_run_evidence']:
            lease, worker, result = [read_pin(row[key]) for key in ('lease', 'worker', 'result')]
            if lease.get('status') != 'CLOSED' or lease.get('worker_exit_code') != worker.get('exit_code') or worker.get('exit_code') != 0:
                raise ValueError('ALL_PROOF_WORKERS_MUST_BE_ACTUALLY_CLOSED0')
            if row['kind'] == 'CUDA_MODEL' and (result.get('schema') != 'SLA_BATCH_FINITE_OPERATION_ACTUAL_V2'
                    or not result['residency']['all_GPU_resident'] or result['actual']['optimizer_updates'] <= 0):
                raise ValueError('REAL_MODEL_UPDATES_REQUIRED_NOT_CPU_CHECKS')
    return q


def run(spec):
    stop = operation_contract(spec)
    if spec['operation'] == 'R1_CANDIDATE500_V2':
        candidate_admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    actual_site = validate(spec)  # Before parent construction or CUDA model load.
    root = Path(spec['registered_root'])
    (root / 'SITE_ACTUAL.private.json').write_text(json.dumps(actual_site, indent=2))
    import jax
    import flax
    import optax
    from go2arx5_vla.sla_adapter.runtime_v3 import DeviceResidentEngine as BatchedEngine
    from scripts.sla.train_v1 import probes
    global ACTIVE_ENGINE
    began = time.time()
    local = dict(spec, deadline_epoch=spec['deadline_epoch'] - spec['closeout_reserve_s'])
    if spec['operation'] == 'COMPARE_BATCH_V2':
        local['gradient_capture_steps'] = [0]
    elif spec['operation'] == 'QUALIFY_CONTINUOUS_BATCH_V2':
        local['gradient_capture_steps'] = [0, 1]
    elif spec['operation'] == 'PROFILE_BATCH_V2' and (
            spec['microbatch'] == 1 or spec.get('training_matmul_precision') == 'highest'):
        # Export the initial gradient during warmup for same-allocation
        # comparisons. The eight timed updates exclude that export.
        local['gradient_capture_steps'] = [0]
    else:
        local['gradient_capture_steps'] = []
    engine = ACTIVE_ENGINE = BatchedEngine.__new__(BatchedEngine)
    BatchedEngine.__init__(engine, local, root / 'work')
    engine.bindings['training_site'] = dict(GPU_uuid=actual_site['GPU_uuid'], GPU_name=actual_site['GPU_name'],
                                             jax=jax.__version__, flax=flax.__version__, optax=optax.__version__)
    if spec['operation'] == 'R1_CANDIDATE500_V2':
        engine.bindings['candidate_lineage'] = spec['candidate_lineage']
    if spec.get('resume'):
        engine.resume(spec['resume'])
    if spec['operation'] == 'QUALIFY_RESUME_BATCH_V2' and (engine.step, engine.cursor) != (2, 64):
        raise ValueError('RESUME_ORIGINAL_STEP2_CURSOR64_REQUIRED')
    if engine.step >= stop:
        raise ValueError('NO_FINISHED_ENDPOINT_REPLAY')
    result = dict(schema='SLA_BATCH_FINITE_OPERATION_ACTUAL_V2', operation=spec['operation'],
                  noncandidate=spec['operation'] != 'R1_CANDIDATE500_V2', requested_updates=stop - engine.step,
                  started_epoch=began, initial_step=engine.step, source_sha=spec['source_sha'])
    if engine.step == 0:
        result['initial_checkpoint'] = engine.checkpoint('step000')
        if not result['noncandidate']:
            result['initial_RGB'] = probes(engine, 'step0-RGB')
    rows = []
    try:
        while engine.step < stop:
            if STOP_REQUESTED or time.time() >= local['deadline_epoch']:
                raise TimeoutError('FINITE_V2_WINDOW_SAVE_COMMITTED_STATE')
            row = engine.update()
            rows.append(row)
            print(json.dumps(dict(event='ACTUAL_BATCH_V2_UPDATE', step=engine.step, cursor=engine.cursor,
                                  FM32=row['mean_FM32_loss'], microbatch=engine.microbatch, wall_s=row['wall_s'])), flush=True)
            saves = (2, 4) if result['noncandidate'] else (250, 500)
            if engine.step in saves:
                result['last_checkpoint'] = engine.checkpoint(f'step{engine.step:03d}')
                if engine.step == 4 and spec['operation'] == 'PROFILE_BATCH_V2':
                    result['profile_step4_checkpoint'] = result['last_checkpoint']
            if not result['noncandidate'] and engine.step == 250:
                result['midpoint_RGB'] = probes(engine, 'step250-RGB')
    except TimeoutError:
        result['last_checkpoint'] = engine.checkpoint(f'LIMIT-step{engine.step:03d}')
        result['status'] = 'LIMIT_SAVED_COMMITTED_STATE_NOT_COMPLETE'
        result['exit_code'] = 75
    else:
        result['status'] = 'COMPLETE_FINITE_V2_MODEL_OPERATION'
        result['exit_code'] = 0
        if spec['operation'] in ('PROFILE_BATCH_V2', 'COMPARE_BATCH_V2'):
            result['last_checkpoint'] = engine.checkpoint(f'step{engine.step:03d}')
        if spec['operation'] in ('QUALIFY_CONTINUOUS_BATCH_V2', 'R1_CANDIDATE500_V2'):
            result['endpoint_RGB'] = probes(engine, f'step{engine.step}-RGB')
    if spec['operation'] == 'PROFILE_BATCH_V2' and len(rows) == 10:
        timed = rows[2:]
        seconds = sum(row['wall_s'] for row in timed)
        result['throughput'] = dict(warmup_updates=2, timed_updates=len(timed), samples_per_s=32*len(timed) / seconds,
                                    updates_per_s=len(timed) / seconds, timed_wall_s=seconds,
                                    warmup_wall_s=sum(row['wall_s'] for row in rows[:2]),
                                    host_wait_s=sum(row['host_wait_s'] for row in timed))
    result.update(step=engine.step, cursor=engine.cursor, actual=engine.counts,
                  training_matmul_precision=engine.training_precision,
                  frozen_fingerprint_unchanged=engine.verify_frozen(), residency=engine.residency(),
                  load_wall_s=engine.load_wall_s, wall_s=time.time() - began, site=actual_site,
                  GPU_allocated_wall_hours_upper=(time.time() - began) / 3600,
                  ordinary_RGB_count=engine.counts['ordinary_RGB_generations'], physical_requests=0)
    engine.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['registered_root'])
    if (root / 'RESULT.private.json').exists() or (root / 'FAILED.private.json').exists():
        raise FileExistsError('EXISTING_OR_UNKNOWN_V2_OPERATION_NEVER_REPLAY')
    def stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        if ACTIVE_ENGINE is not None and hasattr(ACTIVE_ENGINE, 'spec'):
            ACTIVE_ENGINE.spec['deadline_epoch'] = time.time()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        result = run(spec)
    except Exception as error:
        result = dict(status='FAILED_V2_OPERATION_NOT_POLICY_SUCCESS', error=repr(error),
                      actual=getattr(ACTIVE_ENGINE, 'counts', {}), step=getattr(ACTIVE_ENGINE, 'step', 0),
                      cursor=getattr(ACTIVE_ENGINE, 'cursor', 0), physical_requests=0)
        with (root / 'FAILED.private.json').open('x') as handle:
            json.dump(result, handle, indent=2)
        raise
    with (root / 'RESULT.private.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
