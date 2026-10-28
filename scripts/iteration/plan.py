"""CPU-only plans for isolated collection and frozen-dataset loader experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('INVALID_'+name)
    if integer and not isinstance(value, int):raise ValueError('INTEGER_'+name+'_REQUIRED')
    return value


def collection_plan(card):
    if card['status'] != 'NOT_AUTHORIZED':raise ValueError('PREPARATION_ONLY_NO_GO_ADOPTION')
    ids = [row['id'] for row in card['episodes']]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', x) for x in ids):
        raise ValueError('UNIQUE_SAFE_FIRST_ATTEMPT_IDS_REQUIRED')
    for row in card['episodes']:
        if row['split'] not in ('TRAIN', 'DEV', 'CANARY'):raise ValueError('EXPLICIT_SPLIT_REQUIRED')
        if type(row['seed']) is not int:raise ValueError('FIXED_SEED_REQUIRED')
    workers = positive(card['workers'], 'WORKERS', True)
    cores = positive(card['cores_per_worker'], 'WORKER_CORES', True)
    total = positive(card['max_concurrent_cores'], 'TOTAL_CORES', True)
    if workers not in (1, 2) or workers*cores+2 > total:
        raise ValueError('RESERVE_COORDINATOR_AND_AUDITOR_CORES_START_WITH_AT_MOST_TWO_WORKERS')
    cost = {key:positive(card['per_episode'][key], key, key != 'cpu_s')
            for key in ('physics_steps', 'rgb_pairs', 'cpu_s', 'output_bytes')}
    limits = {key:positive(card['limits'][key], key, key != 'cpu_s') for key in cost}
    if any(len(ids)*cost[key] > limits[key] for key in cost):raise ValueError('WHOLE_FIRST_ATTEMPT_PLAN_EXCEEDS_LIMIT')
    return dict(schema='ITERATION_COLLECTION_PREPARATION_V1', status='PREPARED_NOT_AUTHORIZED',
        authorizes_execution=False, live_adapter='NOT_IMPLEMENTED_REQUIRES_NEW_TASK_ADMISSION',
        profile='4080-simulation', workers=workers, cores_per_worker=cores,
        audit_workers=1, coordinator_cores=1, episodes=copy.deepcopy(card['episodes']),
        first_attempts_only=True, cost=cost, limits=limits,
        required_before_execution=['fixed-science-task-card-and-GO', 'standard-RUN_MANIFEST-and-entry-release',
            'current-host-registry-and-output-role', 'live-lease-and-original-deadline',
            'global-coordinator-reservations', 'native-worker-budget-adapter',
            'serial-canary-and-parallel-trajectory-equivalence', 'measured-CPU-render-thread-budget'],
        data_policy='NEXT_VERSION_ONLY_NEVER_APPEND_TO_ACTIVE_TRAINING',
        active_PI0292_capsule='UNCHANGED')


def hopper_plan(identity, max_loader_workers=4):
    for key in ('dataset_manifest_sha256', 'statistics_sha256', 'processor_sha256', 'source_seal_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}',identity[key]):raise ValueError('FIXED_'+key+'_REQUIRED')
    if not re.fullmatch(r'[0-9a-f]{40}',identity['hf_revision']):raise ValueError('FIXED_HF_REVISION_REQUIRED')
    if identity['optimization_split'] != 'TRAIN':raise ValueError('DEV_OPTIMIZATION_FORBIDDEN')
    if max_loader_workers not in (0,2,4):raise ValueError('BOUNDED_LOADER_MATRIX_REQUIRED')
    cases=[dict(num_workers=n,persistent_workers=n>0,prefetch_factor=2 if n else None,
                pin_memory=True, sampler='ORIGINAL_CONSUMED_CURSOR', seed=29200,
                multiprocessing_context='spawn' if n else None)
           for n in (0,2,4) if n<=max_loader_workers]
    return dict(schema='ITERATION_HOPPER_PREPARATION_V1',status='PREPARED_NOT_AUTHORIZED',
        authorizes_execution=False, execution='OWN_LEGAL_H200_PBS_ONLY', identity=copy.deepcopy(identity),
        billing_project='CFP04-CF-007_REVERIFY_AT_JOB_BOUNDARY',cases=cases,
        unchanged=['frozen-dataset','TRAIN-only-statistics','native-processor-and-collator',
            'consumed-sample-order','microbatch-2-accumulation-16','effective-batch-32',
            'BF16-native-masked-flow-loss','optimizer-and-RNG-resume','full-checkpoint-retention'],
        baseline='PI0292 already uses persistent workers and prefetch=2; not a new measured speedup',
        proposed_cache='BOUNDED_VERIFIED_DECODED_RGB_LRU_FOR_NEXT_RUN_ONLY',
        required_acceptance=['full-fixed-input-hash-audit','raw-and-processed-byte-equivalence',
            'same-consumed-trace-across-loader-workers-and-resume',
            'real-loss-gradient-update-and-RNG-equivalence-in-allocation',
            'actual-throughput-RSS-scratch-and-checkpoint-cost',
            'fresh-project-personal-quota-owner-job-and-original-budget-gates'],
        deferred=['VLM-feature-cache','torch-compile','attention-kernel-change','DDP','LoRA'],
        model_calls=0,pbs_submissions=0,performance_claim='NOT_MEASURED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['corelab','hopper'],required=True)
    p.add_argument('--input',required=True);a=p.parse_args()
    with open(a.input) as f:card=json.load(f)
    result=collection_plan(card) if a.kind=='corelab' else hopper_plan(card)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
