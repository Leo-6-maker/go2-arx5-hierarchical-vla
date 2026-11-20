"""Benchmark ADAPT from a cold checkpoint without publishing another training endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint as checkpoint, adapt5k_contract as contract
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
from go2arx5_vla.sla_adapter.scale20k_training import UpdateBreakdown
from go2arx5_vla.sla_adapter.site_v2 import validate
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.train_adapt5k import inputs, restored_engine


SCOPE = 'ADAPT5K_MB1_THROUGHPUT_PROBE_V1'
WARMUP, TIMED, PROFILE = 3, 5, 2
VARIANTS = ('serial','prefetch','prefetch_deferred')


def phase_window(variant):
    if variant not in VARIANTS:
        raise ValueError('EXPECTED_ADAPT_ENGINEERING_VARIANT')
    return WARMUP+TIMED+(PROFILE if variant=='serial' else 0)


def state_fingerprint(engine):
    import jax

    leaves,_ = jax.tree_util.tree_flatten_with_path(dict(parameters=engine.parameters,optimizer=engine.states,rng=engine.rng))
    records = {}
    for key,value in leaves:
        array = np.ascontiguousarray(value)
        records[jax.tree_util.keystr(key)] = dict(dtype=str(array.dtype),shape=list(value.shape),
                                                  sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return records


def timing_summary(rows):
    if len(rows)!=TIMED or any('synchronized_intervals_s' in row for row in rows):
        raise ValueError('ADAPT_THROUGHPUT_REQUIRES_THE_FIXED_UNPROFILED_MATCHED_WINDOWS')
    wall = [row['wall_s'] for row in rows]
    count = sum(contract.BATCH.values())
    return dict(updates=len(rows),total_step_wall_s=sum(wall),update_p50_s=statistics.median(wall),
                update_p90_s=statistics.quantiles(wall,n=10,method='inclusive')[8],
                FM_windows_per_second=len(rows)*count/sum(wall),GPU_hours_for_timed_updates=sum(wall)/3600)


def exposure_fingerprint(rows):
    checks = []
    for row in rows:
        domains = row['domains']
        phase = row['phase_step']
        if (set(domains)!=set(contract.DOMAINS) or any(domains[d]['backward_calls']!=contract.BATCH[d] for d in contract.DOMAINS)
                or domains['target']['training_full_generations']!=4 or tuple(domains['target']['auxiliary_slots'])!=contract.auxiliary_slots(phase-1)):
            raise ValueError('ADAPT_PROBE_UPDATE_EXPOSURE_CHANGED')
        checks.append(dict(phase=phase,learning_rate=row['learning_rate'],cursors=row['lineage_cursors'],
                           visits={d:domains[d]['visit_slice_sha256'] for d in contract.DOMAINS},
                           auxiliary_slots=domains['target']['auxiliary_slots']))
    return canonical_hash(checks)


def compare(receipts):
    if len(receipts)!=len(VARIANTS) or {row['variant'] for row in receipts}!=set(VARIANTS):
        raise ValueError('ADAPT_THROUGHPUT_REQUIRES_ONE_COMPLETE_RECEIPT_PER_VARIANT')
    runs = {row['variant']:row for row in receipts}
    serial = runs['serial']
    identity = ('source_sha','parent_checkpoint_sha256','source_training_card_sha256','engineering_contract',
                'phase_start','device','matched_exposure_sha256')
    checks = {}
    for name, row in runs.items():
        count = phase_window(name)
        expected = {d:count*n for d,n in contract.BATCH.items()}
        if (row['scope']!=SCOPE or row['status']!='ADAPT_ENGINEERING_PROBE_COMPLETE' or row['exit_code']
                or row['actual_optimizer_updates']!=count or row['committed_ledger_updates']!=count
                or row['FM_gradient_windows']!=expected or row['attempted_H40']!=4*count or row['completed_H40']!=4*count
                or row['scientific_updates'] or row['native_requests'] or row['native_ticks']
                or row['throughput']['updates']!=TIMED or any(row[k]!=serial[k] for k in identity)
                or set(row['matched_end_state'])!=set(serial['matched_end_state']) or len(row['matched_end_state'])!=140):
            raise ValueError(f'ADAPT_THROUGHPUT_UNMATCHED_ENGINEERING_RECEIPT:{name}')
        differences = [k for k,v in row['matched_end_state'].items() if serial['matched_end_state'][k]!=v]
        speedup = row['throughput']['FM_windows_per_second']/serial['throughput']['FM_windows_per_second']
        checks[name] = dict(throughput=row['throughput'],speedup_vs_serial=speedup,state_identical=not differences,
                            states_verified=140,first_differing_leaves=differences[:12])
    return dict(scope=SCOPE,engineering_only=True,matched_exposure_sha256=serial['matched_exposure_sha256'],
                candidate_checks=checks,new_optimizer_updates=0,new_native_requests=0)


def benchmark(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad

    root = Path(spec['registered_root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    variant = spec['variant']
    count = phase_window(variant)
    card = read_pin(spec['training_card'])
    prior = read_pin(spec['sealed_predecessor'])
    plan = read_pin(spec['engineering_contract'])
    start = prior['end_phase']
    if (root.stat().st_uid!=os.getuid() or any(root.iterdir()) or spec['plan']!=contract.PLAN
            or spec['scope']!=SCOPE or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or (source/'SOURCE_SHA.private').read_text().strip()!=spec['source_sha']
            or plan['scope']!=SCOPE or plan['parent_checkpoint_sha256']!=prior['checkpoint']['sha256']
            or plan['training_card_sha256']!=spec['training_card']['sha256'] or plan['update_limit'][variant]!=count
            or prior['exit_code'] or prior['status']!='ADAPT_FIXED_SEGMENT_SAVED' or prior['native_requests']
            or card['stage_start']!=prior['binding']['stage_start'] or card['stage_stop']!=prior['binding']['stage_stop']
            or not card['stage_start']<start<start+count<=card['stage_stop']
            or card['microbatch']!=dict.fromkeys(contract.DOMAINS,1)):
        raise PermissionError('ADAPT_BENCHMARK_REQUIRES_FRESH_BOUNDED_MATCHED_NONPHYSICAL_SCOPE')
    checkpoint.read_complete(prior['checkpoint'],prior['binding'])
    lock = OwnerLock(spec['neural_owner_lock'])
    started = time.monotonic()
    stopped = False
    engine = None
    rows = []
    result = dict(scope=SCOPE,variant=variant,source_sha=spec['source_sha'],engineering_only=True,
                  eligible_as_formal_parent=False,parent_checkpoint_sha256=prior['checkpoint']['sha256'],
                  source_training_card_sha256=spec['training_card']['sha256'],engineering_contract=spec['engineering_contract'],
                  phase_start=start,scientific_updates=0,native_requests=0,native_ticks=0,exit_code=1)

    def receive_stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,receive_stop)

    def guard():
        if stopped or Path(spec['STOP_path']).exists() or time.time()+300>=spec['deadline_epoch']:
            raise TimeoutError('ADAPT_BENCHMARK_BOUNDED_STOP')

    def append(ledger, tag, row):
        ledger.write(json.dumps(dict(row,measurement=tag),allow_nan=False)+'\n')
        ledger.flush()
        os.fsync(ledger.fileno())
        rows.append(row)

    try:
        guard()
        site = validate(spec)
        if len(jax.devices())!=1 or jax.default_backend()!='gpu':
            raise PermissionError('ADAPT_BENCHMARK_REQUIRES_ONE_ACTUALLY_ALLOCATED_GPU')
        result['device'] = {key:site[key] for key in ('GPU_name','GPU_uuid','host')}
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        datasets,banks = inputs(card,resize_with_pad)
        def progress(stage, **detail):
            write(root/'PROGRESS.private.json',dict(stage=stage,**detail))

        engine,binding = restored_engine(dict(card,previous_train_or_qualification=spec['sealed_predecessor']),datasets,banks,progress)
        if engine.step!=start or binding!=prior['binding']:
            raise ValueError('ADAPT_BENCHMARK_MUST_RESTORE_THE_ORIGINAL_MODEL_ADAM_RNG_AND_CURSORS')
        result['load_and_restore_s'] = time.monotonic()-started
        executor = nullcontext() if variant=='serial' else ThreadPoolExecutor(max_workers=1,thread_name_prefix='adapt-reader')
        with (root/'UPDATES.private.jsonl').open('x') as ledger,executor as input_executor:
            engine.input_executor = input_executor
            engine.defer_fm_loss = variant=='prefetch_deferred'
            for i in range(WARMUP+TIMED):
                append(ledger,'warmup' if i<WARMUP else 'timed',engine.update(guard))
            result['matched_end_state'] = state_fingerprint(engine)
            if len(result['matched_end_state'])!=140:
                raise ValueError('ADAPT_ENGINEERING_STATE_MUST_INCLUDE_ALL_140_LEAVES')
            result['matched_exposure_sha256'] = exposure_fingerprint(rows)
            result['throughput'] = timing_summary(rows[WARMUP:WARMUP+TIMED])
            if variant=='serial':
                trace = root/'trace'
                with jax.profiler.trace(str(trace),create_perfetto_trace=True):
                    for _ in range(PROFILE):
                        guard()
                        engine.update_breakdown = UpdateBreakdown()
                        append(ledger,'synchronized_profile',engine.update(guard))
                result['profiled_updates'] = len(rows)-WARMUP-TIMED
                result['profile_stage_seconds'] = [r['synchronized_intervals_s'] for r in rows[-PROFILE:]]
        result.update(status='ADAPT_ENGINEERING_PROBE_COMPLETE',exit_code=0)
    except TimeoutError as error:
        result.update(status='ADAPT_ENGINEERING_BOUNDED_STOP',exit_code=75,failure=str(error))
    except Exception as error:
        result.update(status='ADAPT_ENGINEERING_TECHNICAL_TERMINAL',failure=f'{type(error).__name__}: {error}')
    finally:
        result.update(wall_s=time.monotonic()-started,committed_ledger_updates=len(rows),
            actual_optimizer_updates=engine.step-start if engine else 0,
            FM_gradient_windows=engine.gradient_windows if engine else dict.fromkeys(contract.DOMAINS,0),
            attempted_H40=engine.attempted_aux if engine else 0,completed_H40=engine.completed_aux if engine else 0)
        try:
            write(root/'RESULT.private.json',result)
        finally:
            lock.close()
    return result['exit_code']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('--compare',action='store_true',help='Compare the three pinned engineering results in this spec')
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec,sha256=args.sha256))
    if args.compare:
        print(json.dumps(compare([read_pin(p) for p in spec['results']]),allow_nan=False))
    else:
        raise SystemExit(benchmark(spec))
