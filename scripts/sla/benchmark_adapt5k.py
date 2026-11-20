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
from go2arx5_vla.sla_adapter.scale20k_training import UpdateBreakdown, compile_fm_accumulation
from go2arx5_vla.sla_adapter.site_v2 import validate
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.train_adapt5k import inputs, restored_engine


SCOPE = 'ADAPT5K_MB1_THROUGHPUT_PROBE_V1'
BATCH_SCOPE = 'ADAPT5K_BATCH_THROUGHPUT_PROBE_V1'
WARMUP, TIMED, PROFILE = 3, 5, 2
BATCH_WARMUP, BATCH_TIMED = 4, 12
VARIANTS = ('serial','prefetch','prefetch_deferred')
BATCH_VARIANTS = {'compiled_mb1': 1, 'compiled_mb2': 2, 'compiled_mb4': 4}
TWO_GPU_VARIANT = 'compiled_two_gpu_mb2'


def probe_variants(scope):
    if scope == SCOPE:
        return VARIANTS
    if scope == BATCH_SCOPE:
        return tuple(BATCH_VARIANTS)
    raise ValueError('EXPECTED_ADAPT_ENGINEERING_SCOPE')


def phase_window(variant, scope=SCOPE):
    if scope==BATCH_SCOPE and (variant in BATCH_VARIANTS or variant==TWO_GPU_VARIANT):
        return BATCH_WARMUP+BATCH_TIMED
    if variant not in probe_variants(scope):
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


def timing_summary(rows, *, timed=TIMED):
    if len(rows)!=timed or any('synchronized_intervals_s' in row for row in rows):
        raise ValueError('ADAPT_THROUGHPUT_REQUIRES_THE_FIXED_UNPROFILED_MATCHED_WINDOWS')
    wall = [row['wall_s'] for row in rows]
    count = sum(contract.BATCH.values())
    memory = [row.get('actual_GPU_memory_stats') or {} for row in rows]
    devices = {row.get('devices',1) for row in rows}
    if len(devices)!=1:
        raise ValueError('ADAPT_TIMED_ROWS_MUST_USE_THE_SAME_GPU_COUNT')
    count_devices, = devices
    per_device_peak = []
    if count_devices==2:
        for device_rows in zip(*(row['per_device_memory_stats'] for row in rows),strict=True):
            per_device_peak.append(max((entry['peak_bytes_in_use'] for entry in device_rows
                                        if entry and 'peak_bytes_in_use' in entry),default=None))
    return dict(updates=len(rows),total_step_wall_s=sum(wall),update_p50_s=statistics.median(wall),
                update_p90_s=statistics.quantiles(wall,n=10,method='inclusive')[8],
                FM_windows_per_second=len(rows)*count/sum(wall),GPU_hours_for_timed_updates=sum(wall)*count_devices/3600,
                GPU_peak_bytes_in_use=max((m['peak_bytes_in_use'] for m in memory if 'peak_bytes_in_use' in m),default=None),
                GPU_peak_pool_bytes=max((m['peak_pool_bytes'] for m in memory if 'peak_pool_bytes' in m),default=None),
                per_device_peak_bytes_in_use=per_device_peak)


def exposure_fingerprint(rows, *, fm_microbatch=1):
    checks = []
    for row in rows:
        domains = row['domains']
        phase = row['phase_step']
        if (set(domains)!=set(contract.DOMAINS) or any(domains[d]['backward_calls']*fm_microbatch!=contract.BATCH[d]
                                                      for d in contract.DOMAINS)
                or domains['target']['training_full_generations']!=4 or tuple(domains['target']['auxiliary_slots'])!=contract.auxiliary_slots(phase-1)):
            raise ValueError('ADAPT_PROBE_UPDATE_EXPOSURE_CHANGED')
        checks.append(dict(phase=phase,learning_rate=row['learning_rate'],cursors=row['lineage_cursors'],
                           visits={d:domains[d]['visit_slice_sha256'] for d in contract.DOMAINS},
                           auxiliary_slots=domains['target']['auxiliary_slots']))
    return canonical_hash(checks)


def require_batched_output_panel(plan):
    from scripts.sla import freeze_adapt5k_performance_panel as frozen
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha

    pin = plan['fixed_output_panel']
    panel = read_pin(pin)
    if (panel['schema'] != frozen.SCHEMA or panel['plan'] != contract.PLAN or panel['old_TEST_access']
            or panel['counts'] != frozen.COUNTS or canonical_hash(panel['rows']) != panel['input_and_noise_sha256']
            or len({row['noise_key'] for rows in panel['rows'].values() for row in rows}) != 48
            or {d:len(rows) for d,rows in panel['rows'].items()} != frozen.COUNTS
            or panel['ordinary_per_variant'] != 48 or panel['independent_restore_per_variant'] != 48
            or sha(panel['noise']['path']) != panel['noise']['sha256']
            or plan.get('export_matched_engineering') is not True):
        raise ValueError('ADAPT_BATCH_REQUIRES_PINNED_48_OUTPUTS_AND_INDEPENDENT_RESTORE')
    return pin


def paired_reference(spec, prior, panel):
    from scripts.sla.adapt5k_vanda import pbs_history
    from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry

    scope = read_pin(spec['paired_single_scope'])
    job = read_pbs_with_auth_retry(pbs_history,scope['job_id'])
    pin = scope['completed']['compiled_mb2']
    reference = read_pin(pin)
    if (scope['scope']!=BATCH_SCOPE or scope['exit_code'] or not scope['engineering_only']
            or job['job_state']!='F' or job.get('Exit_status')!=0
            or job['project']!=spec['project'] or job['Job_Owner'].split('@')[0]!=os.environ.get('USER')
            or reference['variant']!='compiled_mb2' or reference['fm_microbatch']!=2
            or reference['exit_code'] or reference['actual_optimizer_updates']!=16
            or reference['throughput']['updates']!=12 or reference['fixed_output_panel']!=panel
            or reference['parent_checkpoint_sha256']!=prior['checkpoint']['sha256']
            or reference['source_training_card_sha256']!=spec['training_card']['sha256']
            or reference['phase_start']!=prior['end_phase']):
        raise ValueError('ADAPT_PAIR_REQUIRES_COLD_SAME_ROUND_SINGLE_A40_MB2_REFERENCE')
    return pin,reference


def compare(receipts):
    if not receipts:
        raise ValueError('ADAPT_THROUGHPUT_REQUIRES_COMPLETE_RECEIPTS')
    scope = receipts[0]['scope']
    variants = probe_variants(scope)
    timed = BATCH_TIMED if scope == BATCH_SCOPE else TIMED
    if len(receipts)!=len(variants) or {row['variant'] for row in receipts}!=set(variants):
        raise ValueError('ADAPT_THROUGHPUT_REQUIRES_ONE_COMPLETE_RECEIPT_PER_VARIANT')
    runs = {row['variant']:row for row in receipts}
    reference = runs[variants[0]]
    identity = ('source_sha','parent_checkpoint_sha256','source_training_card_sha256','engineering_contract',
                'phase_start','device','matched_exposure_sha256')
    checks = {}
    for name, row in runs.items():
        count = phase_window(name,scope)
        expected = {d:count*n for d,n in contract.BATCH.items()}
        if (row['scope']!=scope or row['status']!='ADAPT_ENGINEERING_PROBE_COMPLETE' or row['exit_code']
                or row['actual_optimizer_updates']!=count or row['committed_ledger_updates']!=count
                or row['FM_gradient_windows']!=expected or row['attempted_H40']!=4*count or row['completed_H40']!=4*count
                or row['scientific_updates'] or row['native_requests'] or row['native_ticks']
                or row['throughput']['updates']!=timed or any(row[k]!=reference[k] for k in identity)
                or set(row['matched_end_state'])!=set(reference['matched_end_state']) or len(row['matched_end_state'])!=140
                or (scope==BATCH_SCOPE and (row['fm_microbatch']!=BATCH_VARIANTS[name] or not row['compiled_fm_accumulation']
                                            or row['fixed_output_panel']!=reference['fixed_output_panel']))):
            raise ValueError(f'ADAPT_THROUGHPUT_UNMATCHED_ENGINEERING_RECEIPT:{name}')
        differences = [k for k,v in row['matched_end_state'].items() if reference['matched_end_state'][k]!=v]
        speedup = row['throughput']['FM_windows_per_second']/reference['throughput']['FM_windows_per_second']
        speedup_key = 'speedup_vs_serial' if scope == SCOPE else 'speedup_vs_compiled_mb1'
        checks[name] = dict(throughput=row['throughput'],**{speedup_key:speedup},state_identical=not differences,
                            states_verified=140,first_differing_leaves=differences[:12])
    return dict(scope=scope,engineering_only=True,eligible_as_formal_parent=False,
                matched_exposure_sha256=reference['matched_exposure_sha256'],
                candidate_checks=checks,new_optimizer_updates=0,new_native_requests=0)


def benchmark(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad

    root = Path(spec['registered_root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    variant,scope = spec['variant'],spec['scope']
    count = phase_window(variant,scope)
    parallel = variant==TWO_GPU_VARIANT
    microbatch = 2 if parallel else BATCH_VARIANTS.get(variant,1)
    compiled = scope == BATCH_SCOPE
    warmup,timed = (BATCH_WARMUP,BATCH_TIMED) if compiled else (WARMUP,TIMED)
    card = read_pin(spec['training_card'])
    prior = read_pin(spec['sealed_predecessor'])
    plan = read_pin(spec['engineering_contract'])
    start = prior['end_phase']
    if (root.stat().st_uid!=os.getuid() or any(root.iterdir()) or spec['plan']!=contract.PLAN
            or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or (source/'SOURCE_SHA.private').read_text().strip()!=spec['source_sha']
            or plan['scope']!=scope or plan['parent_checkpoint_sha256']!=prior['checkpoint']['sha256']
            or plan['training_card_sha256']!=spec['training_card']['sha256'] or plan['update_limit'][variant]!=count
            or prior['exit_code'] or prior['status']!='ADAPT_FIXED_SEGMENT_SAVED' or prior['native_requests']
            or (parallel and spec['expected_pair_parent_sha256']!=prior['checkpoint']['sha256'])
            or card['stage_start']!=prior['binding']['stage_start'] or card['stage_stop']!=prior['binding']['stage_stop']
            or not card['stage_start']<start<start+count<=card['stage_stop']
            or card['microbatch']!=dict.fromkeys(contract.DOMAINS,1)):
        raise PermissionError('ADAPT_BENCHMARK_REQUIRES_FRESH_BOUNDED_MATCHED_NONPHYSICAL_SCOPE')
    checkpoint.read_complete(prior['checkpoint'],prior['binding'])
    output_panel = require_batched_output_panel(plan) if compiled else None
    lock = OwnerLock(spec['neural_owner_lock'])
    started = time.monotonic()
    stopped = False
    engine = None
    rows = []
    result = dict(scope=scope,variant=variant,source_sha=spec['source_sha'],engineering_only=True,
                  fm_microbatch=microbatch,compiled_fm_accumulation=compiled,allocated_GPU_count=2 if parallel else 1,
                  eligible_as_formal_parent=False,parent_checkpoint_sha256=prior['checkpoint']['sha256'],
                  source_training_card_sha256=spec['training_card']['sha256'],engineering_contract=spec['engineering_contract'],
                  phase_start=start,scientific_updates=0,native_requests=0,native_ticks=0,exit_code=1)
    if output_panel is not None:
        result['fixed_output_panel'] = output_panel

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
        if parallel:
            from go2arx5_vla.sla_adapter.adapt5k_pair_site import validate as validate_pair

            site = validate_pair(spec)
            reference_pin,reference = paired_reference(spec,prior,output_panel)
            if any(site[key]!=reference['device'][key] for key in ('GPU_uuid','GPU_name','host')):
                raise PermissionError('ADAPT_PAIR_REFERENCE_A40_MUST_BE_THE_PRIMARY_DEVICE')
            result['pair_site'] = site
            result['paired_single_mb2'] = reference_pin
        else:
            site = validate(spec)
        if len(jax.local_devices())!=result['allocated_GPU_count'] or jax.default_backend()!='gpu':
            raise PermissionError('ADAPT_BENCHMARK_REQUIRES_THE_ACTUALLY_ALLOCATED_GPUS')
        result['device'] = {key:site[key] for key in ('GPU_name','GPU_uuid','host')}
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        datasets,banks = inputs(card,resize_with_pad)
        def progress(stage, **detail):
            write(root/'PROGRESS.private.json',dict(stage=stage,**detail))

        engine,binding = restored_engine(dict(card,previous_train_or_qualification=spec['sealed_predecessor'],
                                              microbatch=dict.fromkeys(contract.DOMAINS,microbatch)),datasets,banks,progress)
        if engine.step!=start or binding!=prior['binding']:
            raise ValueError('ADAPT_BENCHMARK_MUST_RESTORE_THE_ORIGINAL_MODEL_ADAM_RNG_AND_CURSORS')
        result['load_and_restore_s'] = time.monotonic()-started
        executor = nullcontext() if variant=='serial' else ThreadPoolExecutor(max_workers=1,thread_name_prefix='adapt-reader')
        with (root/'UPDATES.private.jsonl').open('x') as ledger,executor as input_executor:
            engine.input_executor = input_executor
            engine.defer_fm_loss = variant=='prefetch_deferred' or compiled
            update = engine.update
            if parallel:
                from go2arx5_vla.sla_adapter.adapt5k_parallel_gradients import ParallelAdaptUpdate

                update = ParallelAdaptUpdate(engine,jax.local_devices()).update
            elif compiled:
                engine.batched_fm_accumulation = {domain:compile_fm_accumulation(gradient)
                                                 for domain,gradient in engine.grad.items()}
            for i in range(warmup+timed):
                append(ledger,'warmup' if i<warmup else 'timed',update(guard))
            result['matched_end_state'] = state_fingerprint(engine)
            if len(result['matched_end_state'])!=140:
                raise ValueError('ADAPT_ENGINEERING_STATE_MUST_INCLUDE_ALL_140_LEAVES')
            result['matched_exposure_sha256'] = exposure_fingerprint(rows,fm_microbatch=microbatch)
            result['throughput'] = timing_summary(rows[warmup:],timed=timed)
            if parallel:
                if result['matched_exposure_sha256']!=reference['matched_exposure_sha256']:
                    raise ValueError('ADAPT_PAIR_EXPOSURE_CHANGED_FROM_MATCHED_SINGLE_MB2')
                single = reference['throughput']
                scaling = result['throughput']['FM_windows_per_second']/single['FM_windows_per_second']
                gpu_hours = result['throughput']['GPU_hours_for_timed_updates']/single['GPU_hours_for_timed_updates']
                result['strong_scaling'] = dict(throughput_ratio=scaling,GPU_hours_ratio=gpu_hours,
                    timed_thresholds_met=scaling>=1.5 and gpu_hours<=4/3,output_qualification_pending=True)
            if plan.get('export_matched_engineering') is True:
                from go2arx5_vla.sla_adapter.scale20k_checkpoint import seal_arrays, training_tree

                result['engineering_checkpoint'] = seal_arrays(
                    root/'engineering_state',training_tree(engine.parameters,engine.states,engine.rng),
                    dict(schema='SLA_ADAPT_ENGINEERING_MATCHED_STATE_V1',eligible_as_formal_parent=False,
                         parent=prior['checkpoint'],variant=variant,phase_step=engine.step,
                         matched_exposure_sha256=result['matched_exposure_sha256'],
                         training_binding=binding,scope=scope,
                         source_sha=spec['source_sha'],physics=False))
            write(root/'MATCHED_INTERVAL.private.json',dict(result,committed_ledger_updates=len(rows),
                  actual_optimizer_updates=engine.step-start,FM_gradient_windows=engine.gradient_windows,
                  attempted_H40=engine.attempted_aux,completed_H40=engine.completed_aux))
            if variant=='serial':
                trace = root/'trace'
                capture = (jax.profiler.trace(str(trace),create_perfetto_trace=True)
                           if plan.get('capture_cuda_trace',True) else nullcontext())
                with capture:
                    for _ in range(PROFILE):
                        guard()
                        engine.update_breakdown = UpdateBreakdown()
                        append(ledger,'synchronized_profile',engine.update(guard))
                result['profiled_updates'] = len(rows)-WARMUP-TIMED
                result['profile_stage_seconds'] = [r['synchronized_intervals_s'] for r in rows[-PROFILE:]]
                result['cuda_trace_requested'] = plan.get('capture_cuda_trace',True)
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
