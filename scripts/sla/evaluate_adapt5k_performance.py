"""Replay the same 48 pinned examples from each isolated ADAPT engineering seal."""
import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, batch_v2, full_h40_v6
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import roundtrip_error
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, HORIZONS, canonical_hash
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha, measure
from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
from go2arx5_vla.sla_adapter.site_v2 import validate
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import freeze_adapt5k_panels as original
from scripts.sla import freeze_adapt5k_performance_panel as frozen_panel
from scripts.sla.benchmark_adapt5k import BATCH_SCOPE, phase_window, require_batched_output_panel
from scripts.sla.evaluate_adapt5k import diagnostic_metrics, load


THRESHOLDS = dict(xyz_m=.001,so3_rad=.010,gripper_m=.0005,crossing_s=.050)


def load_engineering_seal(effect, training, spec):
    checkpoint = effect['engineering_checkpoint']
    root = Path(checkpoint['path']).resolve(strict=True)
    marker = read_pin(dict(path=str(root/'COMPLETE.json'),sha256=checkpoint['sha256']))
    name = effect['variant']
    step = effect['phase_start']+phase_window(name,BATCH_SCOPE)
    if (effect['scope']!=BATCH_SCOPE or effect['exit_code'] or not effect['engineering_only']
            or effect['eligible_as_formal_parent'] or effect['fixed_output_panel']!=spec['fixed_output_panel']
            or effect['source_sha']!=spec['source_sha'] or effect['actual_optimizer_updates']!=16
            or effect['source_training_card_sha256']!=spec['training_card']['sha256']
            or marker['schema']!='SLA_ADAPT_ENGINEERING_MATCHED_STATE_V1' or marker['scope']!=BATCH_SCOPE
            or marker['eligible_as_formal_parent'] or marker['physics'] is not False
            or marker['variant']!=name or marker['phase_step']!=step or len(marker['arrays'])!=140
            or marker['parent']['sha256']!=effect['parent_checkpoint_sha256']
            or marker['matched_exposure_sha256']!=effect['matched_exposure_sha256']
            or marker['source_sha']!=effect['source_sha']):
        raise ValueError('ADAPT_BATCH_EVAL_REQUIRES_ITS_OWN_SEALED_MATCHED_ENGINEERING_STATE')
    binding = marker['training_binding']
    adapt5k_checkpoint.identity(binding,step)
    inputs = {k:training[k] for k in ('tokenizer','parent_manifest','parent_params')}
    return load(inputs,dict(marker,binding=binding),root)


def evaluate(spec, role):
    import flax.nnx as nnx
    import jax
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    root = Path(spec['registered_root']).resolve(strict=True)
    if (root.stat().st_uid!=os.getuid() or any(root.iterdir()) or spec['generation_cap']!=48
            or spec['physics'] is not False or spec['hardware_allowed'] is not False):
        raise PermissionError('ADAPT_BATCH_EVAL_FRESH_NONPHYSICAL_48_OUTPUT_SCOPE_REQUIRED')
    effect = read_pin(spec['engineering_effect'])
    plan = read_pin(effect['engineering_contract'])
    if require_batched_output_panel(plan)!=spec['fixed_output_panel']:
        raise ValueError('ADAPT_BATCH_EVAL_INPUT_OR_NOISE_PIN_CHANGED')
    panel = read_pin(spec['fixed_output_panel'])
    training = read_pin(spec['training_card'])
    assets = read_pin(panel['eval_input'])
    original_panel = read_pin(panel['original_panel'])
    if (assets['panel']!=panel['original_panel'] or panel['tokenizer_sha256']!=training['tokenizer']['sha256']
            or original_panel['bindings']!=original.bindings(assets)):
        raise ValueError('ADAPT_BATCH_EVAL_REQUIRES_PINNED_D1_AND_SOURCE_DEV')
    rows = {d:[] for d in DOMAINS}
    receipt = dict(role=role,variant=effect['variant'],source_sha=spec['source_sha'],panel=spec['fixed_output_panel'],
        engineering_checkpoint=effect['engineering_checkpoint'],rows=rows,optimization_updates=0,
        attempted=0,generated=0,restored_exact=0,native_requests=0,old_TEST_access=False,exit_code=1)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    def guard():
        if stopped or Path(spec['STOP_path']).exists() or time.time()+60>=spec['deadline_epoch']:
            raise TimeoutError('ADAPT_BATCH_EVAL_BOUNDED_STOP')

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    lock = OwnerLock(spec['neural_owner_lock'])
    start = time.monotonic()
    try:
        guard()
        site = validate(spec)
        if len(jax.devices())!=1 or jax.default_backend()!='gpu' or site['GPU_uuid']!=effect['device']['GPU_uuid']:
            raise PermissionError('ADAPT_BATCH_OUTPUT_AND_RESTORE_REQUIRE_THE_MATCHED_A40')
        receipt['site'] = {k:site[k] for k in ('GPU_uuid','GPU_name','host')}
        reference = None
        if role=='restore':
            reference = read_pin(spec['ordinary_result'])
            if (reference['role']!='ordinary' or reference['exit_code'] or reference['generated']!=48
                    or any(reference[k]!=receipt[k] for k in ('site','engineering_checkpoint','panel','source_sha'))):
                raise ValueError('ADAPT_BATCH_RESTORE_REQUIRES_THE_ORIGINAL_EXACT_48_OUTPUTS')
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        datasets = original.datasets(assets,resize_with_pad)
        tokenizer = Tokenizer(training['tokenizer']['path'])
        graphs,params,frozen = load_engineering_seal(effect,training,spec)
        receipt['load_and_verify_s'] = time.monotonic()-start
        fields,_ = extracted_fields(batch_v2,pi0)
        with np.load(panel['noise']['path'],allow_pickle=False) as noise_bank:
            selected = frozen_panel.selected_rows(original_panel)
            for domain,samples in panel['rows'].items():
                sampler = full_h40_v6.make_sampler(graphs[domain],fields,rematerialize=False,precision='highest')
                fitted = nnx.State.merge(params['shared'],params[domain])
                output = root/domain
                output.mkdir()
                for number,(row,(ordinal,source_row)) in enumerate(zip(samples,selected[domain],strict=True)):
                    guard()
                    if row['ordinal']!=ordinal or any(row[k]!=v for k,v in source_row.items()):
                        raise ValueError('ADAPT_BATCH_EVAL_FIXED_ORDER_CHANGED')
                    dataset = datasets[domain]
                    if dataset.identity(row['index'])!=row['identity']:
                        raise ValueError('ADAPT_BATCH_EVAL_DATA_IDENTITY_CHANGED')
                    packed = prepare_host_batch(dataset,tokenizer,domain,(ordinal,),(row['index'],),training=False)
                    noise = noise_bank[row['noise_key']]
                    shape = (1,HORIZONS[domain],32)
                    if (canonical_hash(frozen_panel.input_hash(packed))!=row['input_sha256']
                            or bytes_sha(noise)!=row['noise_sha256'] or noise.shape!=shape):
                        raise ValueError('ADAPT_BATCH_EVAL_OBSERVATION_OR_NOISE_BYTES_CHANGED')
                    obs,label = batch_v2.device_batch(packed)
                    receipt['attempted'] += 1
                    write(root/'COUNTS.private.json',dict(attempted=receipt['attempted'],generated=receipt['generated']))
                    prediction = np.asarray(sampler(fitted,frozen,obs,jax.device_put(noise)))
                    if prediction.shape!=shape or not np.isfinite(prediction).all():
                        raise ValueError('ADAPT_BATCH_EVAL_COMPLETE_FINITE_ACTION_REQUIRED')
                    receipt['generated'] += 1
                    write(root/'COUNTS.private.json',dict(attempted=receipt['attempted'],generated=receipt['generated']))
                    decoded,metrics = measure(domain,dataset,row,prediction[0],np.asarray(label[0]),original_panel)
                    if not np.isfinite(decoded).all():
                        raise ValueError('ADAPT_BATCH_EVAL_NONFINITE_PHYSICAL_ACTION')
                    if domain=='target':
                        metrics['phase'] = diagnostic_metrics(dataset,row,decoded,metrics)
                    path = output/f'{ordinal:03d}.npz'
                    with path.open('xb') as stream:
                        np.savez(stream,normalized=prediction[0],decoded=decoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    record = dict(row,hashes=dict(normalized=bytes_sha(prediction[0]),decoded=bytes_sha(decoded)),
                                  artifact=dict(path=str(path),sha256=sha(path)),metrics=metrics)
                    rows[domain].append(record)
                    if reference is not None:
                        old = reference['rows'][domain][number]
                        if (old['hashes']!=record['hashes'] or any(old[k]!=v for k,v in row.items())
                                or sha(old['artifact']['path'])!=old['artifact']['sha256']):
                            raise ValueError(f'ADAPT_BATCH_RESTORE_OUTPUT_CHANGED:{domain}:{ordinal}')
                        receipt['restored_exact'] += 1
        receipt.update(status='ADAPT_BATCH_FIXED_OUTPUTS_SEALED',exit_code=0)
    except TimeoutError as error:
        receipt.update(status='ADAPT_BATCH_EVAL_BOUNDED_STOP',failure=str(error),exit_code=75)
    except Exception as error:
        receipt.update(status='ADAPT_BATCH_EVAL_TECHNICAL_TERMINAL',failure=f'{type(error).__name__}: {error}')
    finally:
        try:
            write(root/'RESULT.private.json',dict(receipt,wall_s=time.monotonic()-start))
        finally:
            lock.close()
    return receipt['exit_code']


def target_difference(left, right, first, second):
    delta = roundtrip_error(left,right)
    a,b = (row['phase']['predicted_first_crossing_index'] for row in (first,second))

    def signature(metrics):
        envelope = metrics['envelope']
        reasons = {p:{name:bool(value['exceedance_indices']) for name,value in path['channels'].items()}
                   for p,path in envelope['paths'].items()}
        return (envelope['original_full_H40_pass'],reasons,
                metrics['gripper_seam']['original_execution_speed_exceeded'],metrics['phase']['predicted_P10_due_close'])

    values = dict(xyz_m=delta['xyz_m'],so3_rad=delta['so3_rad'],gripper_m=.040*delta['gripper'],
                  crossing_s=None if a is None or b is None else .05*abs(a-b),
                  crossing_presence_equal=(a is None)==(b is None),gate_and_P10_equal=signature(first)==signature(second))
    return dict(values,compatible=values['crossing_presence_equal'] and values['gate_and_P10_equal']
                and all(values[k] is None or values[k]<=limit for k,limit in THRESHOLDS.items()))


def compare_outputs(reference, candidate, restored):
    from scripts.sla.benchmark_scale20k_outputs import read_arrays

    if (any(r['generated']!=48 or r['attempted']!=48 or r['exit_code'] or r['panel']!=reference['panel']
            or r['site']!=reference['site'] for r in (reference,candidate,restored))
            or restored['restored_exact']!=48 or restored['engineering_checkpoint']!=candidate['engineering_checkpoint']):
        raise ValueError('ADAPT_BATCH_MATCHED_OUTPUTS_AND_INDEPENDENT_RESTORE_REQUIRED')
    results = {}
    for domain in DOMAINS:
        results[domain] = []
        pairs = zip(reference['rows'][domain],candidate['rows'][domain],strict=True)
        for old,new in pairs:
            if any(old[k]!=new[k] for k in ('ordinal','identity','input_sha256','noise_sha256')):
                raise ValueError('ADAPT_BATCH_COMPARE_INPUT_OR_NOISE_CHANGED')
            x,a = read_arrays(old)
            y,b = read_arrays(new)
            row = dict(ordinal=old['ordinal'],normalized_bitwise=old['hashes']['normalized']==new['hashes']['normalized'],
                       decoded_bitwise=old['hashes']['decoded']==new['hashes']['decoded'],
                       maximum_normalized_difference=float(np.max(np.abs(x-y))))
            if domain=='target':
                row.update(target_difference(a,b,old['metrics'],new['metrics']))
            else:
                row['max_native_channel_difference'] = np.max(np.abs(a-b),axis=0).tolist()
            results[domain].append(row)
        if len(results[domain])!=frozen_panel.COUNTS[domain]:
            raise ValueError('ADAPT_BATCH_OUTPUT_COMPARISON_DENOMINATOR_CHANGED')
    return dict(domains=results,physical_target_compatible=all(row['compatible'] for row in results['target']),
                independent_restore_exact=48,thresholds=THRESHOLDS)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=('ordinary','restore'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(evaluate(read_pin(dict(path=args.spec,sha256=args.sha256)),args.role))
