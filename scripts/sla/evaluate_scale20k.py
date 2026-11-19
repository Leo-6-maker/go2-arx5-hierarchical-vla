"""Freeze one diagnostic panel or evaluate a formal SCALE20K checkpoint."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6, scale20k_checkpoint, scale20k_model
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, HORIZONS, PLAN, RECIPE, canonical_hash, make_adamw
from go2arx5_vla.sla_adapter.scale20k_data import readers
from go2arx5_vla.sla_adapter.scale20k_evaluation import PER_DOMAIN, RESTORE, bytes_sha, data_sha, measure, prepare, validate
from go2arx5_vla.sla_adapter.scale20k_training import host_batch
from scripts.pi0292.durable import OwnerLock, write


def evidence_spec(spec, mode):
    training = read_pin(spec['training_spec'])
    arm, step = spec['arm'], spec['step']
    pin = spec['checkpoint']
    endpoint = read_pin(dict(path=str(Path(pin['path'])/'COMPLETE.json'), sha256=pin['sha256']))
    binding = endpoint['binding']
    data = data_sha(training['data'])
    expected = dict(arm=arm, plan_sha=PLAN, recipe=RECIPE, source_sha=training['source_sha'],
        parent_manifest_sha256=training['parent_manifest']['sha256'],
        optimizer_sha256=canonical_hash(training['optimizer']), tokenizer_sha256=training['tokenizer']['sha256'],
        learning_rates_sha256=training['learning_rates_sha256'], visit_bank_sha256=training['visit_bank_sha256'],
        data_manifest_sha256=data['manifests'], target_admission_sha256=data['target_admission'])
    if (step not in CHECKPOINTS[arm] or step != endpoint['step'] or training['arm'] != arm
            or any(binding[k] != v for k, v in expected.items()) or spec['plan'] != PLAN
            or spec['model_generation_cap'] != len(DOMAINS)*(PER_DOMAIN if mode == 'evaluate' else len(RESTORE))
            or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or spec['updates_cap'] != 0 or spec['native_request_cap'] != 0):
        raise ValueError('SCALE20K_FORMAL_NONPHYSICAL_ENDPOINT_AND_BOUNDED_ROLE_REQUIRED')
    for key in ('parent_params', 'dataset'):
        if Path(spec[key]).resolve(strict=True) != Path(training[key]).resolve(strict=True):
            raise ValueError('SCALE20K_OFFLINE_SITE_MODEL_AND_DATA_MUST_MATCH_TRAINING')
    return training, binding


def restore_engine(training, binding, pin):
    import jax

    graphs, parameters, frozen, model = scale20k_model.load(training['parent_manifest'], training['parent_params'])
    if (binding['partition_sha256'] != canonical_hash(model['inventory'])
            or binding['initial_state_sha256'] != canonical_hash([model['common_shared_fingerprint'],
                                                                 model['common_private_fingerprint']])):
        raise ValueError('SCALE20K_OFFLINE_RESTORED_PARENT_AND_PARAMETER_PARTITION_CHANGED')
    initialize, _ = make_adamw(**training['optimizer'])
    tree, _, _ = scale20k_checkpoint.restore(pin, arm=training['arm'], parameters=parameters,
        optimizer=initialize(parameters), rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED), binding=binding)
    return graphs, tree['parameters'], frozen


def summarize(rows, domain):
    summary = dict(windows=len(rows), groups=dict(Counter(r['group'] for r in rows)))
    for key in ('normalized_active_mae', 'normalized_padding_rms', 'normalized_action_motion', 'label_action_motion'):
        summary[key] = float(np.mean([r['metrics'][key] for r in rows]))
    if domain == 'target':
        for group in (*summary['groups'], 'all'):
            items = [r['metrics'] for r in rows if group == 'all' or r['group'] == group]
            summary[group] = dict(H40_and_seam_pass=sum(m['H40_and_seam_pass'] for m in items),
                predictions=len(items), label_closure_events=sum(m['width']['expert_crossing_in_window'] is not None for m in items),
                missed_closures=sum(m['width']['missed_in_window_crossing'] for m in items),
                width_MAE_m=float(np.mean([m['width']['width_MAE_m'] for m in items])))
    return summary


def run(spec, mode):
    import flax.nnx as nnx
    import jax
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.site_v2 import validate as validate_site
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    root = Path(spec['registered_root']).resolve(strict=True)
    training, binding = evidence_spec(spec, mode)
    panel = read_pin(spec['panel'])
    if any(root.iterdir()):
        raise FileExistsError('SCALE20K_OFFLINE_ROLE_REQUIRES_FRESH_OUTPUT_DIRECTORY')
    prior = None
    if mode == 'restore':
        prior = read_pin(spec['evaluation_result'])
        if (prior['status'] != 'ENDPOINT_PREDICTIONS_SEALED' or prior['checkpoint'] != spec['checkpoint']
                or prior['panel'] != spec['panel'] or prior['source_sha'] != spec['source_sha']
                or any(len(prior['rows'][d]) != PER_DOMAIN for d in DOMAINS)
                or prior['new_model_generations'] != len(DOMAINS)*PER_DOMAIN):
            raise ValueError('SCALE20K_OFFLINE_FRESH_RESTORE_REQUIRES_ORIGINAL_COMPLETE_DIAGNOSTIC')

    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def guard():
        if stopped or time.time() >= spec['deadline_epoch']-30 or (root/'STOP.json').exists() or Path(spec['STOP_path']).exists():
            raise TimeoutError('SCALE20K_OFFLINE_FINITE_WINDOW_ENDED')

    rows = {d: [] for d in DOMAINS}
    actual = dict(source_sha=spec['source_sha'], arm=spec['arm'], step=spec['step'], mode=mode,
        checkpoint=spec['checkpoint'], panel=spec['panel'], rows=rows, optimization_updates=0, optimization_visits=0,
        new_model_generations=0, generation_attempts=0, new_native_requests=0, new_native_ticks=0, old_TEST_access=False)
    began = time.monotonic()

    def progress(pending):
        write(root/'PROGRESS.private.json', dict(mode=mode, arm=spec['arm'], step=spec['step'],
            pending=pending, generation_attempts=actual['generation_attempts'],
            new_model_generations=actual['new_model_generations'], sealed_rows={d:len(rows[d]) for d in DOMAINS},
            epoch=time.time(), new_native_requests=0))

    lock = OwnerLock(spec['neural_owner_lock'])
    try:
        try:
            guard()
            actual['site'] = validate_site(spec)
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise PermissionError('SCALE20K_OFFLINE_REQUIRES_ONE_ACTUALLY_ALLOCATED_GPU')
            jax.config.update('jax_compilation_cache_dir', spec['compilation_cache_dir'])
            datasets = readers(training['data'], resize_with_pad, source_split='SOURCE_DEV')
            validate(panel, training['data'], datasets)
            token = training['tokenizer']
            if sha(token['path']) != token['sha256']:
                raise ValueError('SCALE20K_OFFLINE_ORIGINAL_TOKENIZER_CHANGED')
            tokenizer = Tokenizer(token['path'])
            graphs, parameters, frozen = restore_engine(training, binding, spec['checkpoint'])
            fields, _ = extracted_fields(batch_v2, pi0)
            for domain in DOMAINS:
                dataset = datasets[domain]
                state = nnx.State.merge(parameters['shared'], parameters[domain])
                sample = full_h40_v6.make_sampler(graphs[domain], fields, rematerialize=False, precision='highest')
                ordinals = range(PER_DOMAIN) if prior is None else RESTORE
                directory = root/domain
                directory.mkdir()
                for ordinal in ordinals:
                    guard()
                    row = panel['rows'][domain][ordinal]
                    obs, labels = host_batch(dataset, tokenizer, domain, (ordinal,), (row['index'],), training=False)
                    shape = (1, HORIZONS[domain], 32)
                    noise = jax.random.normal(jax.random.PRNGKey(row['noise_id']), shape)
                    actual['generation_attempts'] += 1
                    progress([domain, ordinal])
                    sampled = np.asarray(sample(state, frozen, obs, noise))
                    if sampled.shape != shape or not np.isfinite(sampled).all():
                        raise ValueError('SCALE20K_OFFLINE_FINITE_FULL_NATIVE_HORIZON_REQUIRED')
                    actual['new_model_generations'] += 1
                    prediction = sampled[0]
                    physical, metrics = measure(domain, dataset, row, prediction, np.asarray(labels[0]), panel)
                    if not np.isfinite(physical).all():
                        raise ValueError('SCALE20K_OFFLINE_DECODED_ACTION_MUST_BE_FINITE')
                    file = directory/f'{ordinal:03d}.npz'
                    with file.open('xb') as out:
                        np.savez(out, normalized=prediction, decoded=physical)
                        out.flush()
                        os.fsync(out.fileno())
                    record = dict(row, output=dict(path=str(file), sha256=sha(file)), ordinal=ordinal,
                        normalized_sha256=bytes_sha(prediction), decoded_sha256=bytes_sha(physical), metrics=metrics)
                    rows[domain].append(record)
                    if prior is not None:
                        reference = prior['rows'][domain][ordinal]
                        if sha(reference['output']['path']) != reference['output']['sha256']:
                            raise ValueError(f'SCALE20K_OFFLINE_REFERENCE_OUTPUT_CHANGED:{domain}:{ordinal}')
                        if any(reference[key] != record[key] for key in (*row, 'ordinal', 'normalized_sha256', 'decoded_sha256')):
                            raise ValueError(f'SCALE20K_OFFLINE_FRESH_PREDICTION_CHANGED:{domain}:{ordinal}')
                    progress(None)
            actual.update(status='ENDPOINT_PREDICTIONS_SEALED' if prior is None else 'FRESH_RGB48_RESTORED_EXACT',
                          summary={d: summarize(rows[d], d) for d in DOMAINS}, exit_code=0)
        except (Exception, KeyboardInterrupt) as error:
            actual.update(status='BOUNDED_OFFLINE_ROLE_FAILED', failure=f'{type(error).__name__}: {error}', exit_code=1)
        write(root/'RESULT.private.json', dict(actual, wall_s=time.monotonic()-began))
    finally:
        lock.close()
    return actual['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('panel', 'evaluate', 'restore'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('--panel-output', type=Path)
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec, sha256=args.sha256))
    if args.mode == 'panel':
        from openpi_client.image_tools import resize_with_pad

        if args.panel_output is None or args.panel_output.exists():
            parser.error('panel requires an unused --panel-output')
        write(args.panel_output, prepare(spec['data'], resize_with_pad))
        print(sha(args.panel_output))
    else:
        raise SystemExit(run(spec, args.mode))


if __name__ == '__main__':
    main()
