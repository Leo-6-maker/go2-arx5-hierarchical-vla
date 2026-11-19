"""Evaluate sealed raw ADAPT phases against the fixed D1 and source DEV panels."""
import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6, scale20k_checkpoint, scale20k_model
from go2arx5_vla.sla_adapter import adapt5k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, HORIZONS, canonical_hash, make_adamw
from go2arx5_vla.sla_adapter.scale20k_evaluation import PER_DOMAIN, RESTORE, bytes_sha, measure, panel_rows
from go2arx5_vla.sla_adapter.scale20k_training import host_batch
from go2arx5_vla.sla_adapter.site_v2 import validate as validate_site
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import freeze_adapt5k_panels as panels


ROLES = ('formal','formal-restore','phase25-before','phase25-restore')
PROOFS = {0: 'SLA_ADAPT5K_PHASE_ZERO_OUTPUT_RESTORE_ACTUAL_V1',
          25: 'SLA_ADAPT5K_PHASE25_OUTPUT_RESTORE_ACTUAL_V1'}


def diagnostic_metrics(dataset, row, physical, metrics):
    """Count crossings from the actual preceding command, including an index-zero seam."""
    index = row['index']
    before = dataset.reference(index)['width_m']
    state, expert, _, _ = dataset.raw(index)
    del state

    def crossing(action):
        clocks = np.r_[before, .002+.040*np.asarray(action)[:,9]]
        crosses = np.flatnonzero((clocks[:-1] >= .022) & (clocks[1:] < .022))
        return int(crosses[0]) if len(crosses) else None

    expert_cross, model_cross = crossing(expert),crossing(physical)
    error = None if expert_cross is None or model_cross is None else .050*abs(expert_cross-model_cross)
    due = expert_cross is not None and expert_cross < 10
    if row['phase_bucket'] == 'P10_DUE_CLOSE' and not due:
        raise ValueError('ADAPT5K_P10_PHYSICAL_LABEL_NO_LONGER_MATCHES_FROZEN_PHASE_ROW')
    predicted_due = model_cross is not None and model_cross < 10
    return dict(expert_first_crossing_index=expert_cross, predicted_first_crossing_index=model_cross,
                first22mm_crossing_error_s=error, expert_P10_due_close=due, predicted_P10_due_close=predicted_due,
                close_within100ms=error is not None and error <= .100,
                missed_expert_close=expert_cross is not None and model_cross is None,
                timely_P10_close_and_original_gates=due and predicted_due and error is not None
                    and error <= .100 and metrics['H40_and_seam_pass'])


def selections(panel, role):
    if role.startswith('phase25'):
        indices = panels.PHASE25_ORDINALS
    elif role == 'formal-restore':
        indices = RESTORE
    else:
        indices = range(PER_DOMAIN)
    queues = {domain:[(i,panel['formal'][domain][i]) for i in indices] for domain in DOMAINS}
    if role == 'formal':
        queues['target_phase'] = list(enumerate(panel['phase_train']))
    return queues


def load(spec, marker, root):
    import jax

    binding = marker['binding']
    if (binding['tokenizer_sha256'] != spec['tokenizer']['sha256']
            or sha(spec['tokenizer']['path']) != binding['tokenizer_sha256']
            or binding['parent_manifest_sha256'] != spec['parent_manifest']['sha256']
            or binding['optimizer_sha256'] != canonical_hash(OPTIMIZER)):
        raise ValueError('ADAPT5K_OFFLINE_REQUIRES_ORIGINAL_MODEL_TOKENIZER_AND_PARAMETER_LAYOUT')
    graphs, parameters, frozen, model = scale20k_model.load(spec['parent_manifest'],spec['parent_params'])
    if canonical_hash(model['inventory']) != binding['partition_sha256']:
        raise ValueError('ADAPT5K_OFFLINE_PARAMETER_PARTITION_CHANGED')
    initialize,_ = make_adamw(**OPTIMIZER)
    state = scale20k_checkpoint.training_tree(parameters,jax.eval_shape(initialize,parameters),
        jax.random.PRNGKey(scale20k_model.INITIAL_SEED))
    leaves,_ = jax.tree_util.tree_flatten_with_path(state)
    selected = [jax.device_put(value) for key,value in scale20k_checkpoint.verified_arrays(root,marker,leaves)
                if key[0] == jax.tree_util.DictKey('parameters')]
    _,treedef = jax.tree_util.tree_flatten(parameters)
    return graphs, jax.tree_util.tree_unflatten(treedef,selected), frozen


def evaluate(spec, role):
    import flax.nnx as nnx
    import jax
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    root = Path(spec['registered_root']).resolve(strict=True)
    run = read_pin(spec['sealed_training_actual'])
    cp = run['checkpoint']
    path,marker = checkpoint.read_complete(cp, run['binding'])
    phase = marker['phase_step']
    if (spec['plan'] != contract.PLAN or spec['hardware_allowed'] is not False or spec['physics'] is not False
            or root.stat().st_uid != os.getuid() or any(root.iterdir()) or phase != run['end_phase']
            or run['exit_code'] != 0 or cp != spec['checkpoint']
            or (phase != 25 if role.startswith('phase25') else phase not in checkpoint.ENDPOINTS)):
        raise PermissionError('ADAPT5K_OFFLINE_REQUIRES_FRESH_ROLE_AND_REAL_SEALED_PHASE')
    panel = read_pin(spec['panel'])
    queues = selections(panel,role)
    cap = sum(map(len,queues.values()))
    if cap != spec['generation_cap']:
        raise ValueError('ADAPT5K_FIXED_OFFLINE_GENERATION_DENOMINATOR_CHANGED')
    if role.endswith('restore'):
        prior = read_pin(spec['before'])
        expected_role = 'formal' if role == 'formal-restore' else 'phase25-before'
        if (prior['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or prior['role'] != expected_role
                or prior['checkpoint'] != cp or prior['panel'] != spec['panel'] or prior['new_updates']):
            raise ValueError('ADAPT5K_INDEPENDENT_RESTORE_REQUIRES_EXACT_BEFORE_ROLE')
        reference = {d:{r['ordinal']:r for r in prior['rows'][d]} for d in DOMAINS}
    else:
        reference = None
    stat = spec['target_D1']['statistics']['sha256']
    if (panel['schema'] != panels.SCHEMA or panel['plan'] != contract.PLAN or panel['old_TEST_access']
            or panel['bindings'] != panels.bindings(spec) or stat != marker['binding']['target_state_sha256']
            or spec['D1_freeze_sha256'] != marker['binding']['target_D1_sha256']
            or tuple(panel['restore_ordinals']) != RESTORE or tuple(panel['phase25_ordinals']) != panels.PHASE25_ORDINALS):
        raise ValueError('ADAPT5K_REQUIRES_THE_ORIGINAL_D1_PANEL_AND_TRAIN_STATISTICS')
    flag = False

    def receive_stop(*_):
        nonlocal flag
        flag = True
    signal.signal(signal.SIGTERM,receive_stop)
    signal.signal(signal.SIGINT,receive_stop)

    def guard():
        if flag or Path(spec['STOP_path']).exists() or time.time() >= spec['deadline_epoch']-60:
            raise TimeoutError('ADAPT5K_OFFLINE_BOUNDED_STOP')

    rows = {d:[] for d in queues}
    actual = dict(role=role,phase=phase,checkpoint=cp,panel=spec['panel'],source_sha=spec['source_sha'],rows=rows,
                  new_updates=0,optimization_visits=0,native_requests=0,old_TEST_access=False,
                  generation_attempts=0,model_generations=0,matched_output_pairs=0)
    lock = OwnerLock(spec['neural_owner_lock'])
    try:
        guard()
        actual['site'] = validate_site(spec)
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise PermissionError('ADAPT5K_OFFLINE_REQUIRES_ONE_ACTUAL_GPU')
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        datasets = panels.datasets(spec,resize_with_pad)
        if panel_rows(datasets) != panel['formal'] or panels.phase_rows(datasets['target'])[0] != panel['phase_train']:
            raise ValueError('ADAPT5K_FROZEN_D1_WINDOW_SELECTION_CHANGED')
        tokens = Tokenizer(spec['tokenizer']['path'])
        graphs,parameters,frozen = load(spec,marker,path)
        fields,_ = extracted_fields(batch_v2,pi0)
        for kind, queue in queues.items():
            domain = 'target' if kind == 'target_phase' else kind
            dataset = datasets[domain]
            fitted = nnx.State.merge(parameters['shared'],parameters[domain])
            sampler = full_h40_v6.make_sampler(graphs[domain],fields,rematerialize=False,precision='highest')
            out = root/kind
            out.mkdir()
            for ordinal,row in queue:
                guard()
                obs,truth = host_batch(dataset,tokens,domain,(ordinal,),(row['index'],),training=False)
                shape = (1,HORIZONS[domain],32)
                noise = jax.random.normal(jax.random.PRNGKey(row['noise_id']),shape)
                actual['generation_attempts']+=1
                prediction = np.asarray(sampler(fitted,frozen,obs,noise))
                if prediction.shape != shape or not np.isfinite(prediction).all():
                    raise ValueError('ADAPT5K_OFFLINE_REQUIRES_COMPLETE_FINITE_NATIVE_HORIZON')
                actual['model_generations']+=1
                normalized = prediction[0]
                physical,metrics = measure(domain,dataset,row,normalized,np.asarray(truth[0]),panel)
                if not np.isfinite(physical).all():
                    raise ValueError('ADAPT5K_OFFLINE_NONFINITE_PHYSICAL_DECODE')
                if kind == 'target_phase':
                    metrics['phase'] = diagnostic_metrics(dataset,row,physical,metrics)
                path = out/f'{ordinal:03d}.npz'
                with path.open('xb') as stream:
                    np.savez(stream,normalized=normalized,decoded=physical)
                    stream.flush()
                    os.fsync(stream.fileno())
                record = dict(row,ordinal=ordinal,normalized_sha256=bytes_sha(normalized),
                    decoded_sha256=bytes_sha(physical),output=dict(path=str(path),sha256=sha(path)),metrics=metrics)
                rows[kind].append(record)
                if reference is not None:
                    expected = reference[domain][ordinal]
                    hashes = ('normalized_sha256','decoded_sha256')
                    if any(record[key] != expected[key] for key in (*row,*hashes)) or sha(
                            expected['output']['path']) != expected['output']['sha256']:
                        raise ValueError(f'ADAPT5K_INDEPENDENT_RESTORE_OUTPUT_CHANGED:{domain}:{ordinal}')
                    actual['matched_output_pairs']+=1
        actual.update(status='ADAPT_OFFLINE_RESULTS_SEALED',exit_code=0)
    except (Exception,KeyboardInterrupt) as error:
        actual.update(status='ADAPT_OFFLINE_ROLE_FAILED',exit_code=1,failure=f'{type(error).__name__}: {error}')
    finally:
        write(root/'RESULT.private.json',actual)
        lock.close()
    return actual['exit_code']


def restore_proof(ordinary, restored):
    same = ordinary['checkpoint'] == restored['checkpoint'] and ordinary['panel'] == restored['panel']
    counts = {domain:len(ordinary['rows'][domain]) for domain in DOMAINS}
    phase = ordinary['phase']
    base = dict(phase=phase,checkpoint=ordinary['checkpoint'],panel=ordinary['panel'],optimizer_updates=0,
                native_requests=0)
    if (not same or phase != restored['phase'] or ordinary['role'] == restored['role']
            or any(r['exit_code'] or r['new_updates'] or r['native_requests'] for r in (ordinary,restored))):
        raise ValueError('ADAPT5K_RESTORE_PROOF_REQUIRES_TWO_SUCCESSFUL_SAME_IDENTITY_ROLES')
    if phase == 25 and (ordinary['role'],restored['role']) == ('phase25-before','phase25-restore'):
        if ordinary['model_generations'] == restored['model_generations'] == restored['matched_output_pairs'] == 24:
            return dict(base,schema=PROOFS[25],model_generations=48,matched_output_pairs=24)
    if (ordinary['role'],restored['role']) == ('formal','formal-restore'):
        if (counts == dict.fromkeys(DOMAINS,96)
                and restored['model_generations'] == restored['matched_output_pairs'] == 48):
            return dict(base,schema=PROOFS.get(phase,'SLA_ADAPT5K_FORMAL_OUTPUT_RESTORE_ACTUAL_V1'),
                        ordinary_predictions=288,independent_restore_predictions=48,independent_restore_exact=48,
                        phase_train_predictions=len(ordinary['rows']['target_phase']))
    raise ValueError('ADAPT5K_EXACT_OFFLINE_AND_RESTORE_DENOMINATORS_REQUIRED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=(*ROLES,'proof'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec,sha256=args.sha256))
    if args.role == 'proof':
        output = Path(spec['output'])
        if output.exists():
            raise FileExistsError(output)
        write(output,restore_proof(read_pin(spec['ordinary']),read_pin(spec['independent_restore'])))
    else:
        raise SystemExit(evaluate(spec,args.role))


if __name__ == '__main__':
    main()
