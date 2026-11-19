"""Run real ADAPT5K gradients or a finite, independently gated training segment."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter import adapt5k_data, scale20k_checkpoint, scale20k_model
from go2arx5_vla.sla_adapter.adapt5k_training import AdaptEngine, OPTIMIZER
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, bank_sha256, canonical_hash, make_adamw
from go2arx5_vla.sla_adapter.site_v2 import validate as validate_site
from scripts.pi0292.durable import OwnerLock, write


def inputs(spec, resize_with_pad):
    proof = read_pin(spec['train_data_proof'])
    if (proof['schema'] != 'SLA_ADAPT5K_MATCHED_TRAIN_READERS_V1' or proof['core'] != proof['vanda']
            or proof['vanda']['parent_checkpoint_sha256'] != spec['parent_checkpoint']['sha256']
            or proof['vanda']['parent_step'] != spec['parent_step']
            or any(proof['vanda'][k] for k in ('optimizer_updates','optimization_visits','model_generations',
                                               'native_requests','TEST_access'))):
        raise ValueError('ADAPT5K_REQUIRES_IDENTICAL_REAL_CORE_AND_VANDA_TRAIN_INPUTS')
    report = proof['vanda']
    datasets = adapt5k_data.readers(spec['data'],resize_with_pad)
    banks,digests,_ = adapt5k_data.banks_for(datasets,stage_start=spec['stage_start'],
                                         stage_stop=spec['stage_stop'],target_seed=spec['target_seed'])
    if (digests != report['bank_digests'] or digests != spec['stage_bank_sha256']
            or any(report[k] != spec[k] for k in ('stage_start','stage_stop'))
            or report['target_state_sha256'] != spec['data']['target']['statistics']['sha256']
            or report['inherited_R48_statistics_sha256'] != adapt5k_data.R48_STATS
            or report['total_windows'] != {d:len(datasets[d]) for d in DOMAINS}):
        raise ValueError('ADAPT5K_REAL_INPUTS_OR_VISIT_BANK_BYTES_CHANGED')
    return datasets,banks


def stage_binding(spec, original, model):
    data = spec['data']
    target = data['target']
    return dict(plan=contract.PLAN,recipe=checkpoint.RECIPE,source_sha256=spec['source_sha'],
        parent_checkpoint_sha256=spec['parent_checkpoint']['sha256'],parent_step=spec['parent_step'],
        parent_training_spec_sha256=spec['parent_training_spec']['sha256'],
        parent_manifest_sha256=model['parent_manifest_sha256'],partition_sha256=original['partition_sha256'],
        optimizer_sha256=canonical_hash(OPTIMIZER),tokenizer_sha256=spec['tokenizer']['sha256'],
        stage_start=spec['stage_start'],stage_stop=spec['stage_stop'],
        target_state_sha256=target['statistics']['sha256'],target_D1_sha256=spec['target_D1_sha256'],
        stage_target_data_sha256=canonical_hash([row['manifest']['sha256'] for row in target['captures']]),
        R200_membership_sha256=data['robodojo']['manifest']['sha256'],
        LIBERO_membership_sha256=data['libero']['manifest']['sha256'],
        learning_rates_sha256=bank_sha256(contract.phase_rates()),
        stage_bank_sha256=spec['stage_bank_sha256'],frozen_C20_action_statistics_sha256=C20_STATISTICS)


def restored_engine(spec, datasets, banks, progress):
    import jax

    parent_training = read_pin(spec['parent_training_spec'])
    parent = spec['parent_checkpoint']
    receipt = read_pin(dict(path=str(Path(parent['path'])/'COMPLETE.json'),sha256=parent['sha256']))
    original = receipt['binding']
    if (parent_training['arm'] != 'SCALE-MIX20K' or receipt['step'] != spec['parent_step']
            or receipt['step'] not in contract.FORMAL_PARENTS or original['arm'] != 'SCALE-MIX20K'
            or original['source_sha'] != parent_training['source_sha'] or original.get('engineering_only')
            or original['optimizer_sha256'] != canonical_hash(OPTIMIZER) or parent_training['optimizer'] != OPTIMIZER
            or original['tokenizer_sha256'] != spec['tokenizer']['sha256']
            or original['parent_manifest_sha256'] != spec['parent_manifest']['sha256']
            or original['data_manifest_sha256']['target'] != spec['data']['target']['legacy']['manifest']['sha256']
            or original['target_admission_sha256'] != spec['data']['target']['legacy']['admission']['sha256']
            or len(receipt['arrays']) != 140 or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']):
        raise ValueError('ADAPT5K_INHERITS_THE_ORIGINAL_MIX_OPTIMIZER_AND_FIXED_MODEL')
    progress('LOADING_FROZEN_SLA_BACKBONE')
    graphs,parameters,frozen,model = scale20k_model.load(spec['parent_manifest'],spec['parent_params'])
    loaded_fingerprint = canonical_hash([model['common_shared_fingerprint'],model['common_private_fingerprint']])
    if (original['partition_sha256'] != canonical_hash(model['inventory'])
            or original['initial_state_sha256'] != loaded_fingerprint):
        raise ValueError('ADAPT5K_PARENT_MODEL_AND_TRAINABLE_PARTITION_MUST_MATCH')
    binding = stage_binding(spec,original,model)
    initialize,_ = make_adamw(**OPTIMIZER)
    template = dict(parameters=parameters,optimizer=jax.eval_shape(initialize,parameters),
                    rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED))
    prior = spec['previous_train_or_qualification']
    if prior is None:
        if spec['stage_start'] != 0:
            raise ValueError('ADAPT5K_UNQUALIFIED_LATE_PHASE_START')
        progress('RESTORING_COMPLETE_MIX_PARENT_EXPECTED_PHASE_ZERO')
        state,step,_ = scale20k_checkpoint.restore(parent,arm=original['arm'],binding=original,**template)
        if step != spec['parent_step']:
            raise ValueError('ADAPT5K_MIX_PARENT_STEP_CHANGED')
        phase = 0
    else:
        outcome = read_pin(prior)
        if (outcome['status'] not in ('ACTUAL_ADAPT_ZERO_GRADIENTS_SEALED','ADAPT_FIXED_SEGMENT_SAVED',
                                      'ADAPT_BOUNDED_SAVED_PREFIX')
                or outcome['exit_code'] not in (0,75) or not outcome['checkpoint']
                or outcome['native_requests']):
            raise ValueError('ADAPT5K_RESUME_REQUIRES_CLOSED_SAVED_PRIOR_EFFECTS')
        phase = outcome['end_phase']
        old = outcome['binding']
        if old != binding:
            checkpoint.next_stage_binding(old,binding,phase)
        progress('RESTORING_COMPLETE_ADAPT_PHASE_STATE',phase_step=phase)
        state,clocks = checkpoint.restore(outcome['checkpoint'],binding=old,**template)
        if clocks['phase_step'] != phase:
            raise ValueError('ADAPT5K_ACTUAL_PRIOR_PHASE_CHANGED')
    del template,parameters
    engine = AdaptEngine(graphs,state['parameters'],frozen,datasets,banks,Tokenizer(spec['tokenizer']['path']),
        spec['microbatch'],restored_adam=state['optimizer'],restored_rng=state['rng'],
        parent_step=spec['parent_step'],phase_step=phase,stage_start=spec['stage_start'],stage_stop=spec['stage_stop'])
    return engine,binding


def first_apply_gate(spec):
    prior = read_pin(spec['previous_train_or_qualification'])
    result = read_pin(spec['initial_output_and_restore_actual'])
    if (prior['status'] != 'ACTUAL_ADAPT_ZERO_GRADIENTS_SEALED' or prior['new_updates'] != 0
            or result['schema'] != 'SLA_ADAPT5K_PHASE_ZERO_OUTPUT_RESTORE_ACTUAL_V1'
            or result['checkpoint'] != prior['checkpoint'] or result['phase'] != 0
            or result['ordinary_predictions'] != 288 or result['independent_restore_predictions'] != 48
            or result['independent_restore_exact'] != 48 or result['optimizer_updates'] or result['native_requests']):
        raise ValueError('ADAPT5K_FIRST_APPLY_REQUIRES_ACTUAL_ADAPT0_PREDICTIONS_AND_INDEPENDENT_RESTORE')


def verify_nonzero_restore(spec, engine, binding):
    import jax

    parent = read_pin(spec['previous_train_or_qualification'])
    pin = parent['checkpoint']
    path, marker = checkpoint.read_complete(pin,binding)
    if parent['status'] != 'ADAPT_FIXED_SEGMENT_SAVED' or marker['phase_step'] != checkpoint.TECHNICAL_CHECKPOINT:
        raise ValueError('ADAPT5K_NONZERO_RESTORE_REQUIRES_THE_OWN_ORIGINAL_PHASE25_CHECKPOINT')
    stack = scale20k_checkpoint.training_tree(engine.parameters,engine.states,engine.rng)
    for leaf, row in zip(jax.tree.leaves(stack),marker['arrays'],strict=True):
        saved = np.load(path/row['file'],allow_pickle=False)
        restored = np.asarray(leaf)
        if saved.dtype != restored.dtype or saved.shape != restored.shape or not np.array_equal(
                saved.reshape(-1).view(np.uint8), restored.reshape(-1).view(np.uint8)):
            raise ValueError('ADAPT5K_PHASE25_INDEPENDENT_STATE_RESTORE_CHANGED:'+row['key'])
    return pin, len(marker['arrays'])


def first_nonzero_gate(spec):
    prior = read_pin(spec['previous_train_or_qualification'])
    proof = read_pin(spec['phase25_restore_actual'])
    if (proof['status'] != 'ADAPT_PHASE25_INDEPENDENT_STATE_RESTORED_EXACT'
            or proof['checkpoint'] != prior['checkpoint'] or proof['exact_arrays'] != 140
            or proof['end_phase'] != checkpoint.TECHNICAL_CHECKPOINT or proof['new_updates']
            or proof['native_requests']):
        raise ValueError('ADAPT5K_UPDATE_26_REQUIRES_PHASE25_INDEPENDENT_RESTORE')
    outputs = read_pin(spec['phase25_output_restore_actual'])
    if (outputs['schema'] != 'SLA_ADAPT5K_PHASE25_OUTPUT_RESTORE_ACTUAL_V1'
            or outputs['checkpoint'] != prior['checkpoint'] or outputs['phase'] != checkpoint.TECHNICAL_CHECKPOINT
            or outputs['model_generations'] != 48 or outputs['matched_output_pairs'] != 24
            or outputs['optimizer_updates'] or outputs['native_requests']):
        raise ValueError('ADAPT5K_UPDATE_26_REQUIRES_THE_FIXED_PHASE25_OUTPUT_RESTORE')


def run(spec, mode):
    import jax
    import optax
    from openpi_client.image_tools import resize_with_pad

    root = Path(spec['registered_root']).resolve(strict=True)
    deadline,reserve = spec['deadline_epoch'],spec['save_reserve_s']
    if (root.stat().st_uid != os.getuid() or any(root.iterdir()) or spec['plan'] != contract.PLAN
            or spec['physics'] is not False or spec['hardware_allowed'] is not False or not 300 <= reserve
            or deadline-time.time() <= reserve or (mode == 'qualify') != (spec['previous_train_or_qualification'] is None)):
        raise PermissionError('ADAPT5K_FRESH_FINITE_NONPHYSICAL_TRAINING_SCOPE_REQUIRED')
    site = validate_site(spec)
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise PermissionError('ADAPT5K_REQUIRES_ACTUAL_SINGLE_GPU_ALLOCATION')
    jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
    flag = False
    def signal_stop(*_):
        nonlocal flag
        flag = True
    signal.signal(signal.SIGINT,signal_stop)
    signal.signal(signal.SIGTERM,signal_stop)

    def guard():
        if flag or Path(spec['STOP_path']).exists() or time.time() >= deadline-reserve:
            raise TimeoutError('ADAPT5K_SAVE_BEFORE_ACTUAL_ALLOCATION_DEADLINE')

    def progress(stage, **detail):
        write(root/'PROGRESS.private.json',dict(status=stage,epoch=time.time(),**detail))

    lock = OwnerLock(spec['neural_owner_lock'])
    result = dict(scope=checkpoint.RECIPE,mode=mode,source_sha=spec['source_sha'],
                  site=site,new_updates=0,native_requests=0,native_ticks=0)
    engine,latest,start = None,None,0
    began = time.monotonic()
    try:
        guard()
        progress('VALIDATING_TRAIN_DATA_AND_VISIT_BANKS')
        datasets,banks = inputs(spec,resize_with_pad)
        engine,binding = restored_engine(spec,datasets,banks,progress)
        result['binding'] = binding
        start = engine.step

        def seal():
            return checkpoint.save(root/f'phase{engine.step:05d}',phase_step=engine.step,parameters=engine.parameters,
                                   optimizer=engine.states,rng=engine.rng,binding=binding)

        if mode == 'qualify':
            progress('MEASURING_FULL_128_REAL_GRADIENT_NO_ADAM_APPLY')
            gradient,receipt = engine.measure_gradient(guard)
            jax.block_until_ready(gradient)
            norm = float(optax.global_norm(gradient))
            if not 0 < norm < float('inf') or engine.gradient_windows != contract.BATCH or engine.completed_aux != 4:
                raise ValueError('ADAPT5K_REAL_FINITE_WEIGHTED_128_GRADIENT_AND_FOUR_AUX_REQUIRED')
            del gradient
            latest = seal()
            result.update(status='ACTUAL_ADAPT_ZERO_GRADIENTS_SEALED',exit_code=0,checkpoint=latest,
                          qualification=receipt,weighted_gradient_norm=norm)
        elif mode == 'restore25':
            prior, count = verify_nonzero_restore(spec,engine,binding)
            result.update(status='ADAPT_PHASE25_INDEPENDENT_STATE_RESTORED_EXACT',
                          exit_code=0,checkpoint=prior,exact_arrays=count)
        else:
            limit = spec['phase_stop_limit']
            if (not start < limit <= spec['stage_stop']
                    or (start < checkpoint.TECHNICAL_CHECKPOINT and limit != checkpoint.TECHNICAL_CHECKPOINT)):
                raise ValueError('ADAPT5K_REQUIRES_FIRST_25_AND_FORWARD_BOUNDED_SEGMENT')
            if start == 0:
                first_apply_gate(spec)
            if start == checkpoint.TECHNICAL_CHECKPOINT:
                first_nonzero_gate(spec)
            interrupted = False
            with (root/'UPDATES.private.jsonl').open('x') as ledger:
                while engine.step < limit:
                    try:
                        row = engine.update(guard)
                    except TimeoutError:
                        interrupted = True
                        break
                    ledger.write(json.dumps(row,allow_nan=False)+'\n')
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    progress('DURABLE_ADAPT_UPDATE',phase_step=engine.step,cursors=row['phase_cursors'])
                    if engine.step in checkpoint.ENDPOINTS or engine.step == limit:
                        latest = seal()
            if interrupted and engine.step > start and (latest is None or Path(latest['path']).name != f'phase{engine.step:05d}'):
                latest = seal()
            result.update(status='ADAPT_BOUNDED_SAVED_PREFIX' if interrupted else 'ADAPT_FIXED_SEGMENT_SAVED',
                          exit_code=75 if interrupted else 0,checkpoint=latest)
    except (Exception, KeyboardInterrupt):
        result.update(status='ADAPT_TECHNICAL_TERMINAL_RECONCILE_BEFORE_REPLAY',exit_code=1,
                      last_complete_checkpoint=latest,checkpoint=None,failure=traceback.format_exc())
    finally:
        lock.close()
    end = engine.step if engine else 0
    result.update(start_phase=start,end_phase=end,new_updates=end-start,
        optimization_visits={d:(end-start)*contract.BATCH[d] for d in DOMAINS},
        gradient_inputs=engine.gradient_windows if engine else dict.fromkeys(DOMAINS,0),
        complete_H40_generations=engine.completed_aux if engine else 0,
        attempted_H40_generations=engine.attempted_aux if engine else 0,wall_s=time.monotonic()-began)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('qualify','restore25','train'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec,sha256=args.sha256))
    result = run(spec,args.mode)
    write(Path(spec['registered_root'])/'RESULT.private.json',result)
    raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
