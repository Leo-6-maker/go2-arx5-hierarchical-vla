"""Bounded SCALE20K worker. The allocated site supplies the pinned spec."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

from go2arx5_vla.sla_adapter import scale20k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter import scale20k_data, scale20k_model, scale20k_training
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN, RECIPE, bank_sha256, canonical_hash, cursors
from go2arx5_vla.sla_adapter.site_v2 import validate
from scripts.pi0292.durable import OwnerLock, write


def run(spec, mode):
    import jax
    from openpi_client.image_tools import resize_with_pad

    arm = spec['arm']
    root = Path(spec['registered_root']).resolve(strict=True)
    deadline = spec['deadline_epoch']
    reserve = spec['save_reserve_s']
    if (spec['plan_sha'] != PLAN or spec['recipe'] != RECIPE or spec['hardware_allowed'] is not False
            or spec['physics'] is not False or spec['new_native_requests_cap'] != 0
            or type(reserve) is not int or reserve < 300 or deadline-time.time() <= reserve
            or Path(spec['dataset']).resolve(strict=True) != Path(spec['data']['target']['root']).resolve(strict=True)
            or (root/'RESULT.private.json').exists()):
        raise PermissionError('SCALE20K_REQUIRES_EXACT_FRESH_NONPHYSICAL_BOUNDED_SITE_SPEC')
    site = validate(spec)
    if len(jax.devices()) != 1 or jax.default_backend() != 'gpu':
        raise PermissionError('SCALE20K_ACTUALLY_ALLOCATED_SINGLE_GPU_REQUIRED')
    jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
    stopped = False
    heartbeat = root/'PROGRESS.private.json'

    def progress(phase, **detail):
        write(heartbeat,dict(phase=phase,mode=mode,epoch=time.time(),**detail))

    def signal_stop(_signum,_frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,signal_stop)

    def guard():
        if stopped or (root/'STOP.json').exists() or Path(spec['STOP_path']).exists() or time.time() >= deadline-reserve:
            raise TimeoutError('SCALE20K_SAVE_EXIT_AT_FINITE_SITE_BOUNDARY')

    lock = OwnerLock(spec['neural_owner_lock'])
    start = time.monotonic()
    engine = None
    first_step = 0
    initial_counts = {'complete_generations':0,'attempts':0}
    latest = None
    result = dict(scope=RECIPE,arm=arm,mode=mode,site=site,new_native_requests=0,new_native_ticks=0,
                  started_epoch=time.time(),source_sha=spec['source_sha'])
    try:
        guard()
        progress('VALIDATING_FROZEN_DATA')
        datasets = scale20k_data.readers(spec['data'],resize_with_pad)
        banks, bank_digests = scale20k_data.bank_for(datasets)
        if bank_digests != spec['visit_bank_sha256']:
            raise ValueError('SCALE20K_VISIT_BANK_BYTES_DIFFER_FROM_PREREGISTERED_DATA')
        token = spec['tokenizer']
        if sha(token['path']) != token['sha256']:
            raise ValueError('SCALE20K_ORIGINAL_TOKENIZER_BYTES_REQUIRED')
        progress('LOADING_PINNED_SLA_BASE')
        graphs,parameters,frozen,receipt = scale20k_model.load(spec['parent_manifest'],spec['parent_params'])
        optimizer = spec['optimizer']
        engine = scale20k_training.Engine(arm,graphs,parameters,frozen,datasets,banks,Tokenizer(token['path']),
                                           spec['microbatch'],optimizer,rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED))
        del parameters
        binding = dict(recipe=RECIPE,plan_sha=PLAN,arm=arm,source_sha=spec['source_sha'],
            parent_manifest_sha256=receipt['parent_manifest_sha256'],partition_sha256=canonical_hash(receipt['inventory']),
            initial_state_sha256=canonical_hash([receipt['common_shared_fingerprint'],receipt['common_private_fingerprint']]),
            optimizer_sha256=canonical_hash(optimizer),learning_rates_sha256=bank_sha256(engine.rates),
            data_manifest_sha256={d:spec['data'][d]['manifest']['sha256'] for d in DOMAINS},visit_bank_sha256=bank_digests,
            target_admission_sha256=spec['data']['target']['admission']['sha256'],tokenizer_sha256=token['sha256'])
        if binding['learning_rates_sha256'] != spec['learning_rates_sha256']:
            raise ValueError('SCALE20K_ORIGINAL_CONTINUOUS_LR_TABLE_REQUIRED')
        write(root/'MODEL_AT_START.private.json',dict(receipt,binding=binding))

        def seal(label):
            return checkpoint.save(root/label,arm=arm,step=engine.step,parameters=engine.parameters,
                                   optimizer=engine.states,rng=engine.rng,binding=binding)

        if mode=='qualify':
            measured = {}
            for domain in DOMAINS:
                if engine.batch_size[domain]:
                    progress('MEASURING_REAL_GPU_GRADIENTS',domain=domain,completed_domains=list(measured))
                    gradient,measured[domain] = engine.domain_gradient(domain,guard)
                    del gradient
            progress('SAVING_COMPLETE_STEP_ZERO',completed_domains=list(measured))
            latest = seal('step00000')
            return dict(result,exit_code=0,status='ACTUAL_GPU_GRADIENTS_SEALED_NO_UPDATES',checkpoint=latest,
                        receipt=measured,gradient_input_windows=engine.gradient_windows,new_updates=0,new_visits=cursors(arm,0),
                        new_model_generations=engine.completed_aux,attempted_model_generations=engine.attempted_aux,
                        wall_s=time.monotonic()-start)

        prior = read_pin(spec['resume_from_closed_segment'])
        if prior['arm'] != arm or prior['exit_code'] not in (0,75) or not prior['checkpoint']:
            raise ValueError('SCALE20K_RESUME_NEEDS_A_COLD_RESOLVED_PRIOR_EFFECT')
        latest = prior['checkpoint']
        progress('RESTORING_COMPLETE_STATE',prior=latest['sha256'])
        state,engine.step,positions = checkpoint.restore(latest,arm=arm,parameters=engine.parameters,
                                                         optimizer=engine.states,rng=engine.rng,binding=binding)
        if prior['global_step'] != engine.step or prior['global_cursors'] != positions:
            raise ValueError('SCALE20K_PRIOR_LEDGER_AND_CHECKPOINT_DO_NOT_MATCH')
        if prior['mode']=='qualify':
            if engine.step or prior['status']!='ACTUAL_GPU_GRADIENTS_SEALED_NO_UPDATES':
                raise ValueError('SCALE20K_FIRST_UPDATE_REQUIRES_REAL_STATION_GRADIENT_RECEIPT')
        elif prior['mode']!='train':
            raise ValueError('SCALE20K_UNKNOWN_PRIOR_SEGMENT_MODE')
        engine.parameters,engine.states,engine.rng = state['parameters'],state['optimizer'],state['rng']
        del state
        first_step = engine.step
        progress('RESTORED_COMPLETE_STATE',step=engine.step,cursors=positions,checkpoint_sha256=latest['sha256'])
        limit = spec['global_stop_step']
        if type(limit) is not int or not first_step < limit <= CHECKPOINTS[arm][-1] or (not first_step and limit>200):
            raise ValueError('SCALE20K_FIRST_TECHNICAL_PREFIX_AND_FINITE_SEGMENT_REQUIRED')
        initial_counts = dict(complete_generations=engine.completed_aux,attempts=engine.attempted_aux)
        code = 0
        with (root/'UPDATES.private.jsonl').open('x') as journal:
            while engine.step < limit:
                try:
                    row = engine.update(guard)
                except TimeoutError:
                    code = 75
                    break
                progress('COMMITTED_OPTIMIZER_UPDATE',step=engine.step,cursors=row['cursors'])
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                journal.flush()
                os.fsync(journal.fileno())
                if engine.step in CHECKPOINTS[arm] or engine.step==limit:
                    latest = seal(f'step{engine.step:05d}')
            if code==75 and engine.step>first_step and Path(latest['path']).name != f'step{engine.step:05d}':
                latest = seal(f'step{engine.step:05d}')
        result.update(exit_code=code,status='BOUNDED_SAVED_PREFIX' if code else 'FIXED_SEGMENT_SAVED',checkpoint=latest)
    except BaseException:
        result.update(exit_code=1,status='TECHNICAL_TERMINAL_DO_NOT_REPLAY_UNKNOWN_EFFECTS',failure=traceback.format_exc(),
                      last_complete_checkpoint=latest,checkpoint=None)
    finally:
        lock.close()
    step = engine.step if engine is not None else 0
    end = cursors(arm,step)
    before = cursors(arm,first_step)
    return dict(result,global_step=step,global_cursors=end,new_updates=step-first_step,
                new_visits={d:end[d]-before[d] for d in DOMAINS},wall_s=time.monotonic()-start,
                new_model_generations=(engine.completed_aux-initial_counts['complete_generations']) if engine else 0,
                attempted_model_generations=(engine.attempted_aux-initial_counts['attempts']) if engine else 0,
                gradient_input_windows=engine.gradient_windows if engine else dict.fromkeys(DOMAINS,0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('qualify','train'))
    parser.add_argument('spec',type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=str(args.spec),sha256=args.sha256))
    result = run(spec,args.mode)
    if args.mode=='qualify' and result['exit_code']==0:
        result.update(global_step=0,global_cursors=cursors(spec['arm'],0))
    write(Path(spec['registered_root'])/'RESULT.private.json',result)
    raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
