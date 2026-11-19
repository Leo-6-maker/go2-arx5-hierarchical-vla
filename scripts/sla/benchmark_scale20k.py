"""Measure one isolated SCALE-MIX engineering variant from the sealed step-500 state."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import time
import traceback

from go2arx5_vla.sla_adapter import scale20k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter import scale20k_data, scale20k_model, scale20k_training
from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, bank_sha256, canonical_hash, cursors
from go2arx5_vla.sla_adapter.site_v2 import validate
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.evaluate_scale20k import relocated_training


ARM = 'SCALE-MIX20K'
START, WARMUP, UPDATES = 500, 10, 30
SCOPE = 'SCALE_MIX_MATCHED_ENGINEERING30_V1'


def summarize(rows):
    elapsed = rows[WARMUP:]
    if len(rows) != UPDATES or [row['step'] for row in rows] != list(range(START+1, START+UPDATES+1)):
        raise ValueError('BENCHMARK_WARM_METRICS_REQUIRE_ALL_THIRTY_MATCHED_UPDATES')

    def stage(row, key):
        return sum(row['domains'][d][key] for d in DOMAINS)

    columns = dict(update_wall=[r['wall_s'] for r in elapsed],
                   data_wait=[stage(r,'data_wait_s') + stage(r,'auxiliary_input_s') for r in elapsed],
                   fm_backward=[stage(r,'FM_backward_s') for r in elapsed],
                   h40_aux_backward=[stage(r,'auxiliary_backward_s') for r in elapsed],
                   optimizer=[r['optimizer_s'] for r in elapsed])
    return {key:dict(p50=statistics.median(values),p90=statistics.quantiles(values,n=10,method='inclusive')[8])
            for key,values in columns.items()}


def benchmark(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad

    root = Path(spec['registered_root']).resolve(strict=True)
    parent = spec['parent_checkpoint']
    card = read_pin(spec['parent_training_spec'])
    training = relocated_training(card,read_pin(spec['site_assets']))
    scope = read_pin(spec['engineering_contract'])
    size,variant = spec['microbatch'],spec['variant']
    if (root.stat().st_uid != os.getuid() or spec['plan'] != PLAN or scope['plan'] != PLAN
            or spec['scope'] != SCOPE or scope['scope'] != SCOPE or variant not in scope['variants']
            or not (type(size) is int and size in (1,2,4)) or scope['variants'][variant]['microbatch'] != size
            or scope['parent_checkpoint_sha256'] != parent['sha256'] or scope['warmup_updates'] != WARMUP
            or scope['updates_per_variant'] != UPDATES or card['arm'] != ARM
            or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or (root/'RESULT.private.json').exists() or (root/'UPDATES.private.jsonl').exists()):
        raise PermissionError('MIX_BENCHMARK_REQUIRES_FIXED_MATCHED_NONPHYSICAL_ENGINEERING_SCOPE')
    site = validate(spec)
    if len(jax.devices()) != 1 or jax.default_backend() != 'gpu':
        raise PermissionError('MIX_BENCHMARK_REQUIRES_ONE_ACTUALLY_ALLOCATED_GPU')
    jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
    deadline = spec['deadline_epoch']
    stopped = False

    def receive_stop(_number,_frame):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,receive_stop)

    def guard():
        if stopped or Path(spec['STOP_path']).exists() or time.time()>=deadline-300:
            raise TimeoutError('MIX_BENCHMARK_BOUNDED_STOP_RESERVE')

    def report(phase, **fields):
        write(root/'PROGRESS.private.json',dict(phase=phase,epoch=time.time(),**fields))

    lock = OwnerLock(spec['neural_owner_lock'])
    rows = []
    engine = None
    result = dict(scope=SCOPE,engineering_only=True,eligible_as_formal_parent=False,variant=variant,
                  source_sha=spec['source_sha'],parent_sha256=parent['sha256'],microbatch=size,
                  device_name=site['GPU_name'],scientific_updates=0,native_requests=0,native_ticks=0)
    try:
        guard()
        report('VALIDATING_ORIGINAL_THREE_DOMAIN_TRAINS')
        datasets = scale20k_data.readers(training['data'],resize_with_pad)
        banks, ids = scale20k_data.bank_for(datasets)
        if ids != card['visit_bank_sha256']:
            raise ValueError('MIX_BENCHMARK_MUST_REUSE_ORIGINAL_VISIT_BANKS')
        token = training['tokenizer']
        if sha(token['path']) != token['sha256']:
            raise ValueError('MIX_BENCHMARK_MUST_REUSE_THE_ORIGINAL_TOKENIZER')
        marker = read_pin(dict(path=str(Path(parent['path'])/'COMPLETE.json'),sha256=parent['sha256']))
        original = marker['binding']
        if (marker['step'] != START or original['source_sha'] != card['source_sha']
                or original['optimizer_sha256'] != canonical_hash(card['optimizer'])
                or original['tokenizer_sha256'] != token['sha256'] or original['visit_bank_sha256'] != ids
                or original['data_manifest_sha256'] != {d:card['data'][d]['manifest']['sha256'] for d in DOMAINS}
                or original['target_admission_sha256'] != card['data']['target']['admission']['sha256']):
            raise ValueError('MIX_BENCHMARK_ORIGINAL_CHECKPOINT_BINDING_MISMATCH')
        report('LOADING_FROZEN_SLA_BASE')
        load_started = time.monotonic()
        graphs, params, frozen, model = scale20k_model.load(training['parent_manifest'],training['parent_params'])
        if (original['partition_sha256'] != canonical_hash(model['inventory'])
                or original['initial_state_sha256'] != canonical_hash([model['common_shared_fingerprint'],
                                                                        model['common_private_fingerprint']])):
            raise ValueError('MIX_BENCHMARK_REAL_MODEL_GRAPH_MISMATCH')
        engine = scale20k_training.Engine(ARM,graphs,params,frozen,datasets,banks,Tokenizer(token['path']),
                                           dict.fromkeys(DOMAINS,size),card['optimizer'],
                                           rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED))
        del params
        engine.profile_timing = True
        result['model_setup_s'] = time.monotonic()-load_started
        restore_started = time.monotonic()
        state,engine.step,positions = checkpoint.restore(parent,arm=ARM,parameters=engine.parameters,
            optimizer=engine.states,rng=engine.rng,binding=original)
        if engine.step != START or positions != cursors(ARM,START):
            raise ValueError('MIX_BENCHMARK_PARENT_ADAM_AND_CURSORS_MUST_BE_EXACT')
        engine.parameters,engine.states,engine.rng = state['parameters'],state['optimizer'],state['rng']
        jax.block_until_ready((engine.parameters,engine.states,engine.rng))
        del state
        result['restore_parent_s'] = time.monotonic()-restore_started
        result['visit_banks_sha256'] = ids
        result['learning_rates_sha256'] = bank_sha256(engine.rates)
        binding = dict(original,source_sha=spec['source_sha'],engineering_only=True,
                       engineering_contract_sha256=spec['engineering_contract']['sha256'],variant=variant)
        interrupted = False
        with (root/'UPDATES.private.jsonl').open('x') as log:
            for _ in range(UPDATES):
                try:
                    row = engine.update(guard)
                except TimeoutError:
                    interrupted = True
                    break
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                rows.append(row)
                report('ENGINEERING_UPDATE_COMMITTED',actual_updates=len(rows),global_reference_step=engine.step)
        if rows:
            started = time.monotonic()
            result['engineering_checkpoint'] = checkpoint.save(root/'ENGINEERING_ONLY_COMPLETE',arm=ARM,
                step=engine.step,parameters=engine.parameters,optimizer=engine.states,rng=engine.rng,binding=binding)
            result['save_s'] = time.monotonic()-started
        if interrupted:
            result.update(status='ENGINEERING_BOUNDED_STOP',exit_code=75)
        else:
            result.update(status='ENGINEERING_30_COMPLETE',exit_code=0,measured_20=summarize(rows))
    except BaseException:
        result.update(status='ENGINEERING_TECHNICAL_TERMINAL',exit_code=1,failure=traceback.format_exc())
    finally:
        lock.close()
    before,after = cursors(ARM,START),cursors(ARM,START+len(rows))
    result.update(actual_optimizer_updates=len(rows),actual_optimization_visits={d:after[d]-before[d] for d in DOMAINS},
                  actual_aux_generations=engine.completed_aux if engine else 0,
                  attempted_aux_generations=engine.attempted_aux if engine else 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=str(args.spec),sha256=args.sha256))
    result = benchmark(spec)
    write(Path(spec['registered_root'])/'RESULT.private.json',result)
    raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
