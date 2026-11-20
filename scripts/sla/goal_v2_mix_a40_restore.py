"""Qualify a full archived MIX state on a real A40 without applying Adam."""
import argparse
import hashlib
import io
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter import scale20k_checkpoint, scale20k_data, scale20k_model, scale20k_training
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, PLAN, RECIPE, bank_sha256, canonical_hash, cursors
from scripts.pi0292.durable import OwnerLock, write
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job


GOAL = '1b2a25b7c1720051a2627d30831fb18130462374'
TRAINER = '31076fc7a64302b0705ae4d7e526c4341209a100'
CHECKPOINT = '4f4ade751809b375a484a9ba4e4bcb5ad899edfbbe2135b4ac62c41631b76093'
ARM = 'SCALE-MIX20K'
STEP = 500
WALL = 1800


def archived_inputs(config):
    held, assets = map(read_pin,(config['portable_custody'],config['site_assets']))
    checkpoint, spec = held['checkpoint'], read_pin(held['training_spec'])
    export = read_pin(held['transfer'])
    if (held['arm']!=ARM or held['step']!=STEP or held['training_source_sha']!=TRAINER
            or spec['source_sha']!=TRAINER or spec['recipe']!=RECIPE or spec['arm']!=ARM
            or spec['plan_sha']!=PLAN or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or checkpoint['sha256']!=CHECKPOINT or export['step']!=STEP or export['source_sha']!=TRAINER
            or export['members']['checkpoint/COMPLETE.json']['sha256']!=CHECKPOINT
            or any(spec[key]['sha256']!=assets[key]['sha256'] for key in ('parent_manifest','tokenizer'))
            or spec['data']['target']['admission']['sha256']!=assets['data']['target']['admission']['sha256']
            or any(spec['data'][d]['manifest']['sha256']!=assets['data'][d]['manifest']['sha256'] for d in DOMAINS)):
        raise ValueError('MIX_A40_RESTORE_ORIGINAL_CUSTODY_OR_SITE_INPUTS_CHANGED')
    complete = read_pin(dict(path=str(Path(checkpoint['path'])/'COMPLETE.json'),sha256=CHECKPOINT))
    binding = complete['binding']
    expected = dict(source_sha=TRAINER,parent_manifest_sha256=spec['parent_manifest']['sha256'],
        optimizer_sha256=canonical_hash(spec['optimizer']),tokenizer_sha256=spec['tokenizer']['sha256'],
        learning_rates_sha256=spec['learning_rates_sha256'],visit_bank_sha256=spec['visit_bank_sha256'],
        target_admission_sha256=spec['data']['target']['admission']['sha256'],
        data_manifest_sha256={d:spec['data'][d]['manifest']['sha256'] for d in DOMAINS})
    if any(binding[k]!=value for k,value in expected.items()):
        raise ValueError('MIX_A40_RESTORE_BINDING_NOT_FROM_ORIGINAL_RECIPE')
    local = {key:assets[key] for key in ('data','parent_manifest','parent_params','dataset','tokenizer')}
    return checkpoint, dict(spec, **local), binding


def exact_device_arrays(state, manifest):
    """Verify serialization from restored device state against the sealed arrays."""
    import jax

    tree = scale20k_checkpoint.training_tree(state['parameters'],state['optimizer'],state['rng'])
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    count = 0
    for (key,leaf), row in zip(leaves,manifest['arrays'],strict=True):
        if jax.tree_util.keystr(key) != row['key'] or leaf.device.platform != 'gpu':
            raise ValueError('MIX_A40_RESTORE_DEVICE_OR_LEAF_IDENTITY_CHANGED')
        with io.BytesIO() as stream:
            np.save(stream,np.asarray(leaf),allow_pickle=False)
            if hashlib.sha256(stream.getbuffer()).hexdigest()!=row['sha256']:
                raise ValueError('MIX_A40_RESTORE_DEVICE_ROUNDTRIP_MISMATCH:'+row['file'])
        count += 1
    return count


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(config['worker_root']).resolve(strict=True)
    if (root.stat().st_uid!=os.getuid() or (root/'RESULT.private.json').exists()
            or config['job_id']!=os.environ.get('PBS_JOBID') or config['goal']!=GOAL or config['plan']!=PLAN
            or config['arm']!=ARM or config['step']!=STEP or config['worker_sha']!=TRAINER
            or (source/'SOURCE_SHA.private').read_text().strip()!=TRAINER
            or not Path(scale20k_training.__file__).resolve().is_relative_to(source)
            or sha(Path(__file__))!=config['program_sha256']):
        raise PermissionError('MIX_A40_RESTORE_REQUIRES_FRESH_ORIGINAL_TRAINING_SOURCE')
    lock = OwnerLock(root/'PBS_OWNER.lock')
    start = time.monotonic()
    terminal = dict(schema='SLA_GOAL_V2_MIX_A40_FULL_STATE_RESTORE_V1', job=config['job_id'],
        source_sha=config['source_sha'], trainer_sha=TRAINER, optimizer_updates=0,
        optimization_visits=0, training_full_generations=0, native_requests=0, native_ticks=0,
        status='MIX_A40_RESTORE_NOT_QUALIFIED', graduation_scope='MIX500_WITNESS_ONLY', exit_code=1)
    engine = None
    stopped = False

    def signal_stop(*_):
        nonlocal stopped
        stopped = True

    for kind in (signal.SIGINT,signal.SIGTERM):
        signal.signal(kind,signal_stop)

    def guard():
        if stopped or any(Path(s).exists() for s in config['stop_files']) or time.time()+120>=deadline:
            raise TimeoutError('MIX_A40_RESTORE_STOP_OR_SITE_LIMIT')

    def progress(stage):
        write(root/'PROGRESS.private.json',dict(terminal,stage=stage,epoch=time.time()))

    try:
        _, allocation = pbs_job(config)
        deadline = allocation['deadline_epoch']
        if allocation['PBS_wall_s']>WALL or not 450<deadline-time.time()<=WALL:
            raise PermissionError('MIX_A40_RESTORE_REQUIRES_CURRENT_BOUNDED_PBS_WINDOW')
        guard()
        checkpoint, training, binding = archived_inputs(config)
        _, manifest = scale20k_checkpoint.read_complete(checkpoint,ARM,binding)
        if len(manifest['arrays'])!=140 or training['microbatch']!=dict.fromkeys(DOMAINS,1):
            raise ValueError('MIX_A40_RESTORE_REQUIRES_ALL_140_ORIGINAL_ARRAYS_AND_MICROBATCH')
        gpu,name,total,free = actual_gpu()
        fraction = .65
        reserve = math.ceil(total*fraction)
        if not reserve+2*GIB<free:
            raise PermissionError('MIX_A40_RESTORE_ACTUAL_GPU_HEADROOM_UNAVAILABLE')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,JAX_PLATFORMS='cuda',XLA_FLAGS=config['XLA_FLAGS'],
            XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

        import jax
        from openpi_client.image_tools import resize_with_pad

        if len(jax.devices())!=1 or jax.default_backend()!='gpu':
            raise PermissionError('MIX_A40_RESTORE_REQUIRES_THE_ASSIGNED_JAX_DEVICE')
        cache = Path(os.environ['TMPDIR'])/'sla-mix-a40-restore-compile'
        cache.mkdir()
        jax.config.update('jax_compilation_cache_dir',str(cache))
        terminal.update(checkpoint=checkpoint, prior_step=STEP, allocation=allocation,GPU_uuid=gpu,GPU_name=name,
            GPU_total_bytes=total,GPU_free_bytes=free,allocator_bytes=reserve,deadline_epoch=deadline)
        progress('READING_ORIGINAL_TRAIN_ASSETS')
        data = scale20k_data.readers(training['data'],resize_with_pad)
        banks, digests = scale20k_data.bank_for(data)
        token = training['tokenizer']
        if digests!=binding['visit_bank_sha256'] or sha(token['path'])!=binding['tokenizer_sha256']:
            raise ValueError('MIX_A40_RESTORE_ORIGINAL_BANK_OR_TOKENIZER_CHANGED')
        graphs, parameters, frozen, receipt = scale20k_model.load(training['parent_manifest'],training['parent_params'])
        if (binding['partition_sha256']!=canonical_hash(receipt['inventory'])
                or binding['initial_state_sha256']!=canonical_hash([receipt['common_shared_fingerprint'],
                                                                    receipt['common_private_fingerprint']])):
            raise ValueError('MIX_A40_RESTORE_ORIGINAL_PARENT_OR_PARTITION_CHANGED')
        engine = scale20k_training.Engine(ARM,graphs,parameters,frozen,data,banks,Tokenizer(token['path']),
            training['microbatch'],training['optimizer'],rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED))
        del parameters
        if bank_sha256(engine.rates)!=binding['learning_rates_sha256']:
            raise ValueError('MIX_A40_RESTORE_ORIGINAL_LR_CHANGED')
        guard()
        progress('RESTORING_PARAMETERS_ADAM_RNG_ON_A40')
        restored, step, positions = scale20k_checkpoint.restore(checkpoint,arm=ARM,parameters=engine.parameters,
            optimizer=engine.states,rng=engine.rng,binding=binding)
        if step!=STEP or positions!=cursors(ARM,STEP):
            raise ValueError('MIX_A40_RESTORE_GLOBAL_STEP_OR_DOMAIN_CURSOR_CHANGED')
        restored_arrays = exact_device_arrays(restored,manifest)
        engine.parameters,engine.states,engine.rng = (restored[k] for k in ('parameters','optimizer','rng'))
        engine.step = step
        bound_state = engine.parameters,engine.states,engine.rng
        terminal.update(device_arrays_exact=restored_arrays,restored_cursors=positions,
                        next_original_learning_rate=float(engine.rates[step]),domains={})
        for domain in DOMAINS:
            guard()
            progress('MEASURING_ORIGINAL_GRADIENT:'+domain)
            gradient, metrics = engine.domain_gradient(domain,guard)
            jax.block_until_ready(gradient)
            terminal['domains'][domain] = metrics
            del gradient
        if (restored_arrays!=140 or engine.step!=STEP
                or any(a is not b for a,b in zip(bound_state,(engine.parameters,engine.states,engine.rng),strict=True))
                or engine.completed_aux!=4 or engine.gradient_windows!=engine.batch_size):
            raise ValueError('MIX_A40_RESTORE_CHANGED_OPTIMIZER_OR_GRADIENT_DENOMINATOR')
        terminal.update(status='ACTUAL_A40_MIX500_FULL_STATE_AND_NEXT_GRADIENT_QUALIFIED',exit_code=0,
                        actual_GPU_memory_stats=jax.devices()[0].memory_stats())
        return 0
    except (Exception,KeyboardInterrupt) as error:
        terminal['first_failure'] = f'{type(error).__name__}: {error}'
        return 1
    finally:
        terminal.update(wall_s=time.monotonic()-start,gradient_input_windows=engine.gradient_windows if engine else {},
            attempted_H40=engine.attempted_aux if engine else 0,training_full_generations=engine.completed_aux if engine else 0)
        write(root/'RESULT.private.json',terminal)
        lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
