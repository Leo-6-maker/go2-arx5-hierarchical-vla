"""Run one finite segment of the continuous SCALE target arm on VANDA."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_checkpoint import read_complete
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN, RECIPE, cursors
from go2arx5_vla.sla_adapter.scale20k_model import parent_asset
from scripts.pi0292.durable import OwnerLock, write
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job, pin


ARM = 'SCALE-TGT5K'
WALL = 10800


def completed_prefix(config, root):
    """Verify completed updates since the exact previous checkpoint."""
    continuation = config.get('continuation')
    if continuation is None:
        return None, None
    prior = continuation['train_result']
    result_path = Path(prior['path']).resolve(strict=True)
    previous = result_path.parent.parent
    if (result_path != previous/'train/RESULT.private.json' or previous == root
            or previous.stat().st_uid != os.getuid()):
        raise ValueError('SCALE_TARGET_PREFIX_MUST_BE_AN_OWNED_TERMINAL_RUN')
    old_job = continuation['job_id']
    status = subprocess.check_output(['/opt/pbs/bin/qstat','-x','-f','-F','json',old_job],
                                     text=True,stderr=subprocess.PIPE,timeout=20)
    job = json.loads(status)['Jobs'][old_job]
    if (job['job_state'] != 'F' or job.get('Exit_status') not in (0,75) or old_job == config.get('job_id')
            or job['Job_Owner'].split('@')[0] != config['user'] or job['project'] != config['project']):
        raise PermissionError('SCALE_TARGET_ORIGINAL_PBS_PREFIX_MUST_HAVE_FINISHED_AND_SAVED')
    parent = read_pin(continuation['campaign_result'])
    if (pin(previous/'RESULT.private.json') != continuation['campaign_result'] or parent.get('first_failure')
            or parent['PBS_JOBID'] != old_job or parent['stages'].get('train') != prior):
        raise ValueError('SCALE_TARGET_ORIGINAL_PBS_TERMINAL_MUST_BIND_TRAIN_RESULT')
    training_sha = parent.get('training_source_sha',parent['source_sha'])
    old_spec = read_pin(continuation['training_spec'])
    if pin(previous/'train/SPEC.private.json') != continuation['training_spec']:
        raise ValueError('SCALE_TARGET_ORIGINAL_PREFIX_TRAIN_CARD_CHANGED')
    result = read_pin(prior)
    ancestor = old_spec.get('resume_from_closed_segment')
    start = read_pin(ancestor)['global_step'] if ancestor is not None else 0
    step = result['global_step']
    limit = old_spec['global_stop_step']
    if (not 0 <= start < step <= limit <= CHECKPOINTS[ARM][-1] or (not start and limit != 200)
            or (result['status'],result['exit_code'],job['Exit_status']) not in (
                ('FIXED_SEGMENT_SAVED',0,0),('BOUNDED_SAVED_PREFIX',75,75))
            or (result['exit_code'] == 0 and step != limit)):
        raise ValueError('SCALE_TARGET_REQUIRES_A_BOUNDED_SAVED_TRAINING_SEGMENT')
    before, after = cursors(ARM,start), cursors(ARM,step)
    expected = dict(scope=RECIPE,mode='train',arm=ARM,source_sha=training_sha,global_cursors=after,
        new_updates=step-start,new_visits={d:after[d]-before[d] for d in DOMAINS},
        new_native_requests=0,new_native_ticks=0)
    completed_generation_floor = 4*(step-start)
    extra_generations = 4 if result['exit_code'] == 75 else 0
    if (any(result[key] != value for key,value in expected.items()) or old_spec['source_sha'] != training_sha
            or old_spec['arm'] != ARM
            or not completed_generation_floor <= result['new_model_generations'] <= completed_generation_floor+extra_generations
            or old_spec['microbatch'] != dict.fromkeys(DOMAINS,1)):
        raise ValueError('SCALE_TARGET_COMPLETE_TRAINING_EFFECTS_MUST_RECONCILE')
    with (previous/'train/UPDATES.private.jsonl').open('rb') as ledger:
        rows = ledger.readlines()
    if len(rows) != step-start or any(not line.endswith(b'\n') for line in rows):
        raise ValueError('SCALE_TARGET_PREFIX_REQUIRES_ALL_DURABLE_ROWS')
    for expected_step, line in enumerate(rows,start+1):
        row = json.loads(line)
        if (row['step'] != expected_step or row['cursors'] != cursors(ARM,expected_step)
                or set(row['domains']) != {'target'} or row['domains']['target']['training_full_generations'] != 4):
            raise ValueError('SCALE_TARGET_PREFIX_DURABLE_VISITS_OR_GENERATIONS_CHANGED')
    pin_state = result['checkpoint']
    model = read_pin(continuation['model_at_start'])
    if pin(previous/'train/MODEL_AT_START.private.json') != continuation['model_at_start']:
        raise ValueError('SCALE_TARGET_PREFIX_STARTING_IDENTITY_CHANGED')
    checkpoint, manifest = read_complete(pin_state,ARM,model['binding'])
    if checkpoint != previous/'train'/f'step{step:05d}' or manifest['step'] != step or len(manifest['arrays']) != 140:
        raise ValueError('SCALE_TARGET_PREFIX_COMPLETE_CHECKPOINT_REQUIRED')
    worker = Path(continuation['worker_root']).resolve(strict=True)
    members = read_pin(continuation['worker_files'])
    if ((worker/'SOURCE_SHA.private').read_text().strip() != training_sha
            or not {'scripts/sla/train_scale20k.py','src/go2arx5_vla/sla_adapter/scale20k_training.py'}.issubset(members)):
        raise ValueError('SCALE_TARGET_PREFIX_REQUIRES_ORIGINAL_TRAINER_SOURCE')
    for relative,digest in members.items():
        path = (worker/relative).resolve(strict=True)
        if not path.is_relative_to(worker) or sha(path) != digest:
            raise ValueError('SCALE_TARGET_PREFIX_ORIGINAL_TRAINER_BYTES_CHANGED:'+relative)
    return prior, (worker,old_spec)


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    if (root.stat().st_uid != os.getuid() or (root/'RESULT.private.json').exists()
            or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or config['plan_sha'] != PLAN or config['arm'] != ARM):
        raise PermissionError('SCALE_TARGET_FRESH_PBS_EXECUTION_IDENTITY_REQUIRED')
    lock = OwnerLock(root/'PBS_OWNER.lock')
    wall_limit = config.get('max_walltime_s',WALL)
    stages = {}
    status = 'PBS_PREFLIGHT_FAILED_NO_WORKER_START'
    current = None
    failure = None
    training_source_sha = config['source_sha']
    stop = root/'STOP.json'
    cancelled = False

    def cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancel)
    try:
        job_id, window = pbs_job(config)
        deadline = window['deadline_epoch']
        if (type(wall_limit) is not int or not WALL <= wall_limit <= 21600
                or cancelled or stop.exists() or window['PBS_wall_s'] > wall_limit
                or not 900 < deadline-time.time() <= wall_limit):
            status = 'PBS_FINITE_START_WINDOW_ELAPSED_NO_WORKER_START'
            return 75
        custody = read_pin(config['data_custody'])
        data_proof = read_pin(config['data_cpu_proof'])
        if (data_proof['custody_sha256'] != config['data_custody']['sha256']
                or data_proof['visit_bank_sha256'] != custody['visit_bank_sha256']
                or data_proof['lr_sha256'] != custody['learning_rates_sha256']
                or data_proof['windows_by_domain'] != dict(target=1768, robodojo=50979, libero=209146)
                or data_proof['status'] != 'VANDA_MATCHED_TARGET_CPU_DATA_VERIFIED'):
            raise ValueError('SCALE_TARGET_REQUIRES_MATCHED_CROSS_SITE_DATA_PROOF')
        parent_asset(custody['parent_manifest'], custody['parent_params'])
        prior, resume = completed_prefix(config, root)
        first_step = read_pin(prior)['global_step'] if prior else 0
        stop_step = config.get('global_stop_step',500 if resume else 200)
        if (type(stop_step) is not int or not first_step < stop_step <= CHECKPOINTS[ARM][-1]
                or (not resume and stop_step != 200)):
            raise ValueError('SCALE_TARGET_FINITE_ABSOLUTE_STOP_STEP_REQUIRED')
        worker = source
        if resume is not None:
            worker, old_spec = resume
            if any(old_spec[key] != custody[key] for key in ('data','parent_manifest','optimizer','tokenizer',
                                                             'visit_bank_sha256','learning_rates_sha256')):
                raise ValueError('SCALE_TARGET_PREFIX_IMMUTABLE_INPUTS_CHANGED')
            training_source_sha = old_spec['source_sha']

        uuid, name, capacity, free = actual_gpu()
        fraction = .65
        allocator = math.ceil(capacity*fraction)
        if free <= allocator+2*GIB or shutil.disk_usage(root).free <= 24*GIB:
            raise PermissionError('SCALE_TARGET_ACTUAL_GPU_AND_STORAGE_HEADROOM_REQUIRED')
        cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-scale20k-target-compile'
        cache.mkdir()
        dataset = custody['data']['target']['root']
        paths = dict(parent_params=str(Path(custody['parent_params']).resolve(strict=True)),
                     dataset=str(Path(dataset).resolve(strict=True)), compilation_cache_dir=str(cache))
        allocation = root/'PBS_ALLOCATION_OBSERVED.private.json'
        host = os.uname().nodename
        write(allocation,dict(job_id=job_id,host=host,uid=os.getuid(),project=config['project'],state='R',
            allocated_GPU_uuids=[uuid], GPU_free_bytes=free, PID=os.getpid(), observed_epoch=time.time(), **window))
        site = root/'SITE_BINDING.private.json'
        write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=host,uid=os.getuid(),GPU_uuid=uuid,GPU_name=name,
            allocation_deadline_epoch=window['allocation_deadline_epoch'],allocator_limit_bytes=allocator,
            GPU_headroom_bytes=2*GIB,host_headroom_bytes=32*GIB,paths=paths,allocation=dict(kind='PBS',job_id=job_id,
                project=config['project'],actual_allocation=pin(allocation))))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_FLAGS=config['XLA_FLAGS'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if resume is not None:
            inherited = [value for value in env.get('PYTHONPATH','').split(os.pathsep) if value
                         and not Path(value).resolve().is_relative_to(source)]
            env['PYTHONPATH'] = os.pathsep.join([str(worker),str(worker/'src'),*inherited])
        common = dict(arm=ARM,plan_sha=PLAN,recipe=RECIPE,source_sha=training_source_sha,
            global_stop_step=stop_step,microbatch=dict.fromkeys(DOMAINS,1),deadline_epoch=deadline,
            max_walltime_s=wall_limit,max_gpu_hours=wall_limit/3600,save_and_exit_on_limit=True,save_reserve_s=300,
            model_service_timeout_s=1800,RPC_timeout_s=1800,hardware_allowed=False,physics=False,
            new_native_requests_cap=0,STOP_path=str(stop),neural_owner_lock=str(root/'GPU_WORKER.lock'),
            GPU_uuid=uuid,JAX_memory_fraction=fraction,site_binding=pin(site),**paths,
            **{key:custody[key] for key in ('data','parent_manifest','optimizer','tokenizer',
                                           'visit_bank_sha256','learning_rates_sha256')})
        for mode in (('train',) if resume else ('qualify','train')):
            if cancelled or stop.exists() or time.time()+600 >= deadline:
                status = 'STOPPED_AT_COMPLETED_STAGE'
                return 75
            status = 'UNRESOLVED_GPU_EFFECT_NO_AUTOMATIC_REPLAY'
            stage = root/mode
            stage.mkdir()
            card = stage/'SPEC.private.json'
            write(card,dict(common,registered_root=str(stage),resume_from_closed_segment=prior))
            spec = pin(card)
            current = mode
            with (stage/'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable,'-B','-m','scripts.sla.train_scale20k',mode,
                    spec['path'],spec['sha256']],cwd=worker,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(root/'PROGRESS.private.json',dict(stage=mode,child_PID=child.pid,job_id=job_id,
                                                       spec_sha256=spec['sha256'],epoch=time.time()))
                while child.poll() is None:
                    if cancelled and not stop.exists():
                        write(stop,dict(reason='PBS_OPERATOR_STOP',epoch=time.time()))
                    time.sleep(2)
            result = stage/'RESULT.private.json'
            if not result.is_file():
                status = 'UNRESOLVED_GPU_EFFECT_NO_AUTOMATIC_REPLAY'
                return 1
            prior = pin(result)
            stages[mode] = prior
            observed = read_pin(prior)
            status = observed['status']
            failure = observed.get('failure')
            if child.returncode or observed['exit_code']:
                return child.returncode or observed['exit_code']
        return 0
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(root/'RESULT.private.json',dict(status=status,stages=stages,last_stage=current,
            first_failure=failure,source_sha=config['source_sha'],PBS_JOBID=os.environ.get('PBS_JOBID'),
            training_source_sha=training_source_sha,new_native_requests=0,new_native_ticks=0))
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
